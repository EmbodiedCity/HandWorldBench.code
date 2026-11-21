"""Deterministic resident-worker scheduling over selected visible devices."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from handworldbench.runtime.storage import read,write
from handworldbench.runtime.queue import connect,ready_jobs,priority,remaining_paths

FAMILIES={
    'frames':'prepare','reference_frames':'prepare','reference_geometry':'prepare',
    'hand_detection':'hands','hand_keypoints':'hands','hand_masks':'segmentation',
    'entity_features':'regions','object_tracks':'regions','image_quality':'musiq',
    'visual_judgment':'pavrm','video_features':'jepa','hand_anatomy':'anatomy',
    'background_features':'background','background_geometry':'photo',
    'action_evidence':'judge','event_evidence':'judge','causal_evidence':'judge',
    'state_evidence':'state','memory_evidence':'state','score':'prepare',
}
MEMORY={'prepare':0,'hands':6000,'segmentation':14000,'vision':18000,'regions':36000,'musiq':8000,'pavrm':76000,'jepa':16000,'anatomy':16000,'background':18000,'photo':26000,'judge':30000,'state':40000}

def evictions_for_group(idle,group,free_memory,required_memory):
    """Keep idle models resident when a new backend fits without eviction."""
    if all(free_memory.get(d['uuid'],d['free_mib'])>=required_memory for d in group):
        return []
    return idle

def active(run):
    marker=Path(run)/'scheduler.json'
    if not marker.exists():return None
    doc=read(marker)
    try:
        cmd=Path(f"/proc/{doc['pid']}/cmdline").read_bytes().decode(errors='replace')
        if 'handworldbench.runtime.scheduler' in cmd and str(Path(run).resolve()) in cmd:return doc['pid']
    except FileNotFoundError:pass
    return None

def launch(run,stage,foreground=False,retry=False):
    run=Path(run).resolve()
    with (run/'launch.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        pid=active(run)
        if pid:return dict(status='already_running',pid=pid)
        if retry:
            from handworldbench.runtime.queue import retry_failed
            with connect(run) as connection:retry_failed(connection)
        if stage=='score':
            with connect(run) as c:
                missing=[r['id'] for r in c.execute("SELECT * FROM jobs WHERE state!='done' AND state!='superseded'") if json.loads(r['payload'])['kind']=='observe']
            if missing:raise ValueError('Missing observations for score: '+', '.join(missing[:20]))
        cmd=[sys.executable,'-m','handworldbench.runtime.scheduler','--run-dir',str(run),'--stage',stage]
        if foreground:os.execv(sys.executable,cmd)
        with (run/'logs/scheduler.log').open('a') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(run/'scheduler.json',dict(pid=process.pid,stage=stage,started=time.time()))
        return dict(status='started',pid=process.pid,stage=stage)

def run_queue(run,stage):
    run=Path(run).resolve()
    lock=(run/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=read(run/'configs/run.json');runtime=settings['runtime'];devices=settings['devices']
    pool=[]
    for device in sorted(devices,key=lambda d:d['uuid']):
        folder=Path(runtime['cache_root'])/'device_locks';folder.mkdir(parents=True,exist_ok=True)
        guard=(folder/device['uuid']).open('a')
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A selected GPU is reserved by another HandWorldBench run: '+device['uuid'])
        pool.append(guard)
    write(run/'scheduler.json',dict(pid=os.getpid(),stage=stage,started=time.time()))
    c=connect(run);workers={};sequence=0;draining=False;last_monitor=0;last_status=0;last_memory=0;free_memory={};costs={};resource_wait={}
    interval=max(.05,float(runtime.get('scheduler_poll_seconds',.1)))
    def stop_signal(*_):
        nonlocal draining
        draining=True
    signal.signal(signal.SIGTERM,stop_signal)
    log=(run/'logs/events.jsonl').open('a',buffering=1)
    def event(kind,**data):log.write(json.dumps(dict(time=time.time(),event=kind,**data))+'\n')
    for marker in (run/'workers').glob('*/process.json'):
        process=read(marker);pid=process['pid']
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        except FileNotFoundError:continue
        if 'handworldbench.runtime.worker' in cmd and str(run) in cmd:
            os.kill(pid,signal.SIGTERM)
            deadline=time.monotonic()+20
            while Path(f'/proc/{pid}/cmdline').exists() and time.monotonic()<deadline:time.sleep(.2)
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
                if 'handworldbench.runtime.worker' in cmd and str(run) in cmd:os.kill(pid,signal.SIGKILL)
            except FileNotFoundError:pass
    for row in c.execute("SELECT * FROM jobs WHERE state='running'").fetchall():
        response=run/'workers'/str(row['worker'])/'response.json'
        answer=read(response) if response.exists() else None
        if answer and answer.get('task_id')==row['id'] and answer.get('attempt')==row['attempt']:
            c.execute('UPDATE jobs SET state=?,result=?,worker=NULL WHERE id=?',('done' if answer.get('status')=='ok' else 'pending',json.dumps(answer),row['id']))
        else:c.execute("UPDATE jobs SET state='pending',worker=NULL WHERE id=?",(row['id'],))
    c.commit()
    graph_tasks=[]
    for row in c.execute('SELECT payload FROM jobs'):
        task=json.loads(row['payload'])
        graph_tasks.append({key:task[key] for key in ('id','backend','dependencies','estimated_seconds') if key in task})
    def retire(worker):
        write(worker['directory']/'stop.json',{})
        try:worker['process'].wait(timeout=20)
        except subprocess.TimeoutExpired:
            worker['process'].terminate()
            try:worker['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:worker['process'].kill();worker['process'].wait()
        workers.pop(worker['id'],None)
        event('worker_retired',worker=worker['id'])
    def spawn(family,device,group=None):
        nonlocal sequence
        group=group or [device]
        sequence+=1;wid=f"{os.getpid()}_{sequence}_{family}_{device['index']}"
        folder=run/'workers'/wid;folder.mkdir()
        env=os.environ.copy();env.update(runtime.get('env',{}));env['CUDA_VISIBLE_DEVICES']=','.join(d['uuid'] for d in group) if family!='prepare' else '';env['PYTHONUNBUFFERED']='1'
        env.update(runtime.get('backend_env',{}).get(family,{}))
        env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
        python=runtime.get('python',{}).get(family,runtime.get('default_python',sys.executable))
        with (run/'logs'/f'{wid}.log').open('w') as output:
            process=subprocess.Popen([python,'-m','handworldbench.runtime.worker','--run-dir',str(run),'--slot',wid],env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        worker=dict(id=wid,directory=folder,process=process,family=family,device=device,devices=group,job=None)
        workers[wid]=worker;write(folder/'process.json',dict(pid=process.pid,run_dir=str(run),device=device,family=family))
        event('worker_started',worker=wid,pid=process.pid,device=device['uuid'],family=family)
        return worker
    while True:
        now=time.time()
        for worker in list(workers.values()):
            job=worker['job']
            if not job:
                if worker['process'].poll() is not None:workers.pop(worker['id'])
                continue
            response=worker['directory']/'response.json';answer=None
            if response.exists():
                stamp=response.stat().st_mtime_ns
                if stamp!=worker.get('response_stamp'):
                    answer=read(response)
                    worker['response_stamp']=stamp
            if answer and (answer.get('task_id'),answer.get('attempt'))!=(job['id'],job['attempt']):answer=None
            if answer is None and worker['process'].poll() is not None:
                answer=dict(status='execution_error',reason='worker_exit',exitcode=worker['process'].returncode,task_id=job['id'],attempt=job['attempt'])
            if answer is None:continue
            state='done' if answer['status']=='ok' else 'pending' if job['attempt']<job.get('retry_ceiling',3) and answer['status']!='input_binding_error' else 'failed'
            c.execute('UPDATE jobs SET state=?,worker=NULL,result=? WHERE id=?',(state,json.dumps(answer),job['id']));c.commit()
            write(run/'results'/job['id']/f"attempt_{job['attempt']}.json",answer)
            if state in ('done','failed'):write(run/'results'/job['id']/'result.json',answer)
            event('task_finished',task=job['id'],state=state,seconds=answer.get('seconds'),reason=answer.get('reason'))
            if state=='done':
                costs[job['backend']]=.8*costs.get(job['backend'],answer['seconds'])+.2*answer['seconds']
            worker['job']=None
            if state!='done':retire(worker)
        pending=ready_jobs(c,stage)
        states=dict(c.execute('SELECT id,state FROM jobs'))
        needed_families={FAMILIES[t['backend']] for t in graph_tasks if states.get(t['id']) in ('pending','running')}
        for worker in list(workers.values()):
            if not worker['job'] and worker['family']!='prepare' and worker['family'] not in needed_families:
                retire(worker)
        remaining=remaining_paths(graph_tasks,states,costs)
        resident_backends={backend for backend,family in FAMILIES.items() if any(w['family']==family for w in workers.values())}
        pending.sort(key=lambda pair:priority(pair[0],dict(pair[1],estimated_seconds=remaining[pair[1]['id']]),now,resident_backends))
        if pending and now-last_memory>=1:
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
                free_memory={r.split(',')[0].strip():float(r.split(',')[1]) for r in raw.splitlines()}
                last_memory=now
            except (subprocess.SubprocessError,ValueError):pass
        if not draining:
            for record,task in pending:
                family=FAMILIES[task['backend']]
                busy={d['uuid'] for w in workers.values() if w['job'] for d in w['devices']}
                choices=[]
                candidates=([dict(index=-i-1,uuid='cpu_'+str(i),physical_index=None,total_mib=0,free_mib=0) for i in range(runtime.get('cpu_workers',4))] if family=='prepare' else devices)
                requirement=runtime.get('backend_resources',{}).get(family,{})
                count=int(requirement.get('gpu_count',1));memory=float(requirement.get('memory_mib_per_gpu',MEMORY.get(family,16000)))
                if count<1:raise ValueError('Backend GPU count must be positive')
                for group in itertools.combinations(candidates,count):
                    ids={d['uuid'] for d in group}
                    if ids & busy:continue
                    resident=next((w for w in workers.values() if not w['job'] and {d['uuid'] for d in w['devices']}==ids and w['family']==family),None)
                    idle=[w for w in workers.values() if not w['job'] and {d['uuid'] for d in w['devices']} & ids]
                    reclaimable={d['uuid'] for w in idle for d in w['devices']}
                    if not resident and any(d['uuid'] not in reclaimable and free_memory.get(d['uuid'],d['free_mib'])<memory for d in group):continue
                    choices.append((costs.get(task['backend'],task.get('estimated_seconds',60))+(0 if resident else requirement.get('load_seconds',20)),tuple(d['index'] for d in group),resident,idle,group))
                if not choices:
                    resource_wait.setdefault(task['id'],now)
                    if not any(w['job'] for w in workers.values()) and now-resource_wait[task['id']]>=runtime.get('resource_wait_timeout_s',120):
                        answer=dict(status='execution_error',score_0_1=None,reason='Insufficient available GPU memory for '+family)
                        c.execute("UPDATE jobs SET state='failed',result=? WHERE id=?",(json.dumps(answer),task['id']));c.commit();event('resource_unavailable',task=task['id'],family=family)
                    continue
                _,_,worker,idle,group=min(choices,key=lambda x:(x[0],x[1]))
                if worker is None:
                    evictions=idle if record['attempt'] else evictions_for_group(idle,group,free_memory,memory)
                    for stale in evictions:retire(stale)
                    worker=spawn(family,group[0],group)
                task=dict(task,attempt=record['attempt']+1)
                task['dependency_results']={dep:json.loads(c.execute('SELECT result FROM jobs WHERE id=?',(dep,)).fetchone()[0])['value'] for dep in task['dependencies']}
                c.execute("UPDATE jobs SET state='running',attempt=?,worker=? WHERE id=?",(task['attempt'],worker['id'],task['id']));c.commit()
                worker['job']=task;write(worker['directory']/'task.json',task)
                event('task_started',task=task['id'],worker=worker['id'],attempt=task['attempt'])
        counts=dict(c.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        if now-last_status>=1:
            write(run/'reports/live_status.json',dict(time=now,stage=stage,tasks=counts,workers=[dict(id=w['id'],pid=w['process'].pid,device=w['device'],devices=w['devices'],task=w['job']['id'] if w['job'] else None) for w in workers.values()]))
            last_status=now
        if now-last_monitor>=1200:
            snapshot=read(run/'reports/live_status.json')
            try:snapshot['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
            except subprocess.SubprocessError as exc:snapshot['gpu_snapshot_error']=str(exc)
            with (run/'reports/monitor.jsonl').open('a') as monitor:monitor.write(json.dumps(snapshot)+'\n')
            last_monitor=now
        running=any(w['job'] for w in workers.values())
        if not running and (draining or not pending):break
        time.sleep(interval)
    for worker in list(workers.values()):retire(worker)
    write(run/'reports/live_status.json',dict(time=time.time(),stage=stage,tasks=counts,workers=[],status='drained' if draining else 'finished'))
    event('scheduler_stopped',tasks=counts,drained=draining)
    c.close()
    log.close()
    try:
        if stage!='observe':
            from handworldbench.reporting.report import report
            report([run])
    finally:
        for guard in pool:guard.close()
        lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--stage',default='evaluate',choices=['observe','score','evaluate'])
    args=parser.parse_args();run_queue(args.run_dir,args.stage)

