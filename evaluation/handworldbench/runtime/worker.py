"""Persistent model process with task-correlated atomic responses."""
import argparse
import os
from pathlib import Path
import time
import traceback
from handworldbench.runtime.storage import read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--slot',required=True);args=p.parse_args()
    run=Path(args.run_dir);slot=run/'workers'/args.slot;models={};last=None;task_stamp=None
    config=read(run/'configs/run.json')['runtime']
    interval=max(.02,float(config.get('worker_poll_seconds',.05)))
    os.environ['HANDWORLDBENCH_RUNTIME_CONFIG']=str(run/'configs/run.json')
    import sys
    sys.path[:0]=config.get('python_paths',[])
    write(slot/'ready.json',dict(pid=os.getpid()))
    while not (slot/'stop.json').exists():
        if not (slot/'task.json').exists():time.sleep(interval);continue
        stamp=(slot/'task.json').stat().st_mtime_ns
        if stamp==task_stamp:time.sleep(interval);continue
        task=read(slot/'task.json');key=(task['id'],task['attempt'])
        task_stamp=stamp
        if key==last:time.sleep(interval);continue
        last=key;start=time.monotonic()
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():torch.cuda.reset_peak_memory_stats()
        try:
            from handworldbench.runtime.execute import execute
            value=execute(task,run,models)
            result=dict(status='ok',value=value)
        except Exception as exc:
            status='input_binding_error' if isinstance(exc,ValueError) and str(exc).startswith('Input binding error:') else 'execution_error'
            result=dict(status=status,reason=str(exc),traceback=traceback.format_exc(),score_0_1=None)
            print(result['traceback'],flush=True)
        result.update(task_id=task['id'],attempt=task['attempt'],seconds=time.monotonic()-start)
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            result.update(torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write(slot/'response.json',result)

if __name__=='__main__':main()
