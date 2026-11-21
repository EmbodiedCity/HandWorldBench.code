"""Bounded metadata/RGB/export queues, one parent ledger writer.

Scheduling changes only: every source still uses the unchanged complete
temporal and RGB workers. No source/anchor is skipped to save compute.
"""
import collections
import concurrent.futures
import errno
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from .config import config_hash
from .resources import retryable_resource_error
from .storage import append_segment, append_jsonl, read_jsonl, atomic_json, publish_progress, utc_now, clip_id


def unique_sources(records, stage):
    result={}
    for record in records:
        path=record['source_path']
        if path in result:raise RuntimeError(f'duplicate {stage} source ledger: {path}')
        result[path]=record
    return result


def cache_limits(cfg):
    runtime=cfg['runtime'];rgb=cfg.get('rgb',{})
    limit=int(runtime.get('max_cache_gb',8)*1024**3)
    analysis=int(runtime.get('rgb_cache_max_bytes',rgb.get('max_cache_mb',512)*1024**2))
    export=int(min(rgb.get('max_cache_mb',512),
        runtime.get('max_cache_gb',8)*1024/max(1,runtime['workers']))*1024**2)
    if min(limit,analysis,export)<=0 or max(analysis,export)>limit:
        raise ValueError('streaming cache budget cannot accommodate one worker')
    return limit,analysis,export


def stop_owned_analysis(pool, run, cfg):
    """Stop this executor only; collect exact open scratch directories first.

    Never infer ownership from a directory prefix alone. Cleanup requires an
    open fd in an owned worker, persisted inode evidence and every worker dead.
    If evidence cannot be saved, leave scratch for explicit recovery.
    """
    processes=list((getattr(pool,'_processes',None) or {}).values())
    runtime=cfg['runtime'];base=runtime.get('scratch_dir',runtime.get('cache_root'))
    root=Path(base).resolve() if base else None
    owned={}
    if root:
        for process in processes:
            fdroot=Path('/proc')/str(process.pid)/'fd'
            try:fds=list(fdroot.iterdir())
            except OSError:continue
            for fd in fds:
                try:
                    target=Path(os.readlink(fd))
                    # gray.raw is held open throughout RGB analysis.
                    directory=target.parent
                    if directory.parent!=root or not directory.name.startswith('rgb_'):continue
                    if directory.is_symlink() or directory.resolve()!=directory:continue
                    stat=directory.stat()
                    row=owned.setdefault(str(directory),dict(path=str(directory),device=stat.st_dev,
                        inode=stat.st_ino,worker_pids=[]))
                    if process.pid not in row['worker_pids']:row['worker_pids'].append(process.pid)
                except OSError:continue
    evidence=dict(created_at=utc_now(),owned_worker_pids=[p.pid for p in processes],directories=list(owned.values()))
    saved=False
    try:atomic_json(run/'owned_scratch_stop_intent.json',evidence);saved=True
    except Exception:pass
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:process.join(timeout=5)
    for process in processes:
        if process.is_alive():process.kill();process.join(timeout=5)
    cleaned=[]
    if saved and all(not p.is_alive() for p in processes):
        for row in owned.values():
            directory=Path(row['path'])
            try:
                stat=directory.stat()
                if directory.is_symlink() or directory.resolve().parent!=root:continue
                if (stat.st_dev,stat.st_ino)!=(row['device'],row['inode']):continue
                shutil.rmtree(directory);cleaned.append(str(directory))
            except OSError:pass
    try:atomic_json(run/'owned_scratch_stop_result.json',dict(ended_at=utc_now(),cleaned=cleaned,
        all_workers_dead=all(not p.is_alive() for p in processes),intent_persisted=saved))
    except Exception:pass


def deliver_one(run, candidate, cfg):
    from .artifacts import export_one, verify_one
    from .pipeline import check_resources
    check_resources(cfg)
    cid=clip_id(candidate,config_hash(cfg))
    artifact=None
    try:
        artifact=export_one(run,candidate,cfg)
        return {'export':artifact,'verification':verify_one(artifact,cfg)}
    except Exception as error:
        # FFmpeg reports filesystem failures in stderr rather than errno.
        if retryable_resource_error(error):raise
        if any(s in str(error).lower() for s in ('no space left on device','disk quota exceeded','read-only file system')):
            raise OSError(errno.ENOSPC,str(error)) from error
        evidence=dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-2500:],updated_at=utc_now())
        if artifact is None:
            return {'export':dict(clip_id=cid,status='error',candidate=candidate,**evidence)}
        return {'export':artifact,'verification':dict(clip_id=cid,status='error',**evidence)}


def run_streaming(run, files, cfg):
    from . import pipeline, artifacts
    pipeline.check_resources(cfg)
    pipeline._incremental_state(run,cfg,status='scanning',streaming=True,last_batch_stage='stream')
    temporal=unique_sources(read_jsonl(run/'temporal.jsonl'),'temporal')
    rgb=unique_sources(read_jsonl(run/'rgb.jsonl'),'rgb')
    if set(rgb)-set(temporal):raise RuntimeError('RGB ledger lacks a temporal parent')
    metadata=collections.deque(p for p in files if p not in temporal)
    ready_rgb=collections.deque(x for p,x in temporal.items() if x['status']=='ok' and x.get('candidates') and p not in rgb)
    queued_rgb={x['source_path'] for x in ready_rgb}
    buffer=[]
    # Existing files are checked once on resume; newly created files are checked
    # before their verification record is committed. Neither path trusts a PID.
    if read_jsonl(run/'exports.jsonl'):
        artifacts.verify_exports(run,cfg)
    def refresh_verified():
        exports={x['clip_id']:x for x in read_jsonl(run/'exports.jsonl')}
        verification={x['clip_id']:x for x in read_jsonl(run/'verification.jsonl')}
        return {cid for cid,x in exports.items() if x.get('status')=='ok'
                and verification.get(cid,{}).get('status')=='ok'
                and verification[cid].get('video_sha256')==x.get('checksums',{}).get('video.mp4')}
    verified_ids=refresh_verified()
    workers=cfg['runtime']['workers']
    cache_budget,rgb_cache,export_cache=cache_limits(cfg)
    export_workers=min(workers,cfg['runtime'].get('export_workers',2))
    analysis_pool=concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    delivery_pool=concurrent.futures.ThreadPoolExecutor(max_workers=export_workers)
    analysis={};delivery={};delivery_ids=set()
    selected=[]
    last_flush=time.monotonic();last_publish=0.;last_report=0.
    last_result=utc_now();stop_new=False;pool_dirty=True;last_summary=None

    def flush_temporal():
        nonlocal last_flush
        if buffer:
            append_segment(run/'temporal.jsonl',list(buffer))
            buffer.clear()
        last_flush=time.monotonic()

    def reserved_cache():
        return sum(stage=='rgb' for stage,_ in analysis.values())*rgb_cache+len(delivery)*export_cache

    def publish(status='scanning',error=None,flush=True):
        nonlocal last_publish
        if flush:flush_temporal()
        state=dict(stage='stream',status=status,updated_at=utc_now(),last_result_at=last_result,
            pid=os.getpid(),source_total=len(files),temporal_completed=len(temporal),rgb_completed=len(rgb),
            sources_not_submitted=len(metadata),rgb_queued=len(ready_rgb),
            active_temporal=sum(s=='temporal' for s,_ in analysis.values()),
            active_rgb=sum(s=='rgb' for s,_ in analysis.values()),active_exports=len(delivery),
            independently_verified_exports=len(verified_ids),cache_reserved_bytes=reserved_cache(),
            cache_budget_bytes=cache_budget,error=error)
        publish_progress(run/'progress.json',state)
        print(json.dumps(state),flush=True)
        last_publish=time.monotonic()

    def fill_analysis():
        if stop_new:return
        pipeline.check_resources(cfg)
        while len(analysis)<workers:
            if ready_rgb and reserved_cache()+rgb_cache<=cache_budget:
                item=ready_rgb.popleft()
                analysis[analysis_pool.submit(pipeline.analyze_vision,item,cfg)]=('rgb',item['source_path'])
            elif metadata:
                path=metadata.popleft()
                analysis[analysis_pool.submit(pipeline.analyze_temporal,path,cfg)]=('temporal',path)
            else:break

    def schedule_exports():
        nonlocal pool_dirty,selected
        if pool_dirty:
            flush_temporal()
            selected=artifacts.select_export_candidates(run,cfg)
            atomic_json(run/'selected_machine_candidates.json',selected)
            pool_dirty=False
        pipeline.check_resources(cfg)
        for c in selected:
            cid=clip_id(c,config_hash(cfg))
            if cid in verified_ids or cid in delivery_ids:continue
            if len(delivery)>=export_workers:break
            if reserved_cache()+export_cache>cache_budget:break
            delivery[delivery_pool.submit(deliver_one,run,c,cfg)]=cid
            delivery_ids.add(cid)

    def make_report():
        nonlocal last_report,last_summary
        flush_temporal()
        pipeline._incremental_state(run,cfg,last_batch_stage='stream',
            batch_index=len(temporal)//cfg['runtime'].get('source_batch_size',256))
        last_summary=pipeline.report(run,cfg)
        if analysis and (pipeline.target_met(last_summary['review_ready_count'],last_summary['review_ready_length_histogram'],cfg) or last_summary['production_complete']):
            last_summary.update(execution_status='draining_for_review',awaiting_visual_review=False,production_complete=False)
            atomic_json(run/'summary.json',last_summary)
        last_report=time.monotonic()
        return last_summary

    try:
        publish()
        # A resume of an already fulfilled pool must not begin a new source wave.
        initial=make_report()
        if pipeline.target_met(initial['review_ready_count'],initial['review_ready_length_histogram'],cfg) or initial['production_complete']:
            pipeline._publish_incremental_summary(run,cfg,initial)
            return initial
        schedule_exports();fill_analysis()
        while True:
            pending=list(analysis)+list(delivery)
            if not pending:
                schedule_exports()
                if delivery:continue
                summary=make_report()
                if pipeline.target_met(summary['review_ready_count'],summary['review_ready_length_histogram'],cfg) or summary['production_complete']:
                    pipeline._publish_incremental_summary(run,cfg,summary)
                    return summary
                if stop_new:
                    stop_new=False
                    fill_analysis()
                    if analysis:continue
                if not metadata and not ready_rgb:
                    pipeline._publish_incremental_summary(run,cfg,summary)
                    return summary
                fill_analysis()
                continue
            completed,_=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            rgb_changed=False;delivery_changed=False
            for future in completed:
                if future in analysis:
                    stage,path=analysis.pop(future)
                    record=future.result()  # Retryable resource failures never commit this source.
                    if record.get('source_path')!=path:raise RuntimeError('worker source identity mismatch')
                    if stage=='temporal':
                        if path in temporal:raise RuntimeError('duplicate temporal commit')
                        temporal[path]=record;buffer.append(record)
                        if record['status']=='ok' and record.get('candidates'):
                            # RGB must never commit without a durable temporal parent.
                            flush_temporal()
                            if path in queued_rgb:raise RuntimeError('duplicate RGB scheduling')
                            ready_rgb.append(record);queued_rgb.add(path)
                    else:
                        if path in rgb:raise RuntimeError('duplicate RGB commit')
                        append_jsonl(run/'rgb.jsonl',record);rgb[path]=record
                        rgb_changed=True
                    last_result=utc_now()
                else:
                    cid=delivery.pop(future);delivery_ids.remove(cid)
                    outcome=future.result()
                    if outcome['export']['clip_id']!=cid:raise RuntimeError('export identity mismatch')
                    append_jsonl(run/'exports.jsonl',outcome['export'])
                    if 'verification' in outcome:
                        append_jsonl(run/'verification.jsonl',outcome['verification'])
                        if outcome['verification'].get('status')=='ok':verified_ids.add(cid)
                    delivery_changed=True;last_result=utc_now()
            if len(buffer)>=50 or time.monotonic()-last_flush>=30:flush_temporal()
            if rgb_changed or delivery_changed:pool_dirty=True
            # Give completed clips priority for the shared cache reservation;
            # otherwise a continuous RGB queue could starve export forever.
            schedule_exports()
            fill_analysis()
            if delivery_changed or time.monotonic()-last_report>=300:
                summary=make_report()
                stop_new=pipeline.target_met(summary['review_ready_count'],summary['review_ready_length_histogram'],cfg) or summary['production_complete']
                # Drain at most one bounded active wave. Any newly completed RGB
                # may change global selection; re-evaluate after draining before stopping.
            if time.monotonic()-last_publish>=30:publish('draining_for_review' if stop_new else 'scanning')
    except BaseException as error:
        # A write may have committed before readback raised EIO. Never append
        # the buffer again in the exception path; resume reads canonical state.
        try:publish('failed_recoverable',f'{type(error).__name__}: {error}',flush=False)
        except Exception:pass
        stop_owned_analysis(analysis_pool,run,cfg)
        raise
    finally:
        analysis_pool.shutdown(wait=True,cancel_futures=True)
        delivery_pool.shutdown(wait=True,cancel_futures=True)
