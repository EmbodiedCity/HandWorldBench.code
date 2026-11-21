"""Full-corpus, resumable automated curation with a single durable writer.

New user policy: no atomic-event gate, no global count/length cap. Every selected
clip still passes the original numerical camera/hand/RGB gates, actual model
scoring, exact source mapping, lossless export and independent validation.
"""
import argparse
import collections
import concurrent.futures as futures
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import select
import subprocess
import sys
import time
import traceback

import yaml
import numpy as np

from .config import config_hash,code_manifest
from .storage import read_json,read_jsonl,atomic_json,atomic_bytes,append_segment,append_jsonl,exclusive_writer,sha256_file,clip_id,utc_now
from .pipeline import check_resources,stratified_inventory,_source_fields
from .resources import retryable_resource_error
from .length_contract import validate_selection, validate_interval, LengthContractError


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    s=cfg['selection'];r=cfg['runtime'];c=cfg['curation']
    if cfg['policy']['require_atomic_event'] is not False:raise ValueError('this production policy requires explicit atomic-event waiver')
    if s.get('target_count') is not None or s.get('frame_count_quotas'):raise ValueError('full-corpus pipeline has no global clip quota')
    validate_selection(s)
    if type(c['top_k_per_scene_task']) is not int or c['top_k_per_scene_task']<1:raise ValueError('invalid top K')
    if c.get('max_candidates_per_source') is not None and c['max_candidates_per_source']<c['top_k_per_scene_task']:raise ValueError('source candidate cap would be smaller than curated top K')
    for key in ('workers','export_workers'):
        if type(r[key]) is not int or r[key]<1:raise ValueError('invalid '+key)
    if r.get('weighted_cache_admission'):
        if r['max_cache_gb']*1024<cfg['rgb']['max_cache_mb'] or not r.get('cache_budget_root'):raise ValueError('invalid weighted cache budget')
    elif (r['workers']+r['export_workers'])*cfg['rgb']['max_cache_mb']>r['max_cache_gb']*1024:raise ValueError('combined analysis/export reservation exceeds cache budget')
    if cfg['aesthetics'].get('use_as_hard_gate') is not False:raise ValueError('this policy uses aesthetics for ranking, never to replace stability gates')
    return cfg


def source_job(path,cfg,forbidden=()):
    from .data import read_episode,iter_video_frames
    from .window_hands import GeometryIndex
    from .lazy_vision import analyze_lazy_rgb
    started=time.monotonic();check_resources(cfg)
    try:
        before=os.stat(path);ep=read_episode(path);fields=_source_fields(ep)
        from .recording_isolation import load_policy
        isolated,scope_policy=load_policy(cfg.get('recording_isolation'))
        if fields.get('recording_id') in isolated:
            return dict(source_path=path,status='ok',source=fields,candidates=[],source_stat=dict(size=before.st_size,mtime_ns=before.st_mtime_ns),stage='recording_isolation_excluded',rejections={'benchmark_recording_isolated':1},scope_policy=scope_policy,elapsed_s=round(time.monotonic()-started,3))
        if cfg.get('expansion'):
            from .expansion import baseline
            x=cfg['expansion'];seeds=baseline(x['seed_candidates'],x['seed_candidates_sha256'])
            exclusions=[]
            for c in seeds:
                if c['recording_id']==fields['recording_id']:
                    start=c['source_frame_offset']+c['start_frame']-fields['source_frame_offset']
                    end=c['source_frame_offset']+c['end_frame']-fields['source_frame_offset']
                    if start<ep['n'] and end>0:exclusions.append((max(0,start),min(ep['n'],end)))
            forbidden=tuple(forbidden)+tuple(exclusions)
        origin=fields.get('source_frame_offset')
        if type(origin) is int:
            frozen=cfg.get('frozen_recording_intervals',{}).get(fields.get('recording_id'),())
            forbidden=tuple(forbidden)+tuple(
                (max(0,start-origin),min(ep['n'],end-origin))
                for start,end in frozen if start<origin+ep['n'] and end>origin
            )
        geometry=GeometryIndex(ep,cfg)
        if forbidden:
            from .residual_windows import cap_forbidden
            original_starts=int(np.count_nonzero(geometry.upper_bounds))
            geometry.upper_bounds=cap_forbidden(geometry.upper_bounds,forbidden,geometry.minimum,geometry.multiple,cfg['selection'].get('frame_offset',0))
            geometry.diagnostics.update(starts_before_frozen_exclusion=original_starts,starts_after_frozen_exclusion=int(np.count_nonzero(geometry.upper_bounds)),excluded_intervals=len(forbidden))
        result=dict(source_path=path,status='ok',source=fields,candidates=[],geometry=geometry.diagnostics,
            source_stat=dict(size=before.st_size,mtime_ns=before.st_mtime_ns),policy_atomic_event_required=False)
        if not fields['recording_confident'] or type(fields['source_frame_offset']) is not int or type(fields['recording_time_origin_ns']) is not int:
            result.update(stage='unknown_source_identity',rejections={'unknown_source_identity':1})
        elif not any(geometry.upper_bounds):
            result.update(stage='geometry_reject',rejections={'no_geometry_legal_minimum_window':1})
        else:
            source_sha=sha256_file(path)
            key=hashlib.sha256(path.encode()).hexdigest()[:24]
            sample_dir=Path(cfg['runtime']['sample_root'])/key
            rgb=analyze_lazy_rgb(ep,cfg,geometry,iter_video_frames(path,ep),sample_dir,forbidden=forbidden)
            after=os.stat(path)
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or sha256_file(path)!=source_sha:
                raise RuntimeError('source changed during numerical analysis')
            for candidate in rgb['candidates']:
                candidate.update(fields);candidate['source_sha256_at_analysis']=source_sha
                candidate['atomic_event_requirement']='waived_by_user'
            result.update(rgb,source_sha256=source_sha,stage='rgb_pass' if rgb['candidates'] else 'rgb_reject')
        result['elapsed_s']=round(time.monotonic()-started,3);return result
    except Exception as error:
        if retryable_resource_error(error):raise
        return dict(source_path=path,status='error',stage='unknown_error',candidates=[],error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc()[-2000:],elapsed_s=round(time.monotonic()-started,3))


class ModelClient:
    def __init__(self,cfg,run):
        a=cfg['aesthetics'];env=os.environ.copy();device=a.get('device','cuda:0')
        env['CUDA_VISIBLE_DEVICES']='' if device=='cpu' else str(a['gpu']);env['PYTHONUNBUFFERED']='1'
        if device=='cpu':
            env['OMP_NUM_THREADS']=str(a.get('cpu_threads',4));env['MKL_NUM_THREADS']=env['OMP_NUM_THREADS']
        self.log=(run/'aesthetic_service.log').open('ab',buffering=0)
        self.proc=subprocess.Popen([a['python'],'-m','handclip.aesthetic_service','--assets',a['assets'],'--device',device],
            cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        if not select.select([self.proc.stdout],[],[],180)[0]:
            self.proc.terminate();self.proc.wait(timeout=10);self.log.close();raise TimeoutError('aesthetic model startup timeout')
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('aesthetic model startup failed; see aesthetic_service.log')
        self.ready=json.loads(line)
        if self.ready.get('status')!='ready':raise RuntimeError('aesthetic model not ready')
        self.sequence=0
    def score_record(self,record):
        # Called by exactly one service thread, preserving request/reply order.
        result=dict(record);result['candidates']=[]
        for c in record['candidates']:
            self.sequence+=1;request={'id':self.sequence,'path':c['aesthetic_samples']['path']}
            self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
            if not select.select([self.proc.stdout],[],[],180)[0]:raise TimeoutError('aesthetic inference timeout; source remains uncommitted')
            line=self.proc.stdout.readline()
            if not line:raise RuntimeError('aesthetic service disconnected; source remains uncommitted')
            response=json.loads(line)
            if response.get('id')!=self.sequence or response.get('error'):raise RuntimeError('aesthetic inference failed: '+str(response))
            score=response['result'];validate_score(score,c['aesthetic_samples']['source_frame_indices'])
            row=dict(c);row['aesthetics']=score;result['candidates'].append(row)
        result['stage']='scored';return result
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
        self.log.close()


def validate_score(score,expected_indices):
    from .aesthetic_service import SCENES
    if not isinstance(score,dict) or score.get('status')!='ok':raise ValueError('aesthetic result is not a known pass')
    if score.get('source_frame_indices')!=expected_indices or score.get('sample_count')!=len(expected_indices):raise ValueError('aesthetic sampling alignment mismatch')
    values=np.asarray(score.get('frame_scores'),dtype=float)
    if values.shape!=(len(expected_indices),) or not len(values) or not np.isfinite(values).all():raise ValueError('invalid frame scores')
    expected=dict(first=values[0],minimum=values.min(),p10=np.percentile(values,10),median=np.median(values),mean=values.mean())
    if any(not isinstance(score.get(k),(int,float)) or not np.isfinite(score[k]) or abs(score[k]-v)>1e-5 for k,v in expected.items()):raise ValueError('aesthetic aggregate/frame score mismatch')
    scene=score.get('scene',{})
    if scene.get('label') not in set(SCENES)|{'unknown_scene'} or scene.get('top_label') not in SCENES:raise ValueError('invalid inferred scene')
    if not all(isinstance(scene.get(k),(int,float)) and np.isfinite(scene[k]) for k in ('cosine_similarity','cosine_margin')):raise ValueError('invalid scene confidence')
    embedding=np.asarray(score.get('embedding'),dtype=float)
    if embedding.shape!=(768,) or not np.isfinite(embedding).all():raise ValueError('invalid CLIP embedding')
    if score.get('model',{}).get('model_sha256')!={'ViT-L-14.pt':'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836','aesthetic_mlp.pth':'21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0'}:raise ValueError('aesthetic model provenance mismatch')


def export_job(run,candidate,cfg):
    from .cache_budget import reserve
    with reserve(cfg,int(cfg["rgb"]["max_cache_mb"]*1024**2)):
        return _export_job_reserved(run,candidate,cfg)


def _export_job_reserved(run,candidate,cfg):
    from .artifacts import export_one,verify_one
    validate_interval(candidate,cfg['selection'])
    check_resources(cfg)
    # The output must still be the same source whose quality was measured.
    if sha256_file(candidate['source_path'])!=candidate['source_sha256_at_analysis']:
        raise RuntimeError('source changed between analysis and export')
    try:artifact=export_one(run,candidate,cfg)
    except Exception as error:
        # FFmpeg wraps disk errors as stderr; stop recoverably rather than
        # exhausting a good video's retry budget because storage was full.
        if any(s in str(error).lower() for s in ('no space left on device','disk quota exceeded','read-only file system')):
            import errno
            raise OSError(errno.ENOSPC,str(error)) from error
        raise
    if artifact['source_sha256']!=candidate['source_sha256_at_analysis']:raise RuntimeError('source hash changed during export')
    verification=verify_one(artifact,cfg)
    return dict(export=artifact,verification=verification)


def publish_manifest(run,cfg,selected,exports,verification,*,complete=False):
    validate_selection(cfg['selection'])
    rows=[]
    for c in selected:
        try:validate_interval(c,cfg['selection'])
        except LengthContractError:continue
        if c.get('aesthetics',{}).get('status')!='ok':continue
        cid=clip_id(c,config_hash(cfg));a=exports.get(cid,{});v=verification.get(cid,{})
        if a.get('clip_id')!=cid or v.get('clip_id')!=cid:continue
        if a.get('status')!='ok' or v.get('status')!='ok' or a.get('checksums',{}).get('video.mp4')!=v.get('video_sha256'):continue
        if type(v.get('frame_count')) is not int or v['frame_count']!=c['n_frames'] or v.get('media_timing',{}).get('status')!='ok':continue
        if a.get('source_sha256')!=c.get('source_sha256_at_analysis') or not a.get('source_sha256'):continue
        try:validate_interval(a.get('candidate'),cfg['selection'])
        except LengthContractError:continue
        if any(a.get('candidate',{}).get(k)!=c.get(k) for k in ('source_path','start_frame','end_frame','n_frames','fps','recording_id','source_frame_offset')):continue
        rows.append(dict(clip_id=cid,video=a['video'],first_frame=a['first_frame'],annotations=a['annotations'],
            source_path=c['source_path'],source_sha256=a['source_sha256'],recording_id=c['recording_id'],
            source_start_frame=c['start_frame'],source_end_frame_exclusive=c['end_frame'],
            recording_start_frame=c['source_frame_offset']+c['start_frame'],recording_end_frame_exclusive=c['source_frame_offset']+c['end_frame'],
            n_frames=c['n_frames'],fps=c['fps'],seconds=c['n_frames']/c['fps'],task=c.get('task'),
            scene_category=c['scene_category'],scene_group=c.get('scene_group',c['scene_category']),scene_label_source='CLIP inferred; visual cluster is not a human scene label',scene_task_rank=c['scene_task_rank'],
            aesthetic_median=c['aesthetics']['median'],aesthetic_p10=c['aesthetics']['p10'],
            metrics=c['metrics'],participating_hands=c['participating_hands'],config_sha256=config_hash(cfg),
            source_video_topic='/top-left-camera/image-raw',source_container_format='MCAP',
            source_lineage=cfg.get('source_lineage',{}),
            video_sha256=a['checksums']['video.mp4'],checksums=a['checksums'],
            quality_status='automatic_numeric_and_model_scored_independently_verified',manual_visual_approval=False,
            atomic_event_required=False,selection_status='final_for_frozen_input' if complete else 'provisional_during_scan'))
    atomic_json(run/'curated_manifest.json',rows)
    atomic_bytes(run/'curated_manifest.jsonl',(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)).encode())
    keys=['clip_id','video','first_frame','annotations','source_path','source_sha256','recording_id','source_start_frame',
        'source_end_frame_exclusive','recording_start_frame','recording_end_frame_exclusive','n_frames','fps','seconds',
        'task','scene_category','scene_group','scene_task_rank','aesthetic_median','aesthetic_p10','quality_status','selection_status']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    atomic_bytes(run/'curated_manifest.csv',stream.getvalue().encode('utf-8-sig'))
    return rows


def run(cfg,run_id,inventory_path,resume=False,limit=None,stage='all'):
    from .curation import select_curated
    validate_selection(cfg['selection'])
    run_dir=Path(cfg['paths']['output_root'])/'runs'/run_id
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('invalid run id')
    existed=run_dir.exists()
    if existed and not resume:raise ValueError('run exists; resume its exact frozen configuration')
    run_dir.mkdir(parents=True,exist_ok=True);check_resources(cfg)
    with exclusive_writer(run_dir):
        from .source_priority import prioritize
        files=prioritize(stratified_inventory(read_json(inventory_path)['paths']),cfg)
        if limit is not None:files=files[:limit]
        manifest=dict(config_sha256=config_hash(cfg),code_manifest=code_manifest(Path(__file__).resolve().parents[1]),
            input_count=len(files),policy=cfg['policy'])
        if existed:
            previous=read_json(run_dir/'run.json')
            if previous.get('inventory_sha256')!=sha256_file(run_dir/'inventory.json'):raise ValueError('frozen run inventory checksum changed')
            if any(previous[k]!=v for k,v in manifest.items()) or read_json(run_dir/'inventory.json')!=files:raise ValueError('frozen run inputs/code/config changed')
        else:
            atomic_json(run_dir/'inventory.json',files);atomic_json(run_dir/'config.json',cfg)
            atomic_json(run_dir/'run.json',dict(manifest,created_at=utc_now(),inventory_sha256=sha256_file(run_dir/'inventory.json')))
        from .reuse_analysis import seed
        seed(run_dir,cfg,files)
        prior=read_jsonl(run_dir/'analysis.jsonl');records={}
        for row in prior:
            if row['source_path'] in records:raise RuntimeError('duplicate source ledger; never silently overwrite')
            records[row['source_path']]=row
        if set(records)-set(files):raise RuntimeError('analysis ledger contains a source outside frozen input')
        exports={r['clip_id']:r for r in read_jsonl(run_dir/'exports.jsonl')}
        verification={r['clip_id']:r for r in read_jsonl(run_dir/'verification.jsonl')}
        # On resume check committed output bytes/source identity before reuse.
        if exports:
            from .artifacts import verify_exports
            verify_exports(run_dir,cfg)
            verification={r['clip_id']:r for r in read_jsonl(run_dir/'verification.jsonl')}
        todo=collections.deque(p for p in files if p not in records)
        analysis={};scoring={};score_queue=collections.deque();delivery={};delivery_ids=set();buffer=[]
        selected=[];selection_diagnostics={};last_report=0.;last_flush=time.monotonic();last_progress=0.;last_result=utc_now()
        model=None;analysis_pool=None;score_pool=None;export_pool=None;interrupted=False;summary={}

        def flush():
            nonlocal last_flush
            if buffer:
                committed=list(buffer)
                append_segment(run_dir/'analysis.jsonl',committed);buffer.clear()
                # Only delete this run's regenerable model inputs AFTER commit.
                sample_root=Path(cfg['runtime']['sample_root']).resolve()
                for row in committed:
                    for candidate in row.get('candidates',[]):
                        sample=Path(candidate.get('aesthetic_samples',{}).get('path',''))
                        try:
                            resolved=sample.resolve()
                            if sample_root in resolved.parents and resolved.suffix=='.npz':resolved.unlink(missing_ok=True)
                        except OSError:pass
            last_flush=time.monotonic()

        def commit(row):
            nonlocal last_result
            path=row['source_path']
            if path in records:raise RuntimeError('duplicate source completion')
            records[path]=row;buffer.append(row);last_result=utc_now()

        def report(complete=False):
            nonlocal selected,selection_diagnostics,last_report,summary
            flush()
            candidates=[c for row in records.values() if row['status']=='ok' for c in row.get('candidates',[]) if c.get('aesthetics',{}).get('status')=='ok']
            failed={cid for cid,r in verification.items() if r.get('status')=='error'}
            selected,selection_diagnostics=select_curated([c for c in candidates if clip_id(c,config_hash(cfg)) not in failed],cfg)
            atomic_json(run_dir/'selected_candidates.json',selected)
            released=publish_manifest(run_dir,cfg,selected,exports,verification,complete=complete)
            errors=[r for r in records.values() if r['status']!='ok']
            summary=dict(updated_at=utc_now(),policy=cfg['policy'],input_sources=len(files),analyzed_sources=len(records),
                source_hours=sum(r['source']['n']/r['source']['fps']/3600 for r in records.values() if r.get('source')),
                source_hours_scope='aligned source frame count / actual FPS, includes any source overlap',
                source_errors=len(errors),stages=dict(collections.Counter(r.get('stage','unknown') for r in records.values())),
                machine_candidates=len(candidates),selected_candidates=len(selected),published_clips=len(released),
                output_verification_errors=len(failed),
                published_hours=sum(r['seconds']/3600 for r in released),frame_histogram=dict(collections.Counter(str(r['n_frames']) for r in released)),
                selection=selection_diagnostics,all_sources_processed=len(records)==len(files),
                production_complete=bool(complete and len(released)==len(selected) and not errors and not failed),
                status='finished_with_errors' if complete and (errors or failed) else 'complete' if complete and len(released)==len(selected) else 'scanning',
                atomic_event_required=False,manual_visual_approvals=0,last_result_at=last_result)
            if cfg.get('motion_selection',{}).get('target_count') is not None:
                target=cfg['motion_selection']['target_count']
                summary.update(requested_target=target,automatic_delivery_deficit=max(0,target-len(released)),
                    target_met_automatically=len(released)>=target,
                    human_check_status='pending',scope='motion upgrade over inherited camera/hand/RGB gates')
            atomic_json(run_dir/'summary.json',summary);last_report=time.monotonic();return summary

        def progress(status='running',error=None):
            nonlocal last_progress
            state=dict(updated_at=utc_now(),pid=os.getpid(),status=status,last_result_at=last_result,input_sources=len(files),
                source_completed=len(records)-len(buffer),processed_in_memory=len(records),not_submitted=len(todo),active_source_workers=len(analysis),pending_aesthetic_sources=len(score_queue),
                active_aesthetic_sources=len(scoring),active_exports=len(delivery),exported=sum(v.get('status')=='ok' for v in verification.values()),
                error=error)
            atomic_json(run_dir/'progress.json',state);print(json.dumps(state),flush=True);last_progress=time.monotonic()

        def fill():
            check_resources(cfg)
            if stage not in ('export','report'):
                while todo and len(analysis)<cfg['runtime']['workers'] and len(score_queue)<cfg['runtime']['workers']*2:
                    path=todo.popleft();analysis[analysis_pool.submit(source_job,path,cfg)]=path
                if score_queue and not scoring:
                    row=score_queue.popleft();scoring[score_pool.submit(model.score_record,row)]=row['source_path']
            if stage!='scan':
                for c in selected:
                    cid=clip_id(c,config_hash(cfg))
                    if cid in delivery_ids or verification.get(cid,{}).get('status')=='ok':continue
                    if verification.get(cid,{}).get('status')=='error':continue
                    if len(delivery)>=cfg['runtime']['export_workers']:break
                    delivery[export_pool.submit(export_job,run_dir,c,cfg)]=cid;delivery_ids.add(cid)

        try:
            if stage=='report':
                # Derive completion again from frozen sources and verified
                # outputs; reporting a finished run must not downgrade it.
                result=report(complete=len(records)==len(files));progress(result['status']);return result
            report();progress('starting')
            if stage not in ('export','report'):
                # A separate interpreter owns CUDA. CPU workers never inherit it.
                model=ModelClient(cfg,run_dir);atomic_json(run_dir/'aesthetic_model.json',model.ready)
                analysis_pool=futures.ProcessPoolExecutor(max_workers=cfg['runtime']['workers'],mp_context=multiprocessing.get_context('spawn'))
                score_pool=futures.ThreadPoolExecutor(max_workers=1)
                # Lost volatile samples on resume are regenerated by source_job,
                # because a source is committed only after its model scores exist.
            export_pool=futures.ThreadPoolExecutor(max_workers=cfg['runtime']['export_workers'])
            fill()
            while analysis or scoring or score_queue or delivery or (todo and stage not in ('export','report')):
                pending=list(analysis)+list(scoring)+list(delivery)
                done,_=futures.wait(pending,timeout=5,return_when=futures.FIRST_COMPLETED) if pending else ([],[])
                for future in done:
                    if future in analysis:
                        path=analysis.pop(future);row=future.result()
                        if row['source_path']!=path:raise RuntimeError('source worker identity mismatch')
                        if row.get('candidates'):score_queue.append(row)
                        else:commit(row)
                    elif future in scoring:
                        path=scoring.pop(future);row=future.result()
                        if row['source_path']!=path:raise RuntimeError('score worker identity mismatch')
                        commit(row)
                    else:
                        cid=delivery.pop(future);delivery_ids.remove(cid)
                        try:outcome=future.result()
                        except Exception as error:
                            if retryable_resource_error(error):raise
                            result=dict(clip_id=cid,status='error',error=f'{type(error).__name__}: {error}',verified_at=utc_now())
                            append_jsonl(run_dir/'verification.jsonl',result);verification[cid]=result
                        else:
                            if outcome['export']['clip_id']!=cid or outcome['verification'].get('clip_id')!=cid:raise RuntimeError('delivery identity mismatch')
                            append_jsonl(run_dir/'exports.jsonl',outcome['export']);exports[cid]=outcome['export']
                            append_jsonl(run_dir/'verification.jsonl',outcome['verification']);verification[cid]=outcome['verification']
                if len(buffer)>=50 or time.monotonic()-last_flush>=30:flush()
                if time.monotonic()-last_report>=cfg['runtime'].get('report_interval_seconds',300):report()
                fill()
                if time.monotonic()-last_progress>=30:progress()
            # Final global selection can introduce candidates since the last report.
            report()
            if stage!='scan':
                while True:
                    fill()
                    if not delivery:break
                    future=next(futures.as_completed(delivery));cid=delivery.pop(future);delivery_ids.remove(cid)
                    try:outcome=future.result()
                    except Exception as error:
                        if retryable_resource_error(error):raise
                        result=dict(clip_id=cid,status='error',error=f'{type(error).__name__}: {error}',verified_at=utc_now())
                        append_jsonl(run_dir/'verification.jsonl',result);verification[cid]=result
                    else:
                        if outcome['export']['clip_id']!=cid or outcome['verification'].get('clip_id')!=cid:raise RuntimeError('delivery identity mismatch')
                        append_jsonl(run_dir/'exports.jsonl',outcome['export']);exports[cid]=outcome['export']
                        append_jsonl(run_dir/'verification.jsonl',outcome['verification']);verification[cid]=outcome['verification']
                    report();progress('exporting')
            result=report(complete=len(records)==len(files) and stage!='scan');progress(result['status']);return result
        except BaseException as error:
            interrupted=True
            # Never flush a second time after an uncertain filesystem write.
            try:progress('failed_recoverable',f'{type(error).__name__}: {error}')
            except Exception:pass
            if analysis_pool:
                from .streaming import stop_owned_analysis
                stop_owned_analysis(analysis_pool,run_dir,cfg)
            raise
        finally:
            if analysis_pool:analysis_pool.shutdown(wait=True,cancel_futures=True)
            if score_pool:score_pool.shutdown(wait=True,cancel_futures=True)
            if export_pool:export_pool.shutdown(wait=True,cancel_futures=True)
            if model:model.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--inventory',required=True)
    ap.add_argument('--run-id',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--limit',type=int)
    ap.add_argument('--stage',choices=['all','scan','export','report'],default='all');args=ap.parse_args(argv)
    result=run(load_config(args.config),args.run_id,args.inventory,args.resume,args.limit,args.stage)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
