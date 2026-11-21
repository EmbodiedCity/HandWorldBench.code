"""Lossless native-frame exports, annotations, and independent decode validation."""
import collections
import concurrent.futures
import contextlib
import errno
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np

from .config import config_hash
from .media_timing import validate_video_timing
from .length_contract import validate_selection, validate_interval
from .storage import atomic_json, append_jsonl, read_jsonl, read_json, sha256_file, clip_id, utc_now


def _encode_native(path, ep, start, end, output, encoded_input=None):
    """Encode native pixels; the caller must still validate every output frame.

    FFmpeg may close input after satisfying trim. A resulting BrokenPipe is only
    provisional success when FFmpeg exits cleanly; export_one still requires the
    exact frame count and source RGB hashes before committing an artifact.
    """
    from .data import iter_video_packets, _ffmpeg_executable
    # trim decodes in presentation order; lossless encoding keeps the original
    # decoded native pixel format and does not add another quantization pass.
    command=[_ffmpeg_executable(),'-nostdin','-hide_banner','-loglevel','error','-threads','1',
             '-r',format(ep['fps'],'.12g'),'-f','h264','-i',str(encoded_input) if encoded_input else 'pipe:0','-map','0:v:0',
             '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
             '-an','-c:v','libx264','-crf','0','-preset','fast','-threads','1',
             # Input -r sets frame PTS; explicit CFR output also gives the final
             # MP4 sample a full frame duration. Exact source hashes still gate
             # export, so any encoder frame duplication/removal fails closed.
             '-r',format(ep['fps'],'.12g'),'-fps_mode','cfr',
             '-movflags','+faststart','-f','mp4','-y',str(output)]
    with subprocess.Popen(command,stdin=subprocess.DEVNULL if encoded_input else subprocess.PIPE,
                          stdout=subprocess.DEVNULL,stderr=subprocess.PIPE) as proc:
        errors=[]
        closed_after_trim=[]
        def pipe_closed(exc):
            # Windows CRT sometimes reports EINVAL when writing/flushing a pipe
            # whose reader exited; source iteration exceptions never enter here.
            return isinstance(exc,BrokenPipeError) or (os.name=='nt' and exc.errno==errno.EINVAL)
        def feed():
            try:
                for packet in iter_video_packets(path):
                    try:
                        proc.stdin.write(packet)
                    except OSError as exc:
                        if pipe_closed(exc):
                            closed_after_trim.append(True)
                            break
                        raise
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    proc.stdin.close()
                except OSError as exc:
                    if pipe_closed(exc):
                        closed_after_trim.append(True)
                    else:
                        errors.append(exc)
        thread=threading.Thread(target=feed,daemon=True) if encoded_input is None else None
        if thread is not None:
            thread.start()
        stderr=proc.stderr.read().decode('utf-8',errors='replace')
        if thread is not None:
            thread.join(timeout=120)
        if thread is not None and thread.is_alive():
            proc.kill()
            raise RuntimeError('H264 feeder did not finish')
        code=proc.wait()
        if errors or code or stderr.strip():
            raise RuntimeError(f'encode error {code}: {errors}; {stderr[-3000:]}')
        return dict(input_closed_after_trim=bool(closed_after_trim),
                    encoder_returncode=code, exact_source_validation_required=True)


@contextlib.contextmanager
def _selected_native_source(ep,start,end,cfg):
    """Bounded optional IDR cache; unsupported/unavailable caches fall back safely."""
    from .data import _H264PacketGuard, DataError
    runtime=cfg.get('runtime',{})
    reason='disabled'
    packets=ep.get('video_packets')
    profile=ep.get('metadata',{}).get('h264_alignment',{})
    if not runtime.get('export_fastpath',True):
        yield None,dict(mode='full_source_stream',fastpath_fallback=reason)
        return
    if not profile.get('validated_single_idr_picture_per_packet') or packets is None or len(packets)!=ep['n']:
        yield None,dict(mode='full_source_stream',fastpath_fallback='no_verified_independent_idr_packets')
        return
    budget_mb=min(float(cfg.get('rgb',{}).get('max_cache_mb',512)),
                  float(runtime.get('max_cache_gb',8))*1024/max(1,int(runtime.get('workers',1))))
    budget=int(budget_mb*1024*1024) if np.isfinite(budget_mb) and budget_mb>0 else 0
    selected=packets[start:end]
    required=sum(len(packet) for packet in selected)
    if required>budget or required==0:
        yield None,dict(mode='full_source_stream',fastpath_fallback='selected_packet_cache_budget',
                        selected_packet_bytes=required,cache_budget_bytes=budget)
        return
    # A fresh guard per packet proves its SPS/PPS exist before its IDR slice and
    # forbids references to parameter sets residing outside the selected interval.
    try:
        for packet in selected:
            guard=_H264PacketGuard()
            guard.check(packet)
            if guard.dimensions!=(ep['width'],ep['height']):
                raise DataError('selected packet dimensions differ from calibration')
    except DataError:
        yield None,dict(mode='full_source_stream',fastpath_fallback='packets_not_self_contained')
        return
    directory=None
    try:
        cache_root=runtime.get('cache_root')
        if cache_root:
            Path(cache_root).mkdir(parents=True,exist_ok=True)
        directory=tempfile.TemporaryDirectory(prefix='handclip-export-',dir=cache_root)
        selected_file=Path(directory.name)/'selected.h264'
        with selected_file.open('wb') as stream:
            for packet in selected:
                stream.write(packet)
        if selected_file.stat().st_size!=required:
            raise OSError('selected packet cache size mismatch')
    except OSError as exc:
        if directory is not None:
            directory.cleanup()
        yield None,dict(mode='full_source_stream',fastpath_fallback='selected_packet_cache_unavailable',
                        cache_error=f'{type(exc).__name__}: {exc}')
        return
    try:
        yield selected_file,dict(mode='self_contained_idr_selected_packets',selected_packet_bytes=required,
                                 cache_budget_bytes=budget,source_packet_count=end-start)
    finally:
        directory.cleanup()


def _remux_native(selected,ep,output):
    """Copy only a freshly guarded, one-independent-IDR-per-frame interval.

    This is not a keyframe seek: _selected_native_source selected exact frame
    indices and validated every packet's own SPS/PPS/IDR. The caller must still
    decode, compare every source pixel hash and validate the full MP4 timeline.
    """
    from .data import _ffmpeg_executable
    command=[_ffmpeg_executable(),'-hide_banner','-loglevel','error','-fflags','+genpts',
             '-r',format(ep['fps'],'.12g'),'-f','h264','-i',str(selected),'-map','0:v:0',
             '-an','-c:v','copy','-r',format(ep['fps'],'.12g'),'-movflags','+faststart','-y',str(output)]
    result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    if result.returncode or result.stderr.strip():raise RuntimeError(f'native remux error {result.returncode}: {result.stderr[-3000:]}')
    return dict(mode='original_self_contained_idr_remux',encoder_returncode=0,
                exact_source_validation_required=True,reencoded=False)


def _decode_file(path,width,height):
    from .data import _ffmpeg_executable, _read_exact
    command=[_ffmpeg_executable(),'-nostdin','-hide_banner','-loglevel','error','-threads','1',
             '-i',str(path),'-map','0:v:0','-an','-vsync','0','-threads','1',
             '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stderr=collections.deque(maxlen=128)
    def drain():
        try:
            while True:
                chunk=proc.stderr.read(4096)
                if not chunk:
                    return
                stderr.append(chunk)
        except (OSError,ValueError):
            return
    thread=threading.Thread(target=drain,daemon=True)
    thread.start()
    try:
        while True:
            b=_read_exact(proc.stdout,width*height*3)
            if not b:
                break
            if len(b)!=width*height*3:
                raise RuntimeError('partial output frame')
            yield np.frombuffer(b,np.uint8).reshape(height,width,3)
        code=proc.wait(timeout=30)
        thread.join(timeout=5)
        if code or b''.join(stderr).strip():
            raise RuntimeError(f'output decode error {code}: {b"".join(stderr)[-3000:]}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
        thread.join(timeout=5)


def _digest(frame):
    return hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest()


def _preview(video,output,ep,start,end,candidate):
    import cv2
    from .data import _ffmpeg_executable, project_hands
    uv,_=project_hands(ep)
    width=960
    height=int(round(ep['height']*width/ep['width']/2))*2
    command=[_ffmpeg_executable(),'-hide_banner','-loglevel','error','-f','rawvideo',
             '-pix_fmt','rgb24','-s',f'{width}x{height}','-r',format(ep['fps'],'.12g'),
             '-i','pipe:0','-an','-c:v','libx264','-crf','20','-preset','fast',
             '-pix_fmt','yuv420p','-threads','1','-movflags','+faststart','-y',str(output)]
    edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
           (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for local,frame in enumerate(_decode_file(video,ep['width'],ep['height'])):
            canvas=cv2.resize(frame,(width,height))
            for h in candidate['participating_hands']:
                points=uv[start+local,h]*np.array([width/ep['width'],height/ep['height']])
                color=(80,255,80) if h==0 else (255,150,70)
                for a,b in edges:
                    if np.isfinite(points[[a,b]]).all():
                        cv2.line(canvas,tuple(points[a].astype(int)),tuple(points[b].astype(int)),color,2)
            cv2.putText(canvas,f'PENDING RGB REVIEW | source frame {start+local} | {local+1}/{end-start}',
                        (15,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        stderr=proc.stderr.read()
        if proc.wait() or stderr.strip():
            raise RuntimeError(f'preview encode failed: {stderr[-1500:]}')


def export_one(run,candidate,cfg):
    import cv2
    from .data import read_episode, iter_video_frames, project_hands
    validate_interval(candidate,cfg['selection'])
    cv2.setNumThreads(1)
    cid=clip_id(candidate,config_hash(cfg))
    dest=Path(run)/'clips'/cid
    dest.mkdir(parents=True,exist_ok=True)
    artifact=dest/'artifact.json'
    if artifact.exists():
        previous=read_json(artifact)
        validate_interval(previous.get('candidate'),cfg['selection'])
        for key,digest in previous['checksums'].items():
            if sha256_file(dest/key)!=digest:
                raise RuntimeError('existing artifact changed; refusing overwrite')
        validate_video_timing(dest/'video.mp4',candidate['n_frames'],candidate['fps'])
        return previous
    ep=read_episode(candidate['source_path'],include_video=True)
    validate_interval(candidate,cfg['selection'],source_frames=ep['n'])
    start,end=candidate['start_frame'],candidate['end_frame']
    if not 0<=start<end<=ep['n'] or end-start!=candidate['n_frames'] or (end-start-cfg['selection'].get('frame_offset',0))%cfg['selection']['frame_multiple']:
        raise ValueError('invalid export interval')
    source_sha=sha256_file(candidate['source_path'])
    temporary=dest/'video.partial.mp4'
    encoding='libx264 CRF0 native pixel format'
    with _selected_native_source(ep,start,end,cfg) as (selected_file,decode_path):
        if selected_file is not None and cfg.get('runtime',{}).get('native_packet_remux',False):
            encoder_state=_remux_native(selected_file,ep,temporary)
            from .media_timing import MediaTimingError
            try:validate_video_timing(temporary,end-start,ep['fps'])
            except MediaTimingError as error:
                # A raw H264 VUI may advertise a different rate from the aligned
                # source timeline. Never relax timing: use the established exact
                # decode/encode path and retain the remux rejection reason.
                rejected=encoder_state
                encoder_state=_encode_native(candidate['source_path'],ep,0,end-start,temporary,encoded_input=selected_file)
                encoder_state['native_remux_fallback']=dict(reason='media_timing_mismatch',error=str(error),remux_state=rejected)
            else:
                encoding='original H264 packets in MP4; verified independent IDR frames; no re-encoding'
        elif selected_file is not None:
            encoder_state=_encode_native(candidate['source_path'],ep,0,end-start,temporary,encoded_input=selected_file)
        else:
            encoder_state=_encode_native(candidate['source_path'],ep,start,end,temporary)
        decoded=list()  # hashes only, never retain the RGB video
        first=None
        for frame in _decode_file(temporary,ep['width'],ep['height']):
            if first is None:
                first=frame.copy()
            decoded.append(_digest(frame))
        if len(decoded)!=end-start:
            raise RuntimeError(f'exported count {len(decoded)} != {end-start}')
        expected=[]
        if selected_file is not None:
            for frame in _decode_file(selected_file,ep['width'],ep['height']):
                expected.append(_digest(frame))
        else:
            for i,frame in enumerate(iter_video_frames(candidate['source_path'],ep)):
                if start<=i<end:
                    expected.append(_digest(frame))
    if decoded!=expected:
        raise RuntimeError('lossless export differs from the selected original RGB frames; quarantined')
    if candidate.get('source_first_frame_sha256') and decoded[0]!=candidate['source_first_frame_sha256']:
        raise RuntimeError('RGB candidate/source first-frame mismatch')
    media_timing=validate_video_timing(temporary,end-start,ep['fps'])
    os.replace(temporary,dest/'video.mp4')
    if not cv2.imwrite(str(dest/'first_frame.png'),cv2.cvtColor(first,cv2.COLOR_RGB2BGR)):
        raise RuntimeError('PNG write failed')
    uv,depth=project_hands(ep)
    annotations=dest/'annotations.npz'
    np.savez_compressed(annotations,source_frame_indices=np.arange(start,end),times_ns=ep['times_ns'][start:end],
        positions_m=ep['positions_m'][start:end],hand_valid=ep['hand_valid'][start:end],
        vio_t=ep['vio_t'][start:end],vio_q_wxyz=ep['vio_q_wxyz'][start:end],pose_valid=ep['pose_valid'][start:end],
        projected_uv=uv[start:end],positive_depth=depth[start:end],K=ep['K'],D=ep['D'],
        subtasks=np.asarray(ep['subtasks'][start:end],dtype=str),participating_hands=np.array(candidate['participating_hands']))
    atomic_json(dest/'source_frame_hashes.json',expected)
    atomic_json(dest/'source_metadata.json',ep['metadata'])
    _preview(dest/'video.mp4',dest/'preview.mp4',ep,start,end,candidate)
    checksums={name:sha256_file(dest/name) for name in ('video.mp4','first_frame.png','annotations.npz',
                                                      'source_frame_hashes.json','source_metadata.json','preview.mp4')}
    result=dict(status='ok',clip_id=cid,candidate=candidate,video=str(dest/'video.mp4'),
        first_frame=str(dest/'first_frame.png'),annotations=str(annotations),preview=str(dest/'preview.mp4'),
        source_sha256=source_sha,checksums=checksums,exported_at=utc_now(),qc_status='pending',
        export_validation=dict(frame_count=len(decoded),every_decoded_rgb_frame_identical_to_source=True,
                               encoding=encoding,source_decode=decode_path,
                               encoder=encoder_state,media_timing=media_timing))
    atomic_json(artifact,result)
    return result


def select_export_candidates(run,cfg):
    from .temporal import select_diverse
    from .release import load_release_policy, gate_review
    from .review_scope import excludes, validate_scope
    sources=read_jsonl(Path(run)/'rgb.jsonl')
    reviews={r['clip_id']:r for r in read_jsonl(Path(run)/'reviews.jsonl')}
    exports=read_jsonl(Path(run)/'exports.jsonl')
    current_exports={r['clip_id']:r for r in exports}
    verifications={r['clip_id']:r for r in read_jsonl(Path(run)/'verification.jsonl')}
    errors=collections.Counter(r['clip_id'] for r in exports if r['status']=='error')
    refused=[]
    reserved=[]
    policy_state=load_release_policy(run,cfg)
    for cid,review in reviews.items():
        artifact=current_exports.get(cid,{})
        digest=artifact.get('checksums',{}).get('video.mp4')
        if not digest or review.get('video_sha256')!=digest:
            continue
        candidate=artifact.get('candidate')
        if not candidate:
            continue
        if review['verdict'] in ('reject','uncertain'):
            validate_scope(review,candidate['n_frames'])
            refused.append((candidate,review))
        elif review['verdict']=='pass':
            verification=verifications.get(cid,{})
            if (artifact.get('status')=='ok' and verification.get('status')=='ok'
                    and verification.get('video_sha256')==digest
                    and gate_review(run,cfg,artifact,verification,review,{cid},policy_state)['qualified']):
                reserved.append(candidate)
    candidates=[]
    for record in sources:
        if record['status']!='ok':
            continue
        for candidate in record.get('candidates',[]):
            if not candidate.get('recording_confident') or not isinstance(candidate.get('source_frame_offset'),int):
                continue
            cid=clip_id(candidate,config_hash(cfg))
            if errors[cid]>=cfg.get('runtime',{}).get('export_retry_limit',2):
                continue
            if verifications.get(cid,{}).get('status')=='error':
                continue
            if any(excludes(candidate,r,review) for r,review in refused):
                continue
            candidates.append(candidate)
    return select_diverse(candidates,cfg,reserved=reserved)


def export_candidates(run,cfg):
    from .pipeline import check_resources
    check_resources(cfg)
    selected=select_export_candidates(run,cfg)
    atomic_json(Path(run)/'selected_machine_candidates.json',selected)
    prior={r['clip_id']:r for r in read_jsonl(Path(run)/'exports.jsonl')}
    def export_job(c):
        check_resources(cfg)
        cid=clip_id(c,config_hash(cfg))
        try:
            if prior.get(cid,{}).get('status')=='ok':
                # Re-open the committed artifact and verify all files on resume.
                # Never overwrite a changed/missing artifact under the same ID.
                export_one(run,c,cfg)
                return None
            result=export_one(run,c,cfg)
        except Exception as exc:
            from .resources import retryable_resource_error
            if retryable_resource_error(exc):
                raise
            result=dict(clip_id=cid,status='error',candidate=c,error=f'{type(exc).__name__}: {exc}',
                        traceback=traceback.format_exc()[-2000:],updated_at=utc_now())
        return result
    workers=min(cfg.get('runtime',{}).get('workers',2),cfg.get('runtime',{}).get('export_workers',2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(export_job,c) for c in selected]
        for future in concurrent.futures.as_completed(jobs):
            result=future.result()
            if result is None:
                continue
            # Only the parent writes the ledger; each worker has a distinct clip directory.
            append_jsonl(Path(run)/'exports.jsonl',result)
            print(json.dumps({'export':result['clip_id'],'status':result['status'],'error':result.get('error')}),flush=True)


def verify_one(record,cfg):
    import cv2
    from .data import read_episode, project_hands
    c=record['candidate']
    validate_interval(c,cfg['selection'])
    directory=Path(record['video']).parent
    for file,digest in record['checksums'].items():
        if sha256_file(directory/file)!=digest:
            raise RuntimeError(f'checksum mismatch: {file}')
    media_timing=validate_video_timing(record['video'],c['n_frames'],c['fps'])
    hashes=[]
    first=None
    for frame in _decode_file(record['video'],c['width'],c['height']):
        if first is None:
            first=frame.copy()
        hashes.append(_digest(frame))
    expected=read_json(directory/'source_frame_hashes.json')
    if hashes!=expected or len(hashes)!=c['n_frames'] or (len(hashes)-cfg['selection'].get('frame_offset',0))%cfg['selection']['frame_multiple']:
        raise RuntimeError('independent frame/order validation failed')
    png=cv2.imread(record['first_frame'])
    if png is None or not np.array_equal(cv2.cvtColor(png,cv2.COLOR_BGR2RGB),first):
        raise RuntimeError('first-frame PNG is not the decoded conditioning frame')
    ep=read_episode(c['source_path'])
    validate_interval(c,cfg['selection'],source_frames=ep['n'])
    if sha256_file(c['source_path'])!=record['source_sha256']:
        raise RuntimeError('source MCAP changed after export')
    uv,depth=project_hands(ep)
    section=slice(c['start_frame'],c['end_frame'])
    expected_arrays={name:ep[name][section] for name in ('times_ns','positions_m','hand_valid','vio_t','vio_q_wxyz','pose_valid')}
    expected_arrays.update(projected_uv=uv[section],positive_depth=depth[section],K=ep['K'],D=ep['D'],
                           subtasks=np.asarray(ep['subtasks'][section],dtype=str),participating_hands=np.array(c['participating_hands']))
    with np.load(record['annotations'],allow_pickle=False) as a:
        if len(a['times_ns'])!=len(hashes) or not np.array_equal(a['source_frame_indices'],np.arange(c['start_frame'],c['end_frame'])):
            raise RuntimeError('annotation alignment failed')
        for name,expected_array in expected_arrays.items():
            if name not in a or a[name].shape!=expected_array.shape:
                raise RuntimeError(f'annotation shape mismatch: {name}')
            equal=np.array_equal(a[name],expected_array,equal_nan=True) if np.issubdtype(expected_array.dtype,np.inexact) else np.array_equal(a[name],expected_array)
            if not equal:
                raise RuntimeError(f'annotation source mismatch: {name}')
    return dict(clip_id=record['clip_id'],status='ok',verified_at=utc_now(),frame_count=len(hashes),
                video_sha256=record['checksums']['video.mp4'],first_frame_exact=True,source_frame_order_exact=True,
                media_timing=media_timing)


def verify_exports(run,cfg):
    from .pipeline import check_resources
    validate_selection(cfg['selection'])
    check_resources(cfg)
    existing={r['clip_id']:r for r in read_jsonl(Path(run)/'verification.jsonl')}
    def verify_job(record):
        check_resources(cfg)
        if record['status']!='ok':
            return None
        try:
            validate_interval(record.get('candidate'),cfg['selection'])
            previous=existing.get(record['clip_id'],{})
            if (previous.get('status')=='ok' and previous.get('video_sha256')==record['checksums']['video.mp4']
                    and previous.get('media_timing',{}).get('status')=='ok'
                    and previous.get('media_timing',{}).get('schema_version')==1):
                directory=Path(record['video']).parent
                if all(sha256_file(directory/name)==digest for name,digest in record['checksums'].items()):
                    if sha256_file(record['candidate']['source_path'])!=record['source_sha256']:
                        raise RuntimeError('source MCAP changed after the previous verification')
                    validate_video_timing(record['video'],record['candidate']['n_frames'],record['candidate']['fps'])
                    return None
            result=verify_one(record,cfg)
        except Exception as exc:
            from .resources import retryable_resource_error
            if retryable_resource_error(exc):
                raise
            result=dict(clip_id=record['clip_id'],status='error',error=f'{type(exc).__name__}: {exc}',verified_at=utc_now())
        return result
    records=list({r['clip_id']:r for r in read_jsonl(Path(run)/'exports.jsonl')}.values())
    workers=min(cfg.get('runtime',{}).get('workers',2),cfg.get('runtime',{}).get('export_workers',2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(verify_job,records):
            if result is not None:
                append_jsonl(Path(run)/'verification.jsonl',result)
                print(json.dumps({'verification':result}),flush=True)
