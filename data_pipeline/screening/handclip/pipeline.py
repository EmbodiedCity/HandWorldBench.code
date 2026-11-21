"""Single-writer resumable orchestration. Machine passes are never review passes."""
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import socket
import shutil
import sys
import time
import traceback
from pathlib import Path

from .config import code_manifest, config_hash
from .resources import retryable_resource_error
from .storage import atomic_json, atomic_bytes, append_jsonl, append_segment, read_jsonl, read_json, clip_id, utc_now, sha256_file, publish_progress


def target_met(count, histogram, cfg):
    quotas = cfg['selection'].get('frame_count_quotas', {})
    return count >= cfg['selection']['target_count'] and all(
        histogram.get(str(length), histogram.get(int(length), 0)) >= number
        for length, number in quotas.items())


def check_resources(cfg):
    # These are measured local filesystems, never a claimed remote mount quota.
    for guard in cfg.get('runtime', {}).get('free_space_guards', []):
        free = shutil.disk_usage(guard['path']).free
        if free < guard['minimum_free_gib'] * 1024**3:
            raise RuntimeError(f"resource_guard: {guard['path']} has {free} free bytes; preserve state and resume after capacity recovery")


def inventory(source_root):
    root = Path(source_root)
    return sorted(str(root / e.name / 'episode-000000.mcap')
                  for e in os.scandir(root) if e.name.startswith('session_'))


def stratified_inventory(paths):
    """Deterministic session round-robin; this order is frozen once per run.

    Names are used only for sampling order, not to authorize source identity.
    Actual source/offset confidence still comes from data.read_episode.
    """
    if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError('inventory must contain a list of nonempty source paths')
    digest = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    groups = collections.defaultdict(list)
    for path in sorted(set(paths)):
        name = Path(path).parent.name
        match = re.fullmatch(r'(session_[A-Za-z0-9]+_\d+)_task_\d+_\d+_\d+_mcap_v\d+', name)
        group = match.group(1) if match else 'unknown_source:' + path
        groups[group].append(path)
    queues = [collections.deque(sorted(groups[group], key=lambda p: (digest(p), p)))
              for group in sorted(groups, key=lambda g: (digest(g), g))]
    ordered = []
    while queues:
        following = []
        for queue in queues:
            ordered.append(queue.popleft())
            if queue:
                following.append(queue)
        queues = following
    return ordered


def _source_fields(ep):
    fields = {k: ep.get(k) for k in ('source_path', 'episode_id', 'recording_id',
            'recording_confident', 'source_frame_offset', 'task', 'skill_group', 'fps', 'n', 'width', 'height')}
    fields['recording_time_origin_ns'] = ep.get('metadata', {}).get('recording_time_origin_ns')
    return fields


def analyze_temporal(path, cfg):
    from .data import read_episode
    from .temporal import find_candidates
    started = time.monotonic()
    try:
        ep = read_episode(path)
        candidates = find_candidates(ep, cfg)
        fields = _source_fields(ep)
        for c in candidates:
            c.update(fields)
            c['qc_status'] = 'pending'
        return dict(source_path=path, status='ok', source=fields, candidates=candidates,
                    diagnostics=ep.get('temporal_diagnostics', {}),
                    elapsed_s=round(time.monotonic()-started, 3))
    except Exception as exc:
        if retryable_resource_error(exc):
            raise
        return dict(source_path=path, status='error', error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc()[-2500:], elapsed_s=round(time.monotonic()-started, 3))


def analyze_vision(record, cfg):
    from .data import read_episode, iter_video_frames
    from .vision import analyze_rgb
    started = time.monotonic()
    path = record['source_path']
    try:
        import cv2
        cv2.setNumThreads(cfg['runtime'].get('cv_threads', 1))
        ep = read_episode(path)
        result = analyze_rgb(ep, cfg, record['candidates'], iter_video_frames(path, ep))
        for c in result['candidates']:
            c.update(_source_fields(ep))
            c['qc_status'] = 'pending'
        return dict(source_path=path, status='ok', **result, elapsed_s=round(time.monotonic()-started, 3))
    except Exception as exc:
        if retryable_resource_error(exc):
            raise
        return dict(source_path=path, status='error', candidates=[], error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc()[-2500:], elapsed_s=round(time.monotonic()-started, 3))


def _batch(run, stage, inputs, worker, cfg):
    check_resources(cfg)
    ledger = run / f'{stage}.jsonl'
    prior = read_jsonl(ledger)
    completed = {r['source_path'] for r in prior}
    todo = [r for r in inputs if (r if isinstance(r, str) else r['source_path']) not in completed]
    input_sources = {r if isinstance(r, str) else r['source_path'] for r in inputs}
    stage_total = len(completed | input_sources)
    manifest = read_json(run/'run.json') if (run/'run.json').exists() else {}
    counts = collections.Counter(r['status'] for r in prior)
    n_candidates = sum(len(r.get('candidates', [])) for r in prior)
    segments = ledger.with_suffix('.parts')
    segments.mkdir(exist_ok=True)
    buffer = []
    last_flush = time.monotonic()
    state = dict(stage=stage,completed=len(prior),total=stage_total,
                 batch_inputs=len(inputs),source_total=manifest.get('input_count'),updated_at=utc_now())
    def flush():
        nonlocal last_flush
        if buffer:
            append_segment(ledger, list(buffer))
            buffer.clear()
        publish_progress(run/'progress.json',state)
        print(json.dumps(state),flush=True)
        last_flush=time.monotonic()
    print(json.dumps({'stage':stage, 'remaining':len(todo), 'already_done':len(prior)}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['runtime']['workers']) as pool:
        pending = {}
        try:
            iterator = iter(todo)
            def submit_one():
                item = next(iterator, None)
                if item is not None:
                    pending[pool.submit(worker, item, cfg)] = item
                    return True
                return False
            for _ in range(cfg['runtime']['workers'] * 2):
                submit_one()
            while pending:
                done, _ = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    state['updated_at']=utc_now()
                    state['waiting_for_sources']=len(pending)
                    flush()
                for f in done:
                    pending.pop(f)
                    result = f.result()
                    buffer.append(result)
                    prior.append(result)
                    counts[result['status']] += 1
                    n_candidates += len(result.get('candidates', []))
                    state = dict(stage=stage, completed=len(prior), total=stage_total, counts=dict(counts),
                                 batch_inputs=len(inputs),source_total=manifest.get('input_count'),
                                 candidate_starts=n_candidates, updated_at=utc_now(), pid=os.getpid(),
                                 last_result_at=utc_now(),
                                 last_source=result['source_path'], last_elapsed_s=result.get('elapsed_s'),
                                 last_error=result.get('error'))
                    if len(buffer)>=50 or time.monotonic()-last_flush>=30:
                        flush()
                    submit_one()
        except BaseException as exc:
            # __exit__ may wait many minutes for running workers. Publish the
            # failure before that wait so it cannot resemble healthy progress.
            print(json.dumps(dict(event='batch_failed_before_shutdown', level='error',
                  stage=stage, ledger=str(ledger), error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc(),
                  buffered_uncommitted=len(buffer), pending_futures=len(pending),
                  updated_at=utc_now(), pid=os.getpid()), ensure_ascii=False),
                  file=sys.stderr, flush=True)
            raise
    flush()
    return prior


def report(run, cfg):
    from .temporal import select_diverse
    from .artifacts import select_export_candidates
    from .release import load_release_policy, gate_review
    release_state=load_release_policy(run,cfg)
    temporal = read_jsonl(run/'temporal.jsonl')
    rgb = read_jsonl(run/'rgb.jsonl')
    exported = read_jsonl(run/'exports.jsonl')
    reviews = read_jsonl(run/'reviews.jsonl')
    verified = read_jsonl(run/'verification.jsonl')
    review_map = {r['clip_id']: r for r in reviews}
    verification_map = {r['clip_id']: r for r in verified}
    latest_exports = {r['clip_id']: r for r in exported}
    current_reviews = {cid: rv for cid,rv in review_map.items() if cid in latest_exports
                       and rv.get('video_sha256') == latest_exports[cid].get('checksums',{}).get('video.mp4')
                       and rv.get('video_sha256')}
    current_verifications = {cid: ve for cid,ve in verification_map.items() if cid in latest_exports
                       and ve.get('video_sha256') == latest_exports[cid].get('checksums',{}).get('video.mp4')
                       and ve.get('video_sha256')}
    finals = []
    for cid, r in latest_exports.items():
        if r['status'] != 'ok':
            continue
        rv = review_map.get(cid, {})
        ve = verification_map.get(cid, {})
        current_hash = r.get('checksums', {}).get('video.mp4')
        if (rv.get('verdict') == 'pass' and ve.get('status') == 'ok'
                and current_hash and rv.get('video_sha256') == current_hash
                and ve.get('video_sha256') == current_hash):
            finals.append(dict(r, review=rv, independent_verification=ve))
    all_temporal = [c for r in temporal if r['status']=='ok' for c in r.get('candidates',[])]
    all_rgb = [c for r in rgb if r['status']=='ok' for c in r.get('candidates',[])]
    # Only release-qualified passes can reserve intervals in this common pool.
    # A negative review blocks only its declared scope; alternatives need review.
    eligible = select_export_candidates(run, cfg)
    selected_ids = {clip_id(c, config_hash(cfg)) for c in eligible}
    # New sources can invalidate an older recording identity or expose a
    # cross-MCAP conflict; previous reviews never override the current full pool.
    finals = [record for record in finals if record['clip_id'] in selected_ids]
    event_blocked=collections.Counter()
    gated_finals=[]
    for record in finals:
        gate=gate_review(run,cfg,record,record['independent_verification'],record['review'],
                         selected_ids,release_state)
        if gate['qualified']:
            record['release_gate']=gate
            gated_finals.append(record)
        else:
            event_blocked.update(gate['reasons'])
    finals=gated_finals
    review_ready = sum(
        cid in selected_ids and record.get('status') == 'ok'
        and cid in current_verifications and current_verifications[cid].get('status') == 'ok'
        and current_reviews.get(cid, {}).get('verdict') in (None, 'pass')
        for cid, record in latest_exports.items())
    ready_records = [record for cid, record in latest_exports.items()
        if cid in selected_ids and record.get('status') == 'ok'
        and current_verifications.get(cid, {}).get('status') == 'ok'
        and current_reviews.get(cid, {}).get('verdict') in (None, 'pass')]
    ready_hist = dict(collections.Counter(str(r['candidate']['n_frames']) for r in ready_records))
    final_hist = dict(collections.Counter(str(r['candidate']['n_frames']) for r in finals))
    selected_hist = dict(collections.Counter(str(c['n_frames']) for c in eligible))
    upper_selection=dict(cfg['selection'], target_count=10**9)
    upper_selection.pop('frame_count_quotas',None)
    temporal_upper = select_diverse(all_temporal, dict(cfg, selection=upper_selection)) if all_temporal else []
    rejection = collections.Counter()
    temporal_rejection = collections.Counter()
    for r in temporal:
        temporal_rejection.update(r.get('diagnostics', {}).get('start_failure_counts', {}))
    for r in rgb:
        rejection.update(r.get('rejections', {}))
    run_manifest = read_json(run/'run.json') if (run/'run.json').is_file() else {}
    version = hashlib.sha256(json.dumps(run_manifest.get('code_manifest',{}),sort_keys=True).encode()).hexdigest()
    for record in finals:
        record.update(config_sha256=config_hash(cfg),code_manifest_sha256=version,qc_status='pass')
    source_total = int(run_manifest.get('input_count', len({r['source_path'] for r in temporal})))
    source_scanned = len({r['source_path'] for r in temporal})
    rgb_completed = {r['source_path'] for r in rgb}
    rgb_backlog = sum(r['status']=='ok' and bool(r.get('candidates')) and r['source_path'] not in rgb_completed
                      for r in temporal)
    state_path = run/'incremental_state.json'
    incremental = read_json(state_path) if state_path.is_file() else None
    target = cfg['selection']['target_count']
    status = 'stage_finished'
    if target_met(len(finals), final_hist, cfg):
        status = 'complete'
    elif incremental is not None:
        if target_met(review_ready, ready_hist, cfg):
            status = 'awaiting_visual_review'
        elif incremental.get('status') == 'artifact_recovery_required':
            status = 'artifact_recovery_required'
        elif source_scanned >= source_total and rgb_backlog == 0:
            status = 'exhausted'
        else:
            status = 'scanning' if (run/'writer.lock').exists() else 'ready_to_resume'
    summary = dict(updated_at=utc_now(), config_sha256=config_hash(cfg), target=target,
        source_total=source_total,source_scanned=source_scanned,source_remaining=max(0,source_total-source_scanned),
        rgb_source_backlog=rgb_backlog,execution_status=status,awaiting_visual_review=status=='awaiting_visual_review',
        last_batch_stage=(incremental or {}).get('last_batch_stage'),
        batch_index=(incremental or {}).get('batch_index',0),
        source_batch_size=(incremental or {}).get('source_batch_size'),
        temporal_sources_completed=len(temporal), temporal_sources_errors=sum(r['status']!='ok' for r in temporal),
        temporal_sources_with_candidates=sum(bool(r.get('candidates')) for r in temporal),
        temporal_candidate_starts=len(all_temporal), temporal_nonoverlap_capacity=len(temporal_upper),
        rgb_sources_completed=len(rgb), rgb_sources_errors=sum(r['status']!='ok' for r in rgb),
        rgb_machine_pass_starts=len({(c['source_path'],c['start_frame']) for c in all_rgb}),
        rgb_machine_pass_windows=len(all_rgb), selected_machine_pass=len(eligible),
        selected_machine_fingerprint=hashlib.sha256('\n'.join(sorted(selected_ids)).encode()).hexdigest(),
        review_ready_count=review_ready,
        review_ready_length_histogram=ready_hist, selected_length_histogram=selected_hist,
        requested_length_quotas=cfg['selection'].get('frame_count_quotas', {}),
        review_ready_quota_deficits={str(k): max(0,v-ready_hist.get(str(k),0))
            for k,v in cfg['selection'].get('frame_count_quotas',{}).items()},
        final_quota_deficits={str(k): max(0,v-final_hist.get(str(k),0))
            for k,v in cfg['selection'].get('frame_count_quotas',{}).items()},
        selected_unique_recordings=len({c['recording_id'] for c in eligible}),
        selected_unique_tasks=len({c.get('task') for c in eligible}),
        selected_unique_skill_groups=len({c.get('skill_group') for c in eligible}),
        exported_ok=sum(r['status']=='ok' for r in latest_exports.values()),
        export_error_attempts=sum(r.get('status')=='error' for r in exported),
        verification_errors=sum(r.get('status')=='error' for r in verification_map.values()),
        pending_review=sum(r['status']=='ok' and current_reviews.get(cid,{}).get('verdict') not in ('pass','reject')
                           for cid,r in latest_exports.items()),
        visual_review_pass=sum(r.get('verdict')=='pass' for r in current_reviews.values()),
        independent_verification_pass=sum(r.get('status')=='ok' for r in current_verifications.values()),
        final_qualified=len(finals), deficit=max(0,cfg['selection']['target_count']-len(finals)),
        complete_event_policy_required=release_state['required'],
        release_policy_sha256=release_state['policy_sha256'],
        release_policy_blockers=release_state['reasons'], event_review_blockers=dict(event_blocked),
        final_length_histogram=dict(collections.Counter(str(r['candidate']['n_frames']) for r in finals)),
        temporal_length_histogram=dict(collections.Counter(str(c['n_frames']) for c in temporal_upper)),
        temporal_rejections=dict(temporal_rejection), rgb_rejections=dict(rejection),
        production_complete=target_met(len(finals),final_hist,cfg))
    for length in cfg['selection'].get('frame_count_quotas', {}):
        atomic_json(run/'by_length'/f'{length}_frames.json', dict(
            n_frames=int(length), target_new_clips=cfg['selection']['frame_count_quotas'][length],
            status='pending_visual_and_event_review',
            technically_verified_candidates=[r for r in ready_records if r['candidate']['n_frames']==int(length)],
            final_qualified=[r for r in finals if r['candidate']['n_frames']==int(length)]))
    atomic_json(run/'final_manifest.json',finals)
    final_jsonl = ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in finals)
    atomic_bytes(run/'final_manifest.jsonl',final_jsonl.encode('utf-8'))
    with io.StringIO(newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['clip_id','video','first_frame','annotations','source_path','recording_id',
            'source_frame_offset','task','start_frame','end_frame','n_frames','fps','participating_hands',
            'metrics','config_sha256','code_manifest_sha256','qc_status','verification_status','video_sha256','file_checksums'])
        writer.writeheader()
        for r in finals:
            writer.writerow(dict(clip_id=r['clip_id'],video=r['video'],first_frame=r['first_frame'],annotations=r.get('annotations'),
                participating_hands=json.dumps(r['candidate'].get('participating_hands',[])),
                metrics=json.dumps(r['candidate'].get('metrics',{}),ensure_ascii=False,allow_nan=False),
                config_sha256=r['config_sha256'],code_manifest_sha256=r['code_manifest_sha256'],
                qc_status='pass',verification_status='ok',video_sha256=r['checksums']['video.mp4'],
                file_checksums=json.dumps(r['checksums'],sort_keys=True),
                **{k:r['candidate'].get(k) for k in ('source_path','recording_id','source_frame_offset','task','start_frame','end_frame','n_frames','fps')}))
        atomic_bytes(run/'final_manifest.csv',f.getvalue().encode('utf-8'))
    summary['final_manifest_sha256']={name:sha256_file(run/name) for name in
                                    ('final_manifest.json','final_manifest.jsonl','final_manifest.csv')}
    # Publish status only after every corresponding deliverable manifest was
    # fully written and read back; consumers can bind this status to these hashes.
    atomic_json(run/'summary.json',summary)
    return summary


def _incremental_state(run, cfg, **changes):
    path = run/'incremental_state.json'
    state = read_json(path) if path.is_file() else {
        'batch_index': 0, 'source_batch_size': cfg.get('runtime', {}).get('source_batch_size', 256),
        'status': 'scanning', 'last_batch_stage': 'initialized',
    }
    state.update(changes, updated_at=utc_now())
    atomic_json(path, state)
    return state


def _publish_incremental_summary(run, cfg, summary):
    _incremental_state(run, cfg, status=summary['execution_status'])
    publish_progress(run/'progress.json', dict(
        stage='all',status=summary['execution_status'],updated_at=utc_now(),
        source_scanned=summary['source_scanned'],source_total=summary['source_total'],
        source_remaining=summary['source_remaining'],batch_index=summary['batch_index'],
        last_batch_stage=summary['last_batch_stage'],summary=summary))


def _new_rgb(run, cfg):
    done = {r['source_path'] for r in read_jsonl(run/'rgb.jsonl')}
    records = [r for r in read_jsonl(run/'temporal.jsonl')
               if r['status']=='ok' and r.get('candidates') and r['source_path'] not in done]
    _incremental_state(run, cfg, last_batch_stage='rgb')
    if records:
        _batch(run, 'rgb', records, analyze_vision, cfg)


def _deliver_existing_pool(run, cfg):
    """Retry export failures before expanding source inventory unnecessarily.

    artifacts.select_export_candidates removes exhausted export retries and
    failed verification IDs, so the pool can advance to alternative intervals.
    A stable failure signature is a recovery issue, never a reason to claim
    enough reviewable videos or scan the entire dataset repeatedly.
    """
    from .artifacts import export_candidates, verify_exports
    seen = set()
    while True:
        _incremental_state(run, cfg, last_batch_stage='export')
        export_candidates(run, cfg)
        _incremental_state(run, cfg, last_batch_stage='verify')
        verify_exports(run, cfg)
        _incremental_state(run, cfg, last_batch_stage='report')
        summary = report(run, cfg)
        _publish_incremental_summary(run, cfg, summary)
        target = cfg['selection']['target_count']
        if (target_met(summary['final_qualified'], summary['final_length_histogram'], cfg)
                or target_met(summary['review_ready_count'], summary['review_ready_length_histogram'], cfg)):
            return summary
        if not target_met(summary['selected_machine_pass'], summary['selected_length_histogram'], cfg):
            return summary
        signature = (summary['selected_machine_fingerprint'], summary['review_ready_count'],
                     summary['export_error_attempts'], summary['verification_errors'])
        if signature in seen:
            _incremental_state(run, cfg, status='artifact_recovery_required')
            summary = report(run, cfg)
            _publish_incremental_summary(run, cfg, summary)
            raise RuntimeError('machine pool is sufficient but artifact delivery made no progress; '
                               'inspect export/verification errors before resuming')
        seen.add(signature)


def _run_incremental(run, files, cfg):
    size = cfg.get('runtime', {}).get('source_batch_size', 256)
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError('runtime.source_batch_size must be a positive integer')
    _incremental_state(run, cfg, status='scanning', source_batch_size=size)
    # Resume must first finish existing metadata/RGB/delivery work. A review
    # rejection may be replaceable from the accumulated pool without new I/O.
    existing = read_jsonl(run/'temporal.jsonl')
    if existing or read_jsonl(run/'rgb.jsonl') or read_jsonl(run/'exports.jsonl'):
        _new_rgb(run, cfg)
        summary = _deliver_existing_pool(run, cfg)
        if summary['execution_status'] in ('complete', 'awaiting_visual_review', 'exhausted'):
            return summary
    while True:
        completed = {r['source_path'] for r in read_jsonl(run/'temporal.jsonl')}
        remaining = [path for path in files if path not in completed]
        if not remaining:
            _incremental_state(run, cfg, last_batch_stage='report')
            summary = report(run, cfg)
            _publish_incremental_summary(run, cfg, summary)
            return summary
        batch = remaining[:size]
        previous = read_json(run/'incremental_state.json')
        _incremental_state(run, cfg, status='scanning', last_batch_stage='temporal',
                           batch_index=previous['batch_index']+1, batch_source_count=len(batch))
        _batch(run, 'temporal', batch, analyze_temporal, cfg)
        _new_rgb(run, cfg)
        summary = _deliver_existing_pool(run, cfg)
        if summary['execution_status'] in ('complete', 'awaiting_visual_review', 'exhausted'):
            return summary


def run_pipeline(cfg, run_id, stage='all', inventory_path=None, limit=None, resume=False):
    check_resources(cfg)
    root = Path(cfg['paths']['output_root'])
    run = root/'runs'/run_id
    existed = run.exists()
    if existed and not resume:
        raise ValueError(f'run exists: {run}; use --resume with identical config')
    run.mkdir(parents=True, exist_ok=True)
    lock = run/'writer.lock'
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'writer lock exists: {lock}; inspect recorded host/pid before recovery')
    with os.fdopen(fd,'w') as f:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname(),started_at=utc_now()),f)
    try:
        if (run/'recovery_incomplete.json').exists():
            raise RuntimeError('incomplete analysis recovery; preserve this run and create a new run before importing again')
        confighash = config_hash(cfg)
        if existed:
            manifest=read_json(run/'run.json')
            if manifest['config_sha256']!=confighash:
                raise ValueError('config changed; create a new run, never mix thresholds')
            current_code = code_manifest(Path(__file__).resolve().parents[1])
            if manifest['code_manifest'] != current_code:
                raise ValueError('code changed; restore the frozen version or create a new run')
            if manifest.get('inventory_sha256') != sha256_file(run/'inventory.json'):
                raise ValueError('input inventory changed; create a new run')
            files=read_json(run/'inventory.json')
            if inventory_path is not None:
                requested = stratified_inventory(read_json(inventory_path)['paths'])
                if limit is not None:
                    if limit < 1:
                        raise ValueError('limit must be positive')
                    requested = requested[:limit]
                if requested != files:
                    raise ValueError('resume cannot change or append the frozen input inventory')
            elif limit is not None and limit != len(files):
                raise ValueError('resume cannot change the frozen input inventory limit')
        else:
            files=read_json(inventory_path)['paths'] if inventory_path else inventory(cfg['paths']['source_root'])
            files=stratified_inventory(files)
            if limit is not None:
                if limit < 1:
                    raise ValueError('limit must be positive')
                files=files[:limit]
            atomic_json(run/'inventory.json',files)
            atomic_json(run/'config.json',cfg)
            atomic_json(run/'run.json',dict(config_sha256=confighash,created_at=utc_now(),host=socket.gethostname(),
                code_manifest=code_manifest(Path(__file__).resolve().parents[1]), input_count=len(files),
                inventory_sha256=sha256_file(run/'inventory.json'),
                inventory_order='recording_round_robin_sha256_v1',
                required_complete_atomic_event=True,
                calibration_status='engineering_defaults_not_yet_visually_calibrated', quality_status='pending_review'))
            atomic_json(run/'release_policy.json',dict(
                version='strict_first_frame_plus_complete_atomic_event_v2',
                config_sha256=confighash,required_complete_atomic_event=True,
                minimum_pre_context_frames=6,minimum_post_context_frames=6,
                status='draft_requires_frozen_definition_and_calibration_evidence',
                event_definition_file=None,calibration_file=None))
        if stage == 'stream':
            from .streaming import run_streaming
            result=run_streaming(run,files,cfg)
            print(json.dumps(result),flush=True)
            return result
        if stage == 'all':
            result = _run_incremental(run, files, cfg)
            print(json.dumps(result),flush=True)
            return result
        if stage == 'temporal':
            _batch(run,'temporal',files,analyze_temporal,cfg)
        if stage == 'rgb':
            records=[r for r in read_jsonl(run/'temporal.jsonl') if r['status']=='ok' and r.get('candidates')]
            _batch(run,'rgb',records,analyze_vision,cfg)
        if stage == 'export':
            from .artifacts import export_candidates
            export_candidates(run,cfg)
        if stage == 'verify':
            from .artifacts import verify_exports
            verify_exports(run,cfg)
        result=report(run,cfg)
        publish_progress(run/'progress.json',dict(stage=stage,status='stage_finished',updated_at=utc_now(),summary=result))
        print(json.dumps(result),flush=True)
        return result
    finally:
        lock.unlink(missing_ok=True)
