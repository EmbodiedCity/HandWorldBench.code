"""Additive curation: preserve baseline, prefer new scenes and attractive frames.

The aesthetic model is an image-preference proxy, never a replacement for the
unchanged per-frame hand, camera, RGB, or sustained palm-motion constraints.
"""
import collections
import copy
import functools
import math
import numpy as np
from .storage import read_json, sha256_file


@functools.lru_cache(maxsize=8)
def baseline(path, digest):
    if sha256_file(path) != digest:
        raise ValueError('Preserved baseline candidate identity changed')
    return read_json(path)


def beauty_score(c):
    a=c['aesthetics']
    # First-frame appearance matters for the user's final contact sheets.
    return .45*a['first'] + .35*a['p10'] + .20*a['median']


def beauty_rank(c):
    from .curation import rank
    return (-beauty_score(c), -c['n_frames'], rank(c))


def select_expansion(candidates, cfg):
    from .curation import select_curated, FingerprintIndex, task_key
    from .temporal import _fingerprints
    x=cfg['expansion']
    seeds=baseline(x['seed_candidates'],x['seed_candidates_sha256'])
    old_recordings=collections.Counter(c['recording_id'] for c in seeds)
    old_intervals=collections.defaultdict(list)
    for c in seeds:
        old_intervals[c['recording_id']].append((c['source_frame_offset']+c['start_frame'],c['source_frame_offset']+c['end_frame']))
    fp=FingerprintIndex(cfg['selection'].get('near_duplicate_hamming',4))
    for c in seeds:
        f=_fingerprints(c)
        if f is None:raise ValueError('Baseline fingerprints missing')
        fp.add(f)
    rejected=collections.Counter();eligible=[]
    for c in candidates:
        rec=c.get('recording_id')
        s=c.get('source_frame_offset',0)+c.get('start_frame',0)
        e=c.get('source_frame_offset',0)+c.get('end_frame',0)
        if any(s<b and a<e for a,b in old_intervals[rec]):
            rejected['preserved_baseline_overlap']+=1
        elif rec in old_recordings and any(abs(c.get('recording_time_origin_ns',0)-seed['recording_time_origin_ns'])>1e9/seed['fps'] for seed in seeds if seed['recording_id']==rec):
            rejected['preserved_baseline_origin_conflict']+=1
        elif _fingerprints(c) is not None and fp.nearby(_fingerprints(c)):
            rejected['preserved_baseline_near_duplicate']+=1
        else:eligible.append(c)
    inner=copy.deepcopy(cfg);inner.pop('expansion')
    # The base selector still performs all schema/identity/near-duplicate gates.
    # Defer the actual cap until diversity and aesthetics have been considered.
    inner['motion_selection']['target_count']=max(1,len(eligible))
    pool,diag=select_curated(eligible,inner)
    target=len(pool) if x['target_total'] is None else int(x['target_total'])-len(seeds)
    if target<0:raise ValueError('Target smaller than preserved baseline')
    tasks=collections.Counter(task_key(c.get('task')) for c in seeds)
    recordings=collections.Counter(old_recordings);scenes=collections.Counter()
    vectors=[]
    for c in seeds:
        v=np.asarray(c['aesthetics']['embedding'],float)
        if v.shape!=(768,) or not np.isfinite(v).all() or np.linalg.norm(v)<.5:
            raise ValueError('Baseline scene embedding invalid')
        vectors.append(v/np.linalg.norm(v))
    vectors=np.asarray(vectors, dtype=float).reshape(-1,768)
    for c in pool:
        v=np.asarray(c['aesthetics']['embedding'],float);v/=np.linalg.norm(v)
        scenes[c['scene_group']]=max(scenes[c['scene_group']],int(np.count_nonzero(vectors@v>=x['scene_similarity'])))
    chosen=[]
    while pool and len(chosen)<target:
        capped=[c for c in pool if recordings[c['recording_id']]>=cfg['motion_selection']['max_per_recording']]
        rejected['recording_cap_including_baseline']+=len(capped)
        for c in capped:pool.remove(c)
        if not pool:break
        c=min(pool,key=lambda c:(recordings[c['recording_id']],
            max(tasks[c['normalized_task']],scenes[c['scene_group']]),
            tasks[c['normalized_task']]+scenes[c['scene_group']],beauty_rank(c)))
        pool.remove(c)
        c=dict(c)
        c['aesthetic_preference_score']=beauty_score(c)
        c['selection_scope']='new recording/task/visual-scene preference; then first-frame and temporal image aesthetics; length retained within legal source proposals'
        chosen.append(c)
        recordings[c['recording_id']]+=1;tasks[c['normalized_task']]+=1;scenes[c['scene_group']]+=1
    if cfg['curation'].get('length_priority'):
        chosen.sort(key=lambda c:(-c['n_frames'],beauty_rank(c)))
    rejected['expansion_target_cap']+=len(pool)
    rejected.update(diag['rejections'])
    diag.update(rejections=dict(rejected),preserved_baseline=len(seeds),target_total=x['target_total'],
        unique_tasks=len({c['normalized_task'] for c in chosen}),
        unique_recordings=len({c['recording_id'] for c in chosen}),
        selected_visual_scene_groups=len({c['scene_group'] for c in chosen}),
        scene_counts=dict(collections.Counter(c['scene_category'] for c in chosen)),
        preference=('longest quality-passing candidate first within dedup/diversity constraints; aesthetics breaks length ties' if cfg['curation'].get('length_priority') else 'recording/task/visual-scene coverage, then .45*first+.35*p10+.20*median aesthetics, then length'),
        aesthetic_scope='sampled image-preference proxy; background beauty and visual corruption still require human review')
    return chosen,diag
