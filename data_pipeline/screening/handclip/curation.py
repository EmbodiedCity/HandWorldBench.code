"""Scene/task top-K selection over fully checked machine candidates."""
import collections
import math
import re
import numpy as np

from .temporal import _fingerprints
from .length_contract import validate_selection, validate_interval


def task_key(text):
    return re.sub(r'\s+',' ',re.sub(r'[^\w\s]+',' ',str(text or 'unknown_task').casefold())).strip() or 'unknown_task'


class FingerprintIndex:
    """Exact Hamming query via h+1 bands; no approximate false negatives."""
    def __init__(self,threshold):
        self.threshold=threshold;self.rows=[];self.buckets=collections.defaultdict(set)
        count=min(64,max(1,threshold+1));self.bands=[];offset=0
        for i in range(count):
            size=64//count+(i<64%count);self.bands.append((offset,(1<<size)-1));offset+=size
    def nearby(self,fp):
        if self.threshold<0:return False
        possible=set()
        for band,(offset,mask) in enumerate(self.bands):possible.update(self.buckets[(band,(fp[0]>>offset)&mask)])
        if self.threshold>=64:possible=set(range(len(self.rows)))
        return any(all((a^b).bit_count()<=self.threshold for a,b in zip(fp,self.rows[i])) for i in possible)
    def add(self,fp):
        i=len(self.rows);self.rows.append(fp)
        for band,(offset,mask) in enumerate(self.bands):self.buckets[(band,(fp[0]>>offset)&mask)].add(i)


def rank(c):
    a=c['aesthetics'];rgb=c['metrics']['rgb']
    return (-c['n_frames'],-a['p10'],-a['median'],rgb['max_anchor_displacement_diag'],
        -min(rgb['first_hand_laplacian'].values()),c['source_path'],c['start_frame'])


def select_curated(candidates,cfg):
    if cfg.get('expansion'):
        from .expansion import select_expansion
        return select_expansion(candidates,cfg)
    validate_selection(cfg['selection'])
    settings=cfg['curation'];topk=settings['top_k_per_scene_task'];reasons=collections.Counter()
    validated=[];origins=collections.defaultdict(list);identities=collections.defaultdict(set)
    for c in candidates:
        try:
            start,end,offset=(c[k] for k in ('start_frame','end_frame','source_frame_offset'))
            if any(type(v) is not int for v in (start,end,offset)) or min(start,offset)<0 or end<=start:raise ValueError('invalid_interval')
            validate_interval(c,cfg['selection'])
            if cfg.get('motion_salience') is not None and c.get('metrics',{}).get('motion_salience',{}).get('status')!='pass':
                raise ValueError('motion_salience_not_passed')
            if not c.get('recording_confident') or not c.get('recording_id') or not c.get('source_path'):raise ValueError('unknown_recording')
            if c.get('aesthetics',{}).get('status')!='ok':raise ValueError('aesthetic_unknown')
            if not isinstance(c['aesthetics'].get('scene',{}).get('label'),str) or not c['aesthetics']['scene']['label']:raise ValueError('scene_unknown_schema')
            if not all(math.isfinite(c['aesthetics'][k]) for k in ('p10','median','first')):raise ValueError('aesthetic_unknown')
            rgb=c['metrics']['rgb'];sharpness=rgb['first_hand_laplacian']
            if not isinstance(sharpness,dict) or not sharpness or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in sharpness.values()):raise ValueError('first_hand_quality_unknown')
            if not isinstance(rgb.get('max_anchor_displacement_diag'),(int,float)) or not math.isfinite(rgb['max_anchor_displacement_diag']):raise ValueError('camera_quality_unknown')
            if _fingerprints(c) is None:raise ValueError('missing_visual_fingerprints')
            if type(c.get('recording_time_origin_ns')) is not int:raise ValueError('unknown_recording_time_origin')
            if not math.isfinite(c['fps']) or c['fps']<=0:raise ValueError('unknown_fps')
            origins[c['recording_id']].append((c['recording_time_origin_ns'],1e9/c['fps']))
            identities[c['source_path']].add((c['recording_id'],offset));validated.append(c)
        except (KeyError,TypeError,ValueError) as error:reasons[str(error)]+=1
    conflicts={r for r,values in origins.items() if max(x[0] for x in values)-min(x[0] for x in values)>min(x[1] for x in values)}
    selected=[];intervals=collections.defaultdict(list);counts=collections.Counter();group_recordings=collections.defaultdict(set)
    # Avoid rebuilding/copying an ever growing matrix for every candidate.
    cluster_vectors=np.empty((len(validated),768),dtype=np.float64);cluster_ids=[];source_clusters={}
    fingerprints=FingerprintIndex(cfg['selection'].get('near_duplicate_hamming',4));overlap_sources=set()
    recording_counts=collections.Counter()
    if settings.get('length_priority'):
        from .expansion import beauty_rank
        ordering=lambda c:(-c['n_frames'],beauty_rank(c))
    elif settings.get('aesthetic_priority'):
        from .expansion import beauty_rank
        ordering=beauty_rank
    else:ordering=rank
    for c in sorted(validated,key=ordering):
        rec=c['recording_id']
        if recording_counts[rec]>=cfg.get('motion_selection',{}).get('max_per_recording',float('inf')):
            reasons['recording_clip_cap']+=1;continue
        if rec in conflicts or len(identities[c['source_path']])!=1:reasons['recording_identity_conflict']+=1;continue
        scene=c['aesthetics']['scene']['label'];task=task_key(c.get('task'))
        if settings.get('scene_grouping')=='visual_cluster':
            source=c['source_path']
            source_window=(source,c['start_frame'],c['end_frame'])
            if source_window not in source_clusters:
                vector=np.asarray(c['aesthetics'].get('embedding'),dtype=float)
                if vector.shape!=(768,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<.5:
                    reasons['scene_embedding_unknown']+=1;continue
                vector=vector/np.linalg.norm(vector)
                similarities=cluster_vectors[:len(cluster_ids)]@vector if cluster_ids else np.empty(0)
                if len(similarities) and similarities.max()>=settings.get('scene_cluster_cosine',.94):
                    cluster=cluster_ids[int(similarities.argmax())]
                else:
                    # Deterministic leader identity, distinct from a semantic room label.
                    import hashlib
                    cluster='visual_'+hashlib.sha256((source+':'+str(c['start_frame'])+':'+str(c['end_frame'])).encode()).hexdigest()[:12]
                    cluster_vectors[len(cluster_ids)]=vector;cluster_ids.append(cluster)
                source_clusters[source_window]=cluster
            scene_group=source_clusters[source_window]
        else:scene_group=scene
        group=(scene_group,task)
        if counts[group]>=topk:reasons['scene_task_topk']+=1;continue
        if settings.get('distinct_recordings_per_group',True) and rec in group_recordings[group]:reasons['same_recording_in_scene_task']+=1;continue
        start,end=c['source_frame_offset']+c['start_frame'],c['source_frame_offset']+c['end_frame']
        if any(start<b and a<end for a,b in intervals[rec]):
            reasons['cross_source_overlap']+=1;overlap_sources.add(c['source_path']);continue
        fp=_fingerprints(c)
        if fingerprints.nearby(fp):reasons['visual_near_duplicate']+=1;continue
        row=dict(c);row['scene_category']=scene;row['scene_group']=scene_group;row['normalized_task']=task;row['scene_task_rank']=counts[group]+1
        row['scene_identified']=scene!='unknown_scene'
        row['selection_scope']='longest-first within fully checked source proposals; no claim of global aesthetic optimum'
        selected.append(row);counts[group]+=1;group_recordings[group].add(rec);intervals[rec].append((start,end));fingerprints.add(fp)
        recording_counts[rec]+=1
    # Balanced round-robin viewing/dispatch order; preserves the selected set.
    groups=collections.defaultdict(list)
    for row in selected:groups[(row['scene_group'],row['normalized_task'])].append(row)
    balanced=[]
    for level in range(topk):
        for group in sorted(groups):
            if len(groups[group])>level:balanced.append(groups[group][level])
    if cfg.get('motion_selection'):
        # Across groups choose the longest first, then prefer new tasks/recordings.
        # All candidates have passed the same motion gate; no quota relaxation.
        target=cfg['motion_selection']['target_count']
        if target is None:target=len(balanced)
        pool=list(balanced);balanced=[];task_counts=collections.Counter();scene_counts=collections.Counter()
        while pool and len(balanced)<target:
            chosen=min(pool,key=lambda c:(-c['n_frames'],task_counts[c['normalized_task']],scene_counts[c['scene_group']],rank(c)))
            pool.remove(chosen);balanced.append(chosen)
            task_counts[chosen['normalized_task']]+=1;scene_counts[chosen['scene_group']]+=1
        reasons['target_count_cap']+=len(pool)
    return balanced,dict(rejections=dict(reasons),scene_task_groups=len(groups),
        unique_tasks=len({c['normalized_task'] for c in balanced}),unique_recordings=len({c['recording_id'] for c in balanced}),
        scene_counts=dict(collections.Counter(c['scene_category'] for c in balanced)),
        scene_grouping=settings.get('scene_grouping','coarse_inferred_category'),visual_scene_groups=len(cluster_ids),
        overlap_reselection_sources=sorted(overlap_sources))
