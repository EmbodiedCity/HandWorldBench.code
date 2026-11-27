"""Shared object tracks and independently detected hand-region appearance."""
import copy
from pathlib import Path
import time
from handworldbench.runtime.storage import read,write,digest,cache_lock,artifact_digest
from handworldbench.runtime.execute import dependency
from handworldbench.observations.vision import Observer

def declared_reference_entities(config,reference_sha256):
    """Use the reference entity inventory when the dataset provides it.

    Entity identity describes a physical object independently of its changing
    location. The declaration is bound to the reference video and cached as
    evidence; localization still runs independently on each role's own pixels.
    """
    declaration=config.get('background',{}).get('reference_entities')
    if declaration is None:return None
    if not isinstance(declaration,dict) or declaration.get('reference_sha256')!=reference_sha256:
        raise ValueError('Reference entity inventory: reference SHA mismatch')
    entities=declaration.get('entities');localization=declaration.get('localization')
    if not isinstance(entities,list) or not 1<=len(entities)<=6:
        raise ValueError('Reference entity inventory requires one to six objects')
    if any(not isinstance(e,dict) or e.get('type')!='object' for e in entities):
        raise ValueError('Reference entity inventory must contain only object regions')
    if not isinstance(localization,dict) or not localization.get('source_frame_indices') or not localization.get('evidence'):
        raise ValueError('Reference entity inventory requires source-frame evidence')
    from region_observations import clean_entities
    return clean_entities(entities),dict(requests=0,source='dataset_reference_entities',
                                       reference_sha256=reference_sha256,localization=copy.deepcopy(localization))

def precomputed_reference_objects(path,reference_sha256):
    """Load the reference object inventory, masks and per-frame geometry shipped with the dataset.

    When a case provides `reference_objects`, the reference side of Entity Consistency and Trajectory
    Accuracy uses these fixed observations; objects are not selected, localized or tracked again on the
    reference video. The generated video is still observed with the configured models.
    """
    path=Path(path).resolve();data=read(path)
    if data.get('reference_sha256')!=reference_sha256:raise ValueError('Reference objects: reference SHA mismatch')
    entities=data.get('entities')
    if not isinstance(entities,list) or not 1<=len(entities)<=6 or any(not isinstance(e,dict) or e.get('type')!='object' for e in entities):
        raise ValueError('Reference objects require one to six object entities')
    tracks=copy.deepcopy(data.get('tracks') or {})
    if set(tracks)-{e['id'] for e in entities}:raise ValueError('Reference object tracks name unknown entities')
    for series in tracks.values():
        for record in series.values():
            if record.get('mask'):
                mask=(path.parent/record['mask']).resolve();mask.relative_to(path.parent)
                if not mask.is_file():raise ValueError('Reference object mask missing: '+record['mask'])
                record['mask']=str(mask)
    return dict(entities=entities,tracks=tracks,geometry=data.get('geometry') or {},selection=dict(source='dataset_reference_objects'),localization={})

def track(frames,entities,role,folder):
    import sam3_entity_backend
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);receipt=folder/'tracks.json'
    sam3_entity_backend.run(dict(frames=frames,entities=entities,role=role,folder=str(folder),device='cuda:0',result=str(receipt)))
    return read(receipt)['tracks']

def objects(task,models,out):
    from region_observations import locate_objects,clean_tracks
    from entity_selection import select_and_localize
    from handworldbench.backends.context import CONFIG
    frames=dependency(task,'frames')['frames'];reference=dependency(task,'reference_frames');refs=reference['reference_images'];cfg=read(task['config']);root=Path(CONFIG['cache_root']);clock=time.monotonic()
    usage_before=dict(models.get('vlm_usage',{}))
    signature=dict(reference=reference['source']['sha256'],rgb=[f['decoded_rgb_sha256'] for f in refs],model=CONFIG['models']['vision'],segmentation=CONFIG['models']['sam3'],protocol='manipulated_objects')
    preset=cfg.get('reference_objects')
    declared=None if preset else declared_reference_entities(cfg,reference['source']['sha256'])
    if declared is not None:signature['reference_entities']=cfg['background']['reference_entities']
    if preset:refdata=precomputed_reference_objects(preset,reference['source']['sha256'])
    else:
        with cache_lock(root/'reference_objects',digest(signature)) as folder:
            if (folder/'receipt.json').exists():refdata=read(folder/'receipt.json')
            else:
                observer=Observer(models,CONFIG['models']['vision'],'reference',CONFIG.get('batch',{}).get('object_localization',1))
                if declared is None:
                    entities,selection,prompted,audit=select_and_localize(observer,refs,folder)
                else:
                    entities,selection=declared
                    write(folder/'selection/reference_inventory.json',cfg['background']['reference_entities'])
                    prompted,_,audit=locate_objects(observer,refs,entities,folder/'localization',stride=4,workers=1,role='reference')
                geometry={};tracks=clean_tracks(track(refs,prompted,'reference',folder/'tracking'),entities,refs,'reference',geometry_out=geometry) if entities else {}
                refdata=dict(entities=entities,tracks=tracks,geometry=geometry,selection=selection,localization=audit);write(folder/'receipt.json',refdata)
    reference_usage={k:v-usage_before.get(k,0) for k,v in models.get('vlm_usage',{}).items()}
    entities=refdata['entities'];observer=Observer(models,CONFIG['models']['vision'],'prediction',CONFIG.get('batch',{}).get('object_localization',1))
    prompted,_,audit=locate_objects(observer,frames,entities,out/'object_localization',stride=4,workers=1,role='prediction')
    geometry={};tracks=clean_tracks(track(frames,prompted,'prediction',out/'object_tracking'),entities,frames,'prediction',geometry_out=geometry) if entities else {}
    shared=dict(entities=entities,reference=refdata['tracks'],prediction=tracks,reference_geometry=refdata['geometry'],prediction_geometry=geometry,reference_frames=refs,prediction_frames=frames)
    usage={k:v-usage_before.get(k,0) for k,v in models.get('vlm_usage',{}).items()}
    value = dict(shared_objects=shared, vlm_usage=usage, vlm_usage_scope='task',
                 vlm_usage_by_role={'reference': reference_usage,
                                    'prediction': {k: v-reference_usage.get(k, 0) for k, v in usage.items()}})
    if task.get('include_trajectory_score', True):
        from region_scoring import evaluate_geometric_trajectory
        from handworldbench.backends.normalization import trajectory_predictions
        from handworldbench.backends.spatial import trajectory_reference

        indices = list(range(len(frames)))
        filled, interpolation = trajectory_predictions(geometry, indices, frames)
        mapped = trajectory_reference(cfg, refdata['geometry'], refdata['tracks'], frames)
        if entities:
            result = evaluate_geometric_trajectory(entities, mapped, filled, indices, frames)
        else:
            result = dict(score=None, status='missing_reference_object_geometry',
                          reason='No reference object region was identified')
        result['interpolated_units'] = sum(len(v['interpolated_sample_indices']) for v in interpolation.values())
        result['remaining_missing_points'] = sum(
            v['original_missing']-len(v['interpolated_sample_indices']) for v in interpolation.values()
        )
        result['interpolation'] = interpolation
        value.update(metrics={'trajectory_accuracy': result}, score=result.get('score'),
                     interpolation=interpolation)
    value['stage_seconds'] = {'object_localization_tracking': time.monotonic()-clock}
    return value

def entities(task,models,out):
    from region_observations import clean_tracks
    from handworldbench.observations.entity_features import encode
    from region_scoring import evaluate_region_appearance,aggregate_region_appearance
    from handworldbench.backends.context import CONFIG
    shared=task['shared_objects'];hands=task['unified_hands'];cfg=read(task['config']);clock=time.monotonic()
    if not shared['entities']:
        value=dict(score=None,status='missing_reference_entity_region',reason='No reference object region was identified')
        return dict(metrics={'entity_consistency':value},score=None,stage_seconds={'entity_encoding':time.monotonic()-clock},hand_localization_vlm_requests=0)
    from handworldbench.metrics.population import reference_hand_sides
    sides=reference_hand_sides(hands['geometry']['native_frames'])
    population=[dict(id=side+'_hand',type='hand',anatomical_side=side,description=side+' hand',weight=1.) for side in sides]+shared['entities']
    if not population:
        value=aggregate_region_appearance({})
        return dict(metrics={'entity_consistency':value},score=None,stage_seconds={'entity_encoding':time.monotonic()-clock},hand_localization_vlm_requests=0)
    tracks={};geometry={}
    for role in ('reference','prediction'):
        frames=shared[role+'_frames'];hand_entities=[e for e in population if e['type']=='hand'];hand_ids={e['id'] for e in hand_entities}
        raw=copy.deepcopy({eid:series for eid,series in hands[role].items() if eid in hand_ids})
        for eid,series in raw.items():
            for i,record in series.items():record.update(source_rgb_sha256=frames[int(i)]['decoded_rgb_sha256'],mask_source_role=role,mask_sha256=artifact_digest(record['mask']),exists=bool(record['area_pixels']))
        hand_geometry={};clean=clean_tracks(raw,hand_entities,frames,role,geometry_out=hand_geometry)
        tracks[role]={**shared[role],**clean};geometry[role]={**shared[role+'_geometry'],**hand_geometry}
    features,timing=encode(population,{role:shared[role+'_frames'] for role in ('reference','prediction')},tracks,geometry,CONFIG,models)
    value=evaluate_region_appearance(population,features['reference'],features['prediction'],list(range(len(shared['prediction_frames']))))
    return dict(metrics={'entity_consistency':value},score=value.get('score'),stage_seconds={'entity_encoding':time.monotonic()-clock},encoding=timing,hand_localization_vlm_requests=0)

def run(task,models,out):
    return entities(task,models,out) if 'entity_consistency' in task['metrics'] else objects(task,models,out)
