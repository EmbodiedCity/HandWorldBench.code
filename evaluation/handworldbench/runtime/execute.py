"""Metric-scoped observation handlers and scoring over durable artifacts."""
import copy
from pathlib import Path
import time
from handworldbench.runtime.storage import read,write,cache_lock,digest

def dependency(task,name):
    matches=[v for key,v in task['dependency_results'].items() if key.startswith(name+'_')]
    if len(matches)!=1:raise ValueError(f'Require one {name} dependency')
    return matches[0]

def execute(task,run,models):
    config=task['config'];root=Path(config['cache_root']);row=task['row'];name=task.get('observation')
    if task['kind']=='score':return score(task)
    folder=root/'observations'/task['cache_key']
    with cache_lock(folder):
        if (folder/'receipt.json').exists():
            value=read(folder/'receipt.json')
            if name=='frames':
                from handworldbench.data.inputs import validate_output_protocol
                validate_output_protocol(row,value)
            return dict(value,cache_hit=True)
        start=time.monotonic()
        if name=='frames':
            from handworldbench.observations.video import decode
            value=decode(row['video'],row['video_sha256'],root,'prediction',count=config.get('sampling',{}).get('frames',16))
            from handworldbench.data.inputs import validate_output_protocol
            validate_output_protocol(row,value)
        elif name=='reference_frames':
            from handworldbench.observations.reference import images
            value=images(task['reference'],row,dependency(task,'frames')['frames'],root)
        elif name=='reference_geometry':
            from handworldbench.observations.reference import observe
            value=observe(task['reference'],row,dependency(task,'frames')['frames'],root,dependency(task,'reference_frames'))
        elif name=='hand_detection':
            from handworldbench.observations.shared_hands import predict
            from handworldbench.observations.video import images
            frames=dependency(task,'frames');geometry=dependency(task,'reference_geometry')
            prediction,timing=predict(images(frames['frames']),row['category'],config,
                role='prediction',video_sha256=row['video_sha256'])
            value = dict(prediction=prediction, prediction_frames=frames['frames'],
                         reference_frames=geometry['reference_images'], geometry=geometry,
                         timing=timing, hand_localization_vlm_requests=0)
            if task.get('include_reference_detections', True):
                reference, ref_timing = predict(images(geometry['reference_images']), row['category'], config,
                    role='reference', video_sha256=row['reference']['sha256'])
                value['reference'] = reference
                timing['reference'] = ref_timing
        elif name=='hand_keypoints':
            from handworldbench.observations.hands import KeypointEstimator,associate
            from handworldbench.observations.video import images
            detections=dependency(task,'hand_detection')
            load_seconds=0.
            if 'keypoints' not in models:
                started=time.monotonic();models['keypoints']=KeypointEstimator(config['models']['keypoints'],batch=config.get('batch',{}).get('keypoints',32));load_seconds=time.monotonic()-started
            raw,timing=models['keypoints'].predict(images(detections['prediction_frames']),detections['prediction'])
            timing['model_load_seconds']=load_seconds
            value=dict(frames=associate(detections['geometry']['frames'],raw),raw_detections=raw,timing=timing)
        elif name=='hand_masks':
            from handworldbench.observations.segmentation import hands
            value=hands(dependency(task,'hand_detection'),models,config,root)
        else:
            from handworldbench.backends.dispatch import observe
            value=observe(task,models,folder)
        value['observation_seconds']=time.monotonic()-start
        value['cache_hit']=False
        write(folder/'receipt.json',value);return value

def score(task):
    from handworldbench.registry import METRICS
    metric=task['metric_id'];row=task['row']
    if metric in ('pose_control','keypoint_coverage'):
        from handworldbench.metrics.keypoints import score as keypoints
        geometry=dependency(task,'reference_geometry')
        value=keypoints(geometry['frames'],dependency(task,'hand_keypoints')['frames'],metric)
    else:
        observations=list(task['dependency_results'].values())
        if len(observations)!=1:raise ValueError('Expected one metric evidence bundle')
        value=copy.deepcopy(observations[0]['metrics'][metric])
    value.update(case_id=row['case_id'],model=row['model'],category=row['category'],metric_id=metric,metric_protocol=METRICS[metric].protocol,video_protocol=row.get('video_protocol','declared'))
    return value
