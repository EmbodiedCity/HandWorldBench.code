"""Scoped backend calls over portable configuration and shared observations."""
import copy
import os
from pathlib import Path
import sys
from handworldbench.runtime.storage import read,write
from handworldbench.metrics.normalization import normalize
from handworldbench.data.annotations import semantic,memory,state as bind_state

METRICS_BY_OBSERVATION={
    'image_quality':('musiq',),'visual_judgment':('visual_plausibility',),'video_features':('jepa_similarity',),
    'hand_anatomy':('hand_morphology',),'background_features':('background_consistency',),'background_geometry':('photometric_consistency',),
    'action_evidence':('action_adherence',),'event_evidence':('event_adherence',),'causal_evidence':('causal_plausibility',),
    'state_evidence':('state_progression','evolution_correctness','state_regulation'),'memory_evidence':('memory_consistency',),
    'object_tracks':('trajectory_accuracy',),'entity_features':('entity_consistency',),
}
BACKENDS={'image_quality':'musiq','visual_judgment':'pavrm','video_features':'jepa','hand_anatomy':'malformed','background_features':'background','background_geometry':'photo','action_evidence':'aux','event_evidence':'aux','causal_evidence':'aux','state_evidence':'state','memory_evidence':'state','object_tracks':'region','entity_features':'region'}

def initialize(config):
    engine=Path(__file__).resolve().parents[1]/'_vendor/engine';suite=engine/'engines/classic/metric_suite';backend=Path(__file__).parent
    sys.path[:0]=[str(p) for p in [backend,backend/'regions_support',suite,engine/'engines/classic',engine,engine/'engines/new_aux',engine/'engines/state_four/runtime']]
    from handworldbench.backends.context import S
    for name,src in [('models',config.get('models_root')),('data/classic',config.get('classic_data'))]:
        if not src:continue
        target=S/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() and not target.is_symlink():
            try:target.symlink_to(Path(src).resolve(),target_is_directory=True)
            except FileExistsError:pass
        if target.resolve()!=Path(src).resolve():
            raise ValueError('Cache resource binding differs for '+name+'; use a cache_root matching the configured resources')
    return engine

def observe(task,models,folder):
    config=task['config'];engine=initialize(config);item=task['reference'];row=task['row'];name=task['observation'];metrics=METRICS_BY_OBSERVATION[name]
    if name=='state_evidence':metrics=tuple(task['requested_metrics'])
    if name == 'object_tracks' and not task.get('include_trajectory_score', True):
        metrics = ()
    native_cfg=copy.deepcopy(config.get('metric_defaults',{}))
    native_cfg.update(copy.deepcopy(config.get('observation_options',{}).get(name,{})))
    for native,key in [('qwen','judge'),('da3','da3'),('vp','pavrm')]:
        if config['models'].get(key):native_cfg.setdefault('models',{})[native]=config['models'][key]
    if config['models'].get('sam3'):
        native_cfg.setdefault('models',{}).update(sam3=config['models']['sam3']['checkpoint'],sam3_bpe=config['models']['sam3']['vocabulary'])
        native_cfg['sam3_checkpoint']=config['models']['sam3']['checkpoint'];native_cfg['sam3_vocabulary']=config['models']['sam3']['vocabulary']
    if config['models'].get('dinov3'):native_cfg['dinov3_checkpoint']=config['models']['dinov3']
    mapping=row['spatial_mapping']
    if mapping['kind']=='affine':native_cfg['reference_image_affine']=mapping['matrix']
    elif mapping['kind']=='opencv':native_cfg['reference_image_opencv']={k:v for k,v in mapping.items() if k!='kind'}
    elif mapping['kind']=='opencv_calibrated':
        from handworldbench.observations.pinhole_calibration import validate_calibration
        native_cfg['reference_image_opencv_calibrated']=validate_calibration(mapping['spec'])
    elif mapping['kind']=='fisheye':native_cfg['reference_image_fisheye']=mapping['spec']
    elif mapping['kind']=='identity':native_cfg['reference_image_identity_confirmed']=True
    background=item['annotations'].get('background',{})
    from handworldbench.data.annotations import document
    semantics=document(item['annotations']['semantic']) if item['annotations'].get('semantic') else {}
    concepts=background.get('object_concepts') or [e['description'] for e in semantics.get('entities',[]) if e.get('description')]
    native_cfg['task_description']=item.get('task_description') or ' -> '.join(x['description'] for x in semantics.get('actions',[]) if x.get('description'))
    native_cfg.setdefault('background',dict(background,object_concepts=concepts,source='dataset_reference_annotation'))
    native_cfg.setdefault('background_foreground_prompts',(['hand','arm','person','tool'] if row['category']=='human' else ['robotic hand','robotic arm'])+concepts)
    native_cfg.update(video=row['video'],clip_id=row['case_id'],model_id=row['model'],reference_video=row['reference']['video'],reference_sha256=row['reference']['sha256'])
    if name in ('action_evidence','event_evidence','causal_evidence'):native_cfg.update(annotation=str(semantic(item,row,metrics[0],folder)),time_mapping='identity',metrics=list(metrics))
    if item.get('reference_objects'):native_cfg['reference_objects']=item['reference_objects']['path']
    path=folder/'config.json';write(path,native_cfg)
    native_row=dict(row,prediction_id=row['model']+'__'+row['case_id'],reference_video=row['reference']['video'],reference_sha256=row['reference']['sha256'],observation_key=task['cache_key'])
    call=dict(task,backend=BACKENDS[name],metrics=list(metrics),metric=metrics[0] if metrics else None,row=native_row,config=str(path))
    if name in ('state_evidence','memory_evidence'):
        from handworldbench_paths import loads
        defaults=loads((engine/'engines/state_four/configs/state_defaults.json').read_text())
        defaults['sam3'].update(source_root=config['models']['sam3']['source'],weights=config['models']['sam3']['checkpoint'],bpe=config['models']['sam3']['vocabulary'])
        defaults['hand_finder']=dict(device='cuda:0',category=row['category'],role='prediction',video_sha256=row['video_sha256'])
        defaults['shared_hand_runtime']={key:config[key] for key in ('cache_root','models','batch') if key in config}
        if name=='memory_evidence':state=memory(item,row,defaults,folder)
        else:state=bind_state(item,row,defaults,folder,metrics)
        model_config=loads((engine/'engines/state_four/configs/model.json').read_text())
        model_config['model_manifest']={'path':config['models']['state_judge']}
        model_config.pop('verified_content_manifest',None)
        model_path=folder/'state_model.json';write(model_path,model_config);state['model_config']=str(model_path)
        state['metrics']=list(metrics);state['output_root']=str(folder/'state');state['cache_root']=str(folder/'evidence_cache')
        for case in state['cases']:
            case.update(video=row['video'],video_sha256=row['video_sha256'],model=row['model'])
            for field,directory in [('state_evidence','evidence'),('memory_evidence','memory_evidence'),('memory_bundle','bundles')]:case[field]=str(folder/'state'/directory/(case['case_id']+'.json'))
        state_path=folder/'state.json';write(state_path,state);call['state_config']=str(state_path)
    if name=='hand_anatomy':
        from handworldbench.runtime.execute import dependency
        call['reference_geometry']=dependency(task,'reference_geometry')
        call['hand_detection']=dependency(task,'hand_detection')
    if name=='entity_features':
        from handworldbench.runtime.execute import dependency
        call['shared_objects']=dependency(task,'object_tracks')['shared_objects'];call['unified_hands']=dependency(task,'hand_masks')
    if name in ('image_quality','visual_judgment','video_features'):
        from handworldbench.backends.video_models import run
        value=run(call,models,folder)
    else:
        from handworldbench.backends.inference import run_extended
        if name in ('state_evidence','memory_evidence'):
            from handworldbench.runtime.options import thinking_environment
            with thinking_environment(config,name):
                value=run_extended(call,models,folder)
        else:
            value=run_extended(call,models,folder)
    results={}
    for metric in metrics:
        result=copy.deepcopy(value.get('metrics',{}).get(metric,value));raw=result.get('score')
        status=result.get('status','ok')
        if raw is None and status in ('ok','execution_error','runtime_error','invalid_input'):raise RuntimeError(result.get('reason') or result.get('error') or 'Backend returned no score')
        result.update(metric_id=metric,raw_score=raw,score_0_1=normalize(metric,raw,already_normalized=result.get('normalized',False)),status=status)
        results[metric]=result
    return dict(metrics=results,**{k:v for k,v in value.items() if k in ('shared_objects','vlm_usage','vlm_usage_scope','vlm_usage_by_role','shared_prepare_seconds','stage_seconds','vlm_requests','generated_tokens','image_encoder_cache','feature_file','shape','sampling')})
