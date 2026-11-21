"""Freeze one model and construct its metric-scoped observation graph."""
from dataclasses import asdict
from pathlib import Path
from handworldbench.data.inputs import load_dataset, normalize
from handworldbench.data.timeline import Timeline
from handworldbench.registry import METRICS,DEPENDENCIES,observation_order,select
from handworldbench.runtime.storage import read,write,digest
from handworldbench.runtime.devices import select_devices
from handworldbench.runtime.queue import connect,enqueue
from handworldbench.data.annotations import content_signature

MODEL_INPUTS={'hand_detection':('hand_detector',),'hand_keypoints':('keypoints',),'hand_masks':('sam3',),'object_tracks':('vision','sam3'),'entity_features':('clip',),'image_quality':('musiq',),'visual_judgment':('pavrm',),'video_features':('jepa',),'hand_anatomy':('hand_morphology','hand_detector'),'background_features':('sam3','dinov3'),'background_geometry':('sam3','da3'),'action_evidence':('judge',),'event_evidence':('judge',),'causal_evidence':('judge',),'state_evidence':('state_judge','sam3'),'memory_evidence':('state_judge','sam3')}

def reference_issue(item,metric):
    if 'reference_geometry' not in observation_order([metric]):return None
    geometry=item.get('geometry') or {}
    if not geometry.get('path'):return 'Reference hand geometry is unavailable'
    if geometry.get('status') not in (None,'ok','ready','verified'):
        return 'Reference hand geometry is not ready: '+str(geometry['status'])
    return None

def source_information():
    import subprocess
    from handworldbench import __version__
    root=Path(__file__).resolve().parents[2]
    result=dict(package_version=__version__)
    if (root/'.git').exists():
        try:
            result['git_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
            result['tracked_changes']=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True,stderr=subprocess.DEVNULL).strip())
        except (OSError,subprocess.SubprocessError):
            pass
    return result

def prepare(args):
    run=Path(args.run_dir).resolve()
    if (run/'scheduler.json').exists():
        from handworldbench.runtime.scheduler import active
        if active(run):raise ValueError('Cannot modify an active run')
    dataset_info=load_dataset(args.dataset)
    dataset_root,dataset,_=dataset_info
    config=read(args.config) if args.config else dataset.get('runtime',{})
    if not config:raise ValueError('Runtime configuration required through --config or dataset.json')
    from handworldbench.runtime.config import resolve_runtime_config
    config=resolve_runtime_config(config,Path(args.config).resolve().parent if args.config else dataset_root)
    _,catalog,rows=normalize(args,dataset_info=dataset_info,cache_root=config.get('cache_root'))
    devices=select_devices(args.gpus)
    for name in ('configs','logs','results','reports','workers'):(run/name).mkdir(parents=True,exist_ok=True)
    tasks=[];slots=[]
    requested=select(args.metrics)
    for row in rows:
        item=catalog[row['case_id']]
        selected=[m for m in row['metrics'] if m in requested]
        applicable=[]
        for metric in selected:
            status=item['applicability'].get(metric,'not_applicable')
            reason=None
            if row.get('input_error'):status='input_binding_error'
            if status=='applicable':
                reason=reference_issue(item,metric)
                if reason:status='missing_reference'
            slots.append(dict(case_id=row['case_id'],model=row['model'],category=row['category'],metric_id=metric,metric_protocol=METRICS[metric].protocol,applicability=item['applicability'].get(metric,'not_applicable'),status='pending' if status=='applicable' else status,score_0_1=None,raw_score=None,failure_reason=row.get('input_error') or reason))
            if status=='applicable':applicable.append(metric)
        if not applicable:continue
        Timeline(row['time_mapping'])
        observation_ids={}
        observations = observation_order(applicable)
        for name in observations:
            requested_metrics=[m for m in applicable if name in METRICS[m].observations]
            observation_scope = {}
            if name == 'hand_detection' and 'hand_masks' not in observations:
                observation_scope['include_reference_detections'] = False
            if name == 'object_tracks' and 'trajectory_accuracy' not in requested_metrics:
                observation_scope['include_trajectory_score'] = False
            annotation_key={'action_evidence':'semantic','event_evidence':'supplement','causal_evidence':'semantic','state_evidence':'views','memory_evidence':'supplement'}.get(name)
            annotation=None
            if annotation_key:
                annotation=item['annotations'].get(annotation_key)
                if name in ('event_evidence','memory_evidence'):
                    annotation=(annotation or {}).get('event_adherence' if name=='event_evidence' else 'memory_consistency',{})
                    if (name=='event_evidence' and not annotation.get('annotation')) or (name=='memory_evidence' and annotation.get('inputs') is None):
                        annotation=item['annotations'].get('semantic' if name=='event_evidence' else 'views')
            if name in ('object_tracks','background_features','background_geometry'):
                annotation=dict(semantic=item['annotations'].get('semantic'),
                                background=item['annotations'].get('background'),
                                task_description=item.get('task_description'))
            signature=dict(video=row['video_sha256'],role='prediction',category=row['category'],reference=row['reference'],time_mapping=row['time_mapping'],spatial_mapping=row['spatial_mapping'],observation=name,protocol=config.get('observation_protocols',{}).get(name,'1'),sampling=config.get('sampling',{}),annotation=content_signature(annotation),dependencies=[observation_ids[d] for d in DEPENDENCIES[name]],models={k:config.get('models',{}).get(k) for k in MODEL_INPUTS.get(name,())})
            if name in ('hand_detection','hand_anatomy','state_evidence','memory_evidence'):
                signature['implementation_protocol']='shared_dino_v1'
            if name=='object_tracks' and item['annotations'].get('background',{}).get('reference_entities'):
                signature['entity_inventory_protocol']='reference_inventory_v1'
            if name=='state_evidence' and 'state_regulation' in requested_metrics:
                signature['state_regulation_protocol']=METRICS['state_regulation'].protocol
            if name=='reference_geometry':signature['geometry']=item.get('geometry')
            if name in ('hand_keypoints','hand_masks'):signature['instance_population']='reference_in_frame'
            if name in ('state_evidence','memory_evidence'):
                signature['hand_inference_protocol']='stage_batched_dino_v1'
                signature['models']['hand_detector']=config.get('models',{}).get('hand_detector')
                from handworldbench.runtime.options import thinking_tokens
                signature['thinking_tokens']=thinking_tokens(config,name)
                signature['requested_metrics']=requested_metrics
            if name in MODEL_INPUTS and name not in ('hand_detection','hand_keypoints','hand_masks'):
                signature['metric_options']=config.get('observation_options',{}).get(name,{})
                signature['metric_defaults']=config.get('metric_defaults',{})
            # Include the observation scope in the cache binding.
            signature.update(observation_scope)
            stamp=digest(signature)
            tid=f"{name}_{stamp[:24]}"
            observation_ids[name]=tid
            tasks.append(dict(id=tid,kind='observe',observation=name,backend=name,cache_key=stamp,row=row,reference=item,requested_metrics=requested_metrics,dependencies=[observation_ids[d] for d in DEPENDENCIES[name]],unlocks=sum(name in ds for ds in DEPENDENCIES.values()),estimated_seconds=60,config=config,**observation_scope))
        for metric in applicable:
            deps=[observation_ids[x] for x in METRICS[metric].observations]
            tid=f"score_{metric}_{digest(dict(deps=deps,protocol=METRICS[metric].protocol))[:24]}"
            tasks.append(dict(id=tid,kind='score',metric_id=metric,backend='score',row=row,reference=item,dependencies=deps,config=config,estimated_seconds=1))
            next(s for s in slots if s['case_id']==row['case_id'] and s['metric_id']==metric)['task_id']=tid
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task identities')
    downstream={task['id']:0 for task in tasks}
    for task in tasks:
        for dep in task['dependencies']:downstream[dep]+=1
    for task in tasks:task['unlocks']=downstream[task['id']]
    write(run/'configs/run.json',dict(schema_version=1,model=args.model,dataset=str(Path(args.dataset).resolve()),devices=[asdict(d) for d in devices],metrics=requested,runtime=config,code=source_information()))
    write(run/'configs/predictions.json',rows);write(run/'configs/tasks.json',tasks);write(run/'configs/slots.json',slots)
    with connect(run) as c:
        existing={r['id'] for r in c.execute('SELECT id FROM jobs')};current={t['id'] for t in tasks}
        for tid in existing-current:c.execute("UPDATE jobs SET state='superseded' WHERE id=?",(tid,))
        enqueue(c,tasks)
    receipt=dict(model=args.model,videos=len(rows),tasks=len(tasks),slots=len(slots),input_errors=[s for s in slots if s['status']=='input_binding_error'])
    write(run/'reports/prepare.json',receipt)
    return receipt

