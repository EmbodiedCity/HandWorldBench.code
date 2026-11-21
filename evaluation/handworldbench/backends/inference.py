"""Adapters for resident metric inference backends."""
import collections,copy,functools,hashlib,json,pathlib,sys,time
from handworldbench.backends.context import S,ENGINE,CONFIG,read,write,decoded_frames,stage_video
from handworldbench.metrics.normalization import normalize
E=ENGINE

def bundle(metrics):
    first=next(iter(metrics.values()))
    return dict(first,metrics=metrics)

def run_extended(task,models,out):
    import torch
    backend=task['backend'];row=task['row'];cfg=copy.deepcopy(read(task['config']));video=stage_video(row);cfg['video']=str(video)
    if backend=='region':
        from handworldbench.backends.regions import run
        return run(task,models,out)
    if backend in ('malformed','background'):
        shared=decoded_frames(row);frames=shared['frames'];sampling=shared['sampling']
    if backend=='background':
        import sam3_dinov3_background as bg
        return bg.evaluate(frames,video,cfg,out,torch.device('cuda:0'))
    if backend=='malformed':
        import hand_morphology as anatomy
        from handworldbench.observations.hands import associate
        detection=task['hand_detection']
        cfg['shared_hand_frames']=associate(task['reference_geometry']['frames'],detection['prediction'])
        cfg['morphology_batch']=CONFIG.get('batch',{}).get('hand_morphology',8)
        native=anatomy.evaluate(frames,video,cfg,out,torch.device('cuda:0'))
        if native.get('error'):raise RuntimeError(native['error'])
        crops=read(out/'vbench2_hand_per_crop.json');gt=task['reference_geometry']['frames'];values=[];details=[];expected=missing=0
        for frame in frames:
            ref=gt[frame['sample_index']]
            hands=[h for h in ref.get('hands',[]) if any(h.get('valid_2d',[]))] if ref else []
            if not hands:continue
            observed={c['side']:c for c in crops if c['sample_index']==frame['sample_index']}
            scores=[observed[h['side']]['p_normal'] if h['side'] in observed else 0. for h in hands]
            expected+=len(hands);missing+=sum(h['side'] not in observed for h in hands)
            values.append(sum(scores)/len(scores));details.append({'sample_index':frame['sample_index'],'scores':scores})
        return {'score':sum(values)/len(values) if values else None,'normalized':True,
                'status':'ok' if values else 'missing_reference','conditional_native_probability':native.get('score'),
                'classifier_batch':read(out/'morphology_batch.json'),'expected_units':expected,'observed_units':expected-missing,'penalized_units':missing,'frames':details}
    if backend=='photo':
        from handworldbench_aux import core,background
        from handworldbench.backends.scoring import calculate_photometric
        core.configure_runtime();core.writable=lambda p:pathlib.Path(p);background.writable=core.writable
        core.DATA=S/'cache/aux';background.DATA=core.DATA;core.DATA.mkdir(exist_ok=True,parents=True)
        core.sha256=functools.lru_cache(maxsize=256)(core.sha256);background.sha256=core.sha256
        arrays,meta=background.prepare(cfg)
        value=calculate_photometric('photometric_consistency',arrays,meta)
        value['foreground_input_protocol']='recorded_reference_concepts_v1'
        value['foreground_concepts_complete']=cfg['background'].get('concepts_complete',False)
        value['foreground_source']=cfg['background']['source']
        return value
    if backend=='aux':
        pass
        from handworldbench_aux import core
        core.writable=lambda p:pathlib.Path(p)
        from handworldbench_aux.judge import LocalJudge
        from handworldbench_aux import vlm_metrics
        if 'judge' not in models:models['judge']=LocalJudge(cfg,out)
        judge=models['judge'];judge.cfg=cfg;judge.folder=out;judge.calls=[]
        annotation=read(cfg['annotation']) if cfg.get('annotation') else None
        values={}
        for m in task['metrics']:
            try:values[m]=vlm_metrics.evaluate(m,cfg,annotation,judge,{})
            except core.MetricIssue as exc:
                values[m]={'score':None,'status':getattr(exc,'status','execution_error'),'reason':str(exc)}
        encoder_cache=getattr(judge.model,'_handworldbench_image_encoder_cache',None)
        return dict(bundle(values),vlm_requests=len(judge.calls),generated_tokens=sum(x.get('decoded_new_tokens',0) for x in judge.calls),image_encoder_cache=dict(encoder_cache.stats) if encoder_cache else None)
    if backend=='state':
        usage_before=dict(models.get('state_judge',{}).get('usage',{}))
        sys.path.insert(0,str(E/'engines/state_four/runtime'))
        from state_eval.context import EvaluationContext
        from state_eval import pipeline as pm,score_normalization as sp,judge as jm
        from state_eval.prepare import prepare
        if 'state_policy' not in models:
            old=jm.Judge.__init__
            def resident(self,context):
                if 'state_judge' in models:self.__dict__.update(models['state_judge']);self.context=context
                else:old(self,context);models['state_judge']=self.__dict__.copy()
                from handworldbench.backends.visual_cache import install
                install(self.model,context.config['cases'][0]['case_id'])
            jm.Judge.__init__=resident;models['state_policy']=True
        sc=read(task['state_config'])
        from handworldbench.runtime.attempts import state_attempt_config
        sc,cp=state_attempt_config(sc,out,task['attempt'],video,task['metrics'])
        write(cp,sc);context=EvaluationContext.load(cp,'cuda:0')
        start=time.monotonic();prep=prepare(context);prepare_seconds=time.monotonic()-start
        scoring_start=time.monotonic()
        pipeline=pm.Pipeline(context);records=list(pipeline.run());write(out/'state_records.json',records);values={}
        for metric in task['metrics']:
            units=[r['metrics'][metric] for r in records if metric in r.get('metrics',{}) and r['metrics'][metric].get('applicable',True)]
            from handworldbench.metrics.errors import execution_failures
            for unit in units:
                errors=execution_failures(unit)
                if errors:unit.update(score=None,status='execution_error',error=errors)
            failures=[u for u in units if u.get('score') is None]
            values[metric]={'score':sum(u['score'] for u in units)/len(units) if units and not failures else None,'status':'execution_error' if failures else 'ok' if units else 'not_applicable','units':units,
                           'reason':'; '.join(str(u.get('error') or u.get('reason') or u.get('original_status')) for u in failures) if failures else None}
        usage={k:v-usage_before.get(k,0) for k,v in models.get('state_judge',{}).get('usage',{}).items()}
        encoder_cache=getattr(models.get('state_judge',{}).get('model'),'_handworldbench_image_encoder_cache',None)
        return dict(bundle(values),shared_prepare_seconds=prepare_seconds,preparation_records=prep,stage_seconds={'state_shared_prepare':prepare_seconds,'state_scoring':time.monotonic()-scoring_start},vlm_usage=usage,vlm_usage_scope='task',image_encoder_cache=dict(encoder_cache.stats) if encoder_cache else None)
    raise ValueError('Unknown inference backend: '+backend)
