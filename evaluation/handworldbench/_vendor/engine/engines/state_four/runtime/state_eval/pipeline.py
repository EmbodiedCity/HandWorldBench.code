"""Configuration-driven four-metric pipeline with one lazy shared judge."""
import copy,csv,json
from pathlib import Path
from .context import METRICS
from .io import ROOT,read,save,sha
from .judge import Judge
from .score_normalization import publish
from .state_score import StateEvaluator
from .memory_score import evaluate as memory_evaluate
from .hand_bridge import selection_binding,require_current_evidence

PENDING={'pending','not_generated','pending_generation'}
def pending(status):return {'score':None,'raw_score':None,'status':status,'original_status':status,'applicable':True,'zero_imputed':False}
def na(reason):return {**publish(None,'not_applicable',applicable=False),'status':'not_applicable','reason':reason}
def failed(reason,status='runtime_error'):return {**publish(None,status),'status':status,'error':str(reason)}

class Pipeline:
    def __init__(self,context,judge_factory=Judge,numeric_choose=None,binary_choose=None,fact_ask=None):
        self.context=context;self.judge_factory=judge_factory;self._judge=None
        self.numeric_choose=numeric_choose;self.binary_choose=binary_choose;self.fact_ask=fact_ask
    @property
    def judge(self):
        if self._judge is None:self._judge=self.judge_factory(self.context)
        return self._judge
    def _sources(self,row):
        return {k:sha(self.context.path(v)) for k,v in row.items() if k in ('memory_annotation','memory_annotation_source','state_spec','state_protocol','memory_evidence','state_evidence','frame_mapping','memory_bundle') and v and self.context.path(v).exists()}
    def fingerprint(self,row):
        return {'config_sha256':self.context.fingerprint,'case':row,'sources':self._sources(row),
                'memory_protocol_sha256':sha(self.context.path(self.context.config['memory_protocol'])) if 'memory_protocol' in self.context.config else None,
                'model_binding':self.context.model_binding(),'hand_selection_binding':selection_binding(self.context),
                'runtime':{p.name:sha(p) for p in sorted((ROOT/'state_eval').glob('*.py'))}}
    def applicability(self,row):
        out={};explicit=row.get('not_applicable',{})
        annotation=self.context.memory_annotation(row) if row.get('memory_annotation') else None
        spec=self.context.state_spec(row) if row.get('state_spec') else None
        for metric in self.context.metrics:
            if metric in explicit:
                if not isinstance(explicit[metric],str) or not explicit[metric].strip():raise ValueError('Explicit NA requires reason')
                out[metric]={'status':'not_applicable','reason':explicit[metric]}
            elif metric=='memory_consistency':
                out[metric]=annotation.get('metric_applicability',{}).get(metric,{'status':'pending'}) if annotation else {'status':'pending'}
            else:out[metric]=spec.get('applicability',{}).get(metric,{'status':'pending'}) if spec else {'status':'pending'}
        return out
    def _memory_binding(self,row,annotation,evidence):
        if evidence.get('annotation_sha256')!=sha(self.context.field_path(row,'memory_annotation')):
            raise ValueError('Memory evidence annotation binding differs')
    def validate(self,verify_media=False):
        issues=[];waiting=[]
        try:self.context.model_binding()
        except (ValueError,KeyError,OSError,TypeError) as exc:issues.append({'model_error':str(exc)})
        for row in self.context.config['cases']:
            try:
                app=self.applicability(row)
                if any(x['status'] not in ('applicable','not_applicable') for x in app.values()):
                    raise ValueError('Applicability is pending; complete explicit task annotation before scoring')
                case=self.context.case(row)
                if case['status'] in PENDING:
                    waiting.append({'case_id':case['case_id'],'status':'pending_generation'});continue
                if not Path(case['video']).exists():raise ValueError('Available video missing')
                if verify_media and sha(case['video'])!=case['video_sha256']:raise ValueError('Video SHA differs')
                for metric,a in app.items():
                    if a['status']!='applicable':continue
                    key='memory_evidence' if metric=='memory_consistency' else 'state_evidence'
                    ep=self.context.field_path(row,key)
                    if not ep.exists():
                        waiting.append({'case_id':case['case_id'],'metric':metric,'status':'pending_evidence'});continue
                    e=read(ep)
                    require_current_evidence(e,self.context)
                    if metric=='memory_consistency':
                        annotation=self.context.memory_annotation(row);self._memory_binding(row,annotation,e)
                        from .memory_score import load_inputs
                        load_inputs(self.context,row,annotation,e,verify_media=verify_media)
                        if e.get('video_sha256')!=case['video_sha256']:raise ValueError('Memory video binding differs')
                        for unit in annotation['memory']:
                            hidden=e.get('unit_hidden_windows',{}).get(unit['id'],[])
                            if not hidden or any(not e['windows'].get(w) for w in hidden):raise ValueError('Memory requires explicit nonempty hidden evidence')
                    else:
                        from .retention_gate_data_driven import validate_state_policies
                        validate_state_policies(self.context.state_spec(row))
                        self.context.field_path(row,'state_protocol')
                        if e['binding']['case']['video_sha256']!=case['video_sha256']:raise ValueError('State video binding differs')
                        if e['binding'].get('specs_sha256') and e['binding']['specs_sha256']!=sha(self.context.field_path(row,'state_spec')):raise ValueError('State evidence specifications differ')
                    if verify_media:
                        frames=[f for values in e['windows'].values() for f in values] if metric=='memory_consistency' else e['frames']
                        for frame in frames:
                            if not frame.get('sha256') or sha(frame['path'])!=frame['sha256']:raise ValueError('Evidence image differs')
                            index=frame.get('frame_index',frame.get('source_index'))
                            if type(index) is not int or index<0:raise ValueError('Evidence frame index invalid')
            except (ValueError,KeyError,OSError,TypeError) as exc:issues.append({'case_id':row.get('case_id'),'error':str(exc)})
        return {'valid':not issues,'issues':issues,'waiting':waiting,'model_loaded':False,'case_count':len(self.context.config['cases'])}
    def run_case(self,row):
        app=self.applicability(row);case=self.context.case(row);fp=self.fingerprint(row)
        final=self.context.output_root/'cases'/case['case_id']/'summary.json'
        if final.exists():
            old=read(final)
            if old.get('fingerprint')!=fp:raise ValueError('Existing result binding changed; choose a new output root')
            return old
        result={'case':case,'scope':row.get('scope','configured'),'fingerprint':fp,'metrics':{},
                'diagnostic':True,
                'execution_kind':'injected_answers_not_vlm_inference' if self.numeric_choose or self.binary_choose or self.fact_ask else 'local_qwen'}
        for metric,a in app.items():
            if a['status']=='not_applicable':result['metrics'][metric]=na(a.get('reason','Explicit task annotation'))
            elif case['status'] in PENDING:result['metrics'][metric]=pending('pending_generation')
        active=[m for m in self.context.metrics if m not in result['metrics']]
        ready_for_binding=active and all(self.context.field_path(row,'memory_evidence' if m=='memory_consistency' else 'state_evidence').exists() for m in active)
        if ready_for_binding:
            bp=self.context.output_root/'bindings'/(case['case_id']+'.json')
            if bp.exists() and read(bp)!=fp:raise ValueError('Partial case input/version binding changed')
            save(bp,fp)
        state=[m for m in active if m!='memory_consistency']
        if state:
            if not self.context.field_path(row,'state_evidence').exists():
                result['metrics'].update({m:pending('pending_evidence') for m in state})
            else:
                try:
                    spec=self.context.state_spec(row);evaluator=StateEvaluator(self.context,row,self.numeric_choose,self.binary_choose)
                    # Shared state graph computes progress needed by correctness/retention.
                    scored=evaluator.evaluate(self.judge,case,spec)
                    for metric in state:result['metrics'][metric]=scored['metrics'][metric]
                except Exception as exc:result['metrics'].update({m:failed(exc) for m in state})
        if 'memory_consistency' in active:
            metric='memory_consistency';ep=self.context.field_path(row,'memory_evidence')
            if not ep.exists():result['metrics'][metric]=pending('pending_evidence')
            else:
                try:
                    annotation=self.context.memory_annotation(row);e=read(ep);self._memory_binding(row,annotation,e)
                    cc=dict(case);cc['status']='ready'
                    result['metrics'][metric]=memory_evaluate(self.judge,cc,annotation,e,self.context.output_root/'memory',
                        mode='graded',choose_fn=self.numeric_choose,context=self.context,row=row,fact_ask=self.fact_ask)
                except Exception as exc:result['metrics'][metric]=failed(exc)
        terminal=all(not m.get('status',m.get('original_status','')).startswith('pending') for m in result['metrics'].values())
        if terminal:save(final,result)
        # Do not cache pending case summaries as terminal evaluation failures.
        return result
    def run(self):
        # Normal execution must verify actual input bytes before loading the judge.
        report=self.validate(verify_media=True)
        if not report['valid']:raise ValueError('Input validation failed: '+json.dumps(report['issues'],ensure_ascii=False))
        marker=self.context.output_root/'run_manifest.json'
        binding={'config_sha256':self.context.fingerprint,'runtime':{p.name:sha(p) for p in sorted((ROOT/'state_eval').glob('*.py'))},'model_binding':self.context.model_binding(),'hand_selection_binding':selection_binding(self.context)}
        if marker.exists():
            if read(marker)!=binding:raise ValueError('Run configuration/runtime changed; use a new output root')
        elif any((self.context.output_root/x).exists() for x in ('state','memory','cases','bindings')):
            raise ValueError('Unregistered existing result tree; choose a clean output root')
        else:save(marker,binding)
        return [self.run_case(row) for row in self.context.config['cases']]
    def export(self):
        rows=[]
        for row in self.context.config['cases']:
            app=self.applicability(row);path=self.context.output_root/'cases'/row['case_id']/'summary.json'
            d=read(path) if path.exists() else None
            if d and d['fingerprint']!=self.fingerprint(row):raise ValueError('Cannot export stale summary')
            for metric,a in app.items():
                if d:m=d['metrics'][metric]
                elif a['status']=='not_applicable':m=na(a.get('reason',''))
                else:m=pending('pending_generation' if row.get('status') in PENDING else 'pending_evaluation')
                rows.append({'scope':row.get('scope','configured'),'model':row.get('model',row.get('model_id')),'case_id':row['case_id'],
                             'clip_id':row['clip_id'],'metric':metric,'score':m['score'],'status':m.get('status',m.get('original_status')),
                             'applicable':a['status']=='applicable','planned_available':row.get('status') not in PENDING,'zero_imputed':bool(m.get('zero_imputed',m.get('invalid_as_zero',False))),
                             'source':str(path) if d else None,'source_sha256':sha(path) if d else None})
        means=[];common_sets={}
        for scope,metric in sorted({(r['scope'],r['metric']) for r in rows}):
            subset=[r for r in rows if r['scope']==scope and r['metric']==metric];models=sorted({r['model'] for r in subset})
            groups={model:[r for r in subset if r['model']==model and r['applicable']] for model in models}
            # Freeze comparison membership from configured available/applicable tasks,
            # not successful or already completed evaluations. Pending inference stays in it.
            planned=[{r['clip_id'] for r in group if r['planned_available']} for group in groups.values()]
            common=set.intersection(*planned) if planned else set();common_sets[scope+'/'+metric]=sorted(common)
            completed_sets=[{r['clip_id'] for r in group if r['score'] is not None} for group in groups.values()]
            common_complete=bool(common) and all(common<=finished for finished in completed_sets)
            for model,group in groups.items():
                done=[r for r in group if r['score'] is not None];shared=[r for r in done if r['clip_id'] in common]
                means.append({'scope':scope,'model':model,'metric':metric,'applicable':len(group),'scored':len(done),'pending':len(group)-len(done),
                              'failed_zero':sum(r['zero_imputed'] for r in done),'own_valid_mean':sum(r['score'] for r in done)/len(done) if done else None,
                              'common_task_count':len(common),'common_scored_count':len(shared),'common_pending_count':len(common)-len(shared),
                              'common_cohort_complete':common_complete,
                              'common_task_mean':sum(r['score'] for r in shared)/len(common) if common_complete else None})
        out=self.context.output_root/'export';out.mkdir(parents=True,exist_ok=True)
        for name,data in [('scores',rows),('means',means)]:
            if data:
                with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        save(out/'index.json',{'config_sha256':self.context.fingerprint,'common_sets':common_sets,'means':means,'no_composite':True,
                               'diagnostic':True,
                               'common_cohort_normalization':'intersection of frozen configured available and applicable tasks; pending inference retained; all model means withheld until complete'})
        return rows
