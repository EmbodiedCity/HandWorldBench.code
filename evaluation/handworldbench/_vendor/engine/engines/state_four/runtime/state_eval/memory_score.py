"""Memory scoring from opportunity gates and independent facts."""
from pathlib import Path
from .io import sha,read,save
from .memory_protocol import _frames,_images,_prompt,binary_score,digest,prompt_for,validate_checklist,validate_frames,score_pair
from .memory_transport import compatible_model,gate_answer,FactJudge
VERSION='memory_gate_facts_v1'

def load_inputs(context,row,annotation,evidence,verify_media=True):
 bundle=read(context.field_path(row,'memory_bundle'));protocol=read(context.path(context.config['memory_protocol']))
 if bundle['case']['case_id']!=row['case_id'] or bundle['case']['clip_id']!=row['clip_id'] or bundle['case']['video_sha256']!=row['video_sha256']:raise ValueError('Memory bundle case differs')
 if bundle['roi_protocol']!=protocol['roi_protocol']:raise ValueError('Per-case ROI protocol differs')
 for ref in bundle['source_bindings']:
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Memory frozen input changed')
 if digest(annotation)!=bundle['annotation_digest'] or digest(evidence)!=bundle['evidence_digest']:raise ValueError('Annotation/evidence content differs')
 if verify_media and sha(bundle['case']['video'])!=bundle['case']['video_sha256']:raise ValueError('Memory video changed')
 ids=[u['id'] for u in annotation['memory']]
 if len(set(ids))!=len(ids) or set(ids)!=set(bundle['units']):raise ValueError('Memory unit mapping differs')
 for u in annotation['memory']:
  cfg=bundle['units'][u['id']];checklist=read(cfg['checklist'])
  validate_checklist(checklist)
  if not checklist.get('object_locator') or checklist['clip_id']!=row['clip_id']:raise ValueError('Missing entity locator or wrong clip')
  if not cfg['returns']:raise ValueError('Missing planned return windows')
  names=[r['id'] for r in cfg['returns']]
  if len(set(names))!=len(names):raise ValueError('Duplicate return windows')
  if verify_media:
   validate_frames(cfg['history_frames'])
   for r in cfg['returns']:validate_frames(r['frames'])
 return bundle,protocol

def evaluate(judge,case,annotation,evidence,run_dir,*,mode='graded',choose_fn=None,context=None,row=None,fact_ask=None):
 app=annotation.get('metric_applicability',{}).get('memory_consistency',{});common={'metric':'memory_consistency','version':VERSION,'score':None,'direction':'higher_is_better','units':[],'diagnostic':True}
 if app.get('status')=='not_applicable':return {**common,'status':'not_applicable','applicable':False}
 if case.get('status','').startswith('pending'):return {**common,'status':'pending_generation'}
 if app.get('status')!='applicable':return {**common,'status':'pending_annotation'}
 if context is None or row is None:raise ValueError('Explicit memory run context and row required')
 bundle,protocol=load_inputs(context,row,annotation,evidence);compatible_model(context,protocol)
 oldcase=bundle['case'];oldcase={**oldcase,'status':'ready'}
 ask=FactJudge(judge,context,protocol,run_dir,inject=fact_ask);units=[]
 for unit in annotation['memory']:
  uid=unit['id'];result={'id':uid,'score':0.0,'checks':[]}
  try:
   hidden_ids=evidence.get('unit_hidden_windows',{}).get(uid,unit.get('hidden_window_ids',[]))
   if not hidden_ids:raise ValueError('No explicit intervening opportunity window')
   groups=[('HISTORY',_frames(evidence,unit['history_window_ids'],3)),('INTERVENING',_frames(evidence,hidden_ids,2)),('RETURN',_frames(evidence,unit['reappearance']['window_ids'],3))]
   binding={'metric':'memory_consistency','version':'memory-numeric-v1','case':oldcase,'annotation_sha256':digest(annotation),'evidence_sha256':digest(evidence),'unit_id':uid,'phase':'reappearance_gate','images':groups,'opportunity_check_enabled':True,'code_sha256':sha(Path(__file__).with_name('memory_protocol.py'))}
   pictures=_images(groups)
   try:g=gate_answer(judge,_prompt(unit,unit['reappearance']['question'],'binary',True),pictures,binding,run_dir,protocol,context,choose=choose_fn)
   finally:
    for _,im in pictures:im.close()
   gate=binary_score(g,unit['reappearance']['expected']);result.update(gate=gate,gate_answer=g,status=gate['status'],invalid_as_zero=gate['invalid_as_zero'])
   if gate['score']==0:result['detail_status']='not_run_gate_zero';units.append(result);continue
   cfg=bundle['units'][uid];cl=read(cfg['checklist']);results=[]
   for check in cl['checks']:
    def observe(frames):
     call={'kind':'fact','images':frames,'labels':['Image '+str(i+1) for i in range(len(frames))],'prompt':prompt_for(cl,check),'allowed_codes':sorted(map(int,check['codebook'])),'pixels':1048576}
     try:return ask(call)
     except Exception as exc:return {'status':'runtime_error','choice':None,'error':str(exc)}
    history=observe(cfg['history_frames']);returns=[]
    for win in cfg['returns']:
     returned=observe(win['frames']);returns.append({'window_id':win['id'],'observations':{'history':history,'return':returned},**score_pair(history,returned,check['codebook'],check.get('absence_codes',[]))})
    results.append({'id':check['id'],'returns':returns,'score':sum(r['score'] for r in returns)/len(returns),'return_denominator':len(returns)})
   failed=any(not r['known_pair'] for c in results for r in c['returns'])
   result.update(checks=results,score=sum(c['score'] for c in results)/len(results),check_denominator=len(results),status='partial_evaluation_failure' if failed else 'ok',invalid_as_zero=failed)
  except Exception as exc:result.update(status='runtime_error',invalid_as_zero=True,error=str(exc))
  units.append(result)
 if not units:return {**common,'status':'missing_annotation','score':0.0,'invalid_as_zero':True}
 invalid=any(u.get('invalid_as_zero',False) for u in units)
 return {**common,'status':'partial_evaluation_failure' if invalid else 'ok','score':sum(u['score'] for u in units)/len(units),'units':units,'unit_denominator':len(units),'invalid_as_zero':invalid,'applicable':True,'annotation_digest':digest(annotation),'evidence_digest':digest(evidence),'memory_bundle_sha256':sha(context.field_path(row,'memory_bundle')),'memory_protocol_sha256':sha(context.path(context.config['memory_protocol'])),'aggregation':'equal checks, equal planned returns within check, equal units; unknown/error0; pending/NA null'}


def prepare_bundle(context,row,evidence):
 """Create explicit unit/checklist associations for new configured cases, without answers."""
 path=context.field_path(row,'memory_bundle')
 if path.exists():return read(path)
 if not path.resolve().is_relative_to(context.output_root.resolve()) and not path.resolve().is_relative_to(context.cache_root.resolve()):raise ValueError('New bundle must stay in output/cache root')
 annotation=context.memory_annotation(row);mapping=row.get('memory_unit_checklists',{})
 if set(mapping)!={u['id'] for u in annotation['memory']}:raise ValueError('Provide one explicit checklist per memory unit')
 refs=[{'path':str(context.field_path(row,k)),'sha256':sha(context.field_path(row,k))} for k in ['memory_annotation','memory_evidence']];units={}
 for u in annotation['memory']:
  cfg=mapping[u['id']];cp=context.path(cfg['checklist']);checklist=read(cp);validate_checklist(checklist)
  if checklist['clip_id']!=row['clip_id'] or not checklist.get('object_locator'):raise ValueError('Checklist clip/object locator missing')
  refs.append({'path':str(cp),'sha256':sha(cp)})
  windows=cfg.get('returns')
  if not windows:raise ValueError('Explicit planned return-window groups required')
  units[u['id']]={'checklist':str(cp),'history_frames':_frames(evidence,u['history_window_ids'],3),'returns':[{'id':w['id'],'frames':_frames(evidence,w['window_ids'],3)} for w in windows]}
 protocol=read(context.path(context.config['memory_protocol']));bundle={'case':context.case(row),'annotation_digest':digest(annotation),'evidence_digest':digest(evidence),'roi_protocol':protocol['roi_protocol'],'source_bindings':refs,'units':units,'prepared_by_sha256':sha(__file__)}
 save(path,bundle);return bundle
