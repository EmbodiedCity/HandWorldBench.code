from __future__ import annotations
import json,copy
from pathlib import Path
from .core import ANNOTATED,PHYSICS_IDS,MetricIssue,read_json,sha256,ROOT

def schema():
 text={'type':'string','minLength':1}; ids={'type':'array','items':text,'minItems':1,'uniqueItems':True}
 def obj(props,required):return {'type':'object','properties':props,'required':required,'additionalProperties':False}
 q=obj({'id':text,'question':text,'expected':{'type':'boolean'},'window_ids':ids},['id','question','expected','window_ids'])
 action=obj({'id':text,'description':text,'instruction_source':text,'window_ids':ids},['id','description','window_ids'])
 state=obj({'id':text,'entity_id':text,'initial_state':text,'expected_progress':text,'success_evidence':text,'allowed_changes':{'type':'array','items':text},'window_ids':ids},['id','entity_id','initial_state','expected_progress','success_evidence','window_ids'])
 memory=obj({'id':text,'entity_id':text,'invariant':text,'history_window_ids':ids,'reappearance':q,'questions':{'type':'array','items':q,'minItems':1},'allowed_changes':{'type':'array','items':text}},['id','entity_id','invariant','history_window_ids','reappearance','questions','allowed_changes'])
 persistence=obj({'id':text,'entity_id':text,'state_description':text,'establishment':q,'questions':{'type':'array','items':q,'minItems':1},'allowed_reversal_actions':{'type':'array','items':text}},['id','entity_id','state_description','establishment','questions','allowed_reversal_actions'])
 app=obj({'status':{'enum':['pending','applicable','not_applicable']},'reason':{'type':'string'}},['status','reason'])
 props={
 'schema_version':{'const':1},'clip_id':text,'reference_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'reference_duration_s':{'type':'number','exclusiveMinimum':0},'annotation_version':text,'is_synthetic':{'type':'boolean'},
 'metric_applicability':obj({m:app for m in sorted(ANNOTATED)},sorted(ANNOTATED)),
 'entities':{'type':'array','items':obj({'id':text,'description':text},['id','description'])},
 'windows':{'type':'array','items':obj({'id':text,'start_s':{'type':'number','minimum':0},'end_s':{'type':'number','exclusiveMinimum':0}},['id','start_s','end_s'])},
 'actions':{'type':'array','items':action},'events':{'type':'array','items':action},'state_events':{'type':'array','items':state},
 'causal':obj({'status':{'enum':['not_annotated','annotated','not_applicable']},'dimensions':{'type':'array','items':text,'uniqueItems':True},'rationale':{'type':'string'}},['status','dimensions','rationale']),
 'memory':{'type':'array','items':memory},'persistence':{'type':'array','items':persistence}
 }
 return {'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'handworldbench-new-annotation-v1',**obj(props,list(props))}

def template(clip_id='REPLACE_WITH_CLIP_ID',reference_sha256='0'*64,duration=1.0):
 return {'schema_version':1,'clip_id':clip_id,'reference_sha256':reference_sha256,'reference_duration_s':duration,'annotation_version':'draft-1','is_synthetic':False,
 'metric_applicability':{m:{'status':'pending','reason':''} for m in sorted(ANNOTATED)},
 'entities':[],'windows':[],'actions':[],'events':[],'state_events':[],
 'causal':{'status':'not_annotated','dimensions':[],'rationale':''},'memory':[],'persistence':[]}

SECTIONS={'action_adherence':'actions','event_adherence':'events','state_progression':'state_events','evolution_correctness':'state_events','memory_consistency':'memory','state_regulation':'persistence'}

def validate_annotation(a,cfg=None):
 import jsonschema
 issues=[{'path':'.'.join(map(str,e.absolute_path)),'message':e.message} for e in jsonschema.Draft202012Validator(schema()).iter_errors(a)]
 if issues:return issues
 windows={w['id']:w for w in a['windows']};entities={e['id'] for e in a['entities']}
 if len(windows)!=len(a['windows']):issues.append({'path':'windows','message':'duplicate window id'})
 if len(entities)!=len(a['entities']):issues.append({'path':'entities','message':'duplicate entity id'})
 for w in a['windows']:
  if w['end_s']<=w['start_s'] or w['end_s']>a['reference_duration_s']:issues.append({'path':'windows.'+w['id'],'message':'invalid or out-of-range half-open reference window'})
 for section in ['actions','events','state_events','memory','persistence']:
  seen=set()
  for item in a[section]:
   if item['id'] in seen:issues.append({'path':section,'message':'duplicate unit id '+item['id']})
   seen.add(item['id'])
   if item.get('entity_id') and item['entity_id'] not in entities:issues.append({'path':section,'message':'unknown entity '+item['entity_id']})
   refs=list(item.get('window_ids',[]))+list(item.get('history_window_ids',[]));qs=list(item.get('questions',[]))
   for k in ['establishment','reappearance']:
    if k in item:qs.append(item[k])
   qids=[]
   for q in qs:refs+=q['window_ids'];qids.append(q['id'])
   if len(set(qids))!=len(qids):issues.append({'path':section,'message':'duplicate question id'})
   for wid in refs:
    if wid not in windows:issues.append({'path':section,'message':'unknown window '+wid})
 for section,gate_name,history_name in [('memory','reappearance','history_window_ids'),('persistence','establishment',None)]:
  for unit in a[section]:
   gate=unit[gate_name]
   if gate['expected'] is not True:issues.append({'path':section,'message':'Gate must positively establish visibility/state: expected=true'})
   earlier=unit.get(history_name,[]) if history_name else gate['window_ids']
   later=gate['window_ids'] if history_name else [x for q in unit['questions'] for x in q['window_ids']]
   if earlier and later and all(x in windows for x in earlier+later):
    if max(windows[x]['end_s'] for x in earlier)>min(windows[x]['start_s'] for x in later):issues.append({'path':section,'message':'History/establishment must precede later memory checks'})
 for m,v in a['metric_applicability'].items():
  if v['status']=='not_applicable' and not v['reason'].strip():issues.append({'path':m,'message':'not_applicable requires a reason'})
  if v['status']=='applicable' and not a[SECTIONS[m]]:issues.append({'path':m,'message':'applicable requires nonempty '+SECTIONS[m]})
 if a['causal']['status']=='annotated' and (not a['causal']['dimensions'] or not a['causal']['rationale'].strip()):issues.append({'path':'causal','message':'annotated causal dimensions require dimensions and rationale'})
 for dim in a['causal']['dimensions']:
  if dim not in PHYSICS_IDS:issues.append({'path':'causal.dimensions','message':'unknown official dimension '+dim})
 if cfg:
  if a['clip_id']!=cfg['clip_id']:issues.append({'path':'clip_id','message':'annotation belongs to another clip'})
  if cfg.get('reference_video') and sha256(cfg['reference_video'])!=a['reference_sha256']:issues.append({'path':'reference_sha256','message':'reference checksum mismatch'})
  if cfg.get('reference_sha256') and cfg['reference_sha256']!=a['reference_sha256']:issues.append({'path':'reference_sha256','message':'declared reference checksum mismatch'})
 return issues

def load_annotation(cfg):
 if not cfg.get('annotation'):return None
 if not Path(cfg['annotation']).is_file():raise MetricIssue('missing_annotation','Annotation file missing')
 a=read_json(cfg['annotation']);issues=validate_annotation(a,cfg)
 if issues:raise MetricIssue('invalid_input','Invalid annotation',{'validation_errors':issues})
 return a

def require_annotation(metric,cfg,a):
 if a is None:raise MetricIssue('missing_annotation','No annotation provided; automatic annotation is disabled')
 app=a['metric_applicability'][metric]
 if app['status']=='not_applicable':raise MetricIssue('not_applicable',app['reason'])
 if app['status']=='pending':raise MetricIssue('missing_annotation','Applicability and required targets are pending')
 return a[SECTIONS[metric]]
