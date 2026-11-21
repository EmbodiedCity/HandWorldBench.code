from __future__ import annotations
from types import SimpleNamespace
from .core import extract_symbols,MetricIssue,result
from .video import sample,window_frames
from .annotations import require_annotation
from .judge import parse_binary,parse_score,json_score

INTERACTION='WBench/src/metrics/interaction/vlm_interaction.py'
CAUSAL='WBench/src/metrics/physical/causal_plausibility.py'

# Judge instructions for the Yes/No adherence questions. The question generators come from the vendored WBench file;
# the system prompt and the answer format below are HandWorldBench's (the upstream format text refers to scores).
ADHERENCE_SYSTEM_PROMPT=(
 "You are evaluating frames from a video of hand-object manipulation. The frames are sampled in temporal order "
 "from the time window in which a specific action or event should happen.\n\n"
 "Your task: answer each question with Yes or No, in the required output format, based on what you observe in the "
 "provided frames. Follow these guidelines:\n"
 "- The video may be AI-generated. Minor visual artifacts, slight inconsistencies, or imperfect rendering should NOT "
 "cause a 'No' if the overall intent is recognizable.\n"
 "- Focus on whether the CORE semantic meaning of the described action or event is visually present, not whether "
 "every literal detail matches perfectly.\n"
 "- A long or detailed description may only be partially depicted. Judge based on whether the main action or "
 "change is recognizable.\n"
 "- When in doubt between Yes and No, choose the answer that better matches what you actually see, not what you "
 "expect to see.")
BINARY_FORMAT=('Use this EXACT format (at most 50 words in total):\n'
 'Reason: <the specific visual evidence that decides the answer>\n'
 'Answer: Yes or No')
SCORE_FORMAT=('Use this EXACT format (at most 50 words in total):\n'
 'Reason: <only describe specific issues found; if the score is not full marks, name the exact problem; '
 'do not describe what looks good>\n'
 'Score: <a single integer from 0 to 3>')

def score_binary_answers(rows):
 if not rows:raise MetricIssue('insufficient_evidence','No questions')
 if any(type(r.get('answer')) is not bool and not r.get('unknown') for r in rows):raise MetricIssue('runtime_error','Invalid parsed answer')
 return sum(r.get('answer') is not None and r['answer']==r['expected'] for r in rows)/len(rows)

def evaluate(metric,cfg,a,judge,shared):
 if metric=='causal_plausibility':return causal(cfg,a,judge)
 units=require_annotation(metric,cfg,a)
 if metric in ['action_adherence','event_adherence']:
  n=extract_symbols(INTERACTION,['generate_subject_action_questions','generate_event_edit_questions']);key='generate_subject_action_questions' if metric=='action_adherence' else 'generate_event_edit_questions'
  details=[]
  for unit in units:
   fs,rs=window_frames(cfg,a,unit['window_ids']);rows=[]
   for q in n[key](unit['description']):
    v=judge.ask(q['question']+'\n'+BINARY_FORMAT,fs,parse_binary,ADHERENCE_SYSTEM_PROMPT,rs)
    rows.append({'id':q['sub_id'],'expected':q['expected']=='yes',**v})
   details.append({'unit_id':unit['id'],'score':score_binary_answers(rows),'questions':rows,'frame_count':len(fs)})
  return result(metric,score=sum(d['score'] for d in details)/len(details),details={'units':details,'official_questions':5,'model_replaced':True})
 raise ValueError(metric)

def causal(cfg,a,judge):
 n=extract_symbols(CAUSAL,['PHYSICS_DIMENSIONS','TRACK1_SYSTEM_PROMPT','TRACK2_SYSTEM_PROMPT'])
 fs,rs,_=sample(cfg['video'],3,cfg.get('evaluation_window'))
 actions=cfg.get('task_description','')
 if not actions and a:actions=' -> '.join(x['description'] for x in a['actions'])
 context='Actions in this video: '+(actions or 'No instruction context supplied; evaluate observed rendering physics without assuming a task goal.')
 q=context+'\nRate the overall rendering physics AND causal consistency of this ENTIRE video from 0 to 3. Any physics violation at any point should be noted and penalized.'
 t1=judge.ask(q+'\n'+SCORE_FORMAT,fs,parse_score,n['TRACK1_SYSTEM_PROMPT'],rs)
 cf=a.get('causal') if a else None;dims=[];special_status='missing_annotation'
 if cf and cf['status']=='annotated':dims=cf['dimensions'];special_status='annotated'
 elif cf and cf['status']=='not_applicable':special_status='not_applicable'
 rows=[]
 for dim in dims:
  if dim not in n['PHYSICS_DIMENSIONS']:raise MetricIssue('invalid_input','Unknown official physics dimension '+dim)
  d=n['PHYSICS_DIMENSIONS'][dim];q=context+'\nEvaluating: '+d['name']+'\n'+d['prompt']+'\n'+SCORE_FORMAT
  rows.append({'dimension':dim,**judge.ask(q,fs,parse_score,n['TRACK2_SYSTEM_PROMPT'],rs)})
 raw=(t1['score']+sum(x['score'] for x in rows)/len(rows))/2 if rows else t1['score']
 out=result('causal_plausibility',score=raw/3,details={'raw_score':raw,'track1':t1,'track2':rows,'track2_status':special_status,'aggregation':'dual_track' if rows else 'track1_only'})
 out['protocol']+='-dual' if rows else '-global';return out
