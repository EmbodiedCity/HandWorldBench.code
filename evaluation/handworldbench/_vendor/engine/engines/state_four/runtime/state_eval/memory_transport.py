"""Shared model transport for memory gate and fact requests."""

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import json,re,time,hashlib
from pathlib import Path
from .io import sha,read,save
from .memory_protocol import digest
from .numeric_answer import choose_budgeted

def compatible_model(context,protocol):
 return context.model_binding()

def verify_images(rows):
 for f in rows:
  if sha(f['path'])!=f['sha256']:raise ValueError('Memory pixels changed')

def content(call):
 rows=[]
 for label in call['labels']:rows.extend([{'type':'text','text':label},{'type':'image'}])
 rows.append({'type':'text','text':call['prompt']});return rows

def gate_binding(source,prompt,pictures,protocol):
 runtime=protocol['gate_decoding']
 return {'source':source,'prompt':prompt,'labels':['1','2','3'],'image_labels':[label for label,im in pictures],'image_pixel_sha256':[hashlib.sha256(im.tobytes()).hexdigest() for label,im in pictures],**runtime}

def gate_answer(judge,prompt,pictures,source,out,protocol,context,choose=None):
 expected=gate_binding(source,prompt,pictures,protocol);key=digest(expected);dest=Path(out)/'gate'/(key+'.json')
 if dest.exists():
  saved=read(dest)
  if saved['request_binding']!=_hb_loads(json.dumps(expected,ensure_ascii=False)):raise ValueError('Gate cache mismatch')
  return saved['answer']
 if judge is None and choose is None:raise ValueError('A shared Judge is required for a new gate call')
 if choose is None:
  tok=judge.processor.tokenizer;rt=protocol['gate_decoding']
  actual={'close_ids':tok.encode('</think>',add_special_tokens=False),'separator_ids':tok.encode('\n\n',add_special_tokens=False),'label_ids':[tok.encode(x,add_special_tokens=False)[0] for x in ['1','2','3']],'eos_id':tok.eos_token_id}
  if any(rt[k]!=v for k,v in actual.items()):raise ValueError('Configured numeric token encoding differs')
  choose=choose_budgeted
 answer=choose(judge,prompt,pictures,source,Path(out)/'gate_new'/(key+'.json'),labels=('1','2','3'),thinking_budget=4096)
 answer={**answer,'reuse':{'new_model_inference':True,'request_sha256':key}}
 save(dest,{'request_binding':expected,'answer':answer,'adapter_sha256':sha(__file__)});return answer

class FactJudge:
 def __init__(self,judge,context,protocol,out,inject=None):self.judge=judge;self.context=context;self.protocol=protocol;self.out=Path(out);self.inject=inject
 def __call__(self,call):
  from PIL import Image
  verify_images(call['images'])
  current={'call':call,'model_binding':self.context.model_binding(),'transport_sha256':sha(__file__),'decoding':'natural_greedy_max32','thinking':False}
  dest=self.out/'facts'/(digest(current)+'.json')
  if dest.exists():
   old=read(dest)
   if old['binding']!=current:raise ValueError('Current fact cache differs')
   return old
  if self.inject is not None:
   answer=self.inject(call);return {**answer,'execution_kind':'injected_CPU_test'}
  images=[Image.open(f['path']).convert('RGB') for f in call['images']];j=self.judge
  try:
   text=j.processor.apply_chat_template([{'role':'user','content':content(call)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   batch=j.processor(text=[text],images=images,return_tensors='pt',images_kwargs={'size':{'longest_edge':call['pixels'],'shortest_edge':call['pixels']}}).to(j.model.device);length=int(batch['input_ids'].shape[-1]);start=time.monotonic()
   j.start_request(length)
   with j.torch.inference_mode():tokens=j.model.generate(**batch,max_new_tokens=32,do_sample=False,pad_token_id=j.processor.tokenizer.eos_token_id)
   generated_tokens=int(tokens.shape[-1]-length);j.finish_request(generated_tokens)
   raw=j.processor.decode(tokens[0,length:],skip_special_tokens=True).strip();code=int(raw) if re.fullmatch(r'0|[1-9][0-9]*',raw) else None
   result={'status':'ok' if code in call['allowed_codes'] else 'invalid_output','choice':code if code in call['allowed_codes'] else None,'raw':raw,'binding':current,'input_tokens':length,'generated_tokens':generated_tokens,'image_grid_thw':batch['image_grid_thw'].tolist(),'elapsed_seconds':time.monotonic()-start,'reuse':{'new_model_inference':True}};del batch,tokens
  finally:
   for im in images:im.close()
  save(dest,result);return result
