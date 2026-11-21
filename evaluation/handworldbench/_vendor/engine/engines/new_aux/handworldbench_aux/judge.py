from __future__ import annotations

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import json,re,gc,time
from .core import MetricIssue,model_path,stage_model,digest,write_json

def retry_prompt(prompt, parser):
 """Repair the answer format without changing the visual evidence or rubric."""
 if parser is parse_score:
  return prompt + '\nFor this formatting retry, replace the response layout with exactly one line: Score: <integer 0, 1, 2, or 3>. Omit the Reason section and all other text. Keep the same assessment criteria.'
 return prompt + '\nYour previous response did not meet the output format. Return only the required answer format. Do not change the assessment criteria.'

class LocalJudge:
 def __init__(self,cfg,folder):self.cfg=cfg;self.folder=folder;self.model=None;self.processor=None;self.calls=[]
 def _load(self):
  if self.model is not None:return
  import torch
  from transformers import AutoModelForImageTextToText,AutoProcessor
  path=stage_model(self.cfg,'qwen');self.path=path
  self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
  self.model=AutoModelForImageTextToText.from_pretrained(str(path),local_files_only=True,dtype=torch.bfloat16,device_map={'':self.cfg.get('device','cuda:0')},attn_implementation='sdpa').eval()
 def ask(self,prompt,frames,parser,system='',records=None):
  self._load()
  if self.cfg.get('image_encoder_cache',True):
   from handworldbench.backends.visual_cache import install
   scope=(self.cfg.get('model_id'),self.cfg.get('clip_id'),self.cfg.get('video'))
   cache=getattr(self.model,'_handworldbench_image_encoder_cache',None)
   if cache is None or cache.scope!=('prediction',scope):install(self.model,scope)
  elif hasattr(self.model,'_handworldbench_image_encoder_cache'):
   cache=self.model._handworldbench_image_encoder_cache
   cache.core.get_image_features=cache.original
   del self.model._handworldbench_image_encoder_cache
  import torch
  errors=[]
  for attempt in range(2):
   text=prompt if attempt==0 else retry_prompt(prompt,parser)
   token_budget=256 if attempt and parser is parse_score else 2048
   content=[]
   for i,im in enumerate(frames):
    if records:content.append({'type':'text','text':f"Frame {i}, time={records[i]['time_s']:.3f}s, window={records[i].get('window_id','full')}"})
    content.append({'type':'image','image':im})
   content.append({'type':'text','text':text});messages=[]
   if system:messages.append({'role':'system','content':system})
   messages.append({'role':'user','content':content})
   try:
    inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False,processor_kwargs={'images_kwargs':{'size':{'longest_edge':self.cfg.get('vlm_max_pixels',147456),'shortest_edge':65536}}}).to(self.model.device)
    with torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=token_budget,do_sample=False)
    raw=self.processor.batch_decode(out[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
    trace={'prompt':text,'system':system,'prompt_sha256':digest([system,text]),'attempt':attempt,'model_path':str(self.path),'model_id':'Qwen/Qwen3.5-9B','dtype':'bfloat16','max_pixels':self.cfg.get('vlm_max_pixels',147456),'max_new_tokens':token_budget,'response_format_retry':bool(attempt),'frame_records':records,'raw_response':raw,'decoded_new_tokens':out.shape[-1]-inputs['input_ids'].shape[-1]}
    try:parsed=parser(raw);trace.update(parsed=parsed,status='ok')
    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:trace.update(status='parse_error',error=str(e));parsed=None
    self.calls.append(trace);write_json(self.folder/f'call_{len(self.calls):04d}.json',trace)
    del inputs,out
    if parsed is not None:return {**parsed,'trace_file':str(self.folder/f'call_{len(self.calls):04d}.json')}
    errors.append(trace['error'])
   except torch.cuda.OutOfMemoryError as e:
    self.close();raise MetricIssue('runtime_error','Local evaluator GPU out of memory; no model fallback') from e
  raise MetricIssue('runtime_error','VLM format failed after one retry',{'errors':errors})
 def close(self):
  self.model=None;self.processor=None;gc.collect()
  try:
   import torch
   if torch.cuda.is_available():torch.cuda.empty_cache()
  except ImportError:pass

def parse_binary(raw):
 if re.search(r'(?im)^\s*(answer|score)\s*:\s*(unknown|uncertain|cannot determine|n/?a)\s*[.!]?\s*$',raw):return {'answer':None,'unknown':True}
 m=re.findall(r'(?im)^\s*Answer\s*:\s*(Yes|No)\s*[.!]?\s*$',raw)
 if len(m)!=1:raise ValueError('Expected exactly one Answer: Yes/No line')
 return {'answer':m[0].lower()=='yes','reason':raw.split('Answer:')[0].strip()}
def parse_score(raw):
 if re.search(r'(?im)^\s*(answer|score)\s*:\s*(unknown|uncertain|cannot determine|n/?a)\s*[.!]?\s*$',raw):return {'score':0,'unknown':True}
 m=re.findall(r'(?im)^\s*Score\s*:\s*([0-3])\s*$',raw)
 if len(m)!=1:raise ValueError('Expected exactly one Score: 0..3 line')
 return {'score':int(m[0]),'reason':raw.split('Score:')[0].strip()}
def json_score(key):
 def parse(raw):
  s=raw.strip()
  if s.startswith('```'):s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
  d=_hb_loads(s);v=d[key]
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not 1<=v<=5:raise ValueError('Score outside [1,5]')
  if not isinstance(d.get('reason'),str):raise ValueError('reason must be text')
  return {'score':float(v),'reason':d['reason']}
 return parse
