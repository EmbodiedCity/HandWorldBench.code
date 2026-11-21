"""Budgeted reasoning followed by a finite numeric answer."""
import time,re,hashlib,copy
from pathlib import Path
import numpy as np
from .io import ROOT,read,save,sha
from .judge import PresencePenalty
from .finite_answer import FiniteAnswer

def final_digit(raw,labels,truncated=False):
 if truncated:return None
 if '</think>' in raw:text=raw.rsplit('</think>',1)[-1].strip()
 else:text=raw.strip()
 text=re.sub(r'^```(?:text|json)?\s*|\s*```$','',text).strip()
 return text if text in labels else None

def choose_budgeted(judge,prompt,images,binding,dest,labels=('0','1','2','3','4','5'),thinking_budget=4096):
 import os
 thinking_budget=int(os.environ.get('HANDWORLDBENCH_STATE_THINKING_TOKENS',str(thinking_budget)))
 if thinking_budget==0:
  from .constrained_answer import choose
  return choose(judge,prompt,images,binding,dest,labels)
 from transformers import LogitsProcessorList
 dest=Path(dest).resolve()
 if not dest.is_relative_to(judge.context.output_root.resolve()):raise ValueError('Output outside configured run')
 tokenizer=judge.processor.tokenizer
 encoded=[tokenizer.encode(x,add_special_tokens=False) for x in labels]
 if not encoded or any(len(x)!=1 for x in encoded):raise ValueError('Each label must be one token')
 label_ids=[x[0] for x in encoded]
 close_ids=tokenizer.encode('</think>',add_special_tokens=False)
 separator_ids=tokenizer.encode('\n\n',add_special_tokens=False)
 eos_id=tokenizer.eos_token_id
 if eos_id is None:raise ValueError('Tokenizer EOS must be explicit')
 max_tokens=thinking_budget+len(close_ids)+len(separator_ids)+2
 source=copy.deepcopy(binding);pictures=[im for _,im in images]
 provenance={'source':source,'prompt':prompt,'labels':list(labels),'image_labels':[x[0] for x in images],'image_pixel_sha256':[hashlib.sha256(np.array(im).tobytes()).hexdigest() for im in pictures],'numeric_judge_sha256':sha(__file__),'loader_sha256':sha(ROOT/'state_eval/judge.py'),'finite_answer_sha256':sha(ROOT/'state_eval/finite_answer.py'),'close_ids':close_ids,'separator_ids':separator_ids,'label_ids':label_ids,'eos_id':eos_id,'model_config_sha256':sha(judge.config_path),'decoding':{'thinking':True,'thinking_budget':thinking_budget,'max_new_tokens':max_tokens,'final_selection':'argmax over all declared labels, unpenalized logits','forced_closing':'after budget only; never score-triggered','temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':1.5,'seed':1729},'requested_final':'single numeric grade; no explanation'}
 if dest.exists():
  old=read(dest)
  if old['binding']!=provenance:raise ValueError('Existing numeric answer differs')
  return old
 content=[]
 for label,im in images:content.extend([{'type':'text','text':label},{'type':'image'}])
 content.append({'type':'text','text':prompt});text=judge.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 start=time.monotonic();answer={'binding':provenance,'choice':None,'status':'runtime_error','error':None}
 try:
  batch=judge.processor(text=[text],images=pictures,return_tensors='pt',images_kwargs={'size':{'longest_edge':judge.cfg['pixels']['max_per_image'],'shortest_edge':judge.cfg['pixels']['min_per_image']}}).to(judge.model.device)
  n=batch['input_ids'].shape[-1];judge.torch.manual_seed(1729);judge.torch.cuda.manual_seed_all(1729)
  guard=FiniteAnswer(n,thinking_budget,close_ids,separator_ids,label_ids,eos_id)
  judge.start_request(n)
  with judge.torch.inference_mode():tokens=judge.model.generate(**batch,max_new_tokens=max_tokens,eos_token_id=eos_id,pad_token_id=eos_id,do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,logits_processor=LogitsProcessorList([guard,PresencePenalty(n,1.5)]))
  raw=judge.processor.decode(tokens[0,n:],skip_special_tokens=True);count=int(tokens.shape[-1]-n);choice=final_digit(raw,labels,False)
  judge.finish_request(count)
  answer.update(choice=choice,status='ok' if choice is not None else 'invalid_output',raw=raw,input_tokens=int(n),generated_tokens=count,image_grid_thw=batch['image_grid_thw'].tolist(),forced_close=guard.forced_close,selected_token=guard.selected_token,final_label_probability_mass=guard.final_label_probability_mass)
  del batch,tokens
 except Exception as exc:
  answer['error']={'type':type(exc).__name__,'message':str(exc)}
  if 'batch' in locals():del batch
  if 'tokens' in locals():del tokens
  judge.torch.cuda.empty_cache()
 answer['elapsed_seconds']=time.monotonic()-start;save(dest,answer);return answer
