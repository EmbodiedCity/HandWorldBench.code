"""Finite-label visual classification. Probabilities are diagnostics, not grades."""
import hashlib,time
from pathlib import Path
import numpy as np
from .io import ROOT,sha,read,save

def choose(judge,prompt,images,binding,dest,labels=('1','2','3')):
 dest=Path(dest).resolve()
 if not dest.is_relative_to(judge.context.output_root.resolve()):raise ValueError('Output outside configured run')
 token_lists=[judge.processor.tokenizer.encode(x,add_special_tokens=False) for x in labels]
 if any(len(x)!=1 for x in token_lists):raise ValueError('Choice labels must be one token each')
 token_ids=[x[0] for x in token_lists];pictures=[im for _,im in images]
 provenance={'source':binding,'prompt':prompt,'image_pixels_sha256':[hashlib.sha256(np.array(im).tobytes()).hexdigest() for im in pictures],'labels':list(labels),'label_token_ids':token_ids,'classification':{'thinking':False,'do_sample':False,'max_new_tokens':1,'repetition_penalty':1.},'classifier_sha256':sha(__file__),'loader_sha256':sha(ROOT/'state_eval/judge.py'),'model_config_sha256':sha(judge.config_path)}
 if dest.exists():
  old=read(dest)
  if old['binding']!=provenance:raise ValueError('Existing classification binding differs')
  return old
 content=[]
 for label,im in images:content.extend([{'type':'text','text':label},{'type':'image'}])
 content.append({'type':'text','text':prompt});text=judge.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False);started=time.monotonic();result={'binding':provenance,'status':'runtime_error','choice':None,'error':None}
 try:
  batch=judge.processor(text=[text],images=pictures,return_tensors='pt',images_kwargs={'size':{'longest_edge':judge.cfg['pixels']['max_per_image'],'shortest_edge':judge.cfg['pixels']['min_per_image']}}).to(judge.model.device)
  judge.start_request(batch['input_ids'].shape[-1])
  with judge.torch.inference_mode():out=judge.model.generate(**batch,max_new_tokens=1,do_sample=False,repetition_penalty=1.,return_dict_in_generate=True,output_scores=True)
  judge.finish_request(out.sequences.shape[-1]-batch['input_ids'].shape[-1])
  logits=out.scores[0][0].float();all_probs=logits.softmax(-1);selected=logits[token_ids].softmax(-1);top=int(logits.argmax())
  if not bool(judge.torch.isfinite(selected).all()):raise ValueError('Non-finite choice probabilities')
  choice=labels[token_ids.index(top)] if top in token_ids else None
  result.update(status='ok' if choice is not None else 'invalid_output',choice=choice,conditional_label_probabilities=dict(zip(labels,selected.cpu().tolist())),total_label_probability_mass=float(all_probs[token_ids].sum()),unrestricted_top_token=judge.processor.decode([top]),input_tokens=int(batch['input_ids'].shape[-1]),generated_tokens=int(out.sequences.shape[-1]-batch['input_ids'].shape[-1]),image_grid_thw=batch['image_grid_thw'].tolist())
  del batch,out,logits,all_probs,selected
 except Exception as exc:
  result['error']={'type':type(exc).__name__,'message':str(exc)}
  if 'batch' in locals():del batch
  if 'out' in locals():del out
  judge.torch.cuda.empty_cache()
 result['elapsed_seconds']=time.monotonic()-started;save(dest,result);return result
