"""Finite numeric label inference without chain-of-thought generation."""
import time,hashlib,numpy as np
from pathlib import Path
from .io import save
def choose(judge,prompt,images,binding,dest,labels):
 dest=Path(dest).resolve()
 if not dest.is_relative_to(judge.context.output_root.resolve()):raise ValueError('Output outside configured run')
 ids=[judge.processor.tokenizer.encode(x,add_special_tokens=False) for x in labels]
 if any(len(x)!=1 for x in ids):raise ValueError('Labels must be single tokens')
 content=[]
 for label,im in images:content.extend([{'type':'text','text':label},{'type':'image'}])
 content.append({'type':'text','text':prompt});text=judge.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 started=time.perf_counter();result={'binding':{'source':binding,'prompt':prompt,'labels':list(labels),'image_pixel_sha256':[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for _,im in images],'decoding':{'thinking':False,'constrained_labels':True,'max_new_tokens':1}},'status':'runtime_error','choice':None}
 try:
  batch=judge.processor(text=[text],images=[im for _,im in images],return_tensors='pt',images_kwargs={'size':{'longest_edge':judge.cfg['pixels']['max_per_image'],'shortest_edge':judge.cfg['pixels']['min_per_image']}}).to(judge.model.device)
  judge.start_request(batch['input_ids'].shape[-1])
  with judge.torch.inference_mode():out=judge.model(**batch)
  judge.finish_request(1)
  scores=out.logits[0,-1,[x[0] for x in ids]].float()
  if not bool(judge.torch.isfinite(scores).all()):raise ValueError('Nonfinite logits')
  choice=labels[int(scores.argmax())];result.update(status='ok',choice=choice,raw=choice,input_tokens=int(batch['input_ids'].shape[-1]),generated_tokens=1,conditional_label_probabilities=dict(zip(labels,scores.softmax(-1).cpu().tolist())))
 except Exception as e:result['error']={'type':type(e).__name__,'message':str(e)}
 result['elapsed_seconds']=time.perf_counter()-started;save(dest,result);return result
