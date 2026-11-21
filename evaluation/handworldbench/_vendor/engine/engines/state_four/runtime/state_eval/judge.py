"""One shared local Qwen loader. No scoring model is loaded at import time."""
import os
from .io import read
class PresencePenalty:
 def __init__(self,n,p):self.n=n;self.p=p
 def __call__(self,ids,scores):
  import torch
  for row in range(ids.shape[0]):scores[row,torch.unique(ids[row,self.n:])]-=self.p
  return scores
class Judge:
 def start_request(self,input_tokens):
  self.usage['requests']+=1;self.usage['input_tokens']+=int(input_tokens)
 def finish_request(self,generated_tokens):
  self.usage['generated_tokens']+=int(generated_tokens)
 def __init__(self,context):
  self.usage={'requests':0,'input_tokens':0,'generated_tokens':0}
  self.context=context;self.verified_model_binding=context.model_binding(verify_contents=True);self.config_path=context.model_config_path;self.cfg=read(self.config_path);m=self.cfg['model_manifest']
  for key,part in [('HF_HOME','cache/hf'),('TORCH_HOME','cache/torch'),('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
   p=context.cache_root/part;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
  import torch
  from transformers import AutoProcessor,AutoModelForImageTextToText
  self.torch=torch;torch.cuda.set_device(context.device)
  self.processor=AutoProcessor.from_pretrained(m['path'],local_files_only=True)
  self.model=AutoModelForImageTextToText.from_pretrained(m['path'],local_files_only=True,dtype=torch.bfloat16,device_map={'':context.device},attn_implementation='sdpa').eval()
