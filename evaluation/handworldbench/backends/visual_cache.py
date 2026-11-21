"""Reuse exact Qwen image encoder outputs within one prediction case.

No language tokens, labels, scores, or GT observations are cached here.
The installed Transformers implementation computes get_image_features from
pixel_values and image_grid_thw alone; text fusion occurs after this method.
"""
import collections,hashlib,json,time
import torch
class ImageEncoderCache:
 def __init__(self,model,limit=8):
  self.model=model;self.core=model.model;self.original=self.core.get_image_features;self.limit=limit;self.scope=None;self.cache=collections.OrderedDict();self.stats={}
  self.core.get_image_features=self.call
 def reset(self,prediction_id):
  self.scope=('prediction',prediction_id);self.cache.clear();self.stats={'prediction_id':prediction_id,'role':'prediction','hits':0,'misses':0,'bypasses':0,'hash_seconds':0.,'host_encoder_seconds':0.,'entry_limit':self.limit,'key':'case + role + ordered pixels + dtype + shape + grid + encoder kwargs','cached_content':'image encoder features only; language generation always runs'}
 def tensor_key(self,t):
  if t is None:return None
  data=t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
  return (str(t.dtype),tuple(t.shape),str(t.device),hashlib.sha256(data).hexdigest())
 def call(self,pixel_values,image_grid_thw=None,**kwargs):
  if self.model.training or torch.is_grad_enabled():self.stats['bypasses']+=1;return self.original(pixel_values,image_grid_thw,**kwargs)
  try:options=json.dumps(kwargs,sort_keys=True,allow_nan=False)
  except (ValueError,TypeError):self.stats['bypasses']+=1;return self.original(pixel_values,image_grid_thw,**kwargs)
  started=time.perf_counter();key=(self.scope,self.tensor_key(pixel_values),self.tensor_key(image_grid_thw),options);self.stats['hash_seconds']+=time.perf_counter()-started
  if key in self.cache:
   self.stats['hits']+=1;self.cache.move_to_end(key);return self.cache[key]
  self.stats['misses']+=1;started=time.perf_counter();value=self.original(pixel_values,image_grid_thw,**kwargs);self.stats['host_encoder_seconds']+=time.perf_counter()-started
  self.cache[key]=value
  while len(self.cache)>self.limit:self.cache.popitem(last=False)
  return value
def install(model,prediction_id):
 cache=getattr(model,'_handworldbench_image_encoder_cache',None)
 if cache is None:cache=ImageEncoderCache(model);model._handworldbench_image_encoder_cache=cache
 cache.reset(prediction_id);return cache
