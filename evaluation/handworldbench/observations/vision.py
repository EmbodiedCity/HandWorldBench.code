"""Resident visual observations with bounded response-format retries."""
import copy
import json
from pathlib import Path
import threading
import numpy as np
from handworldbench.runtime.storage import write,read,digest,cache_lock

class Observer:
    def __init__(self,models,weights,role,batch_size=1):
        self.models=models;self.weights=weights;self.role=role;self.lock=threading.Lock()
        self.model_name=Path(weights).name
        self.batch_size=max(1,int(batch_size))

    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText,AutoProcessor
        key=('vision',str(self.weights))
        if key not in self.models:
            processor=AutoProcessor.from_pretrained(self.weights,local_files_only=True)
            model=AutoModelForImageTextToText.from_pretrained(self.weights,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
            self.models[key]=(model,processor)
        return self.models[key]

    @staticmethod
    def _messages(images,prompt,labels=None):
        content=[]
        for i,image in enumerate(images):
            if labels:content.append(dict(type='text',text=labels[i]))
            content.append(dict(type='image',image=image))
        content.append(dict(type='text',text=prompt))
        return [dict(role='user',content=content)]

    @staticmethod
    def _parse(raw):
        value=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        if not isinstance(value,dict):raise ValueError('Visual response must contain an object')
        return value

    def ask_raw(self,images,prompt,path,labels=None,tokens=1600,**kwargs):
        import torch
        with self.lock:
            model,processor=self._load()
            inputs=processor.apply_chat_template(self._messages(images,prompt,labels),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'images_kwargs':{'size':{'longest_edge':1048576,'shortest_edge':65536}}}).to(model.device)
            with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=tokens,do_sample=False)
            raw=processor.batch_decode(output[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
            count=int(output.shape[-1]-inputs['input_ids'].shape[-1]);stats=self.models.setdefault('vlm_usage',dict(requests=0,generated_tokens=0));stats['requests']+=1;stats['generated_tokens']+=count
            stats['generation_calls']=stats.get('generation_calls',0)+1
            write(path,dict(role=self.role,prompt=prompt,raw=raw,generated_tokens=count,model=self.model_name));return raw

    def ask(self,images,prompt,path,labels=None,tokens=1600):
        for attempt in range(2):
            raw=self.ask_raw(images,prompt+(' Return valid JSON only.' if attempt else ''),Path(path).with_name(Path(path).stem+f'.{attempt}.json'),labels,tokens)
            try:
                return self._parse(raw)
            except (ValueError,TypeError):pass
        raise ValueError('Visual observation returned invalid JSON after two attempts')

    def _batch_raw(self,requests):
        import torch
        with self.lock:
            model,processor=self._load()
            previous=processor.tokenizer.padding_side
            try:
                processor.tokenizer.padding_side='left'
                messages=[self._messages(r['images'],r['prompt'],r.get('labels')) for r in requests]
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True,processor_kwargs={'images_kwargs':{'size':{'longest_edge':1048576,'shortest_edge':65536}}}).to(model.device)
                with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=requests[0].get('tokens',1600),do_sample=False)
                generated=output[:,inputs['input_ids'].shape[-1]:]
                texts=processor.batch_decode(generated,skip_special_tokens=True)
                eos=model.generation_config.eos_token_id
                eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
                stats=self.models.setdefault('vlm_usage',dict(requests=0,generated_tokens=0))
                stats['generation_calls']=stats.get('generation_calls',0)+1
                for request,raw,ids in zip(requests,texts,generated.tolist()):
                    count=next((i+1 for i,token in enumerate(ids) if token in eos),len(ids))
                    stats['requests']+=1;stats['generated_tokens']+=count
                    path=Path(request['path']).with_name(Path(request['path']).stem+'.0.json')
                    write(path,dict(role=self.role,prompt=request['prompt'],raw=raw,generated_tokens=count,model=self.model_name,batch_size=len(requests)))
                return texts
            finally:
                processor.tokenizer.padding_side=previous

    def ask_batch(self,requests):
        """Batch independent prompts, retaining input order and per-request retries."""
        if self.batch_size==1:
            return [self.ask(r['images'],r['prompt'],r['path'],r.get('labels'),r.get('tokens',1600)) for r in requests]
        import torch
        results=[];position=0
        limit=self.models.get(('vision_batch_limit',str(self.weights)),self.batch_size)
        while position<len(requests):
            chunk=requests[position:position+min(limit,self.batch_size)]
            # Different generation budgets form separate batches.
            while any(r.get('tokens',1600)!=chunk[0].get('tokens',1600) for r in chunk):chunk=chunk[:-1]
            try:
                raw_values=self._batch_raw(chunk)
            except torch.cuda.OutOfMemoryError:
                if len(chunk)==1:raise
                limit=max(1,len(chunk)//2);self.models[('vision_batch_limit',str(self.weights))]=limit
                torch.cuda.empty_cache();continue
            if len(raw_values)!=len(chunk):raise RuntimeError('Visual batch output cardinality changed')
            for request,raw in zip(chunk,raw_values):
                try:value=self._parse(raw)
                except (ValueError,TypeError):
                    path=Path(request['path']).with_name(Path(request['path']).stem+'.1.json')
                    retried=self.ask_raw(request['images'],request['prompt']+' Return valid JSON only.',path,request.get('labels'),request.get('tokens',1600))
                    try:value=self._parse(retried)
                    except (ValueError,TypeError) as exc:raise ValueError('Visual observation returned invalid JSON after two attempts') from exc
                results.append(value)
            position+=len(chunk)
        return results

