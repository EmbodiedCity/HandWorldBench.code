"""Batch CLIP encoding of complete entity crops with role-isolated reuse."""
import copy
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from handworldbench.runtime.storage import read,write,digest,cache_lock

def load_clip(config,device):
    root=Path(__file__).resolve().parents[1]/'_vendor/engine/engines/classic/metric_suite/sources/OpenAI-CLIP/clip'
    name='_handworldbench_clip'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    weights=config['models'].get('clip',str(Path(config['classic_data'])/'metric_suite/weights/ViT-B-32.pt'))
    return sys.modules[name].load(weights,device=device,jit=False)

def crop(image,mask):
    pixels=np.asarray(image);y,x=np.nonzero(mask)
    if not len(x):raise ValueError('A declared valid region has no foreground')
    lo=np.array([x.min(),y.min()]);hi=np.array([x.max()+1,y.max()+1]);padding=(hi-lo)*.1
    lo=np.maximum(0,np.floor(lo-padding)).astype(int);hi=np.minimum([image.width,image.height],np.ceil(hi+padding)).astype(int)
    rgb=np.full_like(pixels,127);rgb[mask]=pixels[mask];cropped=Image.fromarray(rgb[lo[1]:hi[1],lo[0]:hi[0]])
    size=max(cropped.size);square=Image.new('RGB',(size,size),(127,127,127));square.paste(cropped,((size-cropped.width)//2,(size-cropped.height)//2))
    return square

def encode(population,frames,tracks,geometry,config,models,batch=32):
    import torch
    output=copy.deepcopy(geometry);pending=[];start=time.monotonic();stats=dict(encoded_regions=0,cache_hits=0,batches=0,oom_reductions=0)
    def flush():
        if not pending:return
        if 'entity_clip' not in models:models['entity_clip']=load_clip(config,'cuda:0')
        model,transform=models['entity_clip'];offset=0;size=batch
        while offset<len(pending):
            block=pending[offset:offset+size]
            try:
                with torch.inference_mode():values=model.encode_image(torch.stack([transform(x['image']) for x in block]).to('cuda:0')).float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if size==1:raise
                size=max(1,size//2);stats['oom_reductions']+=1;torch.cuda.empty_cache();continue
            if len(values)!=len(block) or not np.isfinite(values).all():raise RuntimeError('Invalid entity feature batch')
            for job,feature in zip(block,values):
                value=feature.tolist();output[job['role']][job['entity']][job['index']]['feature']=value
                with cache_lock(job['folder']):write(job['folder']/'receipt.json',dict(feature=value))
            stats['batches']+=1;stats['encoded_regions']+=len(block);offset+=len(block)
        pending.clear()
    for role in ('reference','prediction'):
        for i,frame in enumerate(frames[role]):
            image=None
            for entity in population:
                eid=entity['id'];g=output[role][eid][str(i)]
                if g['geometry_status']=='absent':continue
                record=tracks[role][eid][str(i)]
                key=digest(dict(role=role,rgb=frame['decoded_rgb_sha256'],mask=g['mask_sha256'],model=config['models'].get('clip','ViT-B-32'),preprocessing='masked_fill127_pad010_square_official_clip'))
                folder=Path(config['cache_root'])/'entity_features'/key
                with cache_lock(folder):
                    if (folder/'receipt.json').exists():g['feature']=read(folder/'receipt.json')['feature'];stats['cache_hits']+=1;continue
                if image is None:image=Image.open(frame['path']).convert('RGB')
                mask=np.asarray(Image.open(record['mask']))>0
                pending.append(dict(image=crop(image,mask),role=role,entity=eid,index=str(i),folder=folder))
                if len(pending)>=batch:flush()
    flush();stats['seconds']=time.monotonic()-start
    return output,stats
