"""Persistent, weight-verified CLIP + LAION aesthetic inference process.

Scores describe sampled image aesthetics. They do not certify hand anatomy,
motion stability, success, or event completeness. Those scopes remain separate.
"""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCENES={
    'kitchen':'a first person view of hands working in a kitchen with cooking utensils',
    'dining_table':'a first person view of hands performing a task on a dining room table',
    'desk_office':'a first person view of hands working at an office desk with stationery or a computer',
    'craft_workbench':'a first person view of hands doing crafts on an indoor workbench',
    'repair_workshop':'a first person view of hands repairing objects in a tool workshop',
    'bathroom_sink':'a first person view of hands working at a bathroom sink',
    'bedroom_laundry':'a first person view of hands handling clothes in a bedroom or laundry room',
    'living_room':'a first person view of hands doing a task in a living room',
    'laboratory':'a first person view of hands doing a task at a laboratory bench',
    'garden_outdoors':'a first person view of hands doing a task outdoors in a garden',
    'shop_shelf':'a first person view of hands handling items near a shop shelf',
    'vehicle':'a first person view of hands doing a task inside a vehicle',
}


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class Predictor:
    def __init__(self,assets,device='cuda:0'):
        import torch
        import torch.nn as nn
        assets=Path(assets);self.torch=torch;self.device=device
        hashes={'ViT-L-14.pt':'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836',
            'aesthetic_mlp.pth':'21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0'}
        for name,sha in hashes.items():
            if digest(assets/name)!=sha:raise ValueError('model asset hash mismatch: '+name)
        sys.path.insert(0,str(assets/'vendor'))
        import clip
        self.clip,self.preprocess=clip.load(str(assets/'ViT-L-14.pt'),device=device,jit=False)
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=nn.Sequential(nn.Linear(768,1024),nn.Dropout(.2),nn.Linear(1024,128),nn.Dropout(.2),
                    nn.Linear(128,64),nn.Dropout(.1),nn.Linear(64,16),nn.Linear(16,1))
            def forward(self,x):return self.layers(x)
        self.head=Head().to(device).eval()
        self.head.load_state_dict(torch.load(assets/'aesthetic_mlp.pth',map_location='cpu',weights_only=True),strict=True)
        self.scene_names=list(SCENES)
        with torch.inference_mode():
            features=self.clip.encode_text(clip.tokenize(list(SCENES.values())).to(device)).float()
            self.text=features/features.norm(dim=-1,keepdim=True)
        self.provenance=dict(name='improved-aesthetic-predictor sac+logos+ava1-l14-linearMSE',
            model_sha256=hashes,model_source='https://github.com/christophschuhmann/improved-aesthetic-predictor',
            clip_source='https://github.com/openai/CLIP',scene_labels=SCENES,device=device,
            scene_label_scope='inferred coarse scene, not ground truth',score_scope='sampled image preference, not video-motion quality')

    def score(self,path,batch_size=64):
        from PIL import Image
        torch=self.torch
        if self.device=='cpu':batch_size=min(batch_size,8)
        with np.load(path,allow_pickle=False) as data:
            frames=np.array(data['frames']);indices=np.array(data['source_frame_indices'])
        if frames.ndim!=4 or frames.shape[1:]!=(224,224,3) or frames.dtype!=np.uint8 or len(frames)!=len(indices) or not len(frames):
            raise ValueError('invalid aesthetic sample/frame contract')
        if np.any(np.diff(indices)<=0):raise ValueError('sample indices must be strictly increasing')
        scores=[];embeddings=[]
        with torch.inference_mode():
            for start in range(0,len(frames),batch_size):
                tensor=torch.stack([self.preprocess(Image.fromarray(x)) for x in frames[start:start+batch_size]]).to(self.device)
                features=self.clip.encode_image(tensor).float();features/=features.norm(dim=-1,keepdim=True)
                values=self.head(features).squeeze(-1).float()
                scores.extend(values.cpu().tolist());embeddings.append(features.mean(dim=0)*len(tensor))
            embedding=torch.stack(embeddings).sum(dim=0);embedding/=embedding.norm()
            similarities=(self.text@embedding).float().cpu().numpy()
            compact_embedding=embedding.cpu().numpy()
        scores=np.array(scores,dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(similarities).all():raise ValueError('nonfinite model output')
        order=np.argsort(-similarities);top=int(order[0]);margin=float(similarities[top]-similarities[order[1]])
        scene=self.scene_names[top] if similarities[top]>=.20 and margin>=.02 else 'unknown_scene'
        return dict(status='ok',model=self.provenance,sample_count=len(scores),source_frame_indices=indices.tolist(),
            frame_scores=scores.tolist(),first=float(scores[0]),minimum=float(scores.min()),p10=float(np.percentile(scores,10)),
            median=float(np.median(scores)),mean=float(scores.mean()),
            scene=dict(label=scene,top_label=self.scene_names[top],cosine_similarity=float(similarities[top]),
                cosine_margin=margin,minimum_cosine=.20,minimum_margin=.02,method='CLIP zero-shot; coarse inferred labels'),
            embedding=[round(float(v),6) for v in compact_embedding])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    with contextlib.redirect_stdout(sys.stderr):predictor=Predictor(args.assets,args.device)
    print(json.dumps({'status':'ready','model':predictor.provenance}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):result=predictor.score(request['path'])
            print(json.dumps({'id':request['id'],'result':result}),flush=True)
        except Exception as error:
            print(json.dumps({'id':request.get('id'),'error':f'{type(error).__name__}: {error}'}),flush=True)


if __name__=='__main__':main()
