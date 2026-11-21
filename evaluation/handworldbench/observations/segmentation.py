"""Shared box-prompted SAM3 masks for independently detected hand instances."""
from pathlib import Path
import sys
import time
import numpy as np
from handworldbench.runtime.storage import cache_lock,digest,read,write
from handworldbench.observations.hands import associate

class Segmenter:
    def __init__(self,config):
        import torch
        sys.path.insert(0,config['source'])
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model=build_sam3_image_model(bpe_path=config['vocabulary'],checkpoint_path=config['checkpoint'],load_from_HF=False,device='cpu',enable_inst_interactivity=True).to('cuda:0').eval()
        if self.model.inst_interactive_predictor is None:raise RuntimeError('SAM3 interactive predictor unavailable')
        self.processor=Sam3Processor(self.model,device='cuda:0')

    def predict(self,image,hands):
        import torch
        if not hands:return []
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=self.processor.set_image(image)
            results=[]
            for hand in hands:
                masks,scores,_=self.model.predict_inst(state,point_coords=None,point_labels=None,box=np.asarray(hand['box'],np.float32),multimask_output=True,normalize_coords=True)
                if isinstance(masks,torch.Tensor):masks=masks.detach().cpu().numpy()
                if isinstance(scores,torch.Tensor):scores=scores.detach().cpu().numpy()
                masks=np.asarray(masks);scores=np.asarray(scores).reshape(-1)
                if not scores.size or not np.isfinite(scores).all():raise RuntimeError('Invalid SAM3 mask quality output')
                mask=np.squeeze(masks[int(np.argmax(scores))])
                if mask.shape!=(image.height,image.width):raise RuntimeError('SAM3 mask dimensions differ from image')
                results.append((hand,mask.astype(bool),float(scores.max())))
        return results

def hands(detections,models,config,root):
    from PIL import Image
    spec=config['models']['sam3']
    output={};load_seconds=0.;inference_seconds=0.;inference_frames=0;cached_frames=0
    for role in ('reference','prediction'):
        images=detections[role+'_frames'];geometry=detections['geometry']['native_frames' if role=='reference' else 'frames']
        assigned=associate(geometry,detections[role]);tracks={}
        for frame,observed in zip(images,assigned):
            signature=dict(role=role,rgb=frame['decoded_rgb_sha256'],hands=observed['hands'],model=spec,protocol='box_prompt_multimask')
            with cache_lock(Path(root)/'hand_masks',digest(signature)) as folder:
                if (folder/'receipt.json').exists():result=read(folder/'receipt.json');cached_frames+=1
                else:
                    result=[];image=Image.open(frame['path']).convert('RGB')
                    predictions=[]
                    if observed['hands']:
                        if 'hand_segmenter' not in models:
                            started=time.monotonic();models['hand_segmenter']=Segmenter(spec);load_seconds+=time.monotonic()-started
                        started=time.monotonic();predictions=models['hand_segmenter'].predict(image,observed['hands']);inference_seconds+=time.monotonic()-started;inference_frames+=1
                    for hand,mask,quality in predictions:
                        path=folder/(hand['side']+'.png');Image.fromarray(mask.astype('uint8')*255).save(path)
                        yy,xx=np.nonzero(mask)
                        result.append(dict(side=hand['side'],mask=str(path),area_pixels=int(mask.sum()),bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if xx.size else None,center_xy=[float(xx.mean()),float(yy.mean())] if xx.size else None,quality=quality))
                    write(folder/'receipt.json',result)
                for hand in result:
                    eid=hand['side']+'_hand';tracks.setdefault(eid,{})[str(frame['sample_index'])]=dict(hand,frame_index=frame['sample_index'],sample_index=frame['sample_index'],status='ok' if hand['area_pixels'] else 'missing')
        output[role]=tracks
    output.update(reference_frames=detections['reference_frames'],prediction_frames=detections['prediction_frames'],geometry=detections['geometry'],hand_localization_vlm_requests=0,timing=dict(model_load_seconds=load_seconds,inference_seconds=inference_seconds,inference_frames=inference_frames,cached_frames=cached_frames))
    return output
