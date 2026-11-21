"""SAM3 foreground masks and DINOv3 background embeddings with first/previous cosine aggregation."""

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
from pathlib import Path
import hashlib,os,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2
from common import CODE,DATA,definitions,official,provenance,sha,statistics,write_json

DEFAULT_SAM=DATA/'weights/sam3/sam3.pt'
DEFAULT_DINO=DATA/'weights/dinov3/dinov3_vits16_pretrain_lvd1689m.pth'
SAM_SHA256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
DINO_SHA256='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
BPE_SHA256='924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'

def foreground_prompts(cfg):
 """Use explicit concepts; the combined entry may supply frozen-reference concepts."""
 if 'background_foreground_prompts' in cfg:
  return cfg['background_foreground_prompts'],'explicit_background_foreground_prompts'
 return None,'missing_explicit_concepts_or_verified_frozen_reference_objects'

def evaluate(frames,video,cfg,folder,device):
 sam=Path(cfg.get('sam3_checkpoint',DEFAULT_SAM));dino=Path(cfg.get('dinov3_checkpoint',DEFAULT_DINO))
 vocab=Path(cfg.get('sam3_vocabulary',DATA/'assets/sam3/bpe_simple_vocab_16e6.txt.gz'))
 prompts,prompt_source=foreground_prompts(cfg);missing=[]
 for path in (sam,dino,vocab):
  if not path.is_file():missing.append(str(path))
 if not isinstance(prompts,list) or not prompts or any(not isinstance(x,str) or not x.strip() for x in prompts):
  missing.append('background_foreground_prompts: explicit SAM3 concepts for hands and manipulated objects')
 base={'model_stack_release':'sam3_dinov3_v1',
       'configured_model_components':['SAM3','DINOv3 ViT-S/16'],
       'implementation_state':'sam3_dinov3','official_reproduction':False,
       'protocol_id':'sam3_dinov3_cosine_v1',
       'implementation_revision':'sam3_dinov3_v1',
       'official_sources':provenance(['SAM3','DINOv3','VBench']),
       'foreground_prompt_source':prompt_source,
       'foreground_prompt_provenance':cfg.get('background_foreground_prompt_provenance'),
       'note':'Official model components and unchanged VBench cosine aggregation; foreground masking and the combination are a project extension.',
       'access_urls':['https://huggingface.co/facebook/sam3','https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/']}
 if missing:return dict(base,status='missing_model_files_or_prompts',score=None,missing=missing)
 if len(frames)<2:return dict(base,status='insufficient_frames',score=None)
 if device.type!='cuda':return dict(base,status='unsupported_device',score=None,reason='Official SAM3 image inference requires a CUDA GPU.')
 # SAM3 also creates unregistered positional caches on the current CUDA device.
 # Selecting it before construction keeps those caches with the requested model.
 torch.cuda.set_device(device)
 base['sam3_inference_precision']='Official CUDA bfloat16 autocast; DINOv3 feature extraction remains float32.'
 weights=[]
 for path,expected in ((sam,SAM_SHA256),(dino,DINO_SHA256)):
  actual=sha(path)
  if actual!=expected:raise ValueError('Checkpoint SHA256 differs from the pinned native model: '+str(path))
  weights.append({'path':str(path),'sha256':actual,'bytes':path.stat().st_size})
 vocab_sha=sha(vocab)
 if vocab_sha!=BPE_SHA256:raise ValueError('SAM3 vocabulary SHA256 differs from the pinned official asset')
 base.update(weights=weights,bpe={'path':str(vocab),'sha256':vocab_sha},foreground_prompts=prompts)
 sys.path[:0]=[str(CODE/'sources/SAM3'),str(CODE/'sources/DINOv3')]
 official('SAM3','sam3/model_builder.py');official('DINOv3','dinov3/hub/backbones.py')
 import sam3.model_builder as builder
 from sam3.model.sam3_image_processor import Sam3Processor
 from dinov3.hub.backbones import dinov3_vits16
 # The official loader tolerates missing keys. Reject them for evaluation so
 # no uninitialized inference tensors can quietly influence a result.
 global _resident_background
 key=(str(sam),str(dino),str(device))
 if key not in _resident_background:
  load_audit={};original_loader=builder._load_checkpoint
  def audited_loader(model,path):
   def check(module,incompatible):
    if incompatible.missing_keys:raise RuntimeError('SAM3 checkpoint missing inference tensors: '+str(incompatible.missing_keys[:8]))
    load_audit.update(missing_inference_keys=[],unused_checkpoint_keys=list(incompatible.unexpected_keys))
   hook=model.register_load_state_dict_post_hook(check)
   try:return original_loader(model,path)
   finally:hook.remove()
  builder._load_checkpoint=audited_loader
  try:
   # Upstream recognizes the exact string "cuda" only; move explicitly to
   # the requested torch.device after construction/loading on CPU.
   model=builder.build_sam3_image_model(bpe_path=str(vocab),checkpoint_path=str(sam),load_from_HF=False,device='cpu').to(device).eval()
  finally:builder._load_checkpoint=original_loader
  processor=Sam3Processor(model,device=str(device))
  encoder=dinov3_vits16(weights=str(dino)).to(device).eval()
  _resident_background[key]=(model,processor,encoder,load_audit)
 model,processor,encoder,load_audit=_resident_background[key]
 base.update(sam_component='SAM3',feature_component='DINOv3 ViT-S/16',
             segmentation_backend='SAM3',segmentation_protocol='sam3_image_text_concepts_union_v1')
 # Official DINOv3 LVD-1689M README image transform (256x256).
 transform=v2.Compose([v2.ToImage(),v2.Resize((256,256),antialias=True),
                       v2.ToDtype(torch.float32,scale=True),
                       v2.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
 features=[];details=[];masksdir=folder/'foreground_masks';masksdir.mkdir()
 overlaydir=folder/'foreground_overlays';overlaydir.mkdir()
 backgrounddir=folder/'masked_backgrounds';backgrounddir.mkdir()
 fill=np.array([.485,.456,.406])*255
 with torch.inference_mode():
  for f in frames:
   rgb=Image.open(f['path']).convert('RGB');array=np.array(rgb)
   union=np.zeros(array.shape[:2],dtype=bool);detections=[]
   # SAM3's official image example runs vision/text under CUDA BF16 autocast.
   # Its vision backbone internally emits BF16 tensors; omitting this context
   # causes a BF16/FP32 linear-layer dtype mismatch with real checkpoints.
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if not details:base['sam3_runtime_autocast']={'enabled':torch.is_autocast_enabled('cuda'),'dtype':str(torch.get_autocast_dtype('cuda'))}
    state=processor.set_image(rgb)
   for concept_index,prompt in enumerate(prompts):
    processor.reset_all_prompts(state)
    with torch.autocast('cuda',dtype=torch.bfloat16):output=processor.set_text_prompt(state=state,prompt=prompt)
    masks=output['masks'].detach().cpu().numpy()
    if masks.shape[-2:]!=union.shape:raise ValueError('SAM3 mask dimensions differ from input image')
    concept=masks.reshape((-1,*union.shape)).any(0) if len(masks) else np.zeros_like(union)
    union|=concept
    conceptpath=masksdir/(str(f['sample_index'])+'_concept'+str(concept_index)+'.png')
    Image.fromarray(concept.astype(np.uint8)*255).save(conceptpath)
    detections.append({'prompt':prompt,'instance_count':len(masks),'scores':output['scores'].detach().cpu().tolist(),
                       'mask':str(conceptpath),'foreground_fraction':float(concept.mean())})
   maskpath=masksdir/(str(f['sample_index'])+'.png');Image.fromarray(union.astype(np.uint8)*255).save(maskpath)
   overlay=array.copy();overlay[union]=np.rint(.6*overlay[union]+.4*np.array([255,64,64])).astype(np.uint8)
   overlaypath=overlaydir/(str(f['sample_index'])+'.png');None
   detail={'sample_index':f['sample_index'],'timestamp_seconds':f['timestamp_seconds'],
           'input_rgb_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
           'foreground_fraction':float(union.mean()),'background_fraction':float((~union).mean()),
           'background_pixels':int((~union).sum()),'foreground_mask':str(maskpath),
           'foreground_overlay':str(overlaypath),'detections':detections,
           'undetected_concepts':[d['prompt'] for d in detections if d['instance_count']==0]}
   details.append(detail)
   # A failed foreground detection must not silently turn into whole-frame scoring.
   detail['measurement_failed']=bool(not union.any() or union.all())
   array[union]=np.rint(fill).astype(np.uint8)
   x=transform(Image.fromarray(array)).unsqueeze(0).to(device)
   features.append(x)
   backgroundpath=backgrounddir/(str(f['sample_index'])+'.png');Image.fromarray(array).save(backgroundpath)
   detail['masked_background']=str(backgroundpath)
 with torch.inference_mode():
  all_inputs=torch.cat(features);batch_size=int(os.environ.get('HANDWORLDBENCH_DINO_BATCH','16'))
  matrix=torch.cat([encoder.forward_features(all_inputs[i:i+batch_size])['x_norm_clstoken'].float() for i in range(0,len(all_inputs),batch_size)])
 if not torch.isfinite(matrix).all() or (matrix.norm(dim=-1)<=0).any():raise RuntimeError('DINOv3 nonfinite or zero features')
 base['dinov3_runtime_precision']={'input_dtype':str(all_inputs.dtype),'weight_dtype':str(next(encoder.parameters()).dtype),'feature_dtype':str(matrix.dtype),'batch_size':batch_size}
 class FeatureModel:
  def encode_image(self,unused):return matrix
 scope={'torch':torch,'np':np,'F':F,'tqdm':tqdm,'get_rank':lambda:0,
        'clip_transform':lambda n:(lambda x:x),'load_video':lambda p:torch.empty((len(matrix),0)),
        'Image':Image,'os':os}
 definitions('VBench','vbench/background_consistency.py',['background_consistency'],scope)
 score,upstream=scope['background_consistency'](FeatureModel(),None,[str(video)],device,False)
 norm=F.normalize(matrix,dim=-1);rows=[]
 for i in range(1,len(matrix)):
  a=max(0.,F.cosine_similarity(norm[i:i+1],norm[i-1:i]).item());b=max(0.,F.cosine_similarity(norm[i:i+1],norm[:1]).item())
  rows.append(dict(details[i],previous_similarity=a,first_similarity=b,score=(a+b)/2))
 if abs(float(score)-np.mean([r['score'] for r in rows]))>1e-6:raise RuntimeError('VBench aggregation mismatch')

 for i,r in enumerate(rows,1):
  if details[i]['measurement_failed'] or details[i-1]['measurement_failed']:r['previous_similarity']=0.
  if details[i]['measurement_failed'] or details[0]['measurement_failed']:r['first_similarity']=0.
  r['score']=(r['previous_similarity']+r['first_similarity'])/2.
 score=float(np.mean([r['score'] for r in rows]))
 np.save(folder/'dinov3_features.npy',matrix.cpu().numpy());write_json(folder/'segmentation.json',details)
 coverage=[{'prompt':p,'frames_detected':sum(d['detections'][i]['instance_count']>0 for d in details),
            'frames_total':len(details)} for i,p in enumerate(prompts)]
 warnings=[]
 if any(c['frames_detected']<c['frames_total'] for c in coverage):
  warnings.append('Some requested concepts were not detected in every frame; review masks for absence, occlusion or missed detections. A nonempty union does not verify complete foreground removal.')
 if any(d['background_fraction']<.05 for d in details):
  warnings.append('Some frames retain less than 5% background pixels; this is a review warning only and does not change the score formula.')
 return dict(base,status='ok',score=float(score),range=[0,1],direction='higher_is_more_consistent',
             summary=statistics([r['score'] for r in rows]),transitions=rows,upstream_aggregation_results=upstream,
             concept_detection_coverage=coverage,warnings=warnings,
             foreground_completeness_verified=False,segmentation_manifest=str(folder/'segmentation.json'),
             masking='Union of all SAM3 masks at official processor confidence=.5; fill foreground with rounded ImageNet mean RGB.',
             dinov3='Official ViT-S/16 LVD-1689M, README 256x256 preprocessing, normalized CLS token.',
             checkpoint_loading_audit=load_audit)

_resident_background={}
