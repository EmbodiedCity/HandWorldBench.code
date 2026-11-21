"""Longest-first exact RGB checks without enumerating overlapping windows.

Camera/hand thresholds and pair/anchor registration are the existing operators.
No strided motion test, composed homography, or stabilization replaces them.
"""
import collections
import errno
import hashlib
import heapq
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import vision as v
from .quality_v4 import native_quality, distributed_features, MotionMask, object_mask
from .registration_v7 import register
from .temporal_hand_quality import longest_soft_end, summary as soft_summary
from .cache_budget import reserve


def next_bad(mask):
    n=len(mask);result=np.full(n+1,n,np.int64);last=n
    for i in range(n-1,-1,-1):
        if mask[i]:last=i
        result[i]=last
    return result


def choose_longest(bounds, minimum, multiple, validate, rgb_cap, *, max_clips=None, forbidden=(), offset=0):
    """Upper-bound best-first search; shorten/requeue BEFORE accepting.

    validate(start, upper) returns the longest temporally legal candidate.
    rgb_cap(candidate) returns the exclusive first failing frame and evidence.
    Neither a percentile gate nor motion activity is assumed monotonic.
    """
    heap=[(-int(b),s,None) for s,b in enumerate(bounds) if b>=minimum]
    heapq.heapify(heap);accepted=[];occupied=list(forbidden)
    while heap and (max_clips is None or len(accepted)<max_clips):
        neg,start,candidate=heapq.heappop(heap);upper=-neg
        if any(a<=start<b for a,b in occupied):continue
        cap=min([upper]+[a-start for a,b in occupied if start<a<start+upper])
        cap=(cap-offset)//multiple*multiple+offset
        if cap<minimum:continue
        if cap!=upper:
            heapq.heappush(heap,(-cap,start,None));continue
        if candidate is None:
            candidate=validate(start,upper)
            if candidate is None:continue
            if candidate['n_frames']<upper:
                heapq.heappush(heap,(-candidate['n_frames'],start,candidate));continue
        boundary,evidence=rgb_cap(candidate)
        limit=(boundary-start-offset)//multiple*multiple+offset
        if limit<minimum:continue
        if limit<candidate['n_frames']:
            heapq.heappush(heap,(-limit,start,None));continue
        candidate=dict(candidate);candidate['metrics']=dict(candidate['metrics'])
        candidate['metrics']['rgb']=evidence
        accepted.append(candidate);occupied.append((start,candidate['end_frame']))
    return accepted


def clip_thumbnail(rgb, size=224):
    """OpenAI CLIP's PIL bicubic resize/center crop, kept as uint8 until GPU."""
    from PIL import Image
    image=Image.fromarray(rgb);width,height=image.size
    if width<=height:shape=(size,int(size*height/width))
    else:shape=(int(size*width/height),size)
    image=image.resize(shape,resample=Image.Resampling.BICUBIC)
    left=round((image.width-size)/2);top=round((image.height-size)/2)
    return np.asarray(image.crop((left,top,left+size,top+size)),dtype=np.uint8)


def analyze_lazy_rgb(ep,cfg,geometry,frame_iterator,sample_output,*,forbidden=()):
    """One full decode, shared frame/pair cache, lazy direct anchor checks.

    Returns only longest-first nonoverlapping proposals. These are reusable
    machine checks, not manual approval. Rerunning with new forbidden intervals
    can reopen alternatives; source metadata and operator provenance are kept.
    """
    settings={**v.DEFAULTS,**cfg.get('rgb',{})};runtime=cfg['runtime']
    offset=cfg.get('selection',{}).get('frame_offset',0)
    cv2.setNumThreads(int(runtime.get('cv_threads',1)))
    n=int(ep['n']);width=min(int(settings['analysis_width']),int(ep['width']))
    height=max(1,round(ep['height']*width/ep['width']));camera=v._camera(ep,width,height)
    bounds=np.asarray(geometry.upper_bounds).copy();hands=geometry.hands
    diagnostics=dict(decoded_frames=0,registration_calls=0,direct_anchor_calls=0,temporal_starts=int(np.count_nonzero(bounds)),
        algorithm='lazy_longest_nonoverlap_v1',all_selected_frames_checked=True,visual_review_performed=False)
    rejections=collections.Counter()
    if not np.any(bounds):return dict(candidates=[],rejections={},diagnostics=diagnostics)
    diff=np.zeros(n+1,np.int64)
    for s in np.flatnonzero(bounds):diff[s]+=1;diff[s+bounds[s]]-=1
    needed=np.cumsum(diff[:-1])>0;cached=int(np.count_nonzero(needed))
    offsets=np.full(n,-1,np.int64);offsets[needed]=np.arange(cached)
    gray_bytes=width*height;mask_bytes=(gray_bytes+7)//8;thumb_bytes=224*224*3
    record_bytes=gray_bytes+mask_bytes+thumb_bytes;total_bytes=cached*record_bytes
    budget=int(runtime.get('rgb_cache_max_bytes',settings['max_cache_mb']*1024**2))
    if total_bytes>budget:
        raise OSError(errno.ENOMEM,f'RGB cache needs {total_bytes} bytes, worker budget {budget}; defer source, never shorten it for resources')
    diagnostics.update(cache_bytes=total_bytes,cache_limit_bytes=budget,cached_frames=cached)
    uv_native,valid=v._project(ep);uv=uv_native*np.array([width/ep['width'],height/ep['height']])
    if uv.shape!=(n,2,21,2) or valid.shape!=(n,2,21):raise ValueError('projection timeline mismatch')
    quality=[None]*n;pairs=[None]*n;hashes={};first_good=np.zeros(n,bool)
    cache_root=Path(runtime['cache_root']);cache_root.mkdir(parents=True,exist_ok=True)
    Path(sample_output).mkdir(parents=True,exist_ok=True)
    with reserve(cfg,int(total_bytes)), tempfile.TemporaryDirectory(prefix='rgb_',dir=cache_root) as temporary:
        with open(Path(temporary)/'gray.raw','w+b',buffering=0) as cache:
            previous=previous_mask=None;previous_index=None;foreground=MotionMask()
            for i,rgb in enumerate(frame_iterator):
                diagnostics['decoded_frames']=i+1
                if i>=n or rgb.dtype!=np.uint8 or rgb.shape!=(ep['height'],ep['width'],3):raise ValueError('source decode frame contract mismatch')
                if not needed[i]:previous=previous_mask=previous_index=None;continue
                gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                if gray.shape!=(height,width):gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
                quality[i]=native_quality(rgb,gray,uv_native[i],valid[i],settings)
                mask=foreground.apply(gray,object_mask(gray.shape,uv[i],valid[i],settings))
                cache.write(gray.tobytes());cache.write(np.packbits(mask.reshape(-1)!=0).tobytes())
                cache.write(clip_thumbnail(rgb).tobytes())
                if previous_index==i-1:
                    pairs[i]=register(previous,gray,previous_mask,mask,camera,settings)
                    diagnostics['registration_calls']+=1
                if bounds[i] and v._quality_reason(quality[i],geometry.frame_hands[i],settings,first=True) is None:
                    first_good[i]=True;hashes[i]=hashlib.sha256(rgb.tobytes()).hexdigest()
                previous,previous_mask,previous_index=gray,mask,i
            if diagnostics['decoded_frames']!=n or cache.tell()!=total_bytes:raise ValueError('complete source decode/cache alignment failed')
            def read(i,kind='gray'):
                if not 0<=i<n or offsets[i]<0:raise ValueError('frame outside complete cache union')
                extra=0 if kind=='gray' else gray_bytes if kind=='mask' else gray_bytes+mask_bytes
                size=gray_bytes if kind=='gray' else mask_bytes if kind=='mask' else thumb_bytes
                cache.seek(int(offsets[i])*record_bytes+extra);raw=cache.read(size)
                if len(raw)!=size:raise OSError(errno.EIO,'truncated analysis cache')
                values=np.frombuffer(raw,np.uint8)
                if kind=='gray':return values.reshape(height,width)
                if kind=='mask':return np.unpackbits(values,count=gray_bytes).reshape(height,width)*np.uint8(255)
                return values.reshape(224,224,3)
            frame_bad=np.ones(n,bool);pair_bad=np.ones(n,bool)
            for i in np.flatnonzero(needed):
                frame_bad[i]=v._quality_reason(quality[i],geometry.frame_hands[i],settings) is not None
                pair=pairs[i]
                pair_bad[i]=pair is None or pair['status']!='ok' or pair['p90_displacement_diag']>settings['max_pair_displacement_diag']
            soft_cfg=settings.get('temporal_hand_quality')
            soft_bad=np.zeros(n,bool)
            if soft_cfg:
                for i in np.flatnonzero(needed):
                    soft_bad[i]=any(quality[i]['hands'][h]['laplacian']<soft_cfg['nominal_min_laplacian'] for h in geometry.frame_hands[i])
            bad_frame=next_bad(frame_bad);bad_pair=next_bad(pair_bad)
            for s in np.flatnonzero(bounds):
                if not first_good[s]:
                    rejections[v._quality_reason(quality[s],geometry.frame_hands[s],settings,first=True) or 'first_frame_unknown']+=1;bounds[s]=0;continue
                cap=min(s+int(bounds[s]),int(bad_frame[s]),int(bad_pair[s+1]))
                if cap<s+int(bounds[s]):
                    if cap==bad_frame[s]:
                        rejections[v._quality_reason(quality[cap],geometry.frame_hands[cap],settings) or 'frame_quality_unknown']+=1
                    elif pairs[cap] is None or pairs[cap]['status']!='ok':
                        rejections['unknown_pair_'+(pairs[cap] or {}).get('reason','missing_registration')]+=1
                    else:rejections['pair_camera_motion']+=1
                bounds[s]=max(0,(cap-s-offset)//geometry.multiple*geometry.multiple+offset)
            # Bound memory even for adversarial starts that fail far into a clip.
            proofs=collections.OrderedDict();failed_frames=collections.OrderedDict()
            def check_anchor(c):
                start,stop=c['start_frame'],c['end_frame'];hands=c['participating_hands']
                if soft_cfg:
                    soft_end=longest_soft_end(soft_bad,start,stop,geometry.minimum,geometry.multiple,offset,soft_cfg['max_fraction'],soft_cfg['max_consecutive_frames'])
                    if soft_end<stop:
                        rejections['temporal_mild_hand_blur_limit']+=1
                        return soft_end,None
                if start not in proofs:
                    reference=read(start);mask=read(start,'mask');features=distributed_features(reference,mask,settings)
                    proofs[start]={'reference':reference,'mask':mask,'features':features,'passed':{},
                        'failure':None if features is not None and len(features)>=settings['min_features'] else start}
                    if proofs[start]['failure'] is not None:rejections['unknown_insufficient_anchor_features']+=1
                    while len(proofs)>8:proofs.popitem(last=False)
                proof=proofs[start];proofs.move_to_end(start)
                # Probe distant frames first. Accepted windows register every selected frame.
                if proof['failure'] is not None and proof['failure']<stop:return proof['failure'],None
                # Use previous failure locations to prioritize checks against the new anchor.
                probes=[i for i in reversed(failed_frames) if start<i<stop]+list(range(stop-1,start,-geometry.multiple))
                for i in probes+list(range(start+1,stop)):
                    if i not in proof['passed']:
                        result=register(proof['reference'],read(i),proof['mask'],read(i,'mask'),camera,settings,proof['features'])
                        diagnostics['registration_calls']+=1;diagnostics['direct_anchor_calls']+=1
                        if result['status']!='ok' or result['p90_displacement_diag']>settings['max_anchor_displacement_diag']:
                            proof['failure']=i if proof['failure'] is None else min(i,proof['failure'])
                            failed_frames[i]=None;failed_frames.move_to_end(i)
                            while len(failed_frames)>32:failed_frames.popitem(last=False)
                            rejections['unknown_anchor_'+result.get('reason','unknown') if result['status']!='ok' else 'anchor_camera_drift']+=1
                            return i,None
                        proof['passed'][i]=result
                regs=[proof['passed'][i] for i in range(start+1,stop)];q=quality[start:stop]
                metrics=dict(analysis_width=width,analysis_height=height,checked_frames=stop-start,direct_anchor_pairs=stop-start-1,
                    first_global_laplacian=q[0]['global_laplacian'],first_hand_laplacian={str(h):q[0]['hands'][h]['laplacian'] for h in hands},
                    min_global_laplacian=min(a['global_laplacian'] for a in q),min_hand_laplacian=min(a['hands'][h]['laplacian'] for a in q for h in hands),
                    max_anchor_displacement_diag=max((a['p90_displacement_diag'] for a in regs),default=0.),
                    max_pair_displacement_diag=max((pairs[i]['p90_displacement_diag'] for i in range(start+1,stop)),default=0.),
                    min_inlier_ratio=min((a['inlier_ratio'] for a in regs),default=1.),
                    min_background_grid_cells=min((a['occupied_cells'] for a in regs),default=0),hand_rgb_semantics_verified=False,hand_quality_method='native_crop_then_single_resize_v4',first_hand_quality={str(h):q[0]['hands'][h] for h in hands},min_legacy_hand_laplacian=min(a['hands'][h]['legacy_laplacian'] for a in q for h in hands),background_methods=dict(collections.Counter(a.get('method','unknown') for a in regs)))
                if soft_cfg:metrics['temporal_hand_quality']=soft_summary(soft_bad,start,stop,soft_cfg['nominal_min_laplacian'],settings['min_hand_laplacian'])
                return stop,metrics
            selected=choose_longest(bounds,geometry.minimum,geometry.multiple,geometry.longest_valid,check_anchor,
                max_clips=cfg['curation'].get('max_candidates_per_source',3),forbidden=forbidden,
                offset=offset)
            for c in selected:
                s,e=c['start_frame'],c['end_frame']
                c.update(source_first_frame_sha256=hashes[s],source_first_frame_fingerprint=v._fingerprint(read(s)),
                    source_middle_frame_fingerprint=v._fingerprint(read(s+(e-s)//2)),source_last_frame_fingerprint=v._fingerprint(read(e-1)))
                stride=max(1,int(round(ep['fps']*cfg['aesthetics'].get('sample_interval_seconds',1.))))
                indices=sorted(set([s,e-1,s+(e-s)//2]+list(range(s,e,stride))))
                key=f'{s:09d}_{e:09d}';sample=Path(sample_output)/(key+'.npz')
                np.savez_compressed(sample,frames=np.stack([read(i,'thumb') for i in indices]),source_frame_indices=np.array(indices,np.int64))
                c['aesthetic_samples']=dict(path=str(sample),source_frame_indices=indices,
                    sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),cache_lifetime='until_scored_record_durably_committed',
                    preprocessing='PIL bicubic short edge 224 and center crop 224, CLIP normalization applied by scorer',
                    semantic_scope='aesthetic/scene sampled evidence; camera and hand checks use every selected frame')
    diagnostics.update(machine_passed=len(selected),rejected_starts=int(np.count_nonzero(geometry.upper_bounds)-np.count_nonzero(bounds)))
    return dict(candidates=selected,rejections=dict(rejections),diagnostics=diagnostics)
