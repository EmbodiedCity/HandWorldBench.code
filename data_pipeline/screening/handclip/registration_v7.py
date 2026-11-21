"""Retry only unknown registration using background visible in both frames.

All homography, support, F/B, projection, photo and motion gates stay unchanged.
No retry selects a lower motion estimate after motion has already been proved.
"""
import math
import cv2
import numpy as np
from . import vision as v
from .quality_v4 import distributed_features

def register(reference,current,reference_mask,current_mask,camera,settings,points=None):
    primary=v._register(reference,current,reference_mask,current_mask,camera,settings,points)
    if primary['status']=='ok':
        primary['method']='lk_reused_anchor_features' if points is not None else 'native_lk'
        return primary
    tiled=distributed_features(reference,reference_mask,settings)
    if points is not None and tiled is not None and np.array_equal(points,tiled):
        retry=dict(primary);retry['duplicate_feature_retry_skipped']=True
    else:
        retry=v._register(reference,current,reference_mask,current_mask,camera,settings,tiled)
    retry['method']='spatially_balanced_lk_retry';retry['primary_unknown_reason']=primary['reason']
    if retry['status']=='ok':return retry
    h,w=reference.shape
    # Exclude current-frame foreground (plus a displacement margin) BEFORE
    # selecting reference points. This does not filter by fitting residuals.
    # ratio denominator remains ALL proposed persistent-background points.
    pad=int(math.ceil(settings['max_anchor_displacement_diag']*math.hypot(w,h)+settings['max_fb_error_px']+2))
    foreground=(current_mask==0).astype(np.uint8)
    expanded=cv2.dilate(foreground,np.ones((2*pad+1,2*pad+1),np.uint8))
    common=reference_mask.copy();common[expanded>0]=0
    persistent=distributed_features(reference,common,settings)
    if persistent is None or len(persistent)<settings['min_features']:
        retry['persistent_background_unknown_reason']='insufficient_persistent_features';return retry
    local_settings=dict(settings,lk_max_level=2)
    result=v._register(reference,current,common,current_mask,camera,local_settings,persistent)
    result.update(method='persistent_background_lk_retry',primary_unknown_reason=primary['reason'],
                  balanced_unknown_reason=retry['reason'],persistent_mask_margin_px=pad,
                  persistent_reference_fraction=float(np.mean(common>0)),
                  confidence_thresholds_unchanged=True,lk_max_level=2)
    if result['status']=='ok':return result
    retry['persistent_background_unknown_reason']=result['reason']
    return retry
