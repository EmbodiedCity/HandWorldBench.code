"""Recorded OpenCV pinhole camera transport into the declared output viewport.

K_target already includes rectification, crop and pixel-centre resize. Apply
exactly once. Source reference observations remain in their native pixels;
cross-view GT21 and trajectory masks use this same recorded calibration.
"""
import copy
import numpy as np

PROTOCOL = 'recorded_opencv_pinhole_to_pinhole_v1'

def validate_calibration(value, source_size=None, target_size=None):
    if not isinstance(value,dict) or value.get('distortion_model')!='opencv_pinhole':
        raise ValueError('Recorded pinhole calibration requires distortion_model=opencv_pinhole')
    for key,size in [('source_wh',source_size),('output_wh',target_size)]:
        a=value.get(key)
        if not isinstance(a,list) or len(a)!=2 or any(type(v)is not int or v<=0 for v in a):
            raise ValueError('Pinhole '+key+' requires positive integer [width,height]')
        if size is not None and list(size)!=a:raise ValueError('Pinhole '+key+' differs from actual pixels')
    for key in ('K_source','K_target','R_source_to_target'):
        a=np.asarray(value.get(key),float)
        if a.shape!=(3,3) or not np.isfinite(a).all():raise ValueError('Pinhole '+key+' requires finite 3x3 matrix')
        if key.startswith('K_'):
            if min(a[0,0],a[1,1])<=0 or not np.allclose(a[2],[0,0,1],rtol=0,atol=1e-10) or abs(a[1,0])+abs(a[0,1])>1e-10:
                raise ValueError('Pinhole intrinsics require positive focal lengths, zero skew, standard last row')
        elif not np.allclose(a@a.T,np.eye(3),rtol=0,atol=1e-7) or abs(np.linalg.det(a)-1)>1e-7:
            raise ValueError('Pinhole rotation must be orthonormal with determinant +1')
    d=value.get('D_source')
    if not isinstance(d,list) or len(d) not in (4,5,8,12,14) or not np.isfinite(np.asarray(d,float)).all():
        raise ValueError('Pinhole D_source requires 4,5,8,12 or 14 finite OpenCV coefficients')
    if any(value.get(k) is not None for k in ('extra_crop','extra_resize')):
        raise ValueError('K_target must include the complete recorded viewport without a second transform')
    return copy.deepcopy(value)

def from_config(cfg,source_size=None,target_size=None):
    value=cfg.get('reference_image_opencv_calibrated')
    if value is None:return None
    if any(cfg.get(k) is not None for k in ('reference_image_opencv','reference_image_fisheye','reference_image_affine','reference_rectification')) or cfg.get('reference_image_normalization')=='scale' or cfg.get('input_conversion',{}).get('converted'):
        raise ValueError('Recorded pinhole calibration cannot combine with an implicit second image transform')
    return validate_calibration(value,source_size,target_size)

def check_source_camera(spec,K,D):
    validate_calibration(spec)
    if not np.allclose(K,spec['K_source'],rtol=0,atol=1e-9) or not np.allclose(np.asarray(D).reshape(-1),spec['D_source'],rtol=0,atol=1e-12):
        raise ValueError('Recorded pinhole source camera differs from reference NPZ calibration')

def maps(spec):
    import cv2
    validate_calibration(spec)
    return cv2.initUndistortRectifyMap(np.asarray(spec['K_source'],float),np.asarray(spec['D_source'],float),np.asarray(spec['R_source_to_target'],float),np.asarray(spec['K_target'],float),tuple(spec['output_wh']),cv2.CV_32FC1)

def warp_pixels(pixels,spec,*,binary=False):
    import cv2
    a=np.asarray(pixels)
    validate_calibration(spec,list(a.shape[1::-1]))
    if binary:
        if a.ndim!=2 or not set(np.unique(a)).issubset({0,1,255}):raise ValueError('Pinhole mask transport requires binary two-dimensional pixels')
    elif a.ndim!=3 or a.shape[2]!=3:raise ValueError('Pinhole RGB transport requires three-channel pixels')
    mx,my=maps(spec)
    return cv2.remap(a,mx,my,cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def project_xyz(xyz,spec):
    validate_calibration(spec)
    a=np.asarray(xyz,float)
    if a.ndim<2 or a.shape[-1]!=3:raise ValueError('Expected XYZ points')
    target=a@np.asarray(spec['R_source_to_target'],float).T
    homogeneous=target@np.asarray(spec['K_target'],float).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=homogeneous[...,:2]/homogeneous[...,2:]
    valid=np.isfinite(target).all(-1)&np.isfinite(uv).all(-1)&(target[...,2]>0)
    return target,np.where(valid[...,None],uv,np.nan),valid
