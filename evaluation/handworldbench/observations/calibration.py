"""Explicit recorded fisheye -> pinhole calibration, shared by pixels and GT21.

No camera estimation or affine approximation is permitted. RGB is transported
linearly; already binary masks are transported with nearest-neighbour sampling.
"""
import copy
import math

PROTOCOL = 'recorded_fisheye_to_pinhole_opencv_v1'
MASK_PROTOCOL = 'reference_mask_recorded_fisheye_nearest_v1'

def validate_calibration(value, source_size=None, target_size=None):
    if not isinstance(value, dict):raise ValueError('Fisheye calibration must be an object')
    if value.get('distortion_model') != 'opencv_fisheye_equidistant':
        raise ValueError('Fisheye distortion model must be explicitly opencv_fisheye_equidistant')
    for key in ('source_wh','output_wh'):
        x=value.get(key)
        if not isinstance(x,list) or len(x)!=2 or any(type(v)is not int or v<=0 for v in x):
            raise ValueError('Fisheye '+key+' requires positive integer [width,height]')
    for requested,key in ((source_size,'source_wh'),(target_size,'output_wh')):
        if requested is not None and list(requested)!=value[key]:
            raise ValueError('Fisheye '+key+' differs from actual frame dimensions')
    for key in ('K_source','K_target','R_source_to_target'):
        a=value.get(key)
        if not isinstance(a,list) or len(a)!=3 or any(not isinstance(r,list) or len(r)!=3 for r in a):
            raise ValueError('Fisheye '+key+' requires a 3x3 matrix')
        if any(type(v)not in (int,float) or not math.isfinite(v) for r in a for v in r):
            raise ValueError('Fisheye '+key+' contains nonfinite entries')
    for key in ('K_source','K_target'):
        a=value[key]
        if min(a[0][0],a[1][1])<=0 or any(abs(a[2][i]-[0,0,1][i])>1e-10 for i in range(3)) or abs(a[1][0])>1e-10 or abs(a[0][1])>1e-10:
            raise ValueError('Fisheye intrinsics require positive focal lengths, zero skew and standard homogeneous row')
    d=value.get('D_source')
    if not isinstance(d,list) or len(d)!=4 or any(type(v)not in(int,float) or not math.isfinite(v) for v in d):
        raise ValueError('Fisheye D_source requires four finite coefficients')
    q=value['R_source_to_target']
    for i in range(3):
        for j in range(3):
            if abs(sum(q[i][k]*q[j][k] for k in range(3))-(1 if i==j else 0))>1e-7:
                raise ValueError('Fisheye camera rotation must be orthonormal')
    det=q[0][0]*(q[1][1]*q[2][2]-q[1][2]*q[2][1])-q[0][1]*(q[1][0]*q[2][2]-q[1][2]*q[2][0])+q[0][2]*(q[1][0]*q[2][1]-q[1][1]*q[2][0])
    if abs(det-1)>1e-7:raise ValueError('Fisheye camera rotation must have determinant +1')
    if value.get('extra_crop') is not None or value.get('extra_resize') is not None:
        raise ValueError('Additional crop/resize needs a separately implemented recorded composition')
    return copy.deepcopy(value)

def from_config(cfg, source_size=None, target_size=None):
    value=cfg.get('reference_image_fisheye')
    if value is None:return None
    if cfg.get('reference_image_affine') is not None or cfg.get('reference_rectification') is not None or cfg.get('reference_image_normalization')=='scale' or cfg.get('input_conversion',{}).get('converted'):
        raise ValueError('Fisheye calibration cannot be combined with a second implicit image transformation')
    return validate_calibration(value,source_size,target_size)

def check_source_camera(spec, K, D):
    import numpy as np
    if not np.allclose(np.asarray(K,float),np.asarray(spec['K_source'],float),rtol=0,atol=1e-9) or not np.allclose(np.asarray(D,float).reshape(-1),np.asarray(spec['D_source'],float),rtol=0,atol=1e-12):
        raise ValueError('Recorded fisheye camera differs from reference NPZ calibration')

def maps(spec):
    import cv2
    import numpy as np
    validate_calibration(spec)
    return cv2.fisheye.initUndistortRectifyMap(np.asarray(spec['K_source'],float),np.asarray(spec['D_source'],float),np.asarray(spec['R_source_to_target'],float),np.asarray(spec['K_target'],float),tuple(spec['output_wh']),cv2.CV_32FC1)

def warp_pixels(pixels, spec, *, binary=False):
    import cv2
    import numpy as np
    a=np.asarray(pixels)
    validate_calibration(spec,list(a.shape[1::-1]))
    if binary:
        if a.ndim!=2 or not set(np.unique(a)).issubset({0,1,255}):
            raise ValueError('Fisheye mask transport requires a binary two-dimensional mask')
    elif a.ndim!=3 or a.shape[2]!=3:raise ValueError('Fisheye RGB transport requires an RGB image')
    mx,my=maps(spec)
    return cv2.remap(a,mx,my,cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def project_xyz(xyz, spec):
    """Rotate actual calibrated 3D points; do not numerically invert fisheye UV."""
    import numpy as np
    validate_calibration(spec)
    a=np.asarray(xyz,float)
    if a.ndim<2 or a.shape[-1]!=3:raise ValueError('Expected XYZ point arrays')
    target=a@np.asarray(spec['R_source_to_target'],float).T
    homogeneous=target@np.asarray(spec['K_target'],float).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=homogeneous[...,:2]/homogeneous[...,2:]
    valid=np.isfinite(target).all(-1)&np.isfinite(uv).all(-1)&(target[...,2]>0)
    return target,uv,valid
