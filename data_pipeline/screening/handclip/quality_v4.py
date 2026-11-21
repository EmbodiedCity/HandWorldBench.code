"""Native-resolution hand evidence and conservative background retries. CPU only."""
import math
import cv2
import numpy as np
from . import vision as v

def native_quality(rgb, gray_analysis, uv_native, valid, settings):
    height,width=rgb.shape[:2]
    scale=np.array([gray_analysis.shape[1]/width,gray_analysis.shape[0]/height])
    result=v._quality(gray_analysis,uv_native*scale,valid,settings)
    original=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    result['hand_quality_method']='native_crop_then_single_resize_v4'
    for h in range(2):
        old=result['hands'][h];old['legacy_laplacian']=old['laplacian']
        pts=uv_native[h]
        if not (valid[h].all() and np.isfinite(pts).all()):continue
        lo,hi=pts.min(axis=0),pts.max(axis=0)
        if (lo<0).any() or hi[0]>=width or hi[1]>=height or (hi-lo).min()<3:
            old['valid']=False;continue
        pad=np.maximum((hi-lo)*.08,2)
        x0,y0=np.maximum(np.floor(lo-pad),0).astype(int)
        x1,y1=np.minimum(np.ceil(hi+pad)+1,[width,height]).astype(int)
        roi=original[y0:y1,x0:x1]
        # No upsampling of a small source ROI, no sharpening or exposure edits.
        ratio=min(1.,settings['roi_size']/max(roi.shape))
        norm=cv2.resize(roi,(max(1,round(roi.shape[1]*ratio)),max(1,round(roi.shape[0]*ratio))),interpolation=cv2.INTER_AREA) if ratio<1 else roi
        lap=v._laplacian(norm)
        smooth=cv2.GaussianBlur(norm,(3,3),.6)
        gx=cv2.Sobel(norm,cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(norm,cv2.CV_32F,0,1,ksize=3)/8
        gradient=np.hypot(gx,gy)
        noise=float(np.median(np.abs(norm.astype(float)-smooth.astype(float)))/.67449)
        old.update(valid=True,laplacian=lap,luma=float(norm.mean()),
            clipped_fraction=float(np.mean((norm<=5)|(norm>=250))),
            native_roi=[int(x0),int(y0),int(x1),int(y1)],normalized_shape=list(norm.shape),
            normalized_scale=float(ratio),noise_sigma_proxy=noise,
            smoothed_laplacian=v._laplacian(smooth),gradient_p95=float(np.percentile(gradient,95)),
            evidence='unmodified native pixels; blur/noise diagnostics are not physical blur labels')
    return result

def distributed_features(gray,mask,settings):
    """Spatially balanced Shi-Tomasi; keep all original reliability thresholds."""
    height,width=gray.shape;pts=[]
    for row in range(settings['grid_rows']):
        for col in range(settings['grid_columns']):
            y0,y1=row*height//settings['grid_rows'],(row+1)*height//settings['grid_rows']
            x0,x1=col*width//settings['grid_columns'],(col+1)*width//settings['grid_columns']
            local=cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],maxCorners=100,qualityLevel=.005,minDistance=5,mask=mask[y0:y1,x0:x1],blockSize=7)
            if local is not None:pts.append(local+np.array([x0,y0],np.float32))
    return np.concatenate(pts).astype(np.float32) if pts else None

def register(reference,current,reference_mask,current_mask,camera,settings,points=None):
    primary=v._register(reference,current,reference_mask,current_mask,camera,settings,points)
    if primary['status']=='ok':
        primary['method']='native_lk';return primary
    tiled=distributed_features(reference,reference_mask,settings)
    retry=v._register(reference,current,reference_mask,current_mask,camera,settings,tiled)
    retry['method']='spatially_balanced_lk_retry';retry['primary_unknown_reason']=primary['reason']
    # Inliers, forward/backward error, coverage, photometric agreement, calibrated
    # grid support and displacement limits are unchanged. Unknown never becomes 0.
    return retry

class MotionMask:
    """Temporal foreground augments (never erodes) the native hand/object mask."""
    def __init__(self):
        self.model=cv2.createBackgroundSubtractorMOG2(history=120,varThreshold=25,detectShadows=False)
        self.frames=0
    def apply(self,gray,base):
        foreground=self.model.apply(gray,learningRate=.02);self.frames+=1
        if self.frames<30:return base
        foreground=cv2.morphologyEx(foreground,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(foreground)
        moving=np.zeros_like(base)
        for k in range(1,count):
            if stats[k,cv2.CC_STAT_AREA]>=36:moving[labels==k]=255
        moving=cv2.dilate(moving,np.ones((7,7),np.uint8))
        # Broad foreground is unreliable under camera motion; retain original mask
        # and full original registration checks rather than claim static pixels.
        if np.mean((moving>0)&(base>0))>.35:return base
        result=base.copy();result[moving>0]=0
        return result

def object_mask(shape,uv,valid,settings):
    height,width=shape;use=valid.copy()
    for h in (0,1):
        pts=uv[h]
        if np.isfinite(pts).all() and ((pts[:,0]<0).all() or (pts[:,0]>=width).all() or (pts[:,1]<0).all() or (pts[:,1]>=height).all()):use[h]=False
    return v._mask(shape,uv,use,settings)
