"""Map object-mask centers into evaluated video coordinates."""
import copy
def trajectory_reference(cfg,geometry,reference_tracks,frames):
    import cv2,numpy as np
    from PIL import Image
    from handworldbench.observations.calibration import warp_pixels
    output=copy.deepcopy(geometry)
    for eid,rows in output.items():
        for i,frame in enumerate(frames):
            item=rows[str(i)];w,h=frame['width'],frame['height']
            if cfg.get('reference_image_opencv_calibrated'):
                from handworldbench.observations.pinhole_calibration import from_config,warp_pixels as warp_pinhole
                spec=from_config(cfg,target_size=[w,h]);method=lambda a:warp_pinhole(a,spec,binary=True)
            elif cfg.get('reference_image_opencv'):
                spec=cfg['reference_image_opencv'];K=np.asarray(spec['K'],float);D=np.asarray(spec['D'],float);sw,sh=spec['source_wh']
                sx,sy=w/sw,h/sh;dx,dy=(.5*sx-.5,.5*sy-.5) if spec.get('resize')=='pillow_pixel_centers' else (0,0)
                mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(sw,sh),cv2.CV_32FC1)
                matrix=np.array([[sx,0,dx],[0,sy,dy],[0,0,1]])
                method=lambda a:cv2.warpPerspective(cv2.remap(a,mx,my,cv2.INTER_NEAREST),matrix,(w,h),flags=cv2.INTER_NEAREST)
            elif cfg.get('reference_image_fisheye'):
                spec=cfg['reference_image_fisheye'];method=lambda a:warp_pixels(a,spec,binary=True)
            elif cfg.get('reference_image_affine'):
                from handworldbench.data.spatial_mapping import affine_matrix_3x3
                matrix=affine_matrix_3x3(cfg['reference_image_affine']);method=lambda a:cv2.warpPerspective(a,matrix,(w,h),flags=cv2.INTER_NEAREST)
            elif [item['width'],item['height']]==[w,h]:
                # Require matching coordinates as well as matching dimensions.
                if cfg.get('reference_image_identity_confirmed') is not True:raise ValueError('Recorded reference image transform missing')
                method=lambda a:a
            else:raise ValueError('Recorded reference image transform missing')
            mask=reference_tracks[eid][str(i)].get('mask');pixels=method(np.asarray(Image.open(mask)).astype('uint8')) if mask else np.zeros((h,w),'uint8')
            yy,xx=np.nonzero(pixels)
            item.update(width=w,height=h,geometry_status='valid' if len(xx) else 'absent',area_pixels=int(len(xx)),
                center_xy=[float(xx.mean()),float(yy.mean())] if len(xx) else None,
                bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None)
    return output
