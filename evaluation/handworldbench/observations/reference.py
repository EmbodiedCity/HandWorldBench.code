"""Reference geometry indexed by source time and transformed into prediction pixels."""
import copy
from pathlib import Path
import numpy as np
from handworldbench.data.timeline import Timeline
from handworldbench.observations.video import stage,decode
from handworldbench.runtime.storage import read,write,digest,cache_lock

def transform(points,spec,width,height,camera_points=None):
    import cv2
    a=np.asarray(points,float)
    if spec['kind']=='identity':return a
    if spec['kind']=='affine':
        from handworldbench.data.spatial_mapping import affine_matrix_3x3
        m=affine_matrix_3x3(spec['matrix'])
        q=np.concatenate([a,np.ones((*a.shape[:-1],1))],axis=-1)@m.T
        return q[...,:2]/q[...,2:]
    if spec['kind']=='opencv':
        k=np.asarray(spec['K'],float);d=np.asarray(spec['D'],float)
        q=cv2.undistortPoints(a.reshape(-1,1,2),k,d,P=k).reshape(a.shape)
        scale=np.array([width/spec['source_wh'][0],height/spec['source_wh'][1]])
        return q*scale+(scale*.5-.5 if spec.get('resize')=='pillow_pixel_centers' else 0)
    if spec['kind']=='opencv_calibrated':
        from handworldbench.observations.pinhole_calibration import validate_calibration,project_xyz
        calibration=validate_calibration(spec['spec'],target_size=[width,height])
        if camera_points is None:raise ValueError('Recorded pinhole mapping requires actual camera-space 3D points')
        xyz=np.asarray(camera_points,float)
        if xyz.shape!=(*a.shape[:-1],3):raise ValueError('Recorded pinhole geometry shape differs from reference points')
        return project_xyz(xyz,calibration)[1]
    if spec['kind']=='fisheye':
        if camera_points is None:raise ValueError('Fisheye mapping requires reference camera-space 3D points')
        calibration=spec['spec']
        if calibration.get('distortion_model')!='opencv_fisheye_equidistant' or calibration['output_wh']!=[width,height]:
            raise ValueError('Fisheye calibration does not match prediction pixels')
        rotation=np.asarray(calibration['R_source_to_target'],float);intrinsics=np.asarray(calibration['K_target'],float)
        xyz=np.asarray(camera_points,float)
        if rotation.shape!=(3,3) or intrinsics.shape!=(3,3) or xyz.shape!=(*a.shape[:-1],3):raise ValueError('Invalid calibrated camera geometry')
        target=xyz@rotation.T;projected=target@intrinsics.T
        with np.errstate(divide='ignore',invalid='ignore'):uv=projected[...,:2]/projected[...,2:]
        return np.where((np.isfinite(target).all(axis=-1)&(target[...,2]>0))[...,None],uv,np.nan)
    raise ValueError('Unsupported spatial mapping')

def images(item,row,frames,root):
    source=item['reference'];timeline=Timeline(row['time_mapping']).inverse()
    times=[timeline.at(f['timestamp_seconds']) for f in frames]
    _,meta=stage(source['video'],source['sha256'],root)
    stamps=np.asarray(meta['times']);indices=[int(np.argmin(abs(stamps-t))) for t in times]
    reference=decode(source['video'],source['sha256'],root,'reference',indices)
    return dict(reference_images=reference['frames'],source=source,source_times=times)



def geometry_indices(data, geom, source, reference_frames, times):
    """Resolve explicitly bound video frames, or use the declared time axis."""
    clock=np.asarray(data[geom['time_key']],float)
    if clock.ndim!=1 or not len(clock) or not np.isfinite(clock).all() or np.any(np.diff(clock)<0):
        raise ValueError('Input binding error: invalid reference geometry time')
    if geom.get('time_origin')=='first_sample':clock=clock-clock[0]
    if len(reference_frames)!=len(times):
        raise ValueError('Input binding error: reference samples and times differ')
    key=geom.get('frame_index_key')
    if key:
        if geom.get('frame_index_reference_sha256')!=source['sha256']:
            raise ValueError('Input binding error: geometry frame indices belong to another video')
        indices=np.asarray(data[key])
        if indices.shape!=clock.shape or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(np.diff(indices)<=0):
            raise ValueError('Input binding error: invalid geometry frame indices')
        lookup={int(index):j for j,index in enumerate(indices)}
        selected=[]
        for frame in reference_frames:
            index=frame['source_frame_index']
            if index not in lookup:
                raise ValueError(f'Input binding error: reference frame {index} has no geometry')
            selected.append(lookup[index])
        return clock,selected
    tolerance=max(2/source['fps'],geom.get('time_tolerance_s',0))
    selected=[]
    for i,t in enumerate(times):
        j=int(np.argmin(abs(clock-t)))
        if abs(clock[j]-t)>tolerance:
            raise ValueError(f'Input binding error: geometry time outside tolerance at sample {i}')
        selected.append(j)
    return clock,selected

def observe(item,row,frames,root,reference=None):
    geom=item['geometry'];source=item['reference']
    if geom.get('status') not in (None,'ok','ready','verified'):
        raise ValueError('Reference geometry is not ready: '+str(geom['status']))
    reference=reference or images(item,row,frames,root)
    times=reference['source_times']
    signature=dict(reference=source['sha256'],geometry=geom,times=times,spatial=row['spatial_mapping'],dimensions=[(f['width'],f['height']) for f in frames])
    folder=Path(root)/'reference_geometry'/digest(signature);folder.mkdir(parents=True,exist_ok=True)
    with cache_lock(folder):
        if (folder/'receipt.json').exists():return read(folder/'receipt.json')
        if geom.get('joint_mapping_status') not in (None,'verified'):raise ValueError('Reference joint semantics not verified')
        ref={'frames':reference['reference_images']}
        data=np.load(geom['path'],allow_pickle=False)
        if row['spatial_mapping']['kind']=='opencv_calibrated':
            from handworldbench.observations.pinhole_calibration import validate_calibration,check_source_camera
            spec=validate_calibration(row['spatial_mapping']['spec'],source_size=[source['width'],source['height']])
            check_source_camera(spec,data['camera_K'],data['camera_distortion'])
        points=np.asarray(data[geom['points_key']],float)
        if points.ndim!=4 or points.shape[1:]!=(2,21,2):raise ValueError('Reference geometry requires frames by 2 by 21 by 2')
        clock,selected=geometry_indices(data,geom,source,ref['frames'],times)
        if len(clock)!=len(points):raise ValueError('Input binding error: geometry time and points differ')
        order=geom.get('joint_indices',list(range(21)))
        if sorted(order)!=list(range(21)):raise ValueError('Invalid joint permutation')
        native=[];mapped=[]
        for i,(frame,t) in enumerate(zip(ref['frames'],times)):
            j=selected[i]
            hands=[]
            for h,side in enumerate(geom['hand_order']):
                p=points[j,h,order];valid=np.isfinite(p).all(axis=1)
                if geom.get('positive_depth_key'):valid &= np.asarray(data[geom['positive_depth_key']][j,h],bool)[order]
                elif geom.get('camera_points_key'):valid &= np.asarray(data[geom['camera_points_key']][j,h])[order,2]>0
                if geom.get('hand_valid_key'):valid &= bool(data[geom['hand_valid_key']][j,h])
                hand=dict(side=side,joints_2d=np.where(np.isfinite(p),p,0).tolist(),valid_2d=valid.tolist())
                hands.append(hand)
            a=dict(sample_index=i,width=frame['width'],height=frame['height'],source_timestamp_seconds=float(clock[j]),source_frame_index=j,hands=hands)
            native.append(a);b=copy.deepcopy(a);b.update(width=frames[i]['width'],height=frames[i]['height'])
            for h,hand in enumerate(b['hands']):
                camera_points=np.asarray(data[geom['camera_points_key']][j,h])[order] if geom.get('camera_points_key') else None
                q=transform(hand['joints_2d'],row['spatial_mapping'],b['width'],b['height'],camera_points)
                hand['valid_2d']=(np.asarray(hand['valid_2d'])&np.isfinite(q).all(axis=1)).tolist()
                hand['joints_2d']=np.where(np.isfinite(q),q,0).tolist()
            mapped.append(b)
        result=dict(frames=mapped,native_frames=native,reference_images=ref['frames'],source=source,cache_key=digest(signature))
        write(folder/'receipt.json',result);return result
