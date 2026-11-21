"""Compute the TI5 background-motion ranking proxy for each source episode."""
import argparse,concurrent.futures as cf,json,math,time
from pathlib import Path
import cv2,numpy as np
from robot_io import absolute_record,atomic,case_id,sha,source_key

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--shard',type=int,default=0)
p.add_argument('--shards',type=int,default=1)
p.add_argument('--workers',type=int,default=4)
a=p.parse_args()
if a.workers<1 or not 0<=a.shard<a.shards:
 p.error('expected positive workers and a shard index in [0, shards)')
root=a.root.resolve();(root/'scan').mkdir(parents=True,exist_ok=True)
progress=root/'progress';progress.mkdir(exist_ok=True)
cv2.setNumThreads(1)
policy=Path(__file__).with_name('ti5_camera_policy.json')
thresholds=json.loads(policy.read_text())
rows=[absolute_record(json.loads(s),a.inventory.resolve().parent)
      for s in a.inventory.read_text().splitlines() if s.strip()][a.shard::a.shards]
inventory_sha=sha(a.inventory)
config=dict(inventory_sha256=inventory_sha,thresholds=thresholds,shards=a.shards)
config_path=root/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=config:
 raise ValueError('scan input or proxy configuration changed; use a new output directory')
atomic(config_path,config)
def one(r):
 key=source_key(r);cid=case_id('ti5',r);out=root/'scan'/(cid+'.json')
 if out.exists():return json.loads(out.read_text())
 result=dict(key=key,id=cid,frames=(r['frames']-1)//80*80+1,status='failed',camera_pose_measured=False,method='RGB background-masked incremental robust affine proxy; not metric translation or measured3D camera pose')
 try:
  if r.get('status')!='metadata_candidate':raise ValueError('source files unavailable')
  if sha(r['source_metadata_path'])!=r['metadata_sha256']:raise ValueError('source metadata changed')
  video_path=Path(r['source_video_path']);raw_sha=sha(video_path);before=video_path.stat()
  result['raw_video_sha256']=raw_sha
  n=result['frames'];fps=float(r['fps']);stride=max(1,round(fps*.5));want=set(range(0,n,stride))|{n-1};cap=cv2.VideoCapture(r['source_video_path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);count=0;measured=[];missing=[];first=None;points=None;diag=None;previous=None;previous_mask=None;previous_index=0;global_pose=np.eye(3);background=cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=25,detectShadows=False)
  try:
   if not cap.isOpened():raise ValueError('decode unavailable')
   while cap.grab():
    if count in want:
     ok,img=cap.retrieve()
     if not ok:raise ValueError('frame extraction failed')
     h,w=img.shape[:2];gray=cv2.cvtColor(cv2.resize(img,(480,round(h*480/w))),cv2.COLOR_BGR2GRAY)
     foreground=background.apply(gray,learningRate=.03)
     background_mask=cv2.erode((foreground==0).astype(np.uint8)*255,np.ones((3,3),np.uint8))
     if first is None:
      first=gray;diag=math.hypot(gray.shape[0],gray.shape[1]);previous=gray;previous_mask=np.full_like(gray,255);previous_index=count
      measured.append((count,np.eye(3)));count+=1;continue
     points=cv2.goodFeaturesToTrack(previous,800,.005,7,mask=previous_mask)
     if points is None or len(points)<8:missing.append(count);count+=1;continue
     q,status,err=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(31,31),maxLevel=4)
     xy=q.reshape(-1,2);xx=np.clip(np.round(xy[:,0]).astype(int),0,gray.shape[1]-1);yy=np.clip(np.round(xy[:,1]).astype(int),0,gray.shape[0]-1)
     good=(status.reshape(-1)>0)&np.isfinite(xy).all(1)&(err.reshape(-1)<30)&(background_mask[yy,xx]>0)
     if good.sum()<8:missing.append(count);count+=1;continue
     source=points.reshape(-1,2)[good];target=xy[good];M,inl=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=2000,confidence=.995)
     if M is None or inl.sum()<8 or inl.mean()<.25:missing.append(count);count+=1;continue
     grid={(min(2,int(x*3/gray.shape[1])),min(2,int(y*3/gray.shape[0]))) for x,y in source[inl.reshape(-1)>0]}
     if len(grid)<3:missing.append(count);count+=1;continue
     step=np.eye(3);step[:2]=M;global_pose=step@global_pose;measured.append((count,global_pose.copy()));previous=gray;previous_mask=background_mask;previous_index=count
    count+=1
  finally:cap.release()
  after=video_path.stat()
  if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):raise IOError('source changed during proxy calculation')
  if count!=r['frames']:raise ValueError('source decoded frame count mismatch')
  if len(measured)<3 or len(missing)>max(1,int(len(want)*.2)):raise ValueError('camera proxy insufficient reliable frames')
  hh,ww=first.shape;grid=np.array([[ww*x,hh*y,1.] for x in [.15,.5,.85] for y in [.15,.5,.85]])
  displacement=[];angles=[];speed=[];angular=[]
  for i,(idx,T) in enumerate(measured):
   displacement.append(float(np.percentile(np.linalg.norm((grid@T.T)[:,:2]-grid[:,:2],axis=1),90))/diag);angles.append(math.degrees(math.atan2(T[1,0],T[0,0])))
   if i:
    prev,A=measured[i-1];dt=(idx-prev)/fps
    if dt>1.51:raise ValueError('camera proxy evidence gap exceeds1.5sec')
    delta=T@np.linalg.inv(A);speed.append(float(np.percentile(np.linalg.norm((grid@delta.T)[:,:2]-grid[:,:2],axis=1),90))/diag/dt);angular.append(abs(math.degrees(math.atan2(delta[1,0],delta[0,0])))/dt)
  stats=dict(anchor_displacement_diag_max=max(displacement),image_plane_rotation_deg_max=max(abs(x) for x in angles),speed_diag_per_sec_p95=float(np.percentile(speed,95)),speed_diag_per_sec_max=max(speed),angular_speed_deg_per_sec_p95=float(np.percentile(angular,95)),angular_speed_deg_per_sec_max=max(angular),measured_samples=len(measured),missing_samples=len(missing),decoded_frames=count,sample_stride=stride)
  checks={k:stats[k]<=v for k,v in thresholds.items()};result.update(status='pass' if all(checks.values()) else 'camera_proxy_rejected',stats=stats,checks=checks,source_frame_range=[0,n],native_mask_policy='preserve source synchronization/sample_quality masks; invalid native supervision must remain masked',video_quality_filters_applied=False)
 except Exception as e:result['error']=type(e).__name__+': '+str(e)
 atomic(out,result);return result
results=[];started=time.time()
with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
 fs={pool.submit(one,r) for r in rows}
 for f in cf.as_completed(fs):
  try:results.append(f.result())
  except Exception as e:results.append(dict(status='infrastructure_error',error=str(e)))
  if len(results)%8==0 or len(results)==len(rows):atomic(progress/f'shard_{a.shard}.json',dict(processed=len(results),total=len(rows),passed=sum(x['status']=='pass' for x in results),elapsed_s=time.time()-started,updated=time.time()))
if any(result['status']=='infrastructure_error' for result in results):raise RuntimeError('proxy receipts could not be published')
