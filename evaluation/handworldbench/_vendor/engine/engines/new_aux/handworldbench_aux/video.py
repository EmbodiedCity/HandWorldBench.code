from __future__ import annotations
from pathlib import Path
import hashlib,math
from .core import MetricIssue,sha256

def probe(video):
 import cv2
 cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
 if fps<=0 or count<1 or w<1 or h<1:raise MetricIssue('invalid_input','Cannot decode video metadata')
 return {'fps':fps,'frame_count':count,'width':w,'height':h,'duration':count/fps}

def sample(video,fps=None,window=None,max_side=None):
 import cv2,numpy as np
 from PIL import Image
 info=probe(video);start,end=window or (0,info['duration'])
 if start<0 or end<=start or end>info['duration']+1/info['fps']:raise MetricIssue('invalid_input','Evaluation window exceeds generated video',{'window':[start,end],'duration':info['duration']})
 first=max(0,math.ceil(start*info['fps']-1e-6));last=min(info['frame_count'],math.ceil(end*info['fps']-1e-6))
 if first>=last:raise MetricIssue('insufficient_evidence','No decoded frame lies inside the requested half-open interval')
 if fps is None:indices=list(range(first,last))
 else:
  if fps<=0:raise MetricIssue('invalid_input','Sampling fps must be positive')
  indices=sorted(set(min(last-1,max(first,int(round(t*info['fps'])))) for t in np.arange(start,end,1/fps)))
 cap=cv2.VideoCapture(str(video));frames=[];records=[]
 for i in indices:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read()
  if not ok:cap.release();raise MetricIssue('invalid_input','Video frame decode failed',{'frame_index':i})
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);original_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
  if max_side and max(rgb.shape[:2])>max_side:
   scale=max_side/max(rgb.shape[:2]);rgb=cv2.resize(rgb,(round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)),interpolation=cv2.INTER_AREA)
  frames.append(Image.fromarray(rgb));records.append({'frame_index':i,'time_s':i/info['fps'],'rgb_sha256':original_sha,'size':[rgb.shape[1],rgb.shape[0]]})
 cap.release()
 if not frames:raise MetricIssue('insufficient_evidence','No frames in evaluation window')
 return frames,records,info

def map_window(cfg,annotation,window_id):
 windows={w['id']:w for w in annotation['windows']}
 if window_id not in windows:raise MetricIssue('invalid_input','Unknown window '+window_id)
 w=windows[window_id];mapping=cfg.get('window_map',{})
 if window_id in mapping:
  value=mapping[window_id]
  if isinstance(value,dict) and value.get('status')=='not_generated':raise MetricIssue('insufficient_evidence','Required window explicitly not generated',{'window_id':window_id,'generation_failure':True,'reason':value.get('reason')})
  return value
 if cfg.get('time_mapping')=='identity':return [w['start_s'],w['end_s']]
 raise MetricIssue('invalid_input','Explicit window_map or time_mapping=identity is required; proportional alignment is forbidden')

def window_frames(cfg,annotation,ids,fps=3):
 frames=[];records=[]
 for wid in ids:
  window=map_window(cfg,annotation,wid);fs,rs,_=sample(cfg['video'],fps,window)
  frames.extend(fs);records.extend([{**v,'window_id':wid} for v in rs])
 paired={}
 for frame,record in zip(frames,records):
  key=record["frame_index"]
  if key not in paired:paired[key]=(frame,{**record,"window_ids":[record["window_id"]]})
  elif record["window_id"] not in paired[key][1]["window_ids"]:paired[key][1]["window_ids"].append(record["window_id"])
 ordered=[paired[k] for k in sorted(paired)]
 return [p[0] for p in ordered],[p[1] for p in ordered]
