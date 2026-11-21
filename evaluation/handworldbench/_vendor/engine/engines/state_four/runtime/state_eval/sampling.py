"""Fixed half-open-window sampling; never align by total-duration proportion."""
import math

def sample_window(fps,frame_count,window,max_frames=3):
 if not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:raise ValueError('invalid_fps')
 if not isinstance(frame_count,int) or frame_count<1:raise ValueError('invalid_frame_count')
 if not isinstance(max_frames,int) or max_frames<1:raise ValueError('invalid_sampling_count')
 if not isinstance(window,(list,tuple)) or len(window)!=2 or not all(isinstance(t,(int,float)) and math.isfinite(t) for t in window) or not 0<=window[0]<window[1]:raise ValueError('invalid_window')
 first=math.ceil(window[0]*fps-1e-7);last=math.ceil(window[1]*fps-1e-7)-1
 if not 0<=first<=last<frame_count:raise ValueError('window_outside_video')
 n=min(max_frames,last-first+1)
 if n==1:return [first]
 return sorted(set(round(first+k*(last-first)/(n-1)) for k in range(n)))

def task_windows(annotation):
 rows=[]
 if annotation['applicability']['state_progression']['status']=='applicable':
  rows.extend({'id':'state:'+k,'window_s':annotation['windows'][k],'role':k} for k in ['before','process','after'])
 if annotation['applicability']['state_regulation']['status']=='applicable':
  for unit in annotation['persistence_units']:
   if not unit['checks']:raise ValueError('missing_retention_check')
   if unit['before'][0]>unit['established'][0] or any(x['window'][0]<unit['established'][1] for x in unit['checks']):raise ValueError('invalid_retention_time_order')
   prefix='persistence:'+unit['id']+':'
   rows.extend([{'id':prefix+'before','window_s':unit['before'],'role':'before'},{'id':prefix+'established','window_s':unit['established'],'role':'established'}])
   rows.extend({'id':prefix+x['id'],'window_s':x['window'],'role':'check'} for x in unit['checks'])
 ids=[x['id'] for x in rows]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate_window_id')
 return rows
