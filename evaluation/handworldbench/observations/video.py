"""Lossless sampled frames with content-keyed staging and explicit time stamps."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from handworldbench.runtime.storage import read,write,digest,cache_lock

def probe(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,width,height','-of','json',str(path)],text=True))['frames']
    if not info:raise ValueError('Video has no decodable frames')
    times=np.array([float(f['best_effort_timestamp_time']) for f in info]);times-=times[0]
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Video timestamps must increase strictly')
    return dict(times=times.tolist(),width=info[0]['width'],height=info[0]['height'],frame_count=len(times))

def stage(video,sha,root):
    root=Path(root)/'videos'/sha;root.mkdir(parents=True,exist_ok=True)
    target=root/('video'+Path(video).suffix)
    with cache_lock(root):
        if not target.exists():
            temp=target.with_suffix('.part');shutil.copyfile(video,temp);temp.replace(target)
        if not (root/'metadata.json').exists():write(root/'metadata.json',probe(target))
        return target,read(root/'metadata.json')

def decode(video,sha,root,role,indices=None,count=16):
    staged,meta=stage(video,sha,root)
    if indices is None:indices=np.linspace(0,meta['frame_count']-1,min(count,meta['frame_count'])).round().astype(int).tolist()
    if not indices or any(i<0 or i>=meta['frame_count'] for i in indices):raise ValueError('Sample outside video')
    folder=Path(root)/'frames'/digest(dict(video=sha,role=role,indices=indices,protocol='lossless_rgb'))
    folder.mkdir(parents=True,exist_ok=True)
    with cache_lock(folder):
        if (folder/'receipt.json').exists():return read(folder/'receipt.json')
        import cv2
        from PIL import Image
        wanted=set(indices);pixels={};cap=cv2.VideoCapture(str(staged))
        try:
            for index in range(max(indices)+1):
                if not cap.grab():raise RuntimeError(f'Decode failed at frame {index}')
                if index in wanted:
                    ok,frame=cap.retrieve()
                    if not ok:raise RuntimeError(f'Decode failed at frame {index}')
                    pixels[index]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        finally:cap.release()
        frames=[]
        for i,index in enumerate(indices):
            rgb=pixels[index];path=folder/f'{i:04d}.png';Image.fromarray(rgb).save(path,compress_level=1)
            frames.append(dict(sample_index=i,source_frame_index=index,timestamp_seconds=meta['times'][index],width=rgb.shape[1],height=rgb.shape[0],decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),path=str(path)))
        result=dict(role=role,video_sha256=sha,video=str(staged),frames=frames,sampling=dict(count=len(frames),indices=indices),metadata=meta)
        write(folder/'receipt.json',result);return result

def images(frames):
    from PIL import Image
    return [Image.open(frame['path']).convert('RGB') for frame in frames]
