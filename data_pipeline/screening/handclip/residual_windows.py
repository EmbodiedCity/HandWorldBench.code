"""Restrict candidate upper bounds before RGB work; never relax quality gates."""
import numpy as np

def cap_forbidden(bounds, forbidden, minimum, multiple=80, offset=1):
    out=np.asarray(bounds,dtype=np.int64).copy();n=len(out)
    normalized=sorted((max(0,int(a)),min(n,int(b))) for a,b in forbidden if int(a)<n and int(b)>0 and int(b)>int(a))
    merged=[]
    for a,b in normalized:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    for a,b in merged:
        out[a:b]=0
        starts=np.arange(a,dtype=np.int64)
        if len(starts):out[:a]=np.minimum(out[:a],a-starts)
    out=np.maximum(0,((out-offset)//multiple)*multiple+offset)
    out[out<minimum]=0
    return out

def source_exclusions(fields,n,seeds):
    offset=fields['source_frame_offset']
    if type(offset) is not int:raise ValueError('source frame offset not established')
    return tuple((max(0,c['start_frame']-offset),min(n,c['end_frame']-offset)) for c in seeds if c['recording_id']==fields['recording_id'] and c['start_frame']<offset+n and c['end_frame']>offset)
