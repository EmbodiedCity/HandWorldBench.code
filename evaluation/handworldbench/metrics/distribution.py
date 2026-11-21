"""Collection-level JEPA similarity from the selected common video features."""
import math
import numpy as np

def polynomial_distance(prediction, reference):
    x,y=np.asarray(prediction,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y),x.shape[1])==0:
        raise ValueError('JEPA feature sets must be nonempty matrices with matching dimensions')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite JEPA features')
    d=x.shape[1]
    return float(100*((x@x.T/d)**2).mean()+100*((y@y.T/d)**2).mean()-200*((x@y.T/d)**2).mean())

def aggregate(rows):
    x=np.concatenate([np.load(r['features']['prediction']['feature_file'],allow_pickle=False) for r in rows])
    y=np.concatenate([np.load(r['features']['reference']['feature_file'],allow_pickle=False) for r in rows])
    distance=polynomial_distance(x,y)
    return dict(raw_score=distance,raw_unit='polynomial_mmd',score_0_1=math.exp(-.4*max(distance,0)),measurement_level='collection')
