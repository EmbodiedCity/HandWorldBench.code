"""Fixed score conversion and single-gap trajectory interpolation."""
import copy,math
def finite(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)

def photo_score(mse):
    return 0. if not finite(mse) else 1./(1.+max(0.,mse)/.01)

def interpolate_single_gaps(points,times):
    """Only original neighboring observations may support interpolation."""
    if len(points)!=len(times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Mismatched points or nonincreasing timestamps')
    out=copy.deepcopy(points);filled=[]
    def valid(p): return isinstance(p,(list,tuple)) and len(p)==2 and all(finite(x) for x in p)
    for i in range(1,len(points)-1):
        if points[i] is None and valid(points[i-1]) and valid(points[i+1]):
            ratio=(times[i]-times[i-1])/(times[i+1]-times[i-1])
            out[i]=[a+ratio*(b-a) for a,b in zip(points[i-1],points[i+1])];filled.append(i)
    return out,filled

def trajectory_predictions(prediction,indices,frames):
    out=copy.deepcopy(prediction);audit={}
    lookup={f.get('input_sample_index',f['sample_index']):f for f in frames}
    times=[lookup[i]['timestamp_seconds'] for i in indices]
    for eid,rows in prediction.items():
        for i in indices:
            out[eid].setdefault(str(i),{'geometry_status':'absent','center_xy':None,'width':lookup[i]['width'],'height':lookup[i]['height']})
        points=[rows.get(str(i),{}).get('center_xy') if rows.get(str(i),{}).get('geometry_status')=='valid' else None for i in indices]
        filled,positions=interpolate_single_gaps(points,times)
        for j in positions:
            i=indices[j];row=out[eid][str(i)]
            row.update(center_xy=filled[j],geometry_status='valid',trajectory_interpolated=True,
                       interpolation_neighbors=[indices[j-1],indices[j+1]])
        audit[eid]={'interpolated_sample_indices':[indices[j] for j in positions],
                    'original_missing':sum(p is None for p in points)}
    return out,audit
