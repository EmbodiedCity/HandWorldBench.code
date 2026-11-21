"""Fixed-branch photometric scoring."""
from handworldbench.backends.normalization import photo_score
def calculate_photometric(metric,a,meta=None):
    import numpy as np
    from handworldbench_aux.background import clouds,project
    if metric!='photometric_consistency':raise ValueError('Deleted background metric requested')
    pointcloud=clouds(a);n=len(pointcloud);rows=[];values=[];missing=0
    for dst in range(1,n):
        branches={}
        previous = project(a, dst, *pointcloud[dst-1])
        first = previous if dst == 1 else project(a, dst, *pointcloud[0])
        for label, p in (('previous', previous), ('first', first)):
            mse = None
            if p['visible_pixels']>=100:
                diff=(p['source_colors'].astype(float)-p['target_colors'].astype(float))/255.
                mse=float(np.mean(diff**2))
            else:missing+=1
            branches[label]={'mse':mse,'score':photo_score(mse),'visible_pixels':p['visible_pixels']}
        value=sum(b['score'] for b in branches.values())/2.;values.append(value);rows.append({'target_frame':dst,'branches':branches,'score':value})
    return {'metric':metric,'status':'ok' if values else 'insufficient_frames','score':sum(values)/len(values) if values else None,
            'normalized':True,'protocol':'fixed_prev_first_equal_mse_0.01_missing_zero_v1',
            'expected_units':2*max(0,n-1),'penalized_units':missing,'details':{'cache':meta,'coverage':rows,'frame_count':n,'sampling_fps':3}}
