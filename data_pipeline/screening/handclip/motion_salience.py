"""Additional palm-motion gate; never replaces the frozen camera/RGB gates.

Positions are transformed through the validated VIO camera-to-world convention
and projected in the first camera. These are skeleton estimates, not human RGB
labels. At least ONE participating hand must satisfy ALL motion requirements;
a static supporting hand still has to pass the inherited completeness gates.
"""
import numpy as np
from .temporal import _rotation_matrices

DEFAULTS = dict(min_palm_extent_lengths=.5, min_screen_extent_diag=.02,
                min_active_fraction=.30, max_static_seconds=5.,
                active_palm_lengths=.10, active_screen_diag=.004,
                activity_lags_seconds=[.2,.4,.6],
                max_isolated_spike_palm_lengths=.35)


def validate_settings(settings):
    cfg = dict(DEFAULTS, **settings)
    if set(cfg)-set(DEFAULTS):raise ValueError('Unknown motion_salience setting')
    for k,v in cfg.items():
        values=v if k=='activity_lags_seconds' else [v]
        if not values or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0 for x in values):
            raise ValueError('Invalid motion_salience.'+k)
    if cfg['min_active_fraction']>1:raise ValueError('Invalid active fraction')
    return cfg


def project_points(points, K, D):
    if not np.isfinite(points).all() or np.any(points[...,2]<=1e-6):
        raise ValueError('Invalid fixed-camera projection depth')
    x,y=points[...,0]/points[...,2],points[...,1]/points[...,2]
    r=np.hypot(x,y);theta=np.arctan(r);t2=theta**2
    d=theta*(1+D[0]*t2+D[1]*t2**2+D[2]*t2**3+D[3]*t2**4)
    scale=np.divide(d,r,out=np.ones_like(r),where=r>1e-12)
    return np.stack((K[0,0]*x*scale+K[0,2],K[1,1]*y*scale+K[1,2]),axis=-1)


def extent(points):
    # PCA gives a rotation-invariant set of axes (up to degenerate eigenspaces).
    centered=points-np.median(points,axis=0)
    _,axes=np.linalg.eigh(centered.T@centered)
    axes_values=centered@axes
    return float(np.linalg.norm(np.percentile(axes_values,95,axis=0)-np.percentile(axes_values,5,axis=0)))


def longest_false_seconds(active,times,dt):
    edges=np.r_[False,~active,False].astype(int)
    starts=np.flatnonzero(np.diff(edges)==1);ends=np.flatnonzero(np.diff(edges)==-1)
    return max((float(times[e-1]-times[s]+dt) for s,e in zip(starts,ends)),default=0.)


class MotionIndex:
    def __init__(self,ep,settings):
        self.cfg=validate_settings(settings)
        self.ep=ep
        p=np.asarray(ep['positions_m'],dtype=float)
        q=np.asarray(ep['vio_q_wxyz'],dtype=float)
        norms=np.linalg.norm(q,axis=1)
        self.rot=_rotation_matrices(q/np.where(norms>0,norms,np.nan)[:,None])
        self.pose_valid=np.asarray(ep['pose_valid'],bool)&np.isfinite(q).all(axis=1)&(np.abs(norms-1)<=.01)
        self.hand_valid=np.asarray(ep['hand_valid'],bool)&np.isfinite(p).all(axis=(2,3))
        self.t=np.asarray(ep['vio_t'],float)
        palm=np.mean(p[:,:,[0,5,9,13,17]],axis=2)
        self.world=np.einsum('nij,nhj->nhi',self.rot,palm)+self.t[:,None,:]
        self.palms=np.linalg.norm(p[:,:,9]-p[:,:,0],axis=-1)
        timestamps=np.asarray(ep['times_ns'],np.int64)
        self.times=(timestamps-timestamps[0]).astype(float)/1e9
        self.K=np.asarray(ep['K'],float);self.D=np.asarray(ep['D'],float)
        self.diag=float(np.hypot(ep['width'],ep['height']))

    def evaluate(self,start,end,hands):
        cfg=self.cfg
        result=dict(version='palm_salience_v1',status='reject',participating_hands=list(hands),
                    active_hands=[],per_hand={},evidence='VIO-compensated skeleton estimate; human RGB review pending')
        if not 0<=start<end<=len(self.times) or end-start<3 or not hands or any(h not in (0,1) for h in hands):
            result['reason']='invalid_interval_or_hands';return result
        ts=self.times[start:end];dt=np.diff(ts)
        if (not np.isfinite(ts).all() or not np.all(dt>0) or not self.pose_valid[start:end].all()
            or not self.hand_valid[start:end,hands].all() or not np.isfinite(self.world[start:end,hands]).all()):
            result['reason']='invalid_time_pose_or_hand';return result
        # Sampling gaps cannot create apparent sustained motion.
        nominal=float(np.median(dt))
        if np.max(dt)>1.5*nominal or np.min(dt)<.5*nominal:
            result['reason']='irregular_timeline';return result
        anchors=self.rot[start].T
        camera=np.einsum('ij,nhj->nhi',anchors,self.world[start:end]-self.t[start])
        try:uv=project_points(camera[:,hands],self.K,self.D)/self.diag
        except ValueError:
            result['reason']='invalid_anchor_projection';return result
        for i,h in enumerate(hands):
            palm=float(np.median(self.palms[start:end,h]))
            if not np.isfinite(palm) or palm<=1e-6:
                result['reason']='invalid_palm_length';return result
            world=self.world[start:end,h]/palm
            screen=uv[:,i]
            jumps=np.linalg.norm(np.diff(world,axis=0),axis=1)
            return_distance=np.linalg.norm(world[2:]-world[:-2],axis=1)
            spikes=(np.minimum(jumps[:-1],jumps[1:])>cfg['max_isolated_spike_palm_lengths'])&(return_distance<cfg['max_isolated_spike_palm_lengths']/2)
            active=np.zeros(len(ts),bool)
            for lag in cfg['activity_lags_seconds']:
                for direction in (-1,1):
                    target=ts+direction*lag
                    indices=np.searchsorted(ts,target)
                    indices=np.clip(indices,0,len(ts)-1)
                    # Clamp only the edges; the actual represented time is saved.
                    dw=np.linalg.norm(world-world[indices],axis=1)
                    ds=np.linalg.norm(screen-screen[indices],axis=1)
                    active|=(dw>=cfg['active_palm_lengths'])&(ds>=cfg['active_screen_diag'])
            weights=np.r_[dt,nominal]
            row=dict(palm_extent_lengths=extent(world),screen_extent_diag=extent(screen),
                     active_fraction=float(np.sum(weights*active)/weights.sum()),
                     longest_static_seconds=longest_false_seconds(active,ts,nominal),
                     isolated_spikes=int(spikes.sum()),median_palm_m=palm)
            row['passed']=(row['palm_extent_lengths']>=cfg['min_palm_extent_lengths']
                           and row['screen_extent_diag']>=cfg['min_screen_extent_diag']
                           and row['active_fraction']>=cfg['min_active_fraction']
                           and row['longest_static_seconds']<=cfg['max_static_seconds']
                           and row['isolated_spikes']==0)
            result['per_hand'][str(h)]=row
            if row['passed']:result['active_hands'].append(h)
        # An isolated jump in ANY participating hand makes the evidence suspect.
        if any(x['isolated_spikes'] for x in result['per_hand'].values()):
            result['reason']='isolated_skeleton_spike'
        elif result['active_hands']:result.update(status='pass',reason='same_hand_amplitude_and_persistence')
        else:result['reason']='no_hand_meets_all_motion_requirements'
        return result
