"""Split at participation changes, retain every visible supporting hand."""
import copy
import numpy as np
from .lazy_geometry import GeometryIndex as NativeGeometry
from .temporal import _project_hands

def participation_codes(ep):
    uv,depth=_project_hands(ep);uv=np.asarray(uv);depth=np.asarray(depth,bool)
    valid=np.asarray(ep['hand_valid'],bool)&np.isfinite(ep['positions_m']).all(axis=(2,3))&np.isfinite(uv).all(axis=(2,3))&np.all(depth,axis=2)
    inside=(uv[...,0]>=0)&(uv[...,0]<ep['width'])&(uv[...,1]>=0)&(uv[...,1]<ep['height'])&depth
    visible=np.any(inside,axis=2)&valid
    # Absence is only geometrically supported when all opposite-hand joints are
    # valid and entirely outside the image. Missing tracking is NEVER absence.
    codes=visible[:,0].astype(int)+2*visible[:,1].astype(int)
    # No visible joints does not prove absence when a huge/invalid projected
    # hull straddles the image. Require the entire bbox outside one image side.
    low=uv.min(axis=2);high=uv.max(axis=2)
    outside=(high[...,0]<0)|(low[...,0]>=ep['width'])|(high[...,1]<0)|(low[...,1]>=ep['height'])
    known=valid&(visible|outside)
    codes[~np.all(known,axis=1)]=-1
    return codes

class GeometryIndex:
    def __init__(self,ep,cfg):
        self.n=ep['n'];self.codes=participation_codes(ep);self.indices={};self.upper_bounds=np.zeros(self.n,np.int64)
        self.hands=[0,1];self.frame_hands=[([0] if x==1 else [1] if x==2 else [0,1]) for x in self.codes]
        self.diagnostics=dict(algorithm='window_participation_v4',source_frames=self.n,bounds_are_not_quality_approvals=True,unknown_participation_frames=int(np.sum(self.codes<0)))
        self.minimum=cfg['selection']['min_frames'];self.multiple=cfg['selection']['frame_multiple']
        changes=np.r_[0,np.flatnonzero(np.diff(self.codes))+1,self.n]
        for code,hands in [(3,[0,1]),(1,[0]),(2,[1])]:
            if not np.any(self.codes==code):continue
            local=copy.deepcopy(cfg);local['hand']['participant_overrides']={ep['episode_id']:hands}
            index=NativeGeometry(ep,local);self.indices[code]=index
            self.minimum=index.minimum;self.multiple=index.multiple
            for a,b in zip(changes[:-1],changes[1:]):
                if self.codes[a]!=code:continue
                starts=np.arange(a,b)
                caps=np.minimum(index.upper_bounds[a:b],b-starts)
                caps=(caps-index.offset)//index.multiple*index.multiple+index.offset
                self.upper_bounds[a:b]=np.where(caps>=index.minimum,caps,0)
        self.diagnostics.update(maximum_hard_upper_bound=int(self.upper_bounds.max(initial=0)),hard_legal_starts=int(np.count_nonzero(self.upper_bounds)))
    def longest_valid(self,start,upper=None):
        code=int(self.codes[start]);index=self.indices.get(code)
        if index is None:return None
        upper=min(int(self.upper_bounds[start]),upper if upper is not None else self.n)
        candidate=index.longest_valid(start,upper)
        if candidate:
            candidate['participant_inference']=dict(rule='window_visible_union_split_at_changes',participating_hands=candidate['participating_hands'],status='geometry_supported_rgb_semantics_pending',static_supporting_hands_are_participants=True,absence_requires_valid_outside_image_skeleton=True,motion_used_to_infer_participation=False)
        return candidate
