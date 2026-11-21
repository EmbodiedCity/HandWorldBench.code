"""Shared DINO hand proposals for state and memory evidence."""
from handworldbench.observations.shared_hands import binding, predict

def selection_binding(context):
    cfg=context.config
    return dict(binding(cfg['shared_hand_runtime'],cfg['hand_finder']['category']),
                selection='confidence_top_two_instances_v1')

def require_current_evidence(evidence,context):
    if evidence.get('hand_selection_binding') != selection_binding(context):
        raise ValueError('Evidence hand detection protocol differs')

class HandFinder:
    def __init__(self,context):
        self.config=context.config['shared_hand_runtime']
        self.options=context.config['hand_finder']
        self.binding=selection_binding(context)
        self.stats={'cache_hits':0,'inferred_images':0,'seconds':0.,'batches':0,'oom_reductions':0}

    def candidates(self,image):
        return self.candidates_many([image])[0]

    def candidates_many(self,images):
        rows,timing=predict(images,self.options['category'],self.config,
            role=self.options['role'],video_sha256=self.options['video_sha256'],
            device=self.options.get('device','cuda:0'))
        for key in self.stats:self.stats[key]+=timing.get(key,0)
        return [self._select(row) for row in rows]

    @staticmethod
    def _select(row):
        # Instance IDs describe proposals, not anatomical left/right labels.
        selected=sorted(row,key=lambda d:(-d['detection_confidence'],tuple(d['box'])))[:2]
        return [dict(concept='hand',instance_id=i,confidence=h['detection_confidence'],
                     bbox=h['box'],area=(h['box'][2]-h['box'][0])*(h['box'][3]-h['box'][1]),
                     detector='Grounding DINO') for i,h in enumerate(selected)]

    def close(self):
        pass

def hand_finder(context):
    return HandFinder(context)
