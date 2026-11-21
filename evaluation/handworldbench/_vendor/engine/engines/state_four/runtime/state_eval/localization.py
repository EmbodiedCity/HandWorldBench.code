import math,copy,os,sys
from .io import sha
from .hand_bridge import hand_finder

def selected_hands(candidates):
    """Return distinct detector instances without inventing anatomical sides."""
    hands=[x for x in candidates if x.get('concept')=='hand']
    ids=[h['instance_id'] for h in hands]
    if len(hands)>2 or len(set(ids))!=len(ids):
        raise ValueError('Expected at most two distinct hand instances')
    return sorted(hands,key=lambda h:h['instance_id'])

def fallback_box(hands, size):
    if len(hands) != 2:
        return ([0, 0, *size], 'missing_hand_pair')
    b = [x['bbox'] for x in hands]
    c = [((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for v in b]
    cx = sum((x[0] for x in c)) / 2
    cy = sum((x[1] for x in c)) / 2
    w = max(b[0][2] - b[0][0], b[1][2] - b[1][0], abs(c[0][0] - c[1][0]))
    h = max(b[0][3] - b[0][1], b[1][3] - b[1][1], abs(c[0][1] - c[1][1]))
    return ([max(0, math.floor(cx - w * 0.6)), max(0, math.floor(cy - h * 0.6)), min(size[0], math.ceil(cx + w * 0.6)), min(size[1], math.ceil(cy + h * 0.6))], 'hand_pair_detail; full_object_coverage_unverified')
def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
def segment_distance(point, a, b):
    dx, dy = (b[0] - a[0], b[1] - a[1])
    den = dx * dx + dy * dy
    t = 0.0 if den == 0 else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / den))
    return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)
def select(candidates, size):
    hands = selected_hands(candidates)
    fallback, reason = fallback_box(hands, size)
    objects = copy.deepcopy([x for x in candidates if x['concept'] != 'hand'])
    if len(hands) != 2:
        return {'bbox': fallback, 'chosen': None, 'object_mask_found': False, 'fallback_reason': 'missing_hand_pair', 'rejected_object_candidates': len(objects)}
    a, b = [center(x['bbox']) for x in hands]
    radius = 0.5 * max((max(x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1]) for x in hands))
    for x in objects:
        x['distance_to_hand_segment'] = segment_distance(center(x['bbox']), a, b)
        x['inside_interaction_corridor'] = x['distance_to_hand_segment'] <= radius
    eligible = [x for x in objects if x['inside_interaction_corridor']]
    base = {'interaction_corridor': {'hand_centers': [a, b], 'radius': radius}, 'rejected_object_candidates': len(objects) - len(eligible)}
    if not eligible:
        return {**base, 'bbox': fallback, 'chosen': None, 'object_mask_found': False, 'fallback_reason': 'no_object_in_hand_interaction_corridor; hand_pair_crop; object_coverage_unverified'}
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    chosen = min(eligible, key=lambda x: (math.dist(center(x['bbox']), mid), -x['confidence']))
    box = chosen['bbox']
    mx = max(48.0, 0.5 * (box[2] - box[0]))
    my = max(48.0, 0.5 * (box[3] - box[1]))
    expanded = [max(0, math.floor(box[0] - mx)), max(0, math.floor(box[1] - my)), min(size[0], math.ceil(box[2] + mx)), min(size[1], math.ceil(box[3] + my))]
    combined = [min(expanded[0], fallback[0]), min(expanded[1], fallback[1]), max(expanded[2], fallback[2]), max(expanded[3], fallback[3])]
    return {**base, 'bbox': combined, 'chosen': chosen, 'object_mask_found': True, 'fallback_reason': None}

class Localizer:
    def __init__(self,context):
        self.hand_finder=hand_finder(context)
        self.keep_hand_finder=bool(context.config.get('hand_worker_root'))
        self.context=context
        config=context.config['sam3']
        self.binding={'weights_sha256':sha(context.path(config['weights'])),
                      'bpe_sha256':sha(context.path(config['bpe'])),
                      'confidence_threshold':config.get('confidence_threshold',.5),
                      'hand_selection_binding':self.hand_finder.binding,
                      'sam3_role':'task_object_localization_only'}
        self.processor=None

    def _ensure_sam(self):
        if self.processor is not None:return
        context=self.context;config=context.config['sam3']
        for key,part in [('HF_HOME','cache/sam_hf'),('TORCH_HOME','cache/torch'),('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
            p=context.cache_root/part;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
        import torch
        source=context.path(config['source_root']);sys.path.insert(0,str(source))
        from sam3 import model_builder
        from sam3.model.sam3_image_processor import Sam3Processor
        weights=str(context.path(config['weights']));bpe=str(context.path(config['bpe']))
        torch.cuda.set_device(context.device);self.torch=torch
        key=(weights,bpe,str(context.device),config.get('confidence_threshold',.5))
        if key not in _LOCALIZER_MODELS:
            model=model_builder.build_sam3_image_model(checkpoint_path=weights,bpe_path=bpe,load_from_HF=False,device='cpu').to(context.device).eval()
            _LOCALIZER_MODELS[key]=(model,Sam3Processor(model,device=context.device,confidence_threshold=key[-1]))
        self.model,self.processor=_LOCALIZER_MODELS[key]
    def candidates(self, image, concepts):
        return self._object_candidates(image,concepts,self.hand_finder.candidates(image))

    @property
    def batch_size(self):
        return max(1,int(self.hand_finder.config.get('batch',{}).get('hand_detector',8)))

    def candidates_many(self,images,concepts):
        hands=self.hand_finder.candidates_many(images)
        if len(hands)!=len(images):raise ValueError('Localization batch output count mismatch')
        return [self._object_candidates(image,concepts,records)
                for image,records in zip(images,hands)]

    def _object_candidates(self,image,concepts,records):
        import numpy as np
        w, h = image.size
        prompts=list(dict.fromkeys(c for c in concepts if c!='hand' and c.strip()))
        if not prompts:return records
        self._ensure_sam()
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            state = self.processor.set_image(image)
            for concept in prompts:
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(state=state, prompt=concept)
                masks = out['masks'].detach().cpu().numpy().reshape((-1, h, w)) > 0
                for mask, conf in zip(masks, out['scores'].detach().float().cpu().numpy()):
                    yy, xx = np.where(mask)
                    if len(xx) < 8:
                        continue
                    records.append({'concept': concept, 'confidence': float(conf), 'bbox': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)], 'area': int(mask.sum())})
        return records

    def close(self):
        if not self.keep_hand_finder:self.hand_finder.close()

_LOCALIZER_MODELS={}
