"""Official SAM3 video instance tracker; fixed entity prompts, no text regrouping.

A bbox is encoded as its two corners with official tracker labels 2 and 3.
This is the identical encoding in Sam3TrackerPredictor.add_new_points_or_box.
It preserves each caller-specified instance ID; the ID is NOT identity truth.
"""

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load

# Resolve metric engine modules.
import sys as _final_sys
from pathlib import Path as _FinalPath
_final_engine = next(p for p in _FinalPath(__file__).resolve().parents if (p/'runtime_paths.py').is_file())
if str(_final_engine) not in _final_sys.path: _final_sys.path.insert(0,str(_final_engine))
from runtime_paths import FINAL_ENGINE, FINAL_ROOT, FINAL_DATA, FINAL_RESULTS, FINAL_SUITE
from pathlib import Path
import argparse
import hashlib
import json
import io
import os
import sys
import tempfile

SUITE = Path(str(FINAL_SUITE))
CHECKPOINT = Path(str(FINAL_DATA/'metric_suite'/'weights/sam3/sam3.pt'))
CHECKPOINT_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
BPE = Path(str(FINAL_DATA/'metric_suite'/'assets/sam3/bpe_simple_vocab_16e6.txt.gz'))
BPE_SHA256 = '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
PROTOCOL = 'sam3_video_instance_points_boxes_native_v1'


from functools import lru_cache
@lru_cache(maxsize=4096)
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def metadata():
    return {'model': 'SAM3 native video instance tracker', 'protocol': PROTOCOL,
            'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'bpe_sha256': BPE_SHA256, 'precision': 'official CUDA BF16 autocast',
            'box_encoding': 'two normalized corners with official tracker labels 2,3',
            'point_limit_normalization': 'Official video tracker default 16 encoded points including 2 box corners; more than 16 is explicitly rejected before prediction, never silently truncated',
            'propagation': 'forward from earliest conditioning frame; unchanged shared sample timestamps, then JPEG100 model adapter',
            'input_recipe': 'Original shared RGB -> JPEG quality=100, subsampling=0 -> official SAM3 PIL RGB decode/1008x1008 resize/normalization',
            'identity_is_verified': False,
            'source_files': {str(p): file_sha(p) for p in [SUITE/'sources/SAM3/sam3/model_builder.py', SUITE/'sources/SAM3/sam3/model/sam3_video_inference.py', SUITE/'sources/SAM3/sam3/model/sam3_tracking_predictor.py', SUITE/'sources/SAM3/sam3/model/io_utils.py']}}


def instance_prompt(prompt, width, height):
    """Encode bbox plus optional points exactly as official SAM3 tracker does."""
    import math
    box = prompt['bbox_xyxy']
    if len(box) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box):
        raise ValueError('SAM3 instance bbox must contain four finite original-pixel values')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('SAM3 bbox is outside the supplied original image')
    points = [[x1/width, y1/height], [x2/width, y2/height]]
    labels = [2, 3]
    extra = prompt.get('points_xy') or []
    extra_labels = prompt.get('point_labels') or []
    if len(extra) != len(extra_labels):
        raise ValueError('SAM3 point and label counts differ')
    for xy, label in zip(extra, extra_labels):
        if len(xy) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in xy):
            raise ValueError('SAM3 instance point must be finite original-pixel xy')
        if not (0 <= xy[0] < width and 0 <= xy[1] < height) or label not in (0, 1):
            raise ValueError('SAM3 point is outside image or has invalid foreground/background label')
        points.append([xy[0]/width, xy[1]/height]); labels.append(label)
    if len(points) > 16:
        raise ValueError(f"SAM3 video tracker official default accepts at most 16 encoded prompt points including the 2 bbox corners; received {len(points)}. Reduce explicit points or use a separately validated full-point protocol; no points were silently truncated.")
    return points, labels


def run(request):
    global CHECKPOINT,BPE
    from handworldbench.backends.context import CONFIG
    CHECKPOINT=Path(CONFIG['models']['sam3']['checkpoint']);BPE=Path(CONFIG['models']['sam3']['vocabulary'])
    import numpy as np
    import torch
    from PIL import Image
    sys.path[:0] = [str(SUITE), str(SUITE/'sources/SAM3')]
    from reliable_io import load_image, save_image, local_tmp_root
    from common import write_json
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    device = request['device']
    if not str(device).startswith('cuda'):
        raise ValueError('Official SAM3 video inference requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    for path, expected in ((CHECKPOINT, CHECKPOINT_SHA256), (BPE, BPE_SHA256)):
        if file_sha(path) != expected:
            raise ValueError('Pinned SAM3 file SHA256 mismatch: '+str(path))
    frames, entities, role = request['frames'], request['entities'], request['role']
    folder = Path(request['folder']); folder.mkdir(parents=True, exist_ok=True)
    mapping = {i+1: e['id'] for i, e in enumerate(entities)}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Duplicate caller entity IDs')
    out = {e['id']: {} for e in entities}
    result_path = Path(request['result'])
    meta = metadata(); meta['frame_count'] = len(frames)
    if not any(e.get(role+'_prompts') for e in entities):
        write_json(result_path, {'tracks': out, 'model': meta, 'status': 'no_instance_prompts'})
        return
    source_hashes = []; encoded_input_hashes = []; decoded_jpeg_hashes = []
    with tempfile.TemporaryDirectory(prefix='sam3-entity-', dir=local_tmp_root()) as temporary:
        local_inputs = Path(temporary)
        input_folder = folder/(role+'_sam3_inputs'); input_folder.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            image = load_image(frame['path'], 'RGB')
            if image.size != (frame['width'], frame['height']):
                raise ValueError('SAM3 sampled RGB dimensions changed')
            rgb_sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            if frame.get('decoded_rgb_sha256') and frame['decoded_rgb_sha256'] != rgb_sha:
                raise ValueError('SAM3 sampled RGB checksum changed')
            source_hashes.append(rgb_sha)
            encoded = save_image(image, input_folder/f'{i:06d}.jpg', quality=100, subsampling=0)
            (local_inputs/f'{i:06d}.jpg').write_bytes(encoded)
            encoded_input_hashes.append(hashlib.sha256(encoded).hexdigest())
            with Image.open(io.BytesIO(encoded)) as encoded_image:
                decoded_jpeg_hashes.append(hashlib.sha256(np.asarray(encoded_image.convert('RGB')).tobytes()).hexdigest())
        global _resident_video_predictor
        if _resident_video_predictor is None:
            _resident_video_predictor=Sam3VideoPredictor(checkpoint_path=str(CHECKPOINT), bpe_path=str(BPE),strict_state_dict_loading=True,async_loading_frames=False,compile=False)
        predictor=_resident_video_predictor
        if predictor.model.tracker.max_point_num_in_prompt_enc != 16:
            raise ValueError('SAM3 video tracker point-limit default changed; protocol must be revalidated')
        meta['prompt_point_counts'] = []
        session = None
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                session = predictor.handle_request({'type': 'start_session', 'resource_path': str(local_inputs),
                    'offload_video_to_cpu': False, 'offload_state_to_cpu': False})['session_id']
                for index, entity in enumerate(entities):
                    for prompt in entity.get(role+'_prompts', []):
                        frame_index = prompt['sample_index']
                        if not isinstance(frame_index, int) or not 0 <= frame_index < len(frames):
                            raise ValueError('SAM3 prompt frame index outside shared samples')
                        frame = frames[frame_index]
                        points, labels = instance_prompt(prompt, frame['width'], frame['height'])
                        meta['prompt_point_counts'].append({'entity_id': entity['id'], 'sample_index': frame_index, 'bbox_corner_points': 2, 'supplied_extra_points': len(prompt.get('points_xy') or []), 'encoded_point_count': len(points), 'effective_point_count': len(points), 'truncated': False})
                        predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                            'frame_index': frame_index, 'obj_id': index+1, 'points': points,
                            'point_labels': labels, 'rel_coordinates': True})
                for response in predictor.handle_stream_request({'type': 'propagate_in_video',
                        'session_id': session, 'propagation_direction': 'forward',
                        'force_tracker_propagation': True,
                        'start_frame_index': min(p['sample_index'] for e in entities for p in e.get(role+'_prompts', []))}):
                    index = int(response['frame_index']); outputs = response['outputs']
                    ids = outputs['out_obj_ids']; masks = outputs['out_binary_masks']
                    for obj_id, mask in zip(ids, masks):
                        obj_id = int(obj_id)
                        if obj_id not in mapping:
                            raise ValueError('SAM3 returned an unrequested entity ID')
                        m = np.asarray(mask, dtype=bool).squeeze()
                        frame = frames[index]
                        if m.shape != (frame['height'], frame['width']):
                            raise ValueError('SAM3 mask shape differs from original sampled image')
                        path = folder/role/'masks'/f'{index:06d}_{mapping[obj_id]}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_image(Image.fromarray(m.astype('uint8')*255), path)
                        yy, xx = np.where(m)
                        box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if len(xx) else None
                        out[mapping[obj_id]][str(index)] = {'mask': str(path), 'mask_sha256': file_sha(path),
                            'bbox_xyxy': box, 'exists': bool(len(xx)),
                            'center_xy': [float(xx.mean()), float(yy.mean())] if len(xx) else None,
                            'tracker_id': obj_id, 'identity_verification': False, 'identity_matches': None,
                            'localization_status': 'unknown', 'mask_model': 'SAM3 native; model estimate',
                            'mask_model_protocol': PROTOCOL, 'mask_checkpoint_sha256': CHECKPOINT_SHA256,
                            'source_rgb_sha256': source_hashes[index],
                            'shared_sample_rgb_sha256': source_hashes[index],
                            'sam3_input_jpeg_sha256': encoded_input_hashes[index],
                            'sam3_input_decoded_jpeg_rgb_sha256': decoded_jpeg_hashes[index],
                            'input_recipe': meta['input_recipe']}
                    print('SAM3_ENTITY_FRAME', role, index, flush=True)
        finally:
            if session is not None:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
            # Session closed above; retain weights for the next request.
    meta.update(source_rgb_sha256=source_hashes, shared_sample_rgb_sha256=source_hashes,
                encoded_jpeg_sha256=encoded_input_hashes, decoded_jpeg_rgb_sha256=decoded_jpeg_hashes,
                source_rgb_sha256_meaning='shared sampler RGB before JPEG; not a claim that the model sees byte-identical RGB',
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
    write_json(result_path, {'tracks': out, 'model': meta, 'status': 'ok'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('request')
    args = parser.parse_args(); run(_hb_loads(Path(args.request).read_text()))

_resident_video_predictor=None
