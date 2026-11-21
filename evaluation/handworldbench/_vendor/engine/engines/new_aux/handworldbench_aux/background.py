from __future__ import annotations
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next((p for p in _HbPath(__file__).resolve().parents if (p / 'handworldbench_paths.py').is_file()))
if str(_hb_root) not in _hb_sys.path:
    _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import sys, gc, math, shutil, tempfile
from pathlib import Path
import numpy as np
from .core import ROOT, DATA, RUNTIME, MetricIssue, model_path, stage_model, sha256, digest, read_json, write_json, writable, result
from .video import sample
PROTOCOL = 'hb-background-rgb-da3-sam3-v1'
_RESIDENT_MODELS = {}

def input_check(config):
    bg = config.get('background', {})
    if not bg.get('source') or not bg.get('object_concepts') or (not all((isinstance(x, str) and x.strip() for x in bg['object_concepts']))):
        raise MetricIssue('invalid_input', 'Recorded foreground concepts and source are required')
    return bg

def cleanup():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def segment(cfg, frames):
    import torch
    bg = input_check(cfg)
    sam = Path(cfg.get('models', {}).get('sam3', _hb_path('runtime_data/classic/metric_suite/weights/sam3/sam3.pt')))
    bpe = Path(cfg.get('models', {}).get('sam3_bpe', _hb_path('runtime_data/classic/metric_suite/assets/sam3/bpe_simple_vocab_16e6.txt.gz')))
    if not sam.is_file() or not bpe.is_file():
        raise MetricIssue('missing_dependency', 'SAM3 checkpoint or BPE unavailable')
    from handworldbench.backends.context import CONFIG
    source = Path(CONFIG['models']['sam3']['source'])
    sys.path.insert(0, str(source))
    model = None
    try:
        import sam3.model_builder as builder
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        device = cfg.get('device', 'cuda:0')
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device)
        original_loader = builder._load_checkpoint

        def checked_loader(model, path):

            def hook(module, incompatible):
                if incompatible.missing_keys:
                    raise MetricIssue('missing_dependency', 'SAM3 checkpoint missing inference parameters', {'missing_keys': incompatible.missing_keys})
            handle = model.register_load_state_dict_post_hook(hook)
            try:
                return original_loader(model, path)
            finally:
                handle.remove()
        builder._load_checkpoint = checked_loader
        try:
            key = ('sam3', str(sam), str(bpe), str(device))
            model = _RESIDENT_MODELS.get(key)
            if model is None:
                model = build_sam3_image_model(checkpoint_path=str(sam), bpe_path=str(bpe), load_from_HF=False, device='cpu').to(device).eval()
                _RESIDENT_MODELS[key] = model
        finally:
            builder._load_checkpoint = original_loader
        processor = Sam3Processor(model, device=device, confidence_threshold=0.5)
        prompts = list(dict.fromkeys(cfg.get('background_foreground_prompts', ['hand', 'arm', 'person', 'tool']) + bg['object_concepts']))
        masks = []
        records = []
        for i, im in enumerate(frames):
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                state = processor.set_image(im)
                fg = np.zeros((im.height, im.width), dtype=bool)
                counts = {}
                for prompt in prompts:
                    processor.reset_all_prompts(state)
                    out = processor.set_text_prompt(state=state, prompt=prompt)
                    v = out.get('masks')
                    count = 0 if v is None else len(v)
                    counts[prompt] = count
                    if count:
                        fg |= v.detach().cpu().numpy().reshape((-1, im.height, im.width)).any(axis=0)
            if not fg.any():
                masks.append(np.zeros_like(fg))
                status = 'no_foreground_detection'
            else:
                masks.append(~fg)
                status = 'ok'
            records.append({'frame': i, 'foreground_instances': counts, 'status': status, 'background_fraction': float(masks[-1].mean())})
        return (np.stack(masks), {'frames': records, 'prompts': prompts, 'source': bg['source'], 'sam3_sha256': sha256(sam), 'bpe_sha256': sha256(bpe)})
    finally:
        if str(source) in sys.path:
            sys.path.remove(str(source))
        if model is not None:
            del model
        cleanup()

def reconstruct(cfg, frames):
    import torch
    source = ROOT / 'vendor/Depth-Anything-3/src'
    sys.path.insert(0, str(source))
    model = None
    try:
        from depth_anything_3.api import DepthAnything3
        key = ('da3', str(model_path(cfg, 'da3')), str(cfg.get('device', 'cuda:0')))
        model = _RESIDENT_MODELS.get(key)
        if model is None:
            model = DepthAnything3.from_pretrained(str(stage_model(cfg, 'da3'))).to(cfg.get('device', 'cuda:0')).eval()
            _RESIDENT_MODELS[key] = model
        with torch.inference_mode():
            pred = model.inference([np.asarray(f) for f in frames], process_res=504, process_res_method='upper_bound_resize', ref_view_strategy='saddle_balanced')
        return {'depth': np.asarray(pred.depth), 'extrinsics': np.asarray(pred.extrinsics), 'intrinsics': np.asarray(pred.intrinsics), 'rgb': np.asarray(pred.processed_images, dtype=np.uint8)}
    finally:
        if str(source) in sys.path:
            sys.path.remove(str(source))
        if model is not None:
            del model
        cleanup()

def prepare(cfg):
    import cv2
    bg = input_check(cfg)
    frames, records, info = sample(cfg['video'], 3, cfg.get('evaluation_window'))
    if len(frames) < 3:
        raise MetricIssue('insufficient_evidence', 'DA3 requires at least three sampled frames')
    signature = {'protocol': PROTOCOL, 'video_sha256': sha256(cfg['video']), 'frames': records, 'object_concepts': bg['object_concepts'], 'concept_source': bg['source'], 'models': cfg.get('models', {}), 'sampling_fps': 3, 'process_res': 504, 'adapter_sha256': sha256(__file__), 'da3_revision': '1c5fde9aed33c2659a620e244d0151b7741e237d', 'sam3_sha256': sha256(Path(cfg.get('models', {}).get('sam3', _hb_path('runtime_data/classic/metric_suite/weights/sam3/sam3.pt'))))}
    key = digest(signature)
    folder = writable(DATA / 'cache/background' / key)
    manifest = folder / 'manifest.json'
    archive = folder / 'arrays.npz'
    if manifest.is_file() and archive.is_file():
        meta = read_json(manifest)
        if meta.get('signature') != signature or sha256(archive) != meta.get('arrays_sha256'):
            raise MetricIssue('invalid_input', 'Background cache signature or payload checksum mismatch')
        with np.load(archive, allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        return (arrays, meta)
    if bg.get('masks_manifest') or bg.get('depth_manifest'):
        raise MetricIssue('invalid_input', 'Cache protocol manifest is incompatible')
    mask, seginfo = segment(cfg, frames)
    arrays = reconstruct(cfg, frames)
    n, h, w = arrays['depth'].shape
    if n != len(frames):
        raise MetricIssue('invalid_input', 'DA3 frame count mismatch')
    arrays['background'] = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in mask])
    arrays['times'] = np.asarray([r['time_s'] for r in records])
    validate_arrays(arrays)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RUNTIME / 'tmp') as tmp:
        staged = Path(tmp) / 'arrays.npz'
        np.savez_compressed(staged, **arrays)
        shutil.copyfile(staged, archive)
    meta = {'signature': signature, 'arrays_sha256': sha256(archive), 'segmentation': seginfo, 'path': str(archive), 'valid_background_fraction': float(arrays['background'].mean()), 'depth_shape': list(arrays['depth'].shape)}
    write_json(manifest, meta)
    return (arrays, meta)

def validate_arrays(a):
    d = a['depth']
    n, h, w = d.shape
    if a['background'].shape != d.shape or a['rgb'].shape != (n, h, w, 3) or len(a['extrinsics']) != n or (a['intrinsics'].shape != (n, 3, 3)) or (len(a['times']) != n):
        raise MetricIssue('invalid_input', 'Frame/mask/depth/camera shapes do not match')
    if a['background'].dtype != bool or a['extrinsics'].shape[1:] not in [(3, 4), (4, 4)] or np.any(np.abs(np.linalg.det(a['intrinsics'])) < 1e-12):
        raise MetricIssue('invalid_input', 'Invalid mask type or singular/malformed cameras')
    if not np.isfinite(a['intrinsics']).all() or not np.isfinite(a['extrinsics']).all():
        raise MetricIssue('invalid_input', 'Invalid camera matrices')
    if np.any(np.diff(a['times']) <= 0):
        raise MetricIssue('invalid_input', 'Frame times are not strictly ordered')

def clouds(a, max_points=5000):
    validate_arrays(a)
    out = []
    rng = np.random.default_rng(0)
    for i, d in enumerate(a['depth']):
        y, x = np.where(a['background'][i] & np.isfinite(d) & (d > 1e-06))
        if len(x) > max_points:
            k = rng.choice(len(x), max_points, replace=False)
            x = x[k]
            y = y[k]
        cam = np.linalg.inv(a['intrinsics'][i].astype(np.float64)) @ np.stack([x, y, np.ones(len(x))]) * d[y, x]
        e = a['extrinsics'][i]
        world = e[:3, :3].T @ (cam - e[:3, 3, None])
        out.append((world.T, a['rgb'][i, y, x]))
    return out

def project(a, target, world, colors):
    h, w = a['depth'].shape[1:]
    e = a['extrinsics'][target]
    cam = (e[:3, :3] @ world.T + e[:3, 3, None]).T
    counts = {'input_points': len(cam)}
    mask = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1e-06)
    counts['behind_camera_or_nonfinite'] = int((~mask).sum())
    cam = cam[mask]
    colors = colors[mask]
    uv = (a['intrinsics'][target] @ cam.T).T
    uv = uv[:, :2] / uv[:, 2, None]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    counts['outside_image'] = int((~ok).sum())
    uv = np.rint(uv[ok]).astype(int)
    z = cam[ok, 2]
    colors = colors[ok]
    x, y = (uv[:, 0], uv[:, 1])
    d = a['depth'][target, y, x]
    ok = a['background'][target, y, x] & np.isfinite(d) & (d > 1e-06)
    counts['target_foreground'] = int((~a['background'][target, y, x]).sum())
    counts['invalid_target_depth'] = int((~(np.isfinite(d) & (d > 1e-06))).sum())
    x, y, z, colors, d = (x[ok], y[ok], z[ok], colors[ok], d[ok])
    pixel = y * w + x
    order = np.lexsort((z, pixel))
    _, first = np.unique(pixel[order], return_index=True)
    visible = order[first]
    counts['zbuffer_occluded'] = len(z) - len(visible)
    return {'filter_counts': counts, 'relative_errors': (np.abs(z - d) / d)[visible], 'source_colors': colors[visible], 'target_colors': a['rgb'][target, y[visible], x[visible]], 'valid_projections': len(z), 'visible_pixels': len(visible), 'target_background_pixels': int(a['background'][target].sum())}
