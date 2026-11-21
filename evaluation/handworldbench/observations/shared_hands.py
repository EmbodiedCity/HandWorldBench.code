"""Resident DINO detection with pixel-exact, role-isolated shared receipts."""
import contextlib
import hashlib
import time
from pathlib import Path
from functools import lru_cache
from handworldbench.runtime.storage import cache_lock, digest, read, write
from handworldbench.observations.hands import HandDetector, PROMPTS

PROTOCOL = 'dino_rgb_025_nms05_v1'
_DETECTORS = {}

@lru_cache(maxsize=16)
def model_binding(weights):
    root = Path(weights).resolve()
    files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
    return dict(path=str(root), files=[[str(p.relative_to(root)) if root.is_dir() else p.name,
                                      p.stat().st_size, p.stat().st_mtime_ns] for p in files])

def binding(config, category):
    return dict(protocol=PROTOCOL, model=model_binding(str(config['models']['hand_detector'])),
                prompt=PROMPTS[category], rgb=True)

def predict(images, category, config, *, role, video_sha256, device='cuda:0'):
    if role not in ('reference', 'prediction') or not video_sha256:
        raise ValueError('Hand detection requires a video fingerprint and explicit role')
    start = time.monotonic()
    images = [im.convert('RGB') for im in images]
    bound = binding(config, category)
    keys = [digest(dict(binding=bound, role=role, video=video_sha256, size=im.size,
                        pixels=hashlib.sha256(im.tobytes()).hexdigest())) for im in images]
    unique = dict(zip(keys, images))
    root = Path(config['cache_root'])/'shared_hand_detection'
    results = {}
    timing = dict(cache_hits=0, inferred_images=0, batches=0, model_load_seconds=0.)
    # Sorted lock acquisition prevents deadlocks for overlapping frame batches.
    with contextlib.ExitStack() as stack:
        folders = {key: stack.enter_context(cache_lock(root, key)) for key in sorted(unique)}
        missing = []
        for key, folder in folders.items():
            if (folder/'receipt.json').exists():
                results[key] = read(folder/'receipt.json')['detections']
            else:
                missing.append(key)
        timing['cache_hits'] = sum(key in results for key in keys)
        if missing:
            model_key = (digest(bound['model']), str(device))
            if model_key not in _DETECTORS:
                loaded = time.monotonic()
                _DETECTORS[model_key] = HandDetector(config['models']['hand_detector'], device=device,
                    batch=config.get('batch', {}).get('hand_detector', 8))
                timing['model_load_seconds'] = time.monotonic()-loaded
            values, inference = _DETECTORS[model_key].predict([unique[key] for key in missing], category)
            if len(values) != len(missing):
                raise ValueError('Shared hand detection output count mismatch')
            timing.update(inference)
            timing['inferred_images'] = len(missing)
            for key, value in zip(missing, values):
                write(folders[key]/'receipt.json', dict(protocol=PROTOCOL, detections=value))
                results[key] = value
    timing['seconds'] = time.monotonic()-start
    return [results[key] for key in keys], timing

