import copy
import hashlib
import json
from pathlib import Path

import yaml


def load_config(path, **overrides):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for name, value in overrides.items():
        if value is not None:
            if name in ('workers','export_workers'):
                cfg['runtime'][name] = value
            else:
                cfg['selection'][name] = value
    s = cfg['selection']
    for name in ('target_count', 'frame_multiple', 'min_frames', 'max_per_recording'):
        if not isinstance(s[name], int) or isinstance(s[name], bool) or s[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if s.get('exact_frames') is not None:
        n = s['exact_frames']
        if not isinstance(n, int) or n < s['min_frames'] or (n-s.get('frame_offset',0)) % s['frame_multiple']:
            raise ValueError('exact_frames must be >= min_frames and divisible by frame_multiple')
        if s.get('max_frames') is not None and n > s['max_frames']:
            raise ValueError('exact_frames exceeds max_frames')
    if s.get('max_frames') is not None and s['max_frames'] < s['min_frames']:
        raise ValueError('max_frames < min_frames')
    quotas = s.get('frame_count_quotas')
    if quotas is not None:
        if not isinstance(quotas, dict) or not quotas:
            raise ValueError('frame_count_quotas must be a nonempty length/count mapping')
        lengths = set()
        for key, count in quotas.items():
            if isinstance(key, bool) or str(key) != str(int(key)):
                raise ValueError('quota lengths must be canonical integer strings or integers')
            length = int(key)
            if (length in lengths or length < s['min_frames'] or (length-s.get('frame_offset',0)) % s['frame_multiple']
                    or (s.get('max_frames') is not None and length > s['max_frames'])
                    or (s.get('exact_frames') is not None and length != s['exact_frames'])):
                raise ValueError('quota length violates the frozen frame selection contract')
            lengths.add(length)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError('quota counts must be positive integers')
        if sum(quotas.values()) != s['target_count']:
            raise ValueError('target_count must equal the sum of frame_count_quotas')
    baseline = s.get('external_exclusions', [])
    if not isinstance(baseline, list):
        raise ValueError('external_exclusions must be a list')
    for item in baseline:
        if not isinstance(item, dict) or not item.get('recording_id') or not item.get('source_path'):
            raise ValueError('every external exclusion needs a recording ID and source path')
        if any(not isinstance(item.get('source_'+position+'_frame_fingerprint'), str)
               or len(item['source_'+position+'_frame_fingerprint']) != 16
               or any(x not in '0123456789abcdefABCDEF' for x in item['source_'+position+'_frame_fingerprint'])
               for position in ('first', 'middle', 'last')):
            raise ValueError('external exclusions require three valid interval fingerprints')
    if baseline and not s.get('exclude_baseline_recordings', False):
        raise ValueError('this additive protocol requires excluding entire baseline recordings')
    for name in ('workers', 'source_batch_size', 'export_retry_limit', 'export_workers'):
        value = cfg['runtime'].get(name, {'source_batch_size': 256, 'export_retry_limit': 2, 'export_workers': 2}.get(name))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'runtime.{name} must be a positive integer')
    quota_mb = float(cfg['runtime'].get('max_cache_gb',8)) * 1024
    per_worker_mb = cfg['runtime'].get('rgb_cache_max_bytes', cfg['rgb'].get('max_cache_mb',512)*1024**2)/1024**2
    if cfg['runtime']['workers'] * per_worker_mb > quota_mb:
        raise ValueError('workers * rgb.max_cache_mb exceeds the shared local cache budget')
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_manifest(root):
    return {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.txt', '.md')
            and not any(part in p.parts for part in ('__pycache__','runs','.pytest_cache','.git','.venv'))}
