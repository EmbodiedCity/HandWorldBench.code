"""Read cases from the published HandWorldBench layouts.

Benchmark (HF dataset ``EmbodiedCity/HandWorldBench``)::

    catalog.json            case_id, category, reference{video, width, height, fps, frame_count}, geometry{path}
    hands/<case_id>.npz     joints_camera_m (N,2,21,3), joints_2d_px (N,2,21,2), valid (N,2,21), ...
    camera/<case_id>.npz    K (3,3), D, distortion_model
    videos/<case_id>.mp4    reference video (optional here; used to cross-check size, fps and frame count)

Corpus (HF dataset ``EmbodiedCity/HandWorldBench-Corpus``, shards extracted into the dataset root)::

    manifest.jsonl[.gz]     id, domain, frames, fps, video{path}, hands{path}, camera{path}
    hands/, camera/         same array schema as the benchmark
    clips/<domain>/<id>/*.mp4

Hand 0 is the left hand and hand 1 the right hand in every array.
"""
import gzip
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

EMBODIMENT = {'human': 'human', 'ego': 'human', 'handson': 'handson', 'real_robot': 'ti5', 'ti5': 'ti5'}
FISHEYE = {'equidistant_fisheye', 'equidistant', 'fisheye'}


@dataclass
class Case:
    case_id: str
    embodiment: str            # 'human' | 'handson' | 'ti5'
    width: int
    height: int
    fps: float
    frame_count: int
    video: Optional[Path]      # reference video (may be None)
    joints: np.ndarray         # (N, 2, 21, 3) camera-space metres
    uv: np.ndarray             # (N, 2, 21, 2) pixels
    hand_valid: np.ndarray     # (N, 2) bool
    K: np.ndarray
    D: np.ndarray
    fisheye: bool
    hand20: Optional[np.ndarray] = None          # HandsOn only: (N, 2, 20) Wuji joint angles
    palm_to_camera: Optional[np.ndarray] = None  # HandsOn only: (N, 2, 4, 4)


def _arrays(hands_path, camera_path):
    with np.load(hands_path) as z:
        joints, uv, valid = z['joints_camera_m'], z['joints_2d_px'], z['valid']
    with np.load(camera_path) as c:
        K, D, model = c['K'], c['D'], str(c['distortion_model'])
    if D.size == 0:  # pinhole
        D = np.zeros(5)
    return dict(joints=joints, uv=uv, hand_valid=valid.any(-1), K=K, D=D, fisheye=model in FISHEYE)


def _probe(path):
    s = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                            'stream=width,height', '-of', 'json', str(path)]))['streams'][0]
    return int(s['width']), int(s['height'])


def detect_layout(root):
    root = Path(root)
    if (root / 'catalog.json').is_file():
        return 'benchmark'
    if (root / 'manifest.jsonl').is_file() or (root / 'manifest.jsonl.gz').is_file():
        return 'corpus'
    raise FileNotFoundError(f'{root}: neither catalog.json (benchmark) nor manifest.jsonl[.gz] (corpus) found')


def list_cases(root, layout):
    """[(case_id, embodiment)] in dataset order."""
    root = Path(root)
    if layout == 'benchmark':
        return [(r['case_id'], EMBODIMENT[r['category']]) for r in json.loads((root / 'catalog.json').read_text())]
    return [(r['id'], EMBODIMENT[r['domain']]) for r in _manifest(root)]


def _manifest(root):
    p = root / 'manifest.jsonl'
    if p.is_file():
        lines = p.read_text().splitlines()
    else:
        with gzip.open(root / 'manifest.jsonl.gz', 'rt') as f:
            lines = f.read().splitlines()
    return [json.loads(s) for s in lines if s.strip()]


class Dataset:
    def __init__(self, root, layout=None):
        self.root = Path(root)
        self.layout = layout or detect_layout(self.root)
        if self.layout == 'benchmark':
            self.rows = {r['case_id']: r for r in json.loads((self.root / 'catalog.json').read_text())}
        else:
            self.rows = {r['id']: r for r in _manifest(self.root)}

    def ids(self):
        return list(self.rows)

    def load(self, case_id):
        r = self.rows[case_id]
        root = self.root
        if self.layout == 'benchmark':
            ref = r['reference']
            hands = root / r.get('geometry', {}).get('path', f'hands/{case_id}.npz')
            a = _arrays(hands, root / 'camera' / f'{case_id}.npz')
            video = root / ref['video'] if ref.get('video') else None
            return Case(case_id=case_id, embodiment=EMBODIMENT[r['category']], width=ref['width'],
                        height=ref['height'], fps=ref['fps'], frame_count=ref['frame_count'], video=video, **a)
        video = root / r['video']['path']
        if not video.is_file():
            raise FileNotFoundError(f'{video}: extract the shard {r.get("shard", "")} into {root} first')
        a = _arrays(root / r['hands']['path'], root / r['camera']['path'])
        w, h = _probe(video)
        return Case(case_id=case_id, embodiment=EMBODIMENT[r['domain']], width=w, height=h, fps=r['fps'],
                    frame_count=r['frames'], video=video, **a)
