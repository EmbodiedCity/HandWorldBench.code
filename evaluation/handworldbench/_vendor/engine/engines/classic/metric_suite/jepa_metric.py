"""Official JEDi features and polynomial MMD, without process-group side effects."""
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next((p for p in _HbPath(__file__).resolve().parents if (p / 'handworldbench_paths.py').is_file()))
if str(_hb_root) not in _hb_sys.path:
    _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
from pathlib import Path
import gc, importlib.util, sys
import types
import ast
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from common import CODE, DATA, official, definitions, provenance, write_json, sha

def load_model(device):
    from vjepa.models import vit_huge
    from vjepa.models.attentive_pooler import AttentiveClassifier
    encoder = vit_huge(img_size=224, patch_size=16, num_frames=16, tubelet_size=2, uniform_power=True, use_sdpa=True, use_SiLU=False, tight_SiLU=False)
    from handworldbench.backends.context import CONFIG
    path = Path(CONFIG['models']['jepa']['encoder'])
    import json
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
    state = {k.replace('module.', '').replace('backbone.', ''): v for k, v in checkpoint['target_encoder'].items()}
    encoder.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()
    encoder = encoder.to(device).eval()
    classifier = AttentiveClassifier(embed_dim=encoder.embed_dim, num_heads=encoder.num_heads, depth=1, num_classes=174)
    probe = Path(CONFIG['models']['jepa']['probe'])
    checkpoint = torch.load(probe, map_location='cpu', weights_only=False, mmap=False)
    state = {k.removeprefix('module.'): v for k, v in checkpoint['classifier'].items()}
    classifier.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()
    aggregator = definitions('JEDi', 'videojedi/V_JEPA_utils.py', ['ClipAggregation'], {'torch': torch, 'nn': torch.nn})['ClipAggregation']
    return (aggregator(encoder, tubelet_size=2, attend_across_segments=True), classifier.to(device).eval())

class WorldArenaVideo:

    def __init__(self, video):
        import decord
        from torchvision.transforms import functional as TF
        cls = definitions('WorldArena', 'video_quality/JEDi/batch.py', ['PairedVideoFolderDataset'], {'Path': Path, 'Dataset': torch.utils.data.Dataset, 'torch': torch, 'VideoReader': decord.VideoReader, 'cpu': decord.cpu, 'TF': TF})['PairedVideoFolderDataset']
        self.video = Path(video)
        self.dataset = cls([self.video.stem], [self.video], num_frames=16, size=224)
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        self.indices = cls._uniform_indices(len(reader), 16)
        if not len(reader):
            raise ValueError('WorldArena input has no decodable frames')
        self.timestamps = np.asarray(reader.get_frame_timestamp(self.indices))[:, 0].tolist()
        self.protocol = {'protocol_id': 'worldarena_official_mp4_16x224_v1', 'scope': 'native_video', 'source': 'WorldArena/video_quality/JEDi/batch.py:PairedVideoFolderDataset defaults', 'frames_per_clip': 16, 'frame_rate': None, 'source_frame_count': len(reader), 'clip_count': 1, 'image_size': [224, 224], 'spatial_interpolation': 'torchvision TF.resize antialias=True, default bilinear', 'temporal_selection': 'Author torch.linspace(0,N-1,16).round(); if N<16, repeat the final frame.', 'repeated_frame_count': max(0, 16 - len(reader)), 'normalization': 'ImageNet RGB; documented channel-axis fix for vjepa 0.1.2', 'image_adapter_sample_count_applies': False}

    def __len__(self):
        return 1

    def get(self, index):
        if index != 0:
            raise IndexError(index)
        import contextlib, io
        messages = io.StringIO()
        with contextlib.redirect_stdout(messages):
            item = self.dataset[0]
        if '[WARN]' in messages.getvalue():
            raise RuntimeError(messages.getvalue().strip())
        tensor = item['video']
        if tensor.shape != (16, 3, 224, 224):
            raise RuntimeError('Unexpected WorldArena input tensor shape')
        return (tensor, {'clip_index': 0, 'source_frame_indices': self.indices, 'timestamps_seconds': self.timestamps})

def encode(tensor, encoder, classifier, device, finetuned=True):
    x = tensor.unsqueeze(0).to(device)
    import vjepa
    path = Path(vjepa.__file__).parent / 'datasets/utils/video/functional.py'
    tree = ast.parse(path.read_text())
    scope = {'torch': torch}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_is_tensor_clip', 'normalize')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

    def normal(clip):
        return scope['normalize'](clip.permute(1, 0, 2, 3), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).permute(1, 0, 2, 3)
    cls = definitions('JEDi', 'videojedi/V_JEPA.py', ['VJEPA'], {'torch': torch})['VJEPA']
    vjepa = cls.__new__(cls)
    vjepa.transforms = normal
    vjepa.encoder = encoder
    vjepa.classifier = types.SimpleNamespace(module=classifier)
    vjepa.finetuned = finetuned
    with torch.inference_mode():
        return vjepa.get_feats(x)
_REFERENCE_FEATURES = {}
