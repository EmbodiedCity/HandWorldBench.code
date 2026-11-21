"""VBench2-Hand classifier over detected hand crops."""
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next((p for p in _HbPath(__file__).resolve().parents if (p / 'handworldbench_paths.py').is_file()))
if str(_hb_root) not in _hb_sys.path:
    _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
from pathlib import Path
import csv, importlib.util, json, math, os, sys
from collections import defaultdict
from typing import Dict, List
import cv2
import numpy as np
import torch
from PIL import Image
from common import CODE, DATA, definitions, official, provenance, sha, statistics, write_json
VERSIONS = ('VBench2-Hand', 'HandCraft')

def resources(selected_only=True):
    from handworldbench.backends.context import CONFIG
    path=CONFIG['models']['hand_morphology']
    if not Path(path).is_file():raise FileNotFoundError(path)
    return {'weights':[{'name':'vbench2_hand','installed_path':path}]}

def _load_vbench(info, device):
    from torchvision import transforms
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    try:
        from timm.data.transforms import _pil_interp
    except ImportError:
        from timm.data.transforms import str_to_pil_interp as _pil_interp

    def module(name, relative):
        spec = importlib.util.spec_from_file_location(name, official('VBench2-Hand', relative))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    config_module = module('_handworldbench_vb2_anatomy_config', 'config.py')
    vit_module = module('_handworldbench_vb2_anatomy_vit', 'models/vision_transformer.py')
    cls = definitions('VBench2-Hand', 'detect.py', ['Analyzer'], {'torch': torch, 'np': np, 'cv2': cv2, 'Image': Image, 'math': math, 'defaultdict': defaultdict, 'transforms': transforms, '_pil_interp': _pil_interp, 'IMAGENET_DEFAULT_MEAN': IMAGENET_DEFAULT_MEAN, 'IMAGENET_DEFAULT_STD': IMAGENET_DEFAULT_STD, 'List': List, 'Dict': Dict, 'DataParallel': torch.nn.DataParallel, 'F': torch.nn.functional, 'get_config': config_module.get_config, 'build_model': lambda cfg, is_pretrain=False: vit_module.build_vit(cfg)})['Analyzer']
    analyzer = cls.__new__(cls)
    analyzer.device = device
    analyzer.batch_size = 8
    original = torch.load

    def compatibility_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original(*args, **kwargs)
    torch.load = compatibility_load
    try:
        model, config = analyzer._build_model(str(official('VBench2-Hand', 'simmim_finetune__vit_base__img224__800ep.yaml')), next((x['installed_path'] for x in info['weights'] if x['name'] == 'vbench2_hand')))
    finally:
        torch.load = original
    analyzer.models = {'hand': model.to(device).eval()}
    analyzer.transforms = {'hand': analyzer._build_transform(config)}
    analyzer.class_threshold = {'hand': 0.3232}
    return (analyzer, config)

def vbench_result(frames, rows, info, cfg, device, folder):
    analyzer, config = load_vbench(info, device)
    details = []
    missed = []
    supplied=cfg['shared_hand_frames']
    by_frame={r['sample_index']:r['hands'] for r in supplied}
    jobs=[]
    for frame in frames:
        hands=by_frame[frame['sample_index']]
        image = cv2.imread(str(frame['path']))
        if image is None:
            raise ValueError('Could not read sampled anatomy image')
        images = []
        retained = []
        for number, hand in enumerate(hands):
            box = hand['box']
            a = np.asarray(box, float)
            if a.shape != (4,) or not np.isfinite(a).all() or np.any(a[2:] <= a[:2]):
                missed.append({'sample_index': frame['sample_index'], 'detection_index': number, 'side': hand.get('side'), 'reason': 'invalid_or_degenerate_box'})
                continue
            try:
                crop = analyzer.smart_cut(image, box)
            except ValueError as error:
                missed.append({'sample_index': frame['sample_index'], 'detection_index': number, 'side': hand.get('side'), 'reason': str(error)})
                continue
            jobs.append((crop,frame,number,hand,image.shape))
    analyzer.batch_size=max(1,int(cfg.get('morphology_batch',8)))
    offset=batches=reductions=0
    while offset<len(jobs):
        block=jobs[offset:offset+analyzer.batch_size]
        try:
            probs=analyzer.predict_batch('hand',[job[0] for job in block])
        except torch.cuda.OutOfMemoryError:
            if analyzer.batch_size==1:raise
            analyzer.batch_size=max(1,analyzer.batch_size//2);reductions+=1
            torch.cuda.empty_cache();continue
        if len(probs)!=len(block):raise ValueError('Morphology batch output count mismatch')
        for (_,frame,number,hand,shape),probability in zip(block,probs):
            a=np.asarray(hand['box'],float)
            if len(probability)!=2 or not all(math.isfinite(x) for x in probability):
                raise ValueError('Invalid morphology classifier output')
            details.append({'sample_index':frame['sample_index'],'timestamp_seconds':frame['timestamp_seconds'],
                'decoded_rgb_sha256':frame['decoded_rgb_sha256'],'detection_index':number,'side':hand.get('side'),
                'bbox_xyxy':hand['box'],'p_abnormal':float(probability[0]),'p_normal':float(probability[1]),
                'touches_image_boundary':bool(a[0]<=0 or a[1]<=0 or a[2]>=shape[1] or a[3]>=shape[0]),
                'abnormal_threshold':.3232,'abnormal_at_author_threshold':bool(probability[0]>.3232)})
        offset+=len(block);batches+=1
    write_json(folder/'morphology_batch.json',dict(crops=len(jobs),batches=batches,batch=analyzer.batch_size,oom_reductions=reductions))
    write_json(folder / 'vbench2_hand_per_crop.json', details)
    values = [x['p_abnormal'] for x in details]
    summary = statistics(values)
    frame_means = [float(np.mean([x['p_abnormal'] for x in details if x['sample_index'] == i])) for i in sorted({x['sample_index'] for x in details})]
    return dict(status='ok' if details else 'no_detected_hands',score=summary['mean'],
        summary=summary,frame_equal_weight_summary=statistics(frame_means),valid_count=len(details),
        sampled_frames=len(frames),excluded_crops=missed,crop_source='shared_grounding_dino',
        protocol='dino_vbench2_hand_v1',details_path=str(folder/'vbench2_hand_per_crop.json'),
        batch=analyzer.batch_size,batches=batches,oom_reductions=reductions)


def evaluate(frames, video, cfg, folder, device):
    info = resources(True)
    return vbench_result(frames, [{'hands': []} for f in frames], info, cfg, device, folder)
_CLASSIFIERS = {}

def load_vbench(info, device):
    key = (str(device), tuple((x['installed_path'] for x in info['weights'] if x['name'] == 'vbench2_hand')))
    if key not in _CLASSIFIERS:
        _CLASSIFIERS[key] = _load_vbench(info, device)
    return _CLASSIFIERS[key]
