"""Conditional legal-termination judgment with fixed evidence and reference eligibility."""
import hashlib
import json
import math
from pathlib import Path
from .io import read, sha

PROTOCOL = 'state_regulation_legal_termination_v1'
UNIFORM_SAMPLES = 12


def uniform_indices(fps, count, start, end, n=UNIFORM_SAMPLES):
    """The native half-open-window rule, extended to twelve fixed samples."""
    if not (math.isfinite(fps) and fps > 0 and count > 0 and 0 <= start < end):
        raise ValueError('Invalid uniform sampling interval')
    first = math.ceil(start * fps - 1e-7)
    last = math.ceil(end * fps - 1e-7) - 1
    if not 0 <= first <= last < count:
        raise ValueError('Sampling interval outside evaluated video')
    n = min(n, last - first + 1)
    return ([first] if n == 1 else
            sorted(set(round(first + i * (last - first) / (n - 1)) for i in range(n))))


def prompt_for(unit, check):
    return (
        'Judge only visible evidence in the chronologically ordered images. '
        'B marks the original initial state, E the originally designated establishment samples, '
        'L the original later check samples, and U fixed uniform samples of the intervening interval. '
        'These labels designate sampling roles, not proof of successful establishment or retention. '
        'Track the SAME object and the NEW configuration specified below relative to B. '
        'Object presence, familiar hand motion, rigid rotation, or a pre-existing configuration alone '
        'does not establish the required new state.\n'
        'Tracked new state: ' + unit['state_description'] + '\n'
        'Original check: ' + check['description'] + '\n'
        'Permitted reversals or changes: ' + ' '.join(unit.get('allowed_reversals', [])) + '\n'
        'First determine from these images whether a visibly supported reversal permitted by the '
        'stated conditions actually terminates the keeping obligation. Do not assume it occurred. '
        'If no such termination is shown, evaluate retention through the original later check. '
        'If it is shown, evaluate how well the new configuration was maintained from establishment '
        'until that permitted termination. Do not require it to remain after the obligation ends. '
        'If an allowed change does not terminate the obligation, continue evaluating the same state '
        'with that allowed change. Words such as "exempt" in the original conditions exempt the '
        'supported event from a loss penalty; they do not exempt this question from scoring or imply '
        'a high score. An earlier unsupported loss, replacement or reset still counts against the '
        'score and cannot be excused by a later permitted action. Do not invent a minimum hold '
        'duration or other conditions absent from the task. A sampling gap is neither proof of '
        'failure nor proof of successful maintenance.\n'
        'Grade the actual preservation of the newly gained configuration during the required '
        'keeping period: 1 = the specified new state was not established, or complete unsupported '
        'reset/loss; 2 = only a small part was preserved; 3 = a substantial part was preserved and '
        'a substantial part was lost; 4 = most was preserved with a small unsupported loss; '
        '5 = it was preserved without an unsupported loss throughout the required period. '
        'A permitted termination alone is not evidence for grade 5. '
        '0 means essential visual evidence cannot determine the preservation. '
        'Reply with a single digit 0, 1, 2, 3, 4, or 5 only. Do not explain.'
    )


def evidence_identity(video_sha256, bbox, interval, indices, unit_id, check_id):
    binding = {'video_sha256': video_sha256, 'bbox_in_source': bbox,
               'fixed_interval_s': interval, 'fixed_uniform_indices': indices,
               'unit_id': unit_id, 'check_id': check_id, 'protocol': PROTOCOL}
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_evidence(evaluator, case, unit, check, original_frames, original_labels):
    """Preserve original B/E/L pixels and add twelve predetermined interval frames."""
    import cv2
    import numpy as np
    from PIL import Image
    evidence = read(evaluator.evidence_path)
    if evidence['binding']['case']['video_sha256'] != case['video_sha256']:
        raise ValueError('Legal-termination evidence/video binding differs')
    if sha(case['video']) != case['video_sha256']:
        raise ValueError('Legal-termination source video changed')
    if len(original_frames) != len(original_labels):
        raise ValueError('Incomplete original B/E/L labels')
    bbox = list(evidence['binding']['temporal_union_bbox'])
    cap = cv2.VideoCapture(case['video'])
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 <= bbox[0] < bbox[2] <= width or not 0 <= bbox[1] < bbox[3] <= height:
            raise ValueError('Legal-termination ROI outside source video')

        def source_image(index):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Cannot decode legal-termination evidence frame')
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(bbox)

        roles, originals = {}, {}
        for frame, label in zip(original_frames, original_labels):
            if sha(frame['path']) != frame['sha256']:
                raise ValueError('Original retention evidence changed')
            index = frame['source_index']
            if frame.get('bbox_in_source', bbox) != bbox:
                raise ValueError('Retention image and fixed source ROI differ')
            with Image.open(frame['path']) as image:
                original = image.convert('RGB').copy()
            if not np.array_equal(np.asarray(original), np.asarray(source_image(index))):
                raise ValueError('Source crop cannot reproduce original retention pixels')
            if index in originals and not np.array_equal(np.asarray(originals[index]), np.asarray(original)):
                raise ValueError('Duplicate source index has different original evidence')
            originals[index] = original
            roles.setdefault(index, []).append(label.split('; source time')[0])
        interval = [unit['established'][0], check['window'][1]]
        fixed = uniform_indices(fps, count, *interval)
        for index in fixed:
            roles.setdefault(index, []).append('U uniform')
        identity = evidence_identity(case['video_sha256'], bbox, interval, fixed, unit['id'], check['id'])
        folder = evaluator.context.cache_root / PROTOCOL / case['case_id'] / identity
        folder.mkdir(parents=True, exist_ok=True)
        frames, labels = [], []
        for index in sorted(roles):
            image = originals[index] if index in originals else source_image(index)
            path = folder / (str(index) + '.png')
            if path.exists():
                with Image.open(path) as previous:
                    if not np.array_equal(np.asarray(previous.convert('RGB')), np.asarray(image)):
                        raise ValueError('Frozen legal-termination evidence pixels differ')
            else:
                image.save(path)
            frames.append({'source_index': index, 'time_s': index / fps, 'path': str(path),
                           'sha256': sha(path), 'bbox_in_source': bbox, 'roles': roles[index],
                           'pixel_sha256': hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                           'sampling_protocol': PROTOCOL})
            labels.append('/'.join(roles[index]))
        audit = {'protocol': PROTOCOL, 'original_B_E_L_preserved': True,
                 'original_source_ROI_pixels_verified': True,
                 'chronological_unique_frames': True, 'output_adaptive_selection': False,
                 'uniform_samples': UNIFORM_SAMPLES, 'fixed_interval_s': interval,
                 'fixed_uniform_indices': fixed, 'bbox_in_source': bbox}
        return frames, labels, audit
    finally:
        cap.release()
