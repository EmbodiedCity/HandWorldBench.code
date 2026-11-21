import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
class LocalizationIdentityError(ValueError):
    """A localization response cannot be matched to the fixed entity inventory."""

ENTITY_FIELDS = ('id', 'type', 'description', 'anatomical_side', 'side', 'weight', 'role', 'trajectory_applicable', 'trajectory_normalized_threshold')
TRACK_FIELDS = ('mask', 'mask_sha256', 'mask_model', 'mask_model_protocol', 'mask_checkpoint_sha256', 'sam3_input_jpeg_sha256', 'sam3_input_decoded_jpeg_rgb_sha256', 'input_recipe', 'hand_selection_normalization')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')

def observation_indices(count, stride=4, *, expected_count=16):
    if type(expected_count) is not int or expected_count < 1 or type(count) is not int or (count not in (8, expected_count)):
        raise ValueError('Object observation requires the fixed 16-frame schedule')
    if type(stride) is not int or stride < 1:
        raise ValueError('object_observation_stride must be a positive integer')
    return sorted(set(range(0, count, stride)) | {count - 1})

def clean_entities(entities):
    from region_scoring import validate_entities
    result = [{k: copy.deepcopy(e[k]) for k in ENTITY_FIELDS if k in e} for e in entities]
    validate_entities(result)
    for e in result:
        if not re.fullmatch('[A-Za-z0-9_.-]+', e['id']) or e['id'] in ('.', '..'):
            raise ValueError('Entity IDs must be safe native mask filename components')
        if not isinstance(e.get('description'), str) or not e['description'].strip():
            raise ValueError('Each fixed entity needs its reference-defined description')
        if e['type'] == 'hand':
            side = e.get('side') or e.get('anatomical_side')
            if side not in ('left', 'right'):
                raise ValueError('Anatomical hand side must be explicit')
    return result

def verify_frames(frames, role, *, expected_count=16):
    from PIL import Image
    if type(expected_count) is not int or expected_count < 1 or len(frames) not in (8, expected_count):
        raise ValueError('Both roles require the fixed 16-frame schedule')
    previous = -math.inf
    for i, frame in enumerate(frames):
        if frame.get('sample_index', i) != i:
            raise ValueError('Frame sample indices must match their fixed order')
        stamp = frame['timestamp_seconds']
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < previous:
            raise ValueError('Frame timestamps must be finite and ordered')
        previous = stamp
        with Image.open(frame['path']) as im:
            image = im.convert('RGB')
            if image.size != (frame['width'], frame['height']):
                raise ValueError(role + ' source frame dimensions changed')
            if hashlib.sha256(image.tobytes()).hexdigest() != frame['decoded_rgb_sha256']:
                raise ValueError(role + ' source frame RGB changed')

def _box(value, width, height):
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any((type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        return None
    if not (0 <= value[0] < value[2] <= 1000 and 0 <= value[1] < value[3] <= 1000):
        return None
    b = [value[0] * width / 1000, value[1] * height / 1000, value[2] * width / 1000, value[3] * height / 1000]
    return b if b[2] - b[0] >= 2 and b[3] - b[1] >= 2 else None

def _point(value, width, height):
    if not isinstance(value, list) or len(value) != 2 or any((type(v) not in (int, float) or not math.isfinite(v) or (not 0 <= v < 1000) for v in value)):
        return None
    return [value[0] * width / 1000, value[1] * height / 1000]

def localization_prompt(objects):
    descriptions = [{'id': e['id'], 'description': e['description']} for e in objects]
    return 'Locate the requested object regions in this current ORIGINAL image: ' + json.dumps(descriptions, ensure_ascii=False) + '. Names are requests, not evidence that an object is present. For each id give a tight box of the visible object itself, excluding the holding hand, sleeve and nearby tool. Coordinates are normalized 0..1000 of this full original image. Give one foreground_point_1000 on a clearly visible object interior pixel, never on the holding hand. Optionally give at most two background_points_1000 on a holding hand or distractor to exclude. If a region cannot be located, return visible=false and null box/foreground point. Only current pixels are supplied. Return JSON {"entities":[{"id":string,"bbox_1000":[x1,y1,x2,y2] or null,"visible":boolean,"foreground_point_1000":[x,y] or null,"background_points_1000":[[x,y]]}]}.'

def locate_objects(observer, frames, objects, folder, stride=4, max_tokens=1600, workers=4, role='prediction', reference_union_count=None):
    """5 single-image requests by default; optional explicit batch API uses 4+1.

    ask_batch receives list[{images,prompt,path,labels,tokens}] and returns parsed
    dicts in the same order. Otherwise ask must be thread-safe (a single-consumer
    dispatcher proxy), or the caller must explicitly select workers=1 for HF.
    No retry or fallback is hidden in this adapter.
    """
    from PIL import Image
    if role not in ('reference', 'prediction'):
        raise ValueError('Explicit reference/prediction localization role required')
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Localization workers must be 1..4')
    if reference_union_count is not None and role != 'reference':
        raise ValueError('Only reference preprocessing accepts a frozen source union')
    indices = observation_indices(len(frames), stride, expected_count=reference_union_count or 16) if objects else []
    folder = Path(folder)
    prompt = localization_prompt(objects)
    requests = []
    for i in indices:
        frame = frames[i]
        with Image.open(frame['path']) as image:
            rgb = image.convert('RGB').copy()
        requests.append(dict(images=[rgb], prompt=prompt, path=folder / 'raw' / ('frame_%03d.json' % i), labels=['Current ORIGINAL frame. Width %s, height %s.' % (rgb.width, rgb.height)], tokens=max_tokens))
    answers = []
    if requests and observer is None:
        raise ValueError('Object localization requires an explicit prediction observer')
    if callable(getattr(observer, 'ask_batch', None)):
        for start in range(0, len(requests), 4):
            batch = requests[start:start + 4]
            response = observer.ask_batch(batch)
            if not isinstance(response, list) or len(response) != len(batch):
                raise ValueError('Localization batch cardinality changed')
            answers.extend(response)
    else:

        def ask(request):
            return observer.ask(request['images'], request['prompt'], request['path'], request['labels'], request['tokens'])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(ask, requests))
    observations = {}
    missing = 0
    ids = {e['id'] for e in objects}
    for i, answer in zip(indices, answers):
        if not isinstance(answer, dict) or not isinstance(answer.get('entities'), list):
            raise ValueError('Malformed localization response; no implicit fallback')
        rows = {}
        for row in answer['entities']:
            if not isinstance(row, dict) or row.get('id') not in ids or row['id'] in rows:
                raise LocalizationIdentityError('Localization entity IDs are unexpected or duplicated')
            rows[row['id']] = row
        observations[str(i)] = {}
        for e in objects:
            r = rows.get(e['id'], {})
            frame = frames[i]
            box = _box(r.get('bbox_1000'), frame['width'], frame['height'])
            visible = r.get('visible') is True
            valid = visible and box is not None
            missing += not valid
            item = dict(visible=visible, bbox_xyxy=box if valid else None, localization_available=valid, source_rgb_sha256=frame['decoded_rgb_sha256'], source_role=role, observation_performed=True)
            foreground = _point(r.get('foreground_point_1000'), frame['width'], frame['height'])
            if valid and foreground is not None:
                raw = r.get('background_points_1000')
                negatives = [_point(p, frame['width'], frame['height']) for p in (raw[:2] if isinstance(raw, list) else [])]
                negatives = [p for p in negatives if p is not None and p != foreground]
                item.update(points_xy=[foreground] + negatives, point_labels=[1] + [0] * len(negatives))
            observations[str(i)][e['id']] = item
    prompted = copy.deepcopy(objects)
    for e in prompted:
        e[role + '_prompts'] = []
        e[role + '_prompt_normalization'] = 'own_sparse_localization_no_identity_v1'
        for i in indices:
            r = observations[str(i)][e['id']]
            if r['localization_available']:
                p = dict(sample_index=i, bbox_xyxy=r['bbox_xyxy'])
                for key in ('points_xy', 'point_labels'):
                    if key in r:
                        p[key] = r[key]
                e[role + '_prompts'].append(p)
    audit = dict(role=role, observed_indices=indices, object_observation_stride=stride, primary_vlm_requests=len(requests), adapter_does_not_retry=True, caller_actual_requests_source='Caller HTTP raw attempts and dispatcher trace; includes caller retries', localization_workers=workers, entity_observations=len(indices) * len(objects), missing_localizations=missing, missing_localization_fraction=missing / (len(indices) * len(objects)) if objects else None, unobserved_indices=[i for i in range(len(frames)) if i not in indices], identity_requests=0, mask_verification_requests=0, state_requests=0)
    write(folder / 'object_localizations.json', observations)
    write(folder / 'localization_audit.json', audit)
    return (prompted, observations, audit)

def clean_tracks(tracks, entities, frames, role, *, geometry_out=None):
    from region_scoring import validate_mask_geometry
    expected = {e['id'] for e in entities}
    if set(tracks) - expected:
        raise ValueError('Tracker returned an undeclared entity')
    result = {}
    for e in entities:
        series = tracks.get(e['id'], {})
        if set(series) - {str(i) for i in range(len(frames))}:
            raise ValueError('Tracker returned an undeclared sample')
        result[e['id']] = {}
        if geometry_out is not None:
            geometry_out[e['id']] = {}
        for i, frame in enumerate(frames):
            row = copy.deepcopy(series.get(str(i), {}))
            if not row:
                row = dict(exists=False, source_rgb_sha256=frame['decoded_rgb_sha256'], mask_source_role=role, missing_reason='No tracker output at fixed sample')
            geometry = validate_mask_geometry(row, frame, role)
            if geometry_out is not None:
                geometry_out[e['id']][str(i)] = geometry
            cleaned = {k: row[k] for k in TRACK_FIELDS if k in row}
            cleaned.update(exists=geometry['geometry_status'] == 'valid', source_rgb_sha256=frame['decoded_rgb_sha256'], mask_source_role=role, gt_guidance_in_prediction=False, bbox_xyxy=geometry['bbox_xyxy'], center_xy=geometry['center_xy'], geometry_status=geometry['geometry_status'], area_pixels=geometry['area_pixels'], identity_checked=False)
            if geometry['geometry_status'] == 'absent':
                cleaned['missing_reason'] = row.get('missing_reason', 'Mask has no positive pixels')
            result[e['id']][str(i)] = cleaned
    return result
