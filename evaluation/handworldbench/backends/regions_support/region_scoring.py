import hashlib
import math
from pathlib import Path
APPEARANCE_PROTOCOL = 'region_appearance_v2_fixed_gt_zero_floor_no_identity'
TRAJECTORY_PROTOCOL = 'mask_centroid_ordered_geometry_v1_no_identity'

def finite(value):
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value)

def validate_entities(entities):
    if not entities or len({e['id'] for e in entities}) != len(entities):
        raise ValueError('Nonempty unique entity definitions required')
    for e in entities:
        if e['type'] not in ('hand', 'object'):
            raise ValueError('Entity type must be hand or object')
        w = e.get('weight', 1.0)
        if not finite(w) or w < 0:
            raise ValueError('Invalid fixed entity weight')

def validate_schedule(indices):
    if not indices or list(indices) != sorted(set(indices)) or any((type(i) is not int or i < 0 for i in indices)):
        raise ValueError('Fixed unique ordered sample indices required')

def validate_mask_geometry(row, frame, role, resolve=lambda x: Path(x)):
    """Read and verify real mask bytes; an explicit absent mask is valid absence.

    Missing files, contradictory exists flags, wrong SHA/shape/role and source
    RGB mismatches are execution errors, never cheap zeros. Empty decoded masks
    produce absent. Returned geometry carries no physical-identity assertions.
    """
    import numpy as np
    from PIL import Image
    if row.get('mask_source_role', role) != role:
        raise ValueError('Mask role mismatch')
    if row.get('gt_guidance_in_prediction') is True and role == 'prediction':
        raise ValueError('Reference guidance on prediction is forbidden')
    if row.get('source_rgb_sha256') != frame['decoded_rgb_sha256']:
        raise ValueError('Mask source RGB mismatch')
    base = {'source_rgb_sha256': frame['decoded_rgb_sha256'], 'role': role, 'width': frame['width'], 'height': frame['height'], 'identity_checked': False, 'sample_index': frame.get('input_sample_index', frame.get('sample_index')), 'source_sample_index': frame.get('sample_index'), 'timestamp_seconds': frame.get('timestamp_seconds')}
    if not row.get('mask'):
        if row.get('exists') is not False:
            raise ValueError('Missing mask requires explicit exists=false')
        return dict(base, geometry_status='absent', area_pixels=0, center_xy=None, bbox_xyxy=None)
    p = Path(resolve(row['mask']))
    from handworldbench.runtime.storage import artifact_digest
    actual = artifact_digest(p)
    if actual != row.get('mask_sha256'):
        raise ValueError('Mask SHA mismatch')
    a = np.asarray(Image.open(p))
    if a.ndim != 2 or a.shape != (frame['height'], frame['width']):
        raise ValueError('Mask must be a single-channel aligned frame')
    ys, xs = np.nonzero(a > 0)
    exists = bool(len(xs))
    if row.get('exists') is not exists:
        raise ValueError('Mask exists flag contradicts decoded pixels')
    if not exists:
        return dict(base, geometry_status='absent', mask_sha256=actual, area_pixels=0, center_xy=None, bbox_xyxy=None)
    return dict(base, geometry_status='valid', mask_sha256=actual, area_pixels=int(len(xs)), center_xy=[float(xs.mean()), float(ys.mean())], bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)])

def state(row):
    value = row.get('geometry_status')
    if value not in ('valid', 'absent'):
        raise ValueError('Validated geometry required for every planned row')
    return value

def cosine_value(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or (not a) or (len(a) != len(b)):
        raise ValueError('Matched nonempty CLIP feature vectors required')
    if not all((finite(v) for v in list(a) + list(b))):
        raise ValueError('Nonfinite CLIP feature')
    aa = math.sqrt(sum((v * v for v in a)))
    bb = math.sqrt(sum((v * v for v in b)))
    if not aa or not bb:
        raise ValueError('Zero CLIP feature')
    cosine = max(-1.0, min(1.0, sum((x * y for x, y in zip(a, b))) / (aa * bb)))
    return cosine

def weighted(rows, field):
    active = [r for r in rows if r['weight'] > 0]
    if not active or any((r.get(field) is None for r in active)):
        return None
    return sum((r[field] * r['weight'] for r in active)) / sum((r['weight'] for r in active))

def evaluate_region_appearance(entities, reference, prediction, indices):
    """Mean across fixed GT-available times; absent prediction contributes zero.

    GT-unavailable frames are explicitly outside the frozen measurable population.
    Required entities with no reference region keep the combined result null.
    """
    validate_entities(entities)
    validate_schedule(indices)
    per = {}
    for e in entities:
        eid = e['id']
        values = []
        rows = []
        missing = 0
        for i in indices:
            rr = reference.get(eid, {}).get(str(i), {})
            pp = prediction.get(eid, {}).get(str(i), {})
            rs = state(rr)
            ps = state(pp)
            if rs == 'absent':
                rows.append({'sample_index': i, 'status': 'reference_unavailable', 'score': None})
                continue
            signed = None
            if ps == 'absent':
                score = 0.0
                missing += 1
                status = 'prediction_region_missing'
            else:
                signed = cosine_value(rr.get('feature'), pp.get('feature'))
                score = max(0.0, signed)
                status = 'region_appearance_measured'
            values.append(score)
            rows.append({'sample_index': i, 'status': status, 'score': score, 'raw_signed_cosine': signed})
        n = len(indices)
        m = len(values)
        per[eid] = {'type': e['type'], 'weight': 1.0 if e['type'] == 'hand' else e.get('weight', 1.0), 'score': sum(values) / m if m else None, 'planned_frames': n, 'reference_available_frames': m, 'reference_unavailable_frames': n - m, 'reference_coverage': m / n, 'prediction_missing_frames': missing, 'observed_region_pairs': m - missing, 'prediction_region_coverage': (m - missing) / m if m else None, 'frames': rows}
    return aggregate_region_appearance(per)


def aggregate_region_appearance(per):
    """Apply fixed entity weights and equal hand/object group weights."""
    groups = {g: {'applicable': any((r['type'] == g and r['weight'] > 0 for r in per.values())), 'score': weighted([r for r in per.values() if r['type'] == g], 'score'), 'reference_coverage': weighted([r for r in per.values() if r['type'] == g], 'reference_coverage')} for g in ('hand', 'object')}
    active = [g for g in groups.values() if g['applicable']]
    scores = [g['score'] for g in active]
    score = sum(scores) / len(scores) if scores and all((v is not None for v in scores)) else None
    coverage = sum((g['reference_coverage'] for g in active)) / len(active) if active and all((g['reference_coverage'] is not None for g in active)) else None
    active_entities=[r for r in per.values() if r['weight']>0]
    counts={'expected_units':sum(r.get('reference_available_frames',0) for r in active_entities),'observed_units':sum(r.get('observed_region_pairs',0) for r in active_entities),'penalized_units':sum(r.get('prediction_missing_frames',0) for r in active_entities),'reference_unavailable_units':sum(r.get('reference_unavailable_frames',0) for r in active_entities)}
    return {'metric': 'entity_consistency', 'protocol': APPEARANCE_PROTOCOL, 'variant': APPEARANCE_PROTOCOL, 'score': score, 'score_name': 'entity_region_appearance', 'status': 'missing_reference_entity_region' if score is None else 'partial_reference_population' if coverage < 1 else 'ok', 'range': [0.0, 1.0], 'direction': 'higher', 'reference_coverage': coverage, 'reference_availability': coverage, 'per_entity': per, 'groups': groups, 'identity_checked': False, 'physical_identity_claim': False, 'definition': 'max(0, CLIP cosine) per entity region; missing prediction=0; fixed GT-available frames; equal declared applicable hand/object groups.', 'reference_unavailable_normalization': 'Report separately; never substitute zero or drop a required entity from aggregation.',**counts}

def evaluate_geometric_trajectory(entities, reference, prediction, indices, frames, threshold=0.1, frame_diagonal_floor=0.005, max_gap_seconds=None):
    """Score all predeclared adjacent mask-centroid segments, no VLM gates.

    Missing prediction mask fails an otherwise GT-available segment. Missing GT
    endpoint makes that planned segment unavailable. Never bridge missing points.
    Coordinates must already share the explicitly prepared reference image grid.
    `frames` is the evaluated-video schedule AFTER validate_time_pairing, not
    the GT source clock. Explicit repeated GT source frames retain every input
    checkpoint; this displacement-only metric never divides by source time.
    """
    validate_entities(entities)
    validate_schedule(indices)
    if not finite(threshold) or threshold <= 0 or (not finite(frame_diagonal_floor)) or (not 0 < frame_diagonal_floor <= 1):
        raise ValueError('Fixed positive trajectory thresholds required')
    if len(indices) < 2:
        raise ValueError('At least two planned frames required')
    if max_gap_seconds is not None and (not finite(max_gap_seconds) or max_gap_seconds <= 0):
        raise ValueError('Invalid fixed max gap')
    byframe = {f.get('input_sample_index', f['sample_index']): f for f in frames}
    if any((i not in byframe for i in indices)):
        raise ValueError('Frame schedule incomplete')
    times = [byframe[i]['timestamp_seconds'] for i in indices]
    if any((not finite(t) for t in times)) or any((b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Increasing paired timestamps required')
    per = {}
    excluded = []
    for e in entities:
        if e['type'] != 'object':
            continue
        if e.get('weight', 1.0) <= 0 or e.get('trajectory_applicable') is False:
            excluded.append({'id': e['id'], 'reason': 'fixed_gt_not_applicable' if e.get('trajectory_applicable') is False else 'fixed_zero_weight'})
            continue
        eid = e['id']
        ref = reference.get(eid, {})
        pred = prediction.get(eid, {})
        for i in indices:
            rr = ref.get(str(i), {})
            pp = pred.get(str(i), {})
            state(rr)
            state(pp)
            f = byframe[i]
            for row in (rr, pp):
                if (row.get('width'), row.get('height')) != (f['width'], f['height']):
                    raise ValueError('Common prepared coordinate grid required; no guessed resize')
                if state(row) == 'valid':
                    p = row.get('center_xy')
                    if not isinstance(p, list) or len(p) != 2 or (not all((finite(v) for v in p))) or (not (0 <= p[0] < f['width'] and 0 <= p[1] < f['height'])):
                        raise ValueError('Invalid validated mask centroid')
        first = next((ref[str(i)] for i in indices if state(ref[str(i)]) == 'valid'), None)
        scale = None
        if first:
            box = first['bbox_xyxy']
            scale = math.hypot(box[2] - box[0], box[3] - box[1])
            if scale <= 0:
                raise ValueError('Nonpositive reference bbox scale')
        points = []
        point_errors = []
        for i in indices:
            rr = ref[str(i)]
            pp = pred[str(i)]
            error = None
            if state(rr) == 'absent':
                point_status = 'reference_unavailable'
            elif state(pp) == 'absent':
                point_status = 'prediction_region_missing'
            else:
                point_status = 'geometry_measured'
                error = math.dist(rr['center_xy'], pp['center_xy']) / scale
                point_errors.append(error)
            points.append({'sample_index': i, 'status': point_status, 'normalized_error': error})
        rows = []
        successes = eligible = missing = unavailable = 0
        for a, b in zip(indices, indices[1:]):
            ra, rb = (ref[str(a)], ref[str(b)])
            pa, pb = (pred[str(a)], pred[str(b)])
            row = {'start_sample_index': a, 'end_sample_index': b, 'success': None}
            if state(ra) == 'absent' or state(rb) == 'absent':
                row['status'] = 'reference_unavailable'
                unavailable += 1
            elif max_gap_seconds is not None and byframe[b]['timestamp_seconds'] - byframe[a]['timestamp_seconds'] > max_gap_seconds:
                row['status'] = 'fixed_time_gap_unavailable'
                unavailable += 1
            else:
                eligible += 1
                if state(pa) == 'absent' or state(pb) == 'absent':
                    row.update(status='prediction_region_missing', success=False)
                    missing += 1
                else:
                    tol = max(scale * threshold, math.hypot(byframe[a]['width'], byframe[a]['height']) * frame_diagonal_floor)
                    ea = math.dist(ra['center_xy'], pa['center_xy'])
                    eb = math.dist(rb['center_xy'], pb['center_xy'])
                    rd = [rb['center_xy'][j] - ra['center_xy'][j] for j in (0, 1)]
                    pd = [pb['center_xy'][j] - pa['center_xy'][j] for j in (0, 1)]
                    em = math.dist(rd, pd)
                    ok = ea <= tol and eb <= tol and (em <= tol)
                    row.update(status='geometry_measured', success=ok, start_error_pixels=ea, end_error_pixels=eb, displacement_error_pixels=em, tolerance_pixels=tol)
                    successes += ok
            rows.append(row)
        planned = len(indices) - 1
        per[eid] = {'weight': e.get('weight', 1.0), 'score': successes / eligible if eligible else None, 'planned_segments': planned, 'reference_available_segments': eligible, 'reference_unavailable_segments': unavailable, 'reference_coverage': eligible / planned, 'prediction_missing_segments': missing, 'geometrically_measured_segments': eligible - missing, 'successful_segments': successes, 'reference_initial_bbox_diagonal': scale, 'segments': rows, 'ADE': sum(point_errors) / len(point_errors) if point_errors else None, 'FDE': points[-1]['normalized_error'], 'points': points, 'ADE_note': 'Conditional diagnostic in GT-initial-bbox-diagonal units; missing prediction is penalized by primary segment score, never inserted as zero error.'}
    score = weighted(list(per.values()), 'score')
    coverage = weighted(list(per.values()), 'reference_coverage')
    return {'metric': 'trajectory_accuracy', 'protocol': TRAJECTORY_PROTOCOL, 'variant': TRAJECTORY_PROTOCOL, 'score': score, 'score_name': 'mask_centroid_ordered_geometry', 'status': 'not_applicable' if not per else 'missing_reference_object_geometry' if score is None else 'partial_reference_population' if coverage < 1 else 'ok', 'direction': 'higher', 'range': [0.0, 1.0], 'reference_coverage': coverage, 'reference_availability': coverage, 'per_object': per, 'identity_checked': False, 'low_occlusion_checked': False, 'required_object_ids': list(per), 'excluded_object_ids': excluded, 'center_type': 'visible_2d_mask_centroid_not_physical_object_center', 'planned_sample_indices': list(indices), 'normalized_threshold': threshold, 'frame_diagonal_floor': frame_diagonal_floor, 'max_gap_seconds': max_gap_seconds, 'definition': 'Correct positions AND displacement / fixed GT-available adjacent segments; missing prediction=failed segment.', 'time_basis': 'evaluated_video_after_verified_pairing', 'reference_unavailable_normalization': 'Retain planned missing segments and coverage; no bridging or selective required-object averaging.'}
