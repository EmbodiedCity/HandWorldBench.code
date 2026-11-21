"""Conservative geometry/pose screening and source-aware clip selection.

Frame intervals are half-open. This stage proposes candidates; projected joints
are not evidence of actual visibility and every proposal still requires RGB and
human review. Camera thresholds are hard gates, never compensated by hand motion.
"""

from collections import defaultdict, deque
import math
import re

import numpy as np


def _project_hands(ep):
    from .data import project_hands
    return project_hands(ep)


def _quat_angles(q, reference):
    """Shortest SO(3) angle in degrees; q and -q represent the same pose."""
    dot = np.abs(np.sum(q * reference, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _rotation_matrices(q):
    w, x, y, z = q.T
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ], axis=-1).reshape(-1, 3, 3)


def _participants(ep, complete, cfg, visible=None):
    """Never drop a still supporting hand or infer involvement from speed.

    The default is both hands. An optional narrow single-hand inference requires
    explicit text saying ONLY the named hand, absence of the other hand in at
    least 98% of frames, and no opposite-hand or bimanual language. It remains
    an inference requiring visual review, not an action annotation.
    """
    texts = [str(ep.get("task", ""))]
    texts.extend(str(x) for x in ep.get("subtasks", []) if x is not None)
    text = " ".join(dict.fromkeys(texts)).lower()
    left = bool(re.search(r"\bleft\s+hand\b", text))
    right = bool(re.search(r"\bright\s+hand\b", text))
    both = bool(re.search(r"\bboth\s+hands\b|\btwo[- ]hand|\bbimanual\b", text))
    explicit = [bool(re.search(r"(?:only\s+(?:the\s+)?left\s+hand|left\s+hand\s+only)", text)),
                bool(re.search(r"(?:only\s+(?:the\s+)?right\s+hand|right\s+hand\s+only)", text))]
    complete_fraction = np.mean(complete, axis=0).tolist()
    appearance = np.mean(complete if visible is None else visible, axis=0).tolist()
    hands = [0, 1]
    reason = "both_hands_conservative_no_reliable_exclusive_evidence"
    overrides = cfg.get("participant_overrides", {})
    override = overrides.get(ep.get("episode_id"))
    if override is not None:
        if (not isinstance(override, (list, tuple)) or not override or
                any(isinstance(h, bool) or not isinstance(h, int) or h not in (0, 1) for h in override)):
            raise ValueError("participant_overrides values must be nonempty lists containing 0 and/or 1")
        hands = sorted(set(override))
        reason = "explicit_user_reviewed_episode_participant_override"
    elif cfg.get("allow_explicit_single_hand_inference", False) and not both:
        for hand in (0, 1):
            opposite_mentioned = right if hand == 0 else left
            if explicit[hand] and not opposite_mentioned and appearance[1-hand] <= .02:
                hands = [hand]
                reason = "explicit_only_one_hand_text_and_opposite_geometry_absent"
                break
    return hands, {
        "status": "inferred_requires_rgb_and_human_review",
        "rule": reason,
        "text_mentions": {"left": left, "right": right, "both": both},
        "episode_complete_hand_fraction": complete_fraction,
        "episode_projected_hand_appearance_fraction": appearance,
        "static_supporting_hands_are_participants": True,
        "skeleton_is_not_visual_visibility_ground_truth": True,
        "episode_id": ep.get("episode_id"),
    }


def _motion_features(world, intrinsic, palms, start, end, hands):
    # Compute each distance once per start and reuse it for every allowed length.
    joints = world[start:end][:, hands][:, :, [0, 4, 8, 12, 16, 20]]
    displacement = np.linalg.norm(joints-joints[:1], axis=-1)
    shape = intrinsic[start:end][:, hands]
    shape = np.abs(shape-shape[:1])
    return np.concatenate((displacement, shape), axis=2), palms[start:end][:, hands]


def _motion_metrics(features, palms, length, hands):
    """Camera-compensated motion, expressed in median palm lengths.

    World-space wrist and individual fingertip displacement capture manipulation;
    changes in wrist-to-joint distances also detect finger articulation.
    Taking a percentile over time avoids a single noisy joint creating activity.
    At least ONE participating hand must move; other hands may support or hold.
    """
    scales = np.median(palms[:length], axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 1e-6):
        return None
    # A single percentile call measures all fingers and both hands, reducing
    # overhead for thousands of overlapping start/length alternatives.
    motion95 = np.percentile(features[:length], 95, axis=0) / scales[:, None]
    per_hand = []
    for index, hand in enumerate(hands):
        item = {
            "hand": hand,
            "palm_length_m": float(scales[index]),
            "wrist_world_motion_p95_palm_lengths": float(motion95[index, 0]),
            "fingertip_world_motion_p95_palm_lengths": float(np.max(motion95[index, 1:6])),
            "finger_articulation_p95_palm_lengths": float(np.max(motion95[index, 6:])),
        }
        item["motion_palm_lengths"] = max(item[k] for k in (
            "wrist_world_motion_p95_palm_lengths", "fingertip_world_motion_p95_palm_lengths",
            "finger_articulation_p95_palm_lengths"))
        per_hand.append(item)
    return {"hand_motion": per_hand,
            "motion_palm_lengths": max(x["motion_palm_lengths"] for x in per_hand)}


def find_candidates(ep, cfg):
    """Return the longest valid 81*k proposal for EVERY valid starting frame.

    `allowed_n_frames` preserves all shorter lengths which passed the same
    metadata gates, for RGB rejection/reselection. RGB must recompute its own
    interval metrics if it shortens a proposal. `metrics_by_n_frames` has string
    length keys; `metrics` describes the longest interval. All thresholds are
    configurable.
    """
    selection = cfg.get("selection", {})
    camera = cfg.get("camera", {})
    hand_cfg = cfg.get("hand", {})
    multiple = int(selection.get("frame_multiple", 81))
    offset = int(selection.get("frame_offset", 0))
    if not 0 <= offset < multiple:
        raise ValueError("invalid frame_offset")
    minimum = int(selection.get("min_frames", multiple))
    maximum = selection.get("max_frames")
    exact = selection.get("exact_frames")
    if multiple <= 0 or minimum <= 0:
        raise ValueError("frame_multiple and min_frames must be positive")
    minimum = ((minimum-offset+multiple-1)//multiple)*multiple+offset
    if exact is not None:
        exact = int(exact)
        if exact < minimum or (exact-offset) % multiple:
            raise ValueError("exact_frames must be a positive frame_multiple multiple >= min_frames")
        minimum = exact
        maximum = exact
    if maximum is not None and int(maximum) < minimum:
        raise ValueError("max_frames is smaller than min_frames")
    n = int(ep["n"])
    failure_counts = defaultdict(int)
    diagnostics = {
        "source_frames": n,
        "minimum_frames": minimum,
        "scanned_starts": max(0, n-minimum+1),
        "starts_without_minimum_remaining_frames": min(n, minimum-1),
        "start_failure_counts": failure_counts,
        "accepted_starts": 0,
        "accepted_length_alternatives": 0,
    }
    ep["temporal_diagnostics"] = diagnostics
    if n < minimum:
        failure_counts["window_too_short"] = n
        return []
    p = np.asarray(ep["positions_m"], dtype=float)
    t = np.asarray(ep["vio_t"], dtype=float)
    q = np.asarray(ep["vio_q_wxyz"], dtype=float)
    ts = np.asarray(ep["times_ns"], dtype=np.int64)
    if p.shape != (n, 2, 21, 3) or t.shape != (n, 3) or q.shape != (n, 4) or ts.shape != (n,):
        raise ValueError("episode arrays do not match the declared frame count")
    uv, depth_valid = _project_hands(ep)
    uv = np.asarray(uv, dtype=float)
    depth_valid = np.asarray(depth_valid, dtype=bool)
    if uv.shape != (n, 2, 21, 2) or depth_valid.shape != (n, 2, 21):
        raise ValueError("project_hands returned unexpected shapes")
    width, height = int(ep["width"]), int(ep["height"])
    border = float(hand_cfg.get("border_fraction", .01))
    min_span = float(hand_cfg.get("min_span_width_fraction", .04))
    max_invalid = int(hand_cfg.get("max_invalid_frames", 0))
    if not 0 <= border < .5 or min_span < 0 or max_invalid < 0:
        raise ValueError("invalid hand gate configuration")
    hand_valid = np.asarray(ep["hand_valid"], dtype=bool)
    finite_hand = np.isfinite(p).all(axis=(2, 3)) & np.isfinite(uv).all(axis=(2, 3))
    inside = ((uv[..., 0] >= width*border) & (uv[..., 0] <= width*(1-border)-1) &
              (uv[..., 1] >= height*border) & (uv[..., 1] <= height*(1-border)-1))
    span = (np.max(uv[..., 0], axis=2) - np.min(uv[..., 0], axis=2)) / width
    complete = hand_valid & finite_hand & np.all(depth_valid & inside, axis=2) & (span >= min_span)
    visible_joints = (np.isfinite(uv).all(axis=-1) & depth_valid &
                      (uv[..., 0] >= 0) & (uv[..., 0] < width) &
                      (uv[..., 1] >= 0) & (uv[..., 1] < height))
    visible = hand_valid & np.any(visible_joints, axis=2)
    diagnostics["hand_valid_fraction"] = np.mean(hand_valid & finite_hand, axis=0).tolist()
    diagnostics["hand_complete_fraction"] = np.mean(complete, axis=0).tolist()
    hands, inference = _participants(ep, complete, hand_cfg, visible)
    # NaNs, invalid skeletons, or points behind the camera cannot be forgiven by
    # max_invalid_frames; that setting only tolerates brief geometric margins.
    hand_hard_ok = np.all(hand_valid[:, hands] & finite_hand[:, hands] & np.all(depth_valid[:, hands], axis=2), axis=1)
    complete_frame = np.all(complete[:, hands], axis=1)
    qnorm = np.linalg.norm(q, axis=1)
    pose_ok = (np.asarray(ep["pose_valid"], dtype=bool) & np.isfinite(t).all(axis=1) &
               np.isfinite(q).all(axis=1) & np.isfinite(qnorm) & (np.abs(qnorm-1) <= .01))
    q = q / np.where(qnorm > 0, qnorm, np.nan)[:, None]
    dt = np.diff(ts).astype(float) / 1e9
    velocity = _quat_angles(q[1:], q[:-1]) / np.where(dt > 0, dt, np.nan)
    velocity_ok = np.isfinite(velocity) & (dt > 0)
    translation_velocity = np.linalg.norm(np.diff(t, axis=0), axis=1) / np.where(dt > 0, dt, np.nan)
    translation_velocity_ok = np.isfinite(translation_velocity) & (dt > 0)
    rotation = _rotation_matrices(q)
    world = np.einsum("nij,nhkj->nhki", rotation, p) + t[:, None, None, :]
    palm_lengths = np.linalg.norm(p[:, :, 9]-p[:, :, 0], axis=-1)
    intrinsic_lengths = np.linalg.norm(p[:, :, 1:]-p[:, :, :1], axis=-1)
    max_translation = float(camera.get("anchor_translation_m", .03))
    max_rotation = float(camera.get("anchor_rotation_deg", 2.0))
    p95_velocity = float(camera.get("angular_velocity_p95_deg_s", 8.0))
    max_velocity = float(camera.get("angular_velocity_max_deg_s", 15.0))
    # Conservative engineering defaults, pending real-video calibration. The
    # anchor envelope alone cannot detect a quick move and return within 3 cm.
    p95_translation_velocity = float(camera.get("translation_velocity_p95_m_s", .10))
    max_translation_velocity = float(camera.get("translation_velocity_max_m_s", .20))
    if (not np.isfinite(p95_translation_velocity) or not np.isfinite(max_translation_velocity)
            or min(p95_translation_velocity, max_translation_velocity) < 0):
        raise ValueError("translation velocity thresholds must be finite and nonnegative")
    min_motion = float(hand_cfg.get("min_motion_palm_lengths", .08))
    if min(max_translation, max_rotation, p95_velocity, max_velocity, min_motion) < 0:
        raise ValueError("camera and activity thresholds cannot be negative")
    proposals = []
    for start in range(n-minimum+1):
        # A first frame with a border/size failure is never rescued by a relaxed
        # invalid-frame allowance: it is the benchmark conditioning image.
        if not pose_ok[start]:
            failure_counts["first_frame_pose"] += 1
            continue
        if not hand_hard_ok[start] or not complete_frame[start]:
            failure_counts["first_frame_hand"] += 1
            continue
        limit = n if maximum is None else min(n, start+int(maximum))
        translation = np.linalg.norm(t[start:limit]-t[start], axis=1)
        angle = _quat_angles(q[start:limit], q[start])
        good = pose_ok[start:limit] & hand_hard_ok[start:limit]
        good &= np.isfinite(translation) & (translation <= max_translation+1e-12)
        good &= np.isfinite(angle) & (angle <= max_rotation+1e-10)
        if limit-start > 1:
            good[1:] &= velocity_ok[start:limit-1] & (velocity[start:limit-1] <= max_velocity+1e-10)
            good[1:] &= (translation_velocity_ok[start:limit-1] &
                         (translation_velocity[start:limit-1] <= max_translation_velocity+1e-12))
        good &= np.cumsum(~complete_frame[start:limit]) <= max_invalid
        bad = np.flatnonzero(~good)
        length_limit = int(bad[0]) if len(bad) else limit-start
        length_limit = ((length_limit-offset) // multiple) * multiple+offset
        if length_limit < minimum:
            reason = "window_too_short"
            if len(bad):
                at = start+int(bad[0])
                if not pose_ok[at]:
                    reason = "pose_missing_or_invalid"
                elif not hand_hard_ok[at]:
                    reason = "hand_missing_or_invalid"
                elif np.count_nonzero(~complete_frame[start:at+1]) > max_invalid:
                    reason = "hand_out_of_frame_or_small"
                elif translation[int(bad[0])] > max_translation or angle[int(bad[0])] > max_rotation:
                    reason = "camera_anchor_envelope"
                elif (at > start and (not translation_velocity_ok[at-1] or
                                      translation_velocity[at-1] > max_translation_velocity+1e-12)):
                    reason = "camera_translation_velocity_max"
                else:
                    reason = "camera_angular_velocity_max"
            failure_counts[reason] += 1
            continue
        allowed = []
        metrics_by_length = {}
        reached_angular_gate = False
        reached_motion_gate = False
        motion_features, motion_palms = _motion_features(
            world, intrinsic_lengths, palm_lengths, start, start+length_limit, hands)
        for length in range(length_limit, minimum-1, -multiple):
            end = start+length
            speeds = velocity[start:end-1]
            angular_p95 = float(np.percentile(speeds, 95)) if len(speeds) else 0.0
            if angular_p95 > p95_velocity+1e-10:
                continue
            reached_angular_gate = True
            translation_speeds = translation_velocity[start:end-1]
            translation_p95 = float(np.percentile(translation_speeds, 95)) if len(translation_speeds) else 0.0
            # Percentiles are not monotonic in window length: a burst can be
            # diluted by later still frames, so every proposed length is checked.
            if not np.isfinite(translation_p95) or translation_p95 > p95_translation_velocity+1e-12:
                continue
            reached_motion_gate = True
            motion = _motion_metrics(motion_features, motion_palms, length, hands)
            if motion is None or not np.isfinite(motion["motion_palm_lengths"]) or motion["motion_palm_lengths"] < min_motion:
                continue
            allowed.append(length)
            metrics_by_length[str(length)] = {
                "anchor_translation_max_m": float(np.max(translation[:length])),
                "anchor_rotation_max_deg": float(np.max(angle[:length])),
                "angular_velocity_p95_deg_s": angular_p95,
                "angular_velocity_max_deg_s": float(np.max(speeds)) if len(speeds) else 0.0,
                "translation_velocity_p95_m_s": translation_p95,
                "translation_velocity_max_m_s": float(np.max(translation_speeds)) if len(translation_speeds) else 0.0,
                "hand_invalid_frames": int(np.count_nonzero(~complete_frame[start:end])),
                "min_hand_span_width_fraction": float(np.min(span[start:end, hands])),
                **motion,
            }
        if allowed:
            proposals.append({
                "start_frame": start, "end_frame": start+allowed[0], "n_frames": allowed[0],
                "allowed_n_frames": allowed,
                "participating_hands": list(hands), "metrics": metrics_by_length[str(allowed[0])],
                "metrics_by_n_frames": metrics_by_length,
                "participant_inference": dict(inference), "qc_status": "pending",
            })
            diagnostics["accepted_starts"] += 1
            diagnostics["accepted_length_alternatives"] += len(allowed)
        else:
            if reached_motion_gate:
                reason = "insufficient_hand_motion"
            elif reached_angular_gate:
                reason = "camera_translation_velocity_p95"
            else:
                reason = "camera_angular_velocity_p95"
            failure_counts[reason] += 1
    diagnostics["start_failure_counts"] = dict(failure_counts)
    return proposals


def _fingerprints(candidate):
    """All three interval-specific samples are required for visual dedup."""
    values = []
    for position in ("first", "middle", "last"):
        value = candidate.get(f"source_{position}_frame_fingerprint")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{16}", value):
            return None
        values.append(int(value, 16))
    return tuple(values)


def select_diverse(candidates, cfg, reserved=None):
    """Task/recording round-robin with length priority and physical-frame fencing.

    Uncertain recording identities/offsets are rejected rather than using MCAP
    filenames as a substitute for a known original recording. Overlap is checked
    across different MCAPs in the SAME original recording coordinates.
    """
    cfg = cfg.get("selection", {})
    target = int(cfg.get("target_count", 500))
    cap = int(cfg.get("max_per_recording", 3))
    multiple = int(cfg.get("frame_multiple", 81))
    length_offset = int(cfg.get("frame_offset", 0))
    minimum = int(cfg.get("min_frames", multiple))
    exact = cfg.get("exact_frames")
    maximum = cfg.get("max_frames")
    hamming = int(cfg.get("near_duplicate_hamming", 4))
    quotas = {int(k): int(v) for k, v in cfg.get('frame_count_quotas', {}).items()}
    baseline = cfg.get('external_exclusions', [])
    excluded_recordings = {str(c['recording_id']) for c in baseline}
    excluded_sources = {str(c['source_path']) for c in baseline}
    if target < 0 or cap <= 0 or multiple <= 0:
        raise ValueError("invalid selection configuration")
    if not -1 <= hamming <= 64:
        raise ValueError("near_duplicate_hamming must be -1 (disabled) or 0..64")
    groups = defaultdict(lambda: defaultdict(list))
    source_identities = defaultdict(set)
    recording_origins = defaultdict(list)
    validated = []
    for c in candidates:
        start, end, offset = c.get("start_frame"), c.get("end_frame"), c.get("source_frame_offset")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (start, end, offset)):
            continue
        rec = c.get("recording_id")
        source = c.get("source_path")
        if not c.get("recording_confident") or not rec or not source or offset < 0 or start < 0 or end <= start:
            continue
        if str(rec) in excluded_recordings or str(source) in excluded_sources:
            continue
        length = end-start
        if length != c.get("n_frames") or length < minimum or (length-length_offset) % multiple:
            continue
        if (exact is not None and length != int(exact)) or (maximum is not None and length > int(maximum)):
            continue
        if quotas and length not in quotas:
            continue
        source_identities[str(source)].add((str(rec), int(offset)))
        origin = c.get("recording_time_origin_ns")
        if isinstance(origin, (int, np.integer)) and not isinstance(origin, bool):
            fps = c.get("fps", 30.)
            if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
                fps = 30.
            recording_origins[str(rec)].append((int(origin), 1e9/float(fps)))
        validated.append(c)
    origin_conflicts = {
        rec for rec, values in recording_origins.items()
        if max(x[0] for x in values)-min(x[0] for x in values) > min(x[1] for x in values)
    }
    for c in validated:
        if str(c["recording_id"]) in origin_conflicts:
            c["selection_rejection_reason"] = "recording_time_origin_conflict"
            continue
        if len(source_identities[str(c["source_path"])]) != 1:
            c["selection_rejection_reason"] = "source_recording_identity_conflict"
            continue
        groups[str(c.get("task") or "unknown_task")][str(c["recording_id"])].append(c)
    for by_recording in groups.values():
        for items in by_recording.values():
            items.sort(key=lambda c: (-c["n_frames"],
                c.get("metrics", {}).get("anchor_rotation_max_deg", float("inf")),
                c["source_frame_offset"]+c["start_frame"], str(c["source_path"])))
    queues = {task: deque(sorted(by_recording, key=lambda rec: (-by_recording[rec][0]["n_frames"], rec)))
              for task, by_recording in groups.items()}
    tasks = deque(sorted(groups))
    used_intervals = defaultdict(list)
    source_intervals = defaultdict(list)
    counts = defaultdict(int)
    fingerprints = []
    for old in baseline:
        fingerprint = _fingerprints(old)
        if fingerprint is None:
            raise ValueError('external baseline fingerprint missing; refusing silent dedup bypass')
        fingerprints.append(fingerprint)
    length_counts = defaultdict(int)
    selected = []
    if reserved:
        safe_keys = {(str(c['source_path']), c['start_frame'], c['end_frame'])
                     for by_recording in groups.values() for items in by_recording.values() for c in items}
        safe_reserved = [c for c in reserved if (str(c['source_path']),c['start_frame'],c['end_frame']) in safe_keys]
        selected = select_diverse(safe_reserved, {'selection': cfg})
        for c in selected:
            rec = str(c['recording_id'])
            source = str(c['source_path'])
            used_intervals[rec].append((c['source_frame_offset']+c['start_frame'],c['source_frame_offset']+c['end_frame']))
            source_intervals[source].append((c['start_frame'],c['end_frame']))
            counts[rec] += 1
            length_counts[c['n_frames']] += 1
            fp = _fingerprints(c)
            if fp is not None:
                fingerprints.append(fp)
    # Fill scarce long bins first. All bins share physical/visual exclusion
    # state; shortening one source four ways can never fill four quotas.
    for length in sorted(quotas, reverse=True) if quotas else [None]:
        active = {task: {rec: [c for c in items if length is None or c['n_frames'] == length]
                        for rec, items in by_recording.items()}
                  for task, by_recording in groups.items()}
        queues = {task: deque(rec for rec in sorted(by_recording,
                      key=lambda rec: (-by_recording[rec][0]['n_frames'], rec)) if by_recording[rec])
                  for task, by_recording in
                  ((t, {r: cs for r, cs in records.items() if cs}) for t, records in active.items())}
        tasks = deque(sorted(task for task, queue in queues.items() if queue))
        while tasks and len(selected) < target and (length is None or length_counts[length] < quotas[length]):
            task = tasks.popleft()
            rec_queue = queues[task]
            picked = None
            for _ in range(len(rec_queue)):
                rec = rec_queue.popleft()
                items = active[task][rec]
                while items and counts[rec] < cap:
                    c = items.pop(0)
                    begin = c['source_frame_offset']+c['start_frame']
                    end = c['source_frame_offset']+c['end_frame']
                    source = str(c['source_path'])
                    fingerprint = _fingerprints(c)
                    visually_duplicate = fingerprint is not None and hamming >= 0 and any(
                        all((a ^ b).bit_count() <= hamming for a, b in zip(fingerprint, seen))
                        for seen in fingerprints)
                    overlap = any(begin < b and a < end for a, b in used_intervals[rec])
                    overlap |= any(c['start_frame'] < b and a < c['end_frame'] for a, b in source_intervals[source])
                    if overlap or visually_duplicate:
                        continue
                    picked = c
                    used_intervals[rec].append((begin, end))
                    source_intervals[source].append((c['start_frame'], c['end_frame']))
                    counts[rec] += 1
                    length_counts[c['n_frames']] += 1
                    if fingerprint is not None:
                        fingerprints.append(fingerprint)
                    break
                if items and counts[rec] < cap:
                    rec_queue.append(rec)
                if picked is not None:
                    selected.append(picked)
                    break
            if rec_queue:
                tasks.append(task)
    return selected
