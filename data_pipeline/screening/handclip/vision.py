"""Streaming RGB gates. Machine approval never means a human visual review.

Thresholds are conservative engineering starting values, not a calibrated claim.
Only grayscale analysis images enter a bounded, temporary, on-disk cache. The
source RGB frames are never geometrically modified. Every retained frame is
tested, including direct registration against the candidate's actual first frame.
"""
from __future__ import annotations

import collections
import hashlib
import math
import os
import tempfile
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np


DEFAULTS = {
    "analysis_width": 640,
    "max_anchor_displacement_diag": 0.02,
    "max_pair_displacement_diag": 0.004,
    "min_features": 40,
    "min_inliers": 24,
    "max_features": 600,
    "min_inlier_ratio": 0.65,
    "min_grid_cells": 6,
    "grid_columns": 4,
    "grid_rows": 3,
    "min_feature_span_fraction": 0.5,
    "max_fb_error_px": 1.0,
    "max_reprojection_error_px": 1.5,
    "max_photometric_error": 25.0,
    "max_projection_roundtrip_error_px": 0.25,
    "min_grid_points": 16,
    "min_global_laplacian": 60.0,
    "min_hand_laplacian": 45.0,
    "min_first_global_laplacian": 100.0,
    "min_first_hand_laplacian": 80.0,
    "roi_size": 128,
    "min_luma": 35.0,
    "max_luma": 225.0,
    "max_clipped_fraction": 0.20,
    "mask_expand_fraction": 0.35,
    "max_cache_mb": 512,
}


def _laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _fingerprint(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = np.packbits((small[:, 1:] > small[:, :-1]).reshape(-1))
    return bits.tobytes().hex()


def _mask(gray_shape, uv, valid, settings):
    """Exclude both hands and the area where a held/operated object may be.

    A convex hull between hands deliberately excludes static supporting hands
    and inter-hand objects, even when just one hand has high motion activity.
    Remaining object ambiguity still requires a separate visual review.
    """
    h, w = gray_shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    hands = []
    for hand in range(2):
        points = uv[hand]
        good = valid[hand] & np.isfinite(points).all(axis=1)
        if np.count_nonzero(good) < 3:
            continue
        points = points[good]
        # Bounded clipping avoids integer overflow on invalid lens projections.
        points = np.clip(points, [-w, -h], [2 * w, 2 * h])
        hands.append(points)
    if hands:
        points = np.concatenate(hands).astype(np.float32)
        hull = cv2.convexHull(points).astype(np.int32)
        exclusion = np.zeros_like(mask)
        cv2.fillConvexPoly(exclusion, hull, 255)
        spans = [float(np.ptp(p, axis=0).max()) for p in hands]
        radius = max(8, int(math.ceil(max(spans) * settings["mask_expand_fraction"])))
        radius = min(radius, max(h, w))
        exclusion = cv2.dilate(exclusion, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        mask[exclusion > 0] = 0
    mask[:4] = mask[-4:] = 0
    mask[:, :4] = mask[:, -4:] = 0
    return mask


def _camera(ep, width, height):
    k = np.asarray(ep["K"], dtype=np.float64).copy()
    k[0] *= width / int(ep["width"])
    k[1] *= height / int(ep["height"])
    d = np.asarray(ep["D"], dtype=np.float64).reshape(4, 1)
    model = ep.get("camera_model", ep.get("metadata", {}).get("camera_model", "equidistant"))
    return k, d, model


def _undistort(points, camera):
    k, d, model = camera
    if model == "pinhole":
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return cv2.fisheye.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), k, d, P=k
    ).reshape(-1, 2)


def _distort(points, camera):
    k, d, model = camera
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if model == "pinhole":
        return points
    normal = np.column_stack(((points[:, 0] - k[0, 2]) / k[0, 0],
                              (points[:, 1] - k[1, 2]) / k[1, 1]))
    return cv2.fisheye.distortPoints(normal.reshape(-1, 1, 2), k, d).reshape(-1, 2)


def _projection_valid(points, camera, settings):
    """OpenCV's finite undistortion output is not proof of a valid camera ray.

    Near/outside the fisheye horizon it may return a large finite sentinel or a
    non-invertible solution. Require a pixel-space round trip; points that fail
    are outside the calibrated, front-facing region used by this pipeline.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ideal = _undistort(points, camera)
        reconstructed = _distort(ideal, camera)
        error = np.linalg.norm(reconstructed - points, axis=1)
    valid = np.isfinite(ideal).all(axis=1) & np.isfinite(error)
    valid &= error <= settings["max_projection_roundtrip_error_px"]
    return ideal, valid


def _inside_hull(points, support):
    hull = cv2.convexHull(np.asarray(support, dtype=np.float32))
    return np.array([cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0
                     for p in points], dtype=bool)


def _coverage(points, shape, settings):
    h, w = shape
    columns, rows = settings["grid_columns"], settings["grid_rows"]
    xx = np.clip((points[:, 0] * columns / w).astype(int), 0, columns - 1)
    yy = np.clip((points[:, 1] * rows / h).astype(int), 0, rows - 1)
    cells = len(set((yy * columns + xx).tolist()))
    span = np.ptp(points, axis=0) / np.array([w, h])
    return cells, float(span.min())


def _features(gray, mask, settings):
    return cv2.goodFeaturesToTrack(gray, maxCorners=int(settings["max_features"]),
                                   qualityLevel=0.01, minDistance=6,
                                   mask=mask, blockSize=7)


def _register(reference, current, reference_mask, current_mask, camera,
              settings, points=None):
    """LK + bidirectional check + robust, spatially supported fisheye mapping.

    Returns unknown on any confidence failure. A low-texture frame is not a
    stationary frame. Homography/grid motion is measured in original image
    coordinates after fitting in calibrated, undistorted feature coordinates.
    """
    def unknown(reason, **extra):
        return {"status": "unknown", "reason": reason, **extra}

    if points is None:
        points = _features(reference, reference_mask, settings)
    if points is None or len(points) < settings["min_features"]:
        return unknown("insufficient_background_features")
    try:
        _, valid_reference = _projection_valid(points, camera, settings)
        points = points[valid_reference]
        if len(points) < settings["min_features"]:
            return unknown("insufficient_features_in_calibrated_fov")
        destination, forward, _ = cv2.calcOpticalFlowPyrLK(
            reference, current, points, None, winSize=(21, 21), maxLevel=int(settings.get("lk_max_level",4)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if destination is None:
            return unknown("lk_failed")
        backward_points, backward, _ = cv2.calcOpticalFlowPyrLK(
            current, reference, destination, None, winSize=(21, 21), maxLevel=int(settings.get("lk_max_level",4)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if backward_points is None:
            return unknown("backward_lk_failed")
        src, dst = points.reshape(-1, 2), destination.reshape(-1, 2)
        fb = np.linalg.norm(backward_points.reshape(-1, 2) - src, axis=1)
        h, w = current.shape
        finite = np.isfinite(dst).all(axis=1)
        rounded = np.rint(np.where(np.isfinite(dst), dst, -1)).astype(np.int64)
        inside = finite & (rounded[:, 0] >= 0) & (rounded[:, 0] < w)
        inside &= (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
        unmasked = np.zeros(len(points), dtype=bool)
        unmasked[inside] = current_mask[rounded[inside, 1], rounded[inside, 0]] > 0
        good = ((forward.ravel() > 0) & (backward.ravel() > 0)
                & (fb <= settings["max_fb_error_px"]) & unmasked)
        if np.count_nonzero(good) < settings["min_inliers"]:
            return unknown("insufficient_consistent_tracks")
        src, dst = src[good], dst[good]
        src_u, valid_src = _projection_valid(src, camera, settings)
        dst_u, valid_dst = _projection_valid(dst, camera, settings)
        calibrated = valid_src & valid_dst
        src, dst = src[calibrated], dst[calibrated]
        src_u, dst_u = src_u[calibrated], dst_u[calibrated]
        if len(src) < settings["min_inliers"]:
            return unknown("insufficient_calibrated_correspondences")
        transform, inlier_mask = cv2.findHomography(
            src_u, dst_u, cv2.RANSAC, settings["max_reprojection_error_px"],
            maxIters=1000, confidence=0.995)
        if transform is None or inlier_mask is None or not np.isfinite(transform).all():
            return unknown("ransac_failed")
        inliers = inlier_mask.ravel().astype(bool)
        count = int(inliers.sum())
        # Denominator includes tracks lost through motion/occlusion, not merely
        # the surviving subset that can make an unreliable fit look perfect.
        ratio = count / len(points)
        if count < settings["min_inliers"] or ratio < settings["min_inlier_ratio"]:
            return unknown("insufficient_inlier_support", inliers=count, inlier_ratio=ratio)
        cells, span = _coverage(src[inliers], reference.shape, settings)
        dest_cells, dest_span = _coverage(dst[inliers], current.shape, settings)
        if (min(cells, dest_cells) < settings["min_grid_cells"] or
                min(span, dest_span) < settings["min_feature_span_fraction"]):
            return unknown("insufficient_spatial_coverage", occupied_cells=cells)
        predicted_u = cv2.perspectiveTransform(src_u.reshape(-1, 1, 2), transform).reshape(-1, 2)
        predicted = _distort(predicted_u, camera)
        residual = np.linalg.norm(predicted[inliers] - dst[inliers], axis=1)
        residual_p90 = float(np.percentile(residual, 90))
        if not np.isfinite(residual_p90) or residual_p90 > settings["max_reprojection_error_px"]:
            return unknown("large_registration_residual")
        # Photometric patch comparison catches many chance correspondences in
        # repeating textures; it is an additional gate, never a replacement.
        photometric = []
        for a, b in zip(src[inliers][::max(1, count // 80)], dst[inliers][::max(1, count // 80)]):
            pa = cv2.getRectSubPix(reference, (5, 5), tuple(float(x) for x in a)).astype(float)
            pb = cv2.getRectSubPix(current, (5, 5), tuple(float(x) for x in b)).astype(float)
            photometric.append(float(np.mean(np.abs(pa - pb))))
        photo = float(np.median(photometric))
        if not np.isfinite(photo) or photo > settings["max_photometric_error"]:
            return unknown("photometric_inconsistency")
        gx, gy = np.meshgrid(np.linspace(.1 * w, .9 * w, 9), np.linspace(.1 * h, .9 * h, 7))
        grid = np.column_stack((gx.ravel(), gy.ravel()))
        grid_u, valid_grid = _projection_valid(grid, camera, settings)
        grid_pixels = np.rint(grid).astype(int)
        supported = (valid_grid & _inside_hull(grid, src[inliers]) &
                     (reference_mask[grid_pixels[:, 1], grid_pixels[:, 0]] > 0))
        grid, grid_u = grid[supported], grid_u[supported]
        if len(grid) < settings["min_grid_points"]:
            return unknown("insufficient_reference_grid_support")
        warped_u = cv2.perspectiveTransform(grid_u.reshape(-1, 1, 2), transform).reshape(-1, 2)
        warped = _distort(warped_u, camera)
        if not np.isfinite(warped).all():
            return unknown("nonfinite_grid_projection")
        if ((warped < 0).any() or (warped[:, 0] >= w).any() or (warped[:, 1] >= h).any()):
            return unknown("grid_leaves_camera_field_of_view")
        _, valid_warped = _projection_valid(warped, camera, settings)
        warped_pixels = np.rint(warped).astype(int)
        warped_pixels = np.minimum(warped_pixels, [w - 1, h - 1])
        supported = (valid_warped & _inside_hull(warped, dst[inliers]) &
                     (current_mask[warped_pixels[:, 1], warped_pixels[:, 0]] > 0))
        grid, warped = grid[supported], warped[supported]
        if len(grid) < settings["min_grid_points"]:
            return unknown("insufficient_target_grid_support")
        grid_cells, grid_span = _coverage(grid, reference.shape, settings)
        target_grid_cells, target_grid_span = _coverage(warped, current.shape, settings)
        if (min(grid_cells, target_grid_cells) < settings["min_grid_cells"] or
                min(grid_span, target_grid_span) < settings["min_feature_span_fraction"]):
            return unknown("insufficient_grid_spatial_coverage")
        displacement = np.linalg.norm(warped - grid, axis=1) / math.hypot(w, h)
        if not np.isfinite(displacement).all():
            return unknown("nonfinite_grid_displacement")
        return {"status": "ok", "p90_displacement_diag": float(np.percentile(displacement, 90)),
                "max_displacement_diag": float(displacement.max()), "inliers": count,
                "inlier_ratio": ratio, "occupied_cells": min(cells, dest_cells),
                "supported_grid_points": len(grid),
                "supported_grid_cells": min(grid_cells, target_grid_cells),
                "reprojection_p90_px": residual_p90, "photometric_error": photo,
                "fb_p95_px": float(np.percentile(fb[good][calibrated][inliers], 95))}
    except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
        return unknown("registration_error", error=type(error).__name__)


def _quality(gray, uv, valid, settings):
    h, w = gray.shape
    mean = float(gray.mean())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    result = {"global_laplacian": _laplacian(gray), "luma": mean,
              "clipped_fraction": clipped, "hands": []}
    for hand in range(2):
        points = uv[hand]
        complete = bool(valid[hand].all() and np.isfinite(points).all())
        hand_result = {"valid": complete, "laplacian": 0.0, "luma": 0.0,
                       "clipped_fraction": 1.0}
        if complete:
            lo, hi = points.min(axis=0), points.max(axis=0)
            if (lo < 0).any() or hi[0] >= w or hi[1] >= h or (hi - lo).min() < 3:
                hand_result["valid"] = False
            else:
                # Little surrounding background, to avoid measuring only a
                # textured tabletop when the actual hand is motion-blurred.
                pad = np.maximum((hi - lo) * 0.08, 2)
                x0, y0 = np.maximum(np.floor(lo - pad), 0).astype(int)
                x1, y1 = np.minimum(np.ceil(hi + pad) + 1, [w, h]).astype(int)
                roi = gray[y0:y1, x0:x1]
                ratio = settings["roi_size"] / max(roi.shape)
                roi = cv2.resize(roi, (max(1, round(roi.shape[1] * ratio)),
                                       max(1, round(roi.shape[0] * ratio))),
                                 interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
                hand_result.update(laplacian=_laplacian(roi), luma=float(roi.mean()),
                                   clipped_fraction=float(np.mean((roi <= 5) | (roi >= 250))))
        result["hands"].append(hand_result)
    return result


def _quality_reason(quality, hands, settings, first=False):
    prefix = "first_" if first else ""
    threshold = settings["min_first_global_laplacian" if first else "min_global_laplacian"]
    if not np.isfinite(quality["global_laplacian"]) or quality["global_laplacian"] < threshold:
        return prefix + "global_blur"
    if not settings["min_luma"] <= quality["luma"] <= settings["max_luma"]:
        return prefix + "global_exposure"
    if quality["clipped_fraction"] > settings["max_clipped_fraction"]:
        return prefix + "global_clipping"
    for index in hands:
        hand = quality["hands"][index]
        if not hand["valid"]:
            return prefix + "hand_geometry_unknown"
        threshold = settings["min_first_hand_laplacian" if first else "min_hand_laplacian"]
        if not np.isfinite(hand["laplacian"]) or hand["laplacian"] < threshold:
            return prefix + "hand_blur"
        if not settings["min_luma"] <= hand["luma"] <= settings["max_luma"]:
            return prefix + "hand_exposure"
        if hand["clipped_fraction"] > settings["max_clipped_fraction"]:
            return prefix + "hand_clipping"
    return None


def _project(ep):
    from .data import project_hands
    projected = project_hands(ep)
    if isinstance(projected, tuple):
        uv, depth_valid = projected
    else:
        uv = projected
        depth_valid = np.isfinite(uv).all(axis=-1)
    valid = np.asarray(depth_valid, dtype=bool) & np.asarray(ep["hand_valid"], dtype=bool)[..., None]
    return np.asarray(uv, dtype=float), valid


def analyze_rgb(ep, cfg, candidates, frame_iterator):
    """Return all independently legal lengths per start, with pending review.

    ``allowed_n_frames`` contains independently validated temporal alternatives;
    arbitrary shorter lengths are never invented by this module. Each length
    gets its own temporal metrics, RGB summaries and interval fingerprints.
    Final selection remains longest first. Only the union
    of complete candidate intervals is cached; no interval is trimmed for space.
    The cache is discarded on success, failure or interruption. Decode/count/alignment errors
    invalidate the entire episode, including seemingly good earlier candidates.
    """
    if not candidates:
        return {"candidates": [], "rejections": {}, "diagnostics": {"decoded_frames": 0}}
    settings = {**DEFAULTS, **cfg.get("rgb", {})}
    runtime = cfg.get("runtime", {})
    cv2.setNumThreads(int(runtime.get("cv_threads", 1)))
    count = int(ep["n"])
    width = min(int(settings["analysis_width"]), int(ep["width"]))
    height = max(1, round(int(ep["height"]) * width / int(ep["width"])))
    camera = _camera(ep, width, height)
    rejections = collections.Counter()
    diagnostics = {"decoded_frames": 0, "analysis_width": width, "analysis_height": height,
                   "threshold_status": "engineering_initial_values_require_visual_calibration",
                   "visual_review_performed": False, "registration_calls": 0}
    normalised = []
    for candidate in candidates:
        start, end = int(candidate["start_frame"]), int(candidate["end_frame"])
        hands = sorted(set(candidate.get("participating_hands", [])))
        allowed = candidate.get("allowed_n_frames", [int(candidate["n_frames"])])
        lengths = sorted(set(int(length) for length in allowed), reverse=True)
        if (not 0 <= start < end <= count or not hands or any(h not in (0, 1) for h in hands)
                or not lengths or min(lengths) <= 0 or start + max(lengths) > end):
            rejections["invalid_candidate_contract"] += 1
            continue
        normalised.append((candidate, start, end, hands, lengths))
    if not normalised:
        return {"candidates": [], "rejections": dict(rejections), "diagnostics": diagnostics}
    # Preserve every frame within each candidate, while avoiding unused gaps.
    # Overlaps and adjacent intervals share records without changing frame IDs.
    intervals = []
    for start, end in sorted((row[1], row[2]) for row in normalised):
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    interval_starts = [start for start, _ in intervals]
    record_bases = []
    cached_frames = 0
    for start, end in intervals:
        record_bases.append(cached_frames)
        cached_frames += end - start
    gray_frame_bytes = width * height
    packed_mask_frame_bytes = (gray_frame_bytes + 7) // 8
    cache_record_bytes = gray_frame_bytes + packed_mask_frame_bytes
    needed = cached_frames * cache_record_bytes
    limit = int(runtime.get("rgb_cache_max_bytes", settings["max_cache_mb"] * 1024 ** 2))
    diagnostics.update(cache_bytes=needed, cache_limit_bytes=limit,
                       gray_cache_bytes=cached_frames * gray_frame_bytes,
                       packed_mask_cache_bytes=cached_frames * packed_mask_frame_bytes,
                       masks_generated=0, mask_cache_reads=0)
    if needed > limit:
        rejections["unknown_cache_budget_exceeded"] += len(normalised)
        return {"candidates": [], "rejections": dict(rejections), "diagnostics": diagnostics}
    uv, valid = _project(ep)
    if uv.shape != (count, 2, 21, 2) or valid.shape != (count, 2, 21):
        raise ValueError("Projected hands do not match the source presentation timeline")
    uv = uv * np.array([width / int(ep["width"]), height / int(ep["height"])])
    cache_root = runtime.get("scratch_dir", runtime.get("cache_root"))
    if cache_root:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
    starts = {c[1] for c in normalised}
    quality, pairs, hashes, fingerprints = {}, {}, {}, {}
    accepted = []
    try:
        with tempfile.TemporaryDirectory(prefix="rgb_", dir=cache_root) as temporary:
            cache_path = os.path.join(temporary, "gray.raw")
            with open(cache_path, "w+b", buffering=0) as cache:
                previous = previous_mask = None
                previous_index = None
                interval_index = 0
                for index, rgb in enumerate(frame_iterator):
                    diagnostics["decoded_frames"] = index + 1
                    if index >= count:
                        raise ValueError("Decoder emitted more frames than the aligned episode")
                    rgb = np.asarray(rgb)
                    if rgb.dtype != np.uint8 or rgb.shape != (int(ep["height"]), int(ep["width"]), 3):
                        raise ValueError("Unexpected RGB frame type or dimensions")
                    while interval_index < len(intervals) and index >= intervals[interval_index][1]:
                        interval_index += 1
                    if interval_index == len(intervals) or index < intervals[interval_index][0]:
                        # A cached frame on either side of a gap is NOT an
                        # adjacent source frame. Never register across it.
                        previous = previous_mask = previous_index = None
                        continue
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    if gray.shape != (height, width):
                        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                    quality[index] = _quality(gray, uv[index], valid[index], settings)
                    current_mask = _mask(gray.shape, uv[index], valid[index], settings)
                    # Generate the exact same binary mask only once per frame.
                    # Pack it alongside the grayscale image on bounded disk;
                    # repeated anchors no longer repeat the expensive dilation.
                    cache.write(gray.tobytes())
                    cache.write(np.packbits(current_mask.reshape(-1) != 0).tobytes())
                    diagnostics["masks_generated"] += 1
                    if previous_index == index - 1:
                        pairs[index] = _register(previous, gray, previous_mask, current_mask, camera, settings)
                        diagnostics["registration_calls"] += 1
                    if index in starts:
                        hashes[index] = hashlib.sha256(rgb.tobytes()).hexdigest()
                        fingerprints[index] = _fingerprint(gray)
                    previous, previous_mask = gray, current_mask
                    previous_index = index
                if diagnostics["decoded_frames"] != count:
                    raise ValueError("Decoder frame count does not match aligned episode")
                if cache.tell() != needed:
                    raise IOError("Temporary cache size does not match the complete candidate union")

                def record_offset(index):
                    interval = bisect_right(interval_starts, index) - 1
                    if interval < 0 or index >= intervals[interval][1]:
                        raise IOError("Requested source frame is outside the candidate cache union")
                    return (record_bases[interval] + index - intervals[interval][0]) * cache_record_bytes

                def read(index):
                    cache.seek(record_offset(index))
                    raw = cache.read(gray_frame_bytes)
                    if len(raw) != gray_frame_bytes:
                        raise IOError("Truncated temporary grayscale cache")
                    return np.frombuffer(raw, np.uint8).reshape(height, width)

                def read_mask(index):
                    cache.seek(record_offset(index) + gray_frame_bytes)
                    raw = cache.read(packed_mask_frame_bytes)
                    if len(raw) != packed_mask_frame_bytes:
                        raise IOError("Truncated temporary packed-mask cache")
                    mask = np.unpackbits(np.frombuffer(raw, np.uint8), count=gray_frame_bytes)
                    diagnostics["mask_cache_reads"] += 1
                    return (mask.reshape(height, width) * np.uint8(255))

                # One temporal candidate per start is the normal contract.
                # Duplicate starts reuse direct results without retaining images.
                grouped = collections.defaultdict(list)
                for row in normalised:
                    grouped[row[1]].append(row)
                for start, rows in sorted(grouped.items()):
                    reference = read(start)
                    reference_mask = read_mask(start)
                    reference_points = _features(reference, reference_mask, settings)
                    direct = {}
                    for candidate, _, end, hands, lengths in rows:
                        reason = _quality_reason(quality[start], hands, settings, first=True)
                        if reason:
                            rejections[reason] += 1
                            continue
                        if reference_points is None or len(reference_points) < settings["min_features"]:
                            rejections["unknown_insufficient_anchor_features"] += 1
                            continue
                        cap = end
                        failure_reason = None
                        for index in range(start, end):
                            reason = _quality_reason(quality[index], hands, settings)
                            if reason:
                                cap, failure_reason = index, reason
                                break
                            if index == start:
                                continue
                            pair = pairs[index]
                            if pair["status"] != "ok":
                                cap, failure_reason = index, "unknown_pair_" + pair["reason"]
                                break
                            if pair["p90_displacement_diag"] > settings["max_pair_displacement_diag"]:
                                cap, failure_reason = index, "pair_camera_motion"
                                break
                            if index not in direct:
                                current = read(index)
                                current_mask = read_mask(index)
                                direct[index] = _register(reference, current, reference_mask, current_mask,
                                                          camera, settings, reference_points)
                                diagnostics["registration_calls"] += 1
                            anchor = direct[index]
                            if anchor["status"] != "ok":
                                cap, failure_reason = index, "unknown_anchor_" + anchor["reason"]
                                break
                            if anchor["p90_displacement_diag"] > settings["max_anchor_displacement_diag"]:
                                cap, failure_reason = index, "anchor_camera_drift"
                                break
                        possible = [length for length in lengths if start + length <= cap]
                        if not possible:
                            rejections[failure_reason or "no_rgb_legal_length"] += 1
                            continue
                        for length in sorted(set(possible), reverse=True):
                            stop = start + length
                            selected = dict(candidate)
                            selected.update(start_frame=start, end_frame=stop, n_frames=length,
                                            participating_hands=hands, qc_status="pending",
                                            source_first_frame_sha256=hashes[start],
                                            source_first_frame_fingerprint=fingerprints[start],
                                            source_middle_frame_fingerprint=_fingerprint(read(start + length // 2)),
                                            source_last_frame_fingerprint=_fingerprint(read(stop - 1)))
                            selected.pop("allowed_n_frames", None)
                            metrics_by_length = selected.pop("metrics_by_n_frames", {})
                            if (str(length) not in metrics_by_length and length not in metrics_by_length
                                    and length != candidate["n_frames"]):
                                rejections["missing_temporal_metrics_for_length"] += 1
                                continue
                            metrics = dict(metrics_by_length.get(str(length), metrics_by_length.get(length,
                                                                                                  candidate.get("metrics", {}))))
                            per_frame = [quality[i] for i in range(start, stop)]
                            registrations = [direct[i] for i in range(start + 1, stop)]
                            metrics["rgb"] = {
                                "analysis_width": width, "analysis_height": height,
                                "checked_frames": length, "direct_anchor_pairs": length - 1,
                                "first_global_laplacian": quality[start]["global_laplacian"],
                                "first_hand_laplacian": {str(h): quality[start]["hands"][h]["laplacian"] for h in hands},
                                "min_global_laplacian": min(q["global_laplacian"] for q in per_frame),
                                "min_hand_laplacian": min(q["hands"][h]["laplacian"] for q in per_frame for h in hands),
                                "max_anchor_displacement_diag": max((r["p90_displacement_diag"] for r in registrations), default=0.0),
                                "max_pair_displacement_diag": max((pairs[i]["p90_displacement_diag"] for i in range(start + 1, stop)), default=0.0),
                                "min_inlier_ratio": min((r["inlier_ratio"] for r in registrations), default=1.0),
                                "min_background_grid_cells": min((r["occupied_cells"] for r in registrations), default=0),
                                "hand_rgb_semantics_verified": False,
                            }
                            selected["metrics"] = metrics
                            accepted.append(selected)
    except (ValueError, IOError, OSError, cv2.error) as error:
        from .resources import retryable_resource_error
        if retryable_resource_error(error):
            raise
        accepted = []
        rejections["unknown_episode_decode_or_cache_error"] += len(normalised)
        diagnostics["episode_error"] = str(error)
    diagnostics["machine_passed"] = len(accepted)
    diagnostics["machine_passed_windows"] = len(accepted)
    diagnostics["machine_passed_starts"] = len({c['start_frame'] for c in accepted})
    diagnostics["unknown_candidates"] = sum(v for k, v in rejections.items() if k.startswith("unknown"))
    return {"candidates": accepted, "rejections": dict(rejections), "diagnostics": diagnostics}
