"""Frame-macro pose accuracy and coverage of all valid in-frame reference points."""
import numpy as np

def score(reference, prediction, metric, confidence_threshold=0.3):
    if metric not in ("pose_control", "keypoint_coverage"):
        raise ValueError(metric)
    if len(reference) != len(prediction):
        raise ValueError("Reference/prediction frame count mismatch")
    frames, pixel_errors = [], []
    expected = detected = 0
    for ref, pred in zip(reference, prediction):
        if ref["sample_index"] != pred["sample_index"]:
            raise ValueError("Frame identity mismatch")
        width, height = ref["width"], ref["height"]
        identities = [h["side"] for h in pred["hands"]]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate predicted hand identity")
        by_side = {h["side"]: h for h in pred["hands"]}
        point_scores, frame_detected = [], 0
        for hand in ref["hands"]:
            xy = np.asarray(hand["joints_2d"], dtype=float)
            if xy.shape != (21, 2):
                raise ValueError("Expected reference shape (21, 2)")
            valid = np.asarray(hand["valid_2d"], dtype=bool)
            if valid.shape != (21,):
                raise ValueError("Expected reference validity shape (21,)")
            valid &= np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            observed = by_side.get(hand["side"])
            q = np.full((21, 2), np.nan) if observed is None else np.asarray(observed["joints_2d"], dtype=float)
            conf = np.zeros(21) if observed is None else np.asarray(observed["confidence"], dtype=float)
            if q.shape != (21, 2) or conf.shape != (21,):
                raise ValueError("Invalid prediction keypoint shape")
            good = np.isfinite(q).all(axis=1) & np.isfinite(conf) & (conf >= confidence_threshold) & (q[:, 0] >= 0) & (q[:, 0] < width) & (q[:, 1] >= 0) & (q[:, 1] < height)
            for j in np.flatnonzero(valid):
                expected += 1
                if good[j]:
                    detected += 1
                    frame_detected += 1
                    error = float(np.linalg.norm(q[j] - xy[j]))
                    pixel_errors.append(error)
                    point_scores.append(1.0 - min(error / (0.1 * max(width, height)), 1.0))
                else:
                    point_scores.append(0.0)
        frame_score=(frame_detected/len(point_scores) if metric=='keypoint_coverage' else float(np.mean(point_scores))) if point_scores else None
        frames.append(dict(sample_index=ref["sample_index"], expected_units=len(point_scores), observed_units=frame_detected, score=frame_score))
    values = [f["score"] for f in frames if f["score"] is not None]
    value = (detected / expected if metric == "keypoint_coverage" else float(np.mean(values))) if expected else None
    return dict(metric_id=metric, score_0_1=value, raw_score=value, raw_unit="normalized", status="missing_reference" if not expected else "scored_with_missing_penalty" if detected < expected else "ok", expected_units=expected, observed_units=detected, penalized_units=expected-detected, missing_rate=(expected-detected)/expected if expected else None, conditional_pixel_error=float(np.mean(pixel_errors)) if pixel_errors else None, frames=frames)
