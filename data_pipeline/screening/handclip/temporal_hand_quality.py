"""Bound mild hand-motion blur separately from the first-frame and hard-floor gates."""
import numpy as np

def longest_soft_end(mask, start, stop, minimum, multiple, offset, max_fraction, max_run):
    """Enumerate valid length bins; a fraction gate is NOT prefix-monotonic."""
    mask = np.asarray(mask, dtype=bool)
    prefix = np.concatenate(([0], np.cumsum(mask[start:stop], dtype=np.int64)))
    # Three consecutive mild frames, for example, make the third a hard end.
    run = 0; cap = stop
    for i in range(start, stop):
        run = run + 1 if mask[i] else 0
        if run > max_run:
            cap = i
            break
    n = (cap-start-offset)//multiple*multiple+offset
    while n >= minimum:
        if prefix[n] <= max_fraction*n + 1e-12:
            return start+n
        n -= multiple
    return start

def summary(mask, start, stop, nominal, hard):
    part = np.asarray(mask[start:stop], bool);run=longest=0
    for value in part:
        run=run+1 if value else 0;longest=max(longest,run)
    return dict(policy='rare_mild_hand_motion_blur_v8',nominal_min_laplacian=nominal,
                hard_min_laplacian=hard,mild_frame_count=int(part.sum()),
                mild_frame_fraction=float(part.mean()) if len(part) else 0.,
                max_consecutive_mild_frames=longest,first_frame_threshold_unchanged=True)
