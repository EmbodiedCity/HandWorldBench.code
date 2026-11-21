"""Check TI5 native signals and primary head-camera media."""

from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np

FALLBACK = "synchronization/sample_quality/invalid_value_fallback"


def stage(source, target):
    with Path(source).open("rb") as original, target.open("wb") as copy:
        shutil.copyfileobj(original, copy, 4 * 1024 * 1024)


def check_native(path, n):
    signals = {}
    with h5py.File(path, "r") as native:
        invalid = native.get(FALLBACK)
        if invalid is None or invalid.shape != (n,) or np.any(invalid[:]):
            raise ValueError("Missing, misaligned, or nonzero invalid-value fallback mask")
        hand = native.get("observations/hand")
        if not isinstance(hand, h5py.Group) or not any(
            isinstance(value, h5py.Dataset) and value.shape == (n, 20) for value in hand.values()
        ):
            raise ValueError("Missing aligned TI5 hand observations")

        def visit(name, value):
            if not name.startswith(("observations/", "actions/")) or not isinstance(value, h5py.Dataset):
                return
            if not value.shape or value.shape[0] != n:
                raise ValueError(f"{name}: native frame count mismatch")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value[:]).all():
                raise ValueError(f"{name}: nonfinite native signal")
            signals[name] = list(value.shape)

        native.visititems(visit)
    return signals


def check_head(path, n, fps):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    means = []
    white = []
    bad = 0
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) >= .01:
            raise ValueError("Head camera cannot be decoded at the metadata FPS")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = float((gray >= 250).mean())
            bad += gray.std() < 1 or bright > .95 or (gray <= 5).mean() > .98
            means.append(float(gray.mean()))
            white.append(bright)
    finally:
        cap.release()
    if len(means) != n:
        raise ValueError(f"Head frame count {len(means)} != {n}")
    if bad:
        raise ValueError(f"{bad} extreme or flat head frames")
    median = float(np.median(means))
    return {
        "decoded_frames": n, "fps": fps, "raw_luma_p50": median, "raw_white_p95": float(np.percentile(white, 95)),
        "planned_gamma": 1.0 if median > 125 else .78,
    }
