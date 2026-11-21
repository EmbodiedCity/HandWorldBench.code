"""Check source media, native Parquet and render metadata for HandsOn episodes."""

from hashlib import sha256
import json
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np
import pyarrow.parquet as pq

from robot_io import sha

VIEWS = ("front", "wrist_left", "wrist_right")
TAR_LOCK = threading.Lock()


def probe_video(path, fps):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path),
    ]
    stream = json.loads(subprocess.check_output(command, timeout=30))["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    if abs(numerator / denominator - fps) > 1e-5:
        raise ValueError(f"{path.name} FPS differs from manifest")
    return {"width": stream["width"], "height": stream["height"], "fps": fps}


def verify_front(path, expected_frames, shape):
    width, height = shape["width"], shape["height"]
    command = [
        "ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-filter_threads", "1", "-threads", "1",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    blur = []
    bad_frame = None
    frame_bytes = height * width * 3
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=errors
    ) as decoder:
        while raw := decoder.stdout.read(frame_bytes):
            if len(raw) != frame_bytes:
                raise ValueError("Partial decoded front frame")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            gray = cv2.cvtColor(cv2.resize(frame, (320, round(height * 320 / width))), cv2.COLOR_BGR2GRAY)
            if bad_frame is None and (gray.mean() < 3 or (gray.std() < 1 and gray.mean() > 250)):
                bad_frame = len(blur)
            blur.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        if decoder.wait() != 0:
            errors.seek(0)
            raise ValueError(f"FFmpeg front decode failed: {errors.read(1000)!r}")
    if len(blur) != expected_frames:
        raise ValueError(f"Front frame count {len(blur)} != {expected_frames}")
    if bad_frame is not None:
        raise ValueError(f"Black or flat white front frame {bad_frame}")
    p10 = float(np.percentile(blur, 10))
    if p10 < 8:
        raise ValueError(f"Front Laplacian P10 {p10:.2f} < 8")
    return p10


def verify_wrist(path, expected_frames):
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-filter_threads", "1", "-threads", "1", "-vsync", "0",
        "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], capture_output=True, text=True, check=True, timeout=120)
    counts = [int(line.split("=")[1]) for line in result.stdout.splitlines() if line.startswith("frame=")]
    if not counts or counts[-1] != expected_frames:
        raise ValueError(f"{path.name} decoded frame count differs from manifest")


def stage(source, target, expected):
    with source.open("rb") as incoming, target.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
    if sha(target) != expected:
        raise ValueError(f"{source} SHA mismatch after read")


def check_local(r, source, archive, members, local):
    n, fps = r["frames"], r["fps"]
    paths = r["video_paths"]
    if set(paths) != set(VIEWS):
        raise ValueError("Unexpected view set")
    views = {}
    for view in VIEWS:
        path = local / (view + ".mp4")
        stage(source / paths[view], path, r["video_sha256"][view])
        shape = probe_video(path, fps)
        if view == "front":
            blur = verify_front(path, n, shape)
        else:
            verify_wrist(path, n)
        views[view] = {**shape, "decoded_frames": n}
    parquet = local / "episode.parquet"
    stage(source / r["data_path"], parquet, r["parquet_sha256"])
    columns = ["frame_index", "timestamp", "observation.state", "action"]
    table = pq.read_table(parquet, columns=columns, use_threads=False)
    if table["frame_index"].to_pylist() != list(range(n)):
        raise ValueError("Native frame index mismatch")
    if not np.allclose(table["timestamp"].to_numpy(), np.arange(n) / fps, atol=.0001, rtol=0):
        raise ValueError("Native timestamp mismatch")
    for name in columns[2:]:
        values = np.asarray(table[name].to_pylist())
        if values.shape != (n, 52) or not np.isfinite(values).all():
            raise ValueError(f"Native {name} is invalid")
    with TAR_LOCK:
        raw = archive.extractfile(members[r["render_metadata_member"]]).read()
    if sha256(raw).hexdigest() != r["render_metadata_sha256"]:
        raise ValueError("Render metadata SHA mismatch")
    return {"technical_status": "pass", "blur_p10_320px": blur, "views": views}
