"""Draw the camera trajectory of one clip as a frustum sequence.

    python render_camera_track.py --camera camera/hw500_0001.npz \
      --first-frame first_frames/hw500_0001.png --out track.mp4

Visualisation only: models consume `c2w` directly.
"""
import argparse, json
from pathlib import Path

import numpy as np


def frustum_edges(scale=0.06, aspect=1.0):
    """Corners of a small pyramid in camera coordinates, +Z forward."""
    x, y = scale * aspect, scale
    near = scale
    far = scale * 2.5
    pts = np.array([[-x, -y, near], [x, -y, near], [x, y, near], [-x, y, near],
                    [-x, -y, far], [x, -y, far], [x, y, far], [-x, y, far]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return pts, edges


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--first-frame", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    with np.load(a.camera, allow_pickle=False) as z:
        c2w = z["c2w"]; K = z["K"]
    T = len(c2w)
    if a.first_frame and Path(a.first_frame).exists():
        bg = cv2.imread(a.first_frame)
    else:
        bg = None
    if bg is None:
        H, W = 480, int(480 * 4 / 3)
        bg = np.full((H, W, 3), 18, np.uint8)
    H, W = bg.shape[:2]

    # world points -> image, using the clip's own intrinsics
    inv = np.linalg.inv(K)
    pts, edges = frustum_edges()

    def project(pw, pose):
        cam = (np.linalg.inv(pose) @ np.c_[pw, np.ones(len(pw))].T).T[:, :3]
        cam = cam[cam[:, 2] > 1e-3]
        if len(cam) == 0:
            return None
        h = (inv @ cam.T).T
        uv = h[:, :2] / h[:, 2:3]
        return uv

    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"avc1"), a.fps, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for t in range(0, T, a.stride):
        img = bg.copy()
        pose = c2w[t]
        uv = project(pts, pose)
        if uv is not None:
            for i, j in edges:
                if i < len(uv) and j < len(uv):
                    cv2.line(img, tuple(np.round(uv[i]).astype(int)),
                             tuple(np.round(uv[j]).astype(int)), (60, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{t}/{T - 1}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    print(json.dumps({"out": a.out, "frames": len(range(0, T, a.stride))}))


if __name__ == "__main__":
    main()
