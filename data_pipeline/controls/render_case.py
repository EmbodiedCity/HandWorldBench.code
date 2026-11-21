"""Render the skeleton, surface and X-ray control videos of one case.

All three signals are drawn in the reference camera (same intrinsics, lens model, resolution, frame rate and
frame count as the reference video):

* skeleton - the 21 projected joints of each hand and their 20 bones, drawn with anti-aliased OpenCV lines/discs.
* surface  - the front-most surface of a hand mesh, flat Lambert shading, one colour per hand.
* xray     - every surface layer of the same mesh, alpha-composited by mesh thickness along the ray.

The hand mesh depends on the embodiment:

* human   - MANO fitted to the 3-D joints on every other frame (``mano_fit.py``); in-between frames reuse the
            neighbouring fits, deformed to the true joints of that frame (``warp``).
* handson - the Wuji hand surface posed by forward kinematics (``robot.py``).
* ti5     - spheres on the joints and tubes along the bones.

Meshes are projected vertex-by-vertex with the camera's own lens model (``cv2.fisheye`` for equidistant fisheye
clips, ``cv2.projectPoints`` otherwise) and rasterised with a small depth-buffer rasteriser (``raster``).
Rendering runs on the CPU.
"""
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

cv2.setNumThreads(1)

SIGNALS = ('skeleton', 'surface', 'xray')
EDGES = [(0, 1 + 4 * i) for i in range(5)] + [(1 + 4 * i + j, 2 + 4 * i + j) for i in range(5) for j in range(3)]
PALETTE = np.array([[245, 144, 62], [95, 129, 250]], np.float32)  # BGR; hand 0 (left) blue, hand 1 (right) orange


class _Holder:
    pass


class _Faces(pickle.Unpickler):
    """Read the MANO face list without importing chumpy."""

    def find_class(self, module, name):
        return _Holder if module.startswith('chumpy') else super().find_class(module, name)


def mano_faces(mano_dir):
    """Triangle list of the MANO hand (from MANO_RIGHT.pkl; left and right share the topology)."""
    with (Path(mano_dir) / 'MANO_RIGHT.pkl').open('rb') as f:
        return np.asarray(_Faces(f, encoding='latin1').load()['f'], dtype=np.int32)


def lift_points(x, k, d, fish):
    x = np.ascontiguousarray(x, np.float64).reshape(-1, 1, 3)
    if fish:
        return cv2.fisheye.projectPoints(x, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
    return cv2.projectPoints(x, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)


def project(x, k, d, fish):
    # projectPoints allocates a per-point Jacobian. Chunk mesh vertices to bound memory.
    x = x.reshape(-1, 3)
    return np.vstack([lift_points(x[i:i + 16000], k, d, fish) for i in range(0, len(x), 16000)])


@njit(cache=True)
def raster(uv, z, faces, light, near, far, intensity):
    """Rasterise triangles; keep the nearest and farthest depth per pixel and the shading of the nearest face."""
    height, width = near.shape
    for f in range(len(faces)):
        i, j, k = faces[f]
        z0, z1, z2 = z[i], z[j], z[k]
        if min(z0, z1, z2) <= .015:
            continue
        ax, ay = uv[i]
        bx, by = uv[j]
        cx, cy = uv[k]
        if not (math.isfinite(ax + ay + bx + by + cx + cy)):
            continue
        minx = max(0, math.ceil(min(ax, bx, cx)))
        maxx = min(width - 1, math.floor(max(ax, bx, cx)))
        miny = max(0, math.ceil(min(ay, by, cy)))
        maxy = min(height - 1, math.floor(max(ay, by, cy)))
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-7:
            continue
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                a = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den
                b = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den
                c = 1 - a - b
                if min(a, b, c) < -1e-7:
                    continue
                depth = 1. / (a / z0 + b / z1 + c / z2)
                if depth < near[y, x]:
                    near[y, x] = depth
                    intensity[y, x] = light[f]
                if depth > far[y, x]:
                    far[y, x] = depth


def spheres(joints):
    """Joint-proxy hand mesh: a sphere per joint and a tube per bone (used for TI5)."""
    vs = []
    fs = []
    off = 0

    def add(vert, face):
        nonlocal off
        vs.append(vert)
        fs.append(face + off)
        off += len(vert)

    angles = 2 * np.pi * np.arange(10) / 10
    ring = np.stack([np.cos(angles), np.sin(angles)], 1)
    for j, c in enumerate(joints):
        if not np.isfinite(c).all() or c[2] <= .015:
            continue
        lat = np.linspace(-np.pi / 2, np.pi / 2, 7)
        sph = np.array([[np.cos(t) * x, np.cos(t) * y, np.sin(t)] for t in lat for x, y in ring])
        add(c + sph * (.010 if j == 0 else .0065),
            np.array([[a, b, c] for h in range(6) for n in range(10)
                      for a, b, c in [(h * 10 + n, h * 10 + (n + 1) % 10, (h + 1) * 10 + n),
                                      ((h + 1) * 10 + n, h * 10 + (n + 1) % 10, (h + 1) * 10 + (n + 1) % 10)]]))
    for a, b in EDGES:
        s, e = joints[a], joints[b]
        v = e - s
        if not np.isfinite(v).all() or min(s[2], e[2]) <= .015 or np.linalg.norm(v) < 1e-7:
            continue
        v /= np.linalg.norm(v)
        basis = np.eye(3)[np.argmin(np.abs(v))]
        u = np.cross(v, basis)
        u /= np.linalg.norm(u)
        w = np.cross(v, u)
        tube = (ring[:, 0, None] * u + ring[:, 1, None] * w) * (.008 if a == 0 else .0055)
        add(np.concatenate([s + tube, e + tube]),
            np.array([[a, b, c] for n in range(10)
                      for a, b, c in [(n, (n + 1) % 10, 10 + n), (10 + n, (n + 1) % 10, 10 + (n + 1) % 10)]]))
    return (np.vstack(vs), np.vstack(fs)) if vs else (None, None)


def warp(v, source, target):
    """Move mesh vertices from a source joint pose to a target joint pose.

    Each vertex is attached to its three closest bones in the source pose; every bone is rotated onto its target
    direction and the three results are blended with inverse-distance weights."""
    A = np.array([a for a, b in EDGES])
    B = np.array([b for a, b in EDGES])
    va = source[B] - source[A]
    vb = target[B] - target[A]
    u = va / np.maximum(np.linalg.norm(va, axis=1)[:, None], 1e-8)
    t = vb / np.maximum(np.linalg.norm(vb, axis=1)[:, None], 1e-8)
    q = v[:, None, :] - source[A]
    along = np.clip((q * va).sum(2) / np.maximum((va * va).sum(1), 1e-10), 0, 1)
    dist = np.linalg.norm(q - along[:, :, None] * va, axis=2)
    which = np.argpartition(dist, 2, axis=1)[:, :3]
    ii = np.arange(len(v))[:, None]
    w = 1 / (dist[ii, which] + .003) ** 3
    w /= w.sum(1)[:, None]
    cross = np.cross(u, t)
    angle = np.arctan2(np.linalg.norm(cross, axis=1), (u * t).sum(1))
    axis = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-8)
    rot = Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
    posed = np.einsum('nvij,nvj->nvi', rot[which], q[ii, which]) + target[A][which]
    return (posed * w[:, :, None]).sum(1)


def aligned_mesh(t, h, ids, verts, mvalid, joints, hide=None):
    """MANO vertices of hand ``h`` at frame ``t``.

    Frames with a fit return it directly. Other frames interpolate the two neighbouring fits (at most 3 frames
    away), each first warped to the true joints of frame ``t``. Returns None when no fit is close enough."""
    use = np.flatnonzero(mvalid[:, h])
    use = use[use != hide] if hide is not None else use
    if not len(use):
        return None
    where = np.searchsorted(ids[use], t)
    lo = use[max(0, where - 1)]
    hi = use[min(where, len(use) - 1)]
    if ids[hi] == t:
        return verts[hi, h]
    if min(abs(int(ids[lo]) - t), abs(int(ids[hi]) - t)) > 3:
        return None

    def deformed(i):
        return warp(verts[i, h], joints[ids[i], h], joints[t, h])

    if lo == hi:
        return deformed(lo)
    frac = np.clip((t - ids[lo]) / (ids[hi] - ids[lo]), 0, 1)
    return (1 - frac) * deformed(lo) + frac * deformed(hi)


def vidprobe(path):
    d = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,nb_frames,avg_frame_rate,codec_name,pix_fmt:format_tags', '-of', 'json', str(path)]))
    return d['streams'][0]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(case, output, mano=None, robots=None, faces=None, previews=False, tmp_root=None):
    """Render the three control videos of ``case`` into ``output/{skeleton,surface,xray}/<case_id>.mp4``.

    case    - a ``datasets.Case``.
    mano    - for human cases: dict with ``source_frame_indices``, ``vertices`` and ``hand_valid`` (``mano_fit.fit``).
    robots  - for HandsOn cases: ``(left, right)`` ``robot.Robot`` instances; ``case.hand20`` / ``case.palm_to_camera``
              must be set (``handson_ik.recover``).
    faces   - for human cases: MANO triangle list (``mano_faces``).
    Returns a small receipt dict.
    """
    cid = case.case_id
    cat = case.embodiment
    n, w, h, fps = case.frame_count, case.width, case.height, case.fps
    output = Path(output)
    vp = case.video
    if vp is not None and Path(vp).is_file():
        refp = vidprobe(vp)
        assert int(refp['nb_frames']) == n and (refp['width'], refp['height']) == (w, h) \
            and Fraction(refp['avg_frame_rate']) == fps, (cid, 'reference video does not match the case metadata')
    human = cat == 'human'
    joints, uv, good, fish = case.joints, case.uv, case.hand_valid, case.fisheye
    if human:
        ids = np.asarray(mano['source_frame_indices'])
        verts = mano['vertices']
        mval = mano['hand_valid']
        face = faces
    elif cat == 'handson':
        angles = case.hand20
        poses = case.palm_to_camera
    assert len(joints) == len(uv) == n and good.shape[:2] == (n, 2)
    K = np.asarray(case.K, np.float64)
    D = np.asarray(case.D, np.float64)

    # The stored 2-D joints must be the projection of the 3-D joints with this camera.
    test = np.isfinite(joints).all(-1) & np.isfinite(uv).all(-1) & (joints[..., 2] > .02) \
        & (good[..., None] if good.ndim == 2 else good)
    js = joints[test][::13]
    us = uv[test][::13]
    spatial = float(np.percentile(np.linalg.norm(lift_points(js, K, D, fish) - us, axis=1), 99)) if len(js) else None
    if spatial is None or spatial > 1.1:
        raise ValueError(f'{cid}: projection p99 {spatial}')

    tmp = Path(tmp_root or tempfile.gettempdir()) / 'hwb_controls' / cid
    tmp.mkdir(exist_ok=True, parents=True)
    writers = {}
    start = time.monotonic()
    skipped = 0
    for name in SIGNALS:
        writers[name] = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
             '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
             '-crf', '18', '-pix_fmt', 'yuv420p', '-map_metadata', '-1', '-movflags', '+faststart',
             str(tmp / (name + '.mp4'))], stdin=subprocess.PIPE)

    def visible(t):
        return np.isfinite(uv[t]).all(-1) & (uv[t, :, :, 0] >= 0) & (uv[t, :, :, 0] < w) & (uv[t, :, :, 1] >= 0) \
            & (uv[t, :, :, 1] < h) & (joints[t, :, :, 2] > .02) & (good[t, :, None] if good.ndim == 2 else good[t])

    shots = set()
    if previews:
        useful = np.where(np.array([visible(t).sum() >= 8 for t in range(n)]))[0]
        if len(useful):
            shots = set(useful[np.linspace(0, len(useful) - 1, min(5, len(useful))).astype(int)].tolist())
    try:
        for t in range(n):
            sk = np.zeros((h, w, 3), np.uint8)
            sol = np.zeros_like(sk)
            xr = np.zeros_like(sk)
            hands = []
            for si in range(2):
                vis = visible(t)[si]
                pts = joints[t, si]
                if not vis.any():
                    continue
                # skeleton
                for a, b in EDGES:
                    if vis[a] and vis[b]:
                        cv2.line(sk, tuple(np.rint(uv[t, si, a]).astype(int)), tuple(np.rint(uv[t, si, b]).astype(int)),
                                 PALETTE[si].tolist(), max(2, round(h / 300)), cv2.LINE_AA)
                for pnt in uv[t, si, vis]:
                    cv2.circle(sk, tuple(np.rint(pnt).astype(int)), max(2, round(h / 320)), PALETTE[si].tolist(), -1,
                               cv2.LINE_AA)
                # hand mesh in camera coordinates
                if human:
                    V = aligned_mesh(t, si, ids, verts, mval, joints)
                    F = face
                elif cat == 'handson':
                    tri, _ = robots[si].pose(angles[t, si])
                    tf = poses[t, si]
                    V = tri.reshape(-1, 3) @ tf[:3, :3].T + tf[:3, 3]
                    F = np.arange(len(V), dtype=np.int32).reshape(-1, 3)
                else:
                    V, F = spheres(pts)
                if V is None:
                    skipped += 1
                    continue
                pix = project(V, K, D, fish)
                tri = V[F]
                norm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                lambert = .35 + .65 * np.abs((norm * np.array([.2, -.3, -.93274])).sum(1)) \
                    / np.maximum(np.linalg.norm(norm, axis=1), 1e-9)
                valid = np.isfinite(pix).all(1) & (V[:, 2] > .015)
                if not valid.any():
                    continue
                x0, y0 = np.maximum(np.floor(pix[valid].min(0)) - 2, 0).astype(int)
                x1, y1 = np.minimum(np.ceil(pix[valid].max(0)) + 3, [w, h]).astype(int)
                if x0 >= x1 or y0 >= y1:
                    continue
                sh = (y1 - y0, x1 - x0)
                near = np.full(sh, np.inf, np.float32)
                far = np.zeros(sh, np.float32)
                inten = np.zeros(sh, np.float32)
                raster(pix - [x0, y0], V[:, 2], F, lambert, near, far, inten)
                hands.append((si, x0, y0, x1, y1, near, far, inten))
            if hands:
                # composite both hands, nearest hand first
                x0 = min(e[1] for e in hands)
                y0 = min(e[2] for e in hands)
                x1 = max(e[3] for e in hands)
                y1 = max(e[4] for e in hands)
                sh = (y1 - y0, x1 - x0)
                zfront = np.full(sh, np.inf, np.float32)
                trans = np.ones(sh, np.float32)
                for si, x, y, ex, ey, near, far, inten in sorted(
                        hands, key=lambda e: np.median(e[5][np.isfinite(e[5])]) if np.isfinite(e[5]).any() else np.inf):
                    roi = np.s_[y - y0:ey - y0, x - x0:ex - x0]
                    active = np.isfinite(near)
                    zf = zfront[roi]
                    front = active & (near < zf)
                    sol[y:ey, x:ex][front] = (inten[front, None] * PALETTE[si]).clip(0, 255).astype(np.uint8)
                    zf[front] = near[front]
                    depth = np.zeros_like(near)
                    depth[active] = np.maximum(far[active] - near[active], 0)
                    alpha = (.24 + .42 * (1 - np.exp(-depth / .023)) + .18 * inten) * active
                    view = xr[y:ey, x:ex]
                    view[:] = np.clip(view + trans[roi][:, :, None] * alpha[:, :, None] * PALETTE[si], 0, 255) \
                        .astype(np.uint8)
                    trans[roi] *= 1 - alpha
            for key, img in zip(SIGNALS, [sk, sol, xr]):
                writers[key].stdin.write(img)
            if t in shots:
                _preview(vp, t, w, h, (sk, sol, xr), output / 'previews' / cid)
    finally:
        for wri in writers.values():
            wri.stdin.close()
        status = [wr.wait() for wr in writers.values()]
    if any(status):
        raise ValueError(f'{cid}: encoding failed {status}')
    if skipped:
        raise ValueError(f'{cid}: visible hand without reconstructible mesh on {skipped} frames')

    verified = {}
    for name in SIGNALS:
        fp = tmp / (name + '.mp4')
        v = vidprobe(fp)
        assert int(v['nb_frames']) == n and v['width'] == w and v['height'] == h \
            and Fraction(v['avg_frame_rate']) == fps and v['codec_name'] == 'h264' and v['pix_fmt'] == 'yuv420p'
        dest = output / name / (cid + '.mp4')
        dest.parent.mkdir(exist_ok=True, parents=True)
        part = dest.with_name(dest.name + '.part')
        shutil.copyfile(fp, part)
        part.replace(dest)
        checksum = _sha256(fp)
        assert _sha256(dest) == checksum
        verified[name] = dict(bytes=dest.stat().st_size, sha256=checksum)
        fp.unlink()
    try:
        tmp.rmdir()
    except OSError:
        pass
    rec = dict(case_id=cid, embodiment=cat, frames=n, seconds=round(time.monotonic() - start, 2),
               projection_p99_px=spatial, videos=verified)
    return rec


def _preview(video, t, w, h, masks, folder):
    """Side-by-side JPEG: reference frame and the three controls blended over it."""
    if video is None or not Path(video).is_file():
        return
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(video), '-vf',
                                   f'select=eq(n\\,{t})', '-frames:v', '1', '-threads', '1', '-f', 'rawvideo',
                                   '-pix_fmt', 'bgr24', '-'])
    bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    cols = [bgr] + [np.where(m.any(2)[:, :, None], (.45 * bgr + .55 * m).astype(np.uint8), bgr) for m in masks]
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f'overlay_{t:06}.jpg'), np.hstack([cv2.resize(q, (320, round(h * 320 / w))) for q in cols]))
