"""Fit MANO to the 3-D hand joints of a human clip (every other frame).

For each hand separately, MANO (``smplx.MANO``, full 45-D pose, flat mean hand) is fitted to the 21 root-relative
camera-space joints of every second frame:

* initial global orientation: Kabsch rotation of the wrist and the four finger bases (MCP) onto the target;
* Adam (lr 0.03, 650 steps) on pose, orientation and one shape vector shared by the whole clip, minimising the mean
  squared joint error over valid frames (tiny L2 terms on pose and shape; shape clamped to [-3, 3]);
* vertices are expressed relative to the fitted wrist and translated to the observed wrist.

MANO's 16 joints are extended with five fingertip vertices and reordered to the HandWorldBench 21-joint order.
Runs on the CPU. The thread count is fixed so repeated runs give the same result.
"""
import inspect
import os
from pathlib import Path

import numpy as np

# MANO's 16 skeleton joints + 5 fingertip vertices, re-ordered to wrist, thumb 1-4, index 1-4, ... pinky 1-4.
TIPS = [745, 317, 444, 556, 673]
ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
STEPS = 650


def _compat():
    """chumpy (needed to unpickle the official MANO files) uses names removed from NumPy >= 1.24 and Python >= 3.11."""
    for k, v in [('bool', bool), ('int', int), ('float', float), ('complex', complex), ('object', object),
                 ('unicode', str), ('str', str)]:
        if k not in np.__dict__:
            setattr(np, k, v)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec


def fit(joints, hand_valid, mano_dir, threads=6, log=print):
    """joints: (N, 2, 21, 3) camera-space metres (hand 0 = left, 1 = right); hand_valid: (N, 2) bool.

    Returns a dict with source_frame_indices (0, 2, 4, ...), vertices (M, 2, 778, 3), joints (M, 2, 21, 3),
    hand_valid (M, 2), pose (2, M, 45), orient (2, M, 3), betas (2, 1, 10), fit_error_mm (per hand mean / max)."""
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads))
    _compat()
    import cv2
    import smplx
    import torch

    torch.set_num_threads(threads)
    models = [smplx.MANO(model_path=str(mano_dir), is_rhand=side == 'right', use_pca=False, flat_hand_mean=True)
              for side in ('left', 'right')]
    tips = torch.tensor(TIPS)
    order = torch.tensor(ORDER)
    ids = list(range(0, len(joints), 2))
    positions = np.nan_to_num(joints[ids])
    valid = hand_valid[ids] & np.isfinite(positions).all((2, 3))
    n = len(ids)
    vertices = []
    fitted = []
    params = []
    errors = []
    for h, model in enumerate(models):
        target = torch.tensor(positions[:, h] - positions[:, h, 0:1], dtype=torch.float32)
        mask = torch.tensor(valid[:, h], dtype=torch.float32)
        pose = torch.zeros((n, 45), requires_grad=True)
        ori = torch.zeros((n, 3), requires_grad=True)
        beta = torch.zeros((1, 10), requires_grad=True)

        def forward():
            result = model(hand_pose=pose, global_orient=ori, betas=beta.expand(n, -1))
            j = torch.cat([result.joints, result.vertices[:, tips]], 1)[:, order]
            return result.vertices, j

        with torch.no_grad():
            _, b = forward()
            b = b - b[:, 0:1]
            p = b[:, [0, 5, 9, 13, 17]].numpy()
            q = target[:, [0, 5, 9, 13, 17]].numpy()
            cov = np.einsum('bni,bnj->bij', p, q)
            u, s, vh = np.linalg.svd(cov)
            rot = vh.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
            for i in range(n):
                if np.linalg.det(rot[i]) < 0:
                    vh[i, -1] *= -1
                    rot[i] = vh[i].T @ u[i].T
                ori[i] = torch.tensor(cv2.Rodrigues(rot[i])[0][:, 0])
        opt = torch.optim.Adam([pose, ori, beta], lr=.03)
        for step in range(STEPS):
            opt.zero_grad()
            vv, j = forward()
            jj = j - j[:, 0:1]
            per = (jj - target).square().mean((1, 2))
            loss = (per * mask).sum() / mask.sum().clamp_min(1) + pose.square().mean() * 1e-7 + beta.square().mean() * 1e-8
            loss.backward()
            opt.step()
            with torch.no_grad():
                beta.clamp_(-3, 3)
        with torch.no_grad():
            vv, j = forward()
            root = j[:, 0:1]
            vv = vv - root
            j = j - root
            err = torch.linalg.vector_norm(j - target, dim=-1).numpy() * 1000
            errors.append(dict(mean_mm=float(err[valid[:, h]].mean()) if valid[:, h].any() else None,
                               max_mm=float(err[valid[:, h]].max()) if valid[:, h].any() else None,
                               absent_hand=not bool(valid[:, h].any())))
            if errors[-1]['mean_mm'] is not None and errors[-1]['mean_mm'] >= 20:
                raise ValueError(f'MANO fit error too large: {errors[-1]}')
            vertices.append(vv.numpy() + positions[:, h, 0:1])
            fitted.append(j.numpy() + positions[:, h, 0:1])
            params.append((pose.detach().numpy(), ori.detach().numpy(), beta.detach().numpy()))
        if log:
            log(f'MANO fit hand {h}: {errors[-1]}')
    return dict(source_frame_indices=np.array(ids), vertices=np.stack(vertices, 1), joints=np.stack(fitted, 1),
                hand_valid=valid, pose=np.stack([p[0] for p in params]), orient=np.stack([p[1] for p in params]),
                betas=np.stack([p[2] for p in params]), fit_error_mm=errors)


def load_or_fit(path, joints, hand_valid, mano_dir, threads=6, log=print):
    """Cached fit: reuse ``path`` (.npz) if it exists, else fit and save it."""
    path = Path(path)
    if path.is_file():
        with np.load(path) as z:
            return {k: z[k] for k in ('source_frame_indices', 'vertices', 'hand_valid')}
    r = fit(joints, hand_valid, mano_dir, threads, log)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part.npz')
    np.savez_compressed(part, **{k: v for k, v in r.items() if k != 'fit_error_mm'})
    part.replace(path)
    return r
