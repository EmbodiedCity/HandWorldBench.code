"""Recover Wuji joint angles and palm pose from the 21 published HandsOn joints.

The HandsOn joints are the ``robot21`` landmarks of the Wuji hand model (``robot.Robot.forward``) placed in the camera
frame. The surface renderer needs the 20 joint angles and the palm-to-camera transform, which are recovered exactly:

1. palm pose: Kabsch fit of the six rigid palm landmarks (wrist and the five finger bases) of the zero pose;
2. joint angles: Gauss-Newton on the 15 moving landmarks in the palm frame (finite-difference Jacobian), restarted
   from a neighbouring solved frame or a random pose for frames that do not converge.

On the benchmark the landmark residual is below 1e-12 m.
"""
import numpy as np

PALM_SITES = [0, 1, 5, 9, 13, 17]


def kabsch(P, Q):
    pc = P.mean(0)
    qc = Q.mean(1)
    H = np.einsum('ki,nkj->nij', P - pc, Q - qc[:, None])
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('nji,nkj->nik', Vt, U)))
    Dm = np.tile(np.eye(3), (len(Q), 1, 1))
    Dm[:, 2, 2] = d
    R = np.einsum('nji,njk,nlk->nil', Vt, Dm, U)
    return R, qc - np.einsum('nij,j->ni', R, pc)


def _solve(robot, local, q, moving, iters=60, tol=1e-12, eps=1e-7):
    N = len(q)

    def res(q):
        return (robot.forward(q)[2][:, moving] - local[:, moving]).reshape(N, -1)

    r = res(q)
    for it in range(iters):
        J = np.stack([(res(q + eps * np.eye(20)[k]) - r) / eps for k in range(20)], -1)
        A = np.einsum('nri,nrj->nij', J, J) + 1e-12 * np.eye(20)
        g = np.einsum('nri,nr->ni', J, r)
        q2 = q - np.linalg.solve(A, g[..., None])[..., 0]
        r2 = res(q2)
        better = np.abs(r2).max(1) <= np.abs(r).max(1)
        q[better] = q2[better]
        r[better] = r2[better]
        if np.abs(r).max() < tol:
            break
    return q, np.abs(r).max(1)


def recover(robot, joints, tol=1e-6, rounds=20):
    """joints: (N, 21, 3) camera-space landmarks of one hand.

    Returns q (N, 20) joint angles, T (N, 4, 4) palm-to-camera transforms and the per-frame max landmark residual (m).
    Frames with non-finite joints (hand not tracked) get q = 0, T = identity and an infinite residual."""
    joints = np.asarray(joints, np.float64)
    finite = np.isfinite(joints).all((1, 2))
    if finite.all():
        return _recover(robot, joints, tol, rounds)
    q = np.zeros((len(joints), 20))
    T = np.tile(np.eye(4), (len(joints), 1, 1))
    err = np.full(len(joints), np.inf)
    if finite.any():
        q[finite], T[finite], err[finite] = _recover(robot, joints[finite], tol, rounds)
    return q, T, err


def _recover(robot, joints, tol, rounds):
    N = len(joints)
    _, _, p0 = robot.forward(np.zeros(20))
    R, t = kabsch(p0[0, PALM_SITES], joints[:, PALM_SITES])
    local = np.einsum('nji,nkj->nki', R, joints - t[:, None])
    moving = [i for i in range(21) if i not in PALM_SITES]
    q, err = _solve(robot, local, np.zeros((N, 20)), moving)
    rng = np.random.default_rng(0)
    for k in range(rounds):
        bad = np.flatnonzero(err > tol)
        if not len(bad):
            break
        good = np.flatnonzero(err <= tol)
        if len(good) and k % 2 == 0:
            init = q[good[np.abs(good[None] - bad[:, None]).argmin(1)]]
        else:
            init = rng.uniform(-1.5, 1.5, (len(bad), 20))
        qb, eb = _solve(robot, local[bad], init.copy(), moving)
        imp = eb < err[bad]
        q[bad[imp]] = qb[imp]
        err[bad[imp]] = eb[imp]
    q = (q + np.pi) % (2 * np.pi) - np.pi
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return q, T, err
