"""Wuji hand forward kinematics and surface meshes.

The kinematic tree, joint axes and the 21 landmark sites (``robot21_XX``) are read from ``assets/{left,right}_landmarks.xml``
(the Wuji MJCF model with the 21 HandWorldBench landmarks added). The link surfaces are the binary STL files of the
official Wuji hand description (https://github.com/wuji-technology/wuji-hand-description, ``meshes/{left,right}/``).

Joint vector ``q`` (20 values): finger ``f`` in 1..5, joint ``j`` in 1..4 -> ``q[(f - 1) * 4 + j - 1]`` (radians).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parent / 'assets'


def vec(s, default=(0, 0, 0)):
    return np.asarray(s.split() if s else default, float)


def quat(s):
    return Rotation.from_quat(vec(s, (1, 0, 0, 0))[[1, 2, 3, 0]]).as_matrix()


class Robot:
    def __init__(self, side, mesh_root=None, assets=ASSETS, meshes=True):
        """side: 'left' or 'right'. mesh_root: the ``meshes`` folder of wuji-hand-description (needed when meshes=True)."""
        self.side = side
        self.nodes = []
        tree = ET.parse(Path(assets) / f'{side}_landmarks.xml').getroot()
        if meshes:
            if mesh_root is None:
                raise ValueError('mesh_root (wuji-hand-description/meshes) is required for surface meshes')
            meshdir = Path(mesh_root) / side
            if not meshdir.is_dir():
                raise FileNotFoundError(f'Wuji hand meshes not found: {meshdir}')
        defs = {e.get('name'): e for e in tree.findall('./asset/mesh')}

        def visit(body, parent):
            idx = len(self.nodes)
            j = body.find('joint')
            name = j.get('name') if j is not None else None
            qi = (int(name.split('finger')[1].split('_')[0]) - 1) * 4 + int(name.rsplit('joint', 1)[1]) - 1 if name else -1
            node = dict(parent=parent, pos=vec(body.get('pos')), rot=quat(body.get('quat')), qi=qi,
                        axis=vec(j.get('axis'), (0, 0, 1)) if j is not None else np.zeros(3),
                        pivot=vec(j.get('pos')) if j is not None else np.zeros(3), tri=[], sites=[])
            if meshes:
                for e in body.findall('geom'):
                    if e.get('group') != '1' or not e.get('mesh'):
                        continue
                    d = defs[e.get('mesh')]
                    p = meshdir / d.get('file')
                    raw = p.read_bytes()
                    n = int.from_bytes(raw[80:84], 'little')
                    assert len(raw) == 84 + 50 * n, f'not a binary STL: {p}'
                    dt = np.dtype([('norm', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
                    tri = np.frombuffer(raw, offset=84, dtype=dt, count=n)['v'].astype(float) * vec(d.get('scale'), (1, 1, 1))
                    tri = tri @ quat(e.get('quat')).T + vec(e.get('pos'))
                    node['tri'].append(tri)
            for s in body.findall('site'):
                if s.get('name', '').startswith('robot21_'):
                    node['sites'].append((int(s.get('name').split('_')[-1]), vec(s.get('pos'))))
            self.nodes.append(node)
            for child in body.findall('body'):
                visit(child, idx)

        visit(tree.find('worldbody/body'), -1)

    def forward(self, q):
        """Link rotations/translations and the 21 landmarks in the palm frame, for a batch of joint vectors."""
        q = np.atleast_2d(q)
        N = len(q)
        Rs = []
        ts = []
        points = np.zeros((N, 21, 3))
        for n in self.nodes:
            R = np.broadcast_to(n['rot'], (N, 3, 3)).copy()
            t = np.broadcast_to(n['pos'], (N, 3)).copy()
            if n['qi'] >= 0:
                a = Rotation.from_rotvec(q[:, n['qi'], None] * n['axis']).as_matrix()
                t = t + (R @ n['pivot']) - (R @ a @ n['pivot'])
                R = R @ a
            if n['parent'] >= 0:
                t = np.einsum('nij,nj->ni', Rs[n['parent']], t) + ts[n['parent']]
                R = Rs[n['parent']] @ R
            Rs.append(R)
            ts.append(t)
            for idx, pt in n['sites']:
                points[:, idx] = R @ pt + t
        return Rs, ts, points

    def pose(self, q):
        """Posed surface triangles (T, 3, 3) and the 21 landmarks (21, 3), palm frame, for one joint vector."""
        Rs, ts, points = self.forward(q)
        parts = []
        for i, n in enumerate(self.nodes):
            for tri in n['tri']:
                parts.append(tri @ Rs[i][0].T + ts[i][0])
        return np.concatenate(parts), points[0]
