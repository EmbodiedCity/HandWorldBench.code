"""Spatial-mapping helpers shared by input binding and the observation backends."""
import numpy as np


def affine_matrix_3x3(matrix):
    """Return an affine spatial mapping as a 3x3 float array.

    Submissions may declare the matrix as 2x3 or 3x3; a 2x3 matrix gets the homogeneous row [0, 0, 1].
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]])
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError('Spatial affine matrix must be a finite 2x3 or 3x3 array')
    if not np.allclose(m[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-10):
        raise ValueError('Spatial affine matrix must have the homogeneous row [0, 0, 1]')
    return m
