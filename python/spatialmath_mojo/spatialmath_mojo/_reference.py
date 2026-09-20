"""Vendored pure-Python reference for batched SE(3)/SO(3) pose algebra.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``SPATIALMATH_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of
the group definitions of SE(3) and SO(3) (compose = matrix product; SE(3)
inverse = [R^T, -R^T t; 0 1]; SO(3) inverse = transpose; point transform =
homogeneous product + Euclidean division), written to be observably
identical to the reference software stack the differential suite checks
against (spatialmath-python's per-pose ``x @ y`` / ``trinv`` / ``h2e(T @
e2h(p))`` dispatch): the fallback performs the same per-pose NumPy calls in
the same IEEE-754 float64 operation order, so it agrees with that oracle to
within a few ulps in practice (frequently bit-exact).

Only the per-pose numerical kernels live here; input validation, batch
broadcasting, and output shaping are shared with the native path in
`spatialmath_mojo.core`, so the two backends can never disagree about
anything but the kernel itself.
"""

from __future__ import annotations

import numpy as np


def compose(a: np.ndarray, b: np.ndarray, n: int, so3: bool) -> np.ndarray:
    """Per-pose matrix products: out[p] = a[p] @ b[p] (k ascending).

    ``a``/``b`` are (1, d, d) or (n, d, d) float64 batches (d = 4 for SE3,
    3 for SO3); a length-1 operand is broadcast, matching the reference
    toolbox's singleton rules. ``so3`` is part of the shared backend
    signature; the matmul is dimension-agnostic.
    """
    del so3  # signature shared with the native backend
    d = a.shape[1]
    out = np.empty((n, d, d), dtype=np.float64)
    for p in range(n):
        out[p] = a[0 if a.shape[0] == 1 else p] @ b[0 if b.shape[0] == 1 else p]
    return out


def inverse(t: np.ndarray, so3: bool) -> np.ndarray:
    """Per-pose inverse: SO(3) transpose; SE(3) structured inverse.

    The SE(3) branch reads only R = T[:3, :3] and t = T[:3, 3] and writes
    [R^T, -R^T t; 0 0 0 1] into a zero matrix — the input's last row is
    never read, exactly like the reference `trinv`. The negation is applied
    after the R^T @ t product, as in the reference.
    """
    n, d = t.shape[0], t.shape[1]
    out = np.empty((n, d, d), dtype=np.float64)
    for p in range(n):
        x = t[p]
        if so3:
            out[p] = x.T
        else:
            r = x[:3, :3]
            ti = np.zeros((4, 4), dtype=np.float64)
            ti[:3, :3] = r.T
            ti[:3, 3] = -r.T @ x[:3, 3]
            ti[3, 3] = 1.0
            out[p] = ti
    return out


def transform(t: np.ndarray, points: np.ndarray, so3: bool) -> np.ndarray:
    """Per-pose point transform: out[i] = t[i] applied to every point.

    SE(3): out[i] = h2e(t[i] @ e2h(P)) — append a row of ones, 4x4 product,
    divide by the last homogeneous row (the division is performed even when
    it is exactly 1, like the reference `h2e`). SO(3): out[i] = t[i] @ P.T.
    ``points`` is (m, 3) row-major; the result is (n, m, 3).
    """
    n = t.shape[0]
    m = points.shape[0]
    out = np.empty((n, m, 3), dtype=np.float64)
    if so3:
        for i in range(n):
            out[i] = (t[i] @ points.T).T
    else:
        homogeneous = np.vstack([points.T, np.ones((1, m), dtype=np.float64)])
        for i in range(n):
            w = t[i] @ homogeneous
            out[i] = (w[:3, :] / w[3, :][np.newaxis, :]).T
    return out
