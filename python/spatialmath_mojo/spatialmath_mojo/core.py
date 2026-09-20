"""Public batch pose algebra API: compose, inverse, transform — arrays in, float64 out.

Each function first tries the native Mojo kernel and transparently falls back
to the vendored pure-Python reference when the kernel is unavailable; both
backends are covered by the same differential suite against the
spatialmath-python oracle (documented tolerance 1e-10; measured agreement is
at the 1e-15 level — ulp differences only, from fused multiply-add
contraction in compiled code, the oracle's own BLAS included).

Semantics matched to the oracle (spatialmath.baseposematrix.__mul__ /
baseposelist._binop / pose3d.SE3.inv / base.trinv / base.h2e):

* compose: full per-pose matrix product ``A[p] @ B[p]`` (4x4 for SE(3)
  batches, 3x3 for SO(3) batches), including the bottom row — not the
  structured shortcut. Batch broadcasting mirrors the toolbox: a singleton
  operand composes against every pose of the other.
* inverse: SE(3) uses the structured inverse ``[R^T, -R^T t; 0 0 0 1]``; the
  input's bottom row is never read and the output's is exactly
  ``[0, 0, 0, 1]`` (the reference ``trinv`` builds the result from a zero
  matrix, so this is observable-identical for valid SE(3) input). SO(3) is
  the exact transpose.
* transform: SE(3) applies ``h2e(T @ e2h(p))`` per (pose, point) pair —
  append a homogeneous 1, 4x4 product, divide by the last component (the
  division happens even when it is exactly 1). SO(3) applies ``R @ p``.
"""

from __future__ import annotations

import numpy as np

from spatialmath_mojo import _native, _reference


def _as_pose_batch(name: str, value) -> tuple[np.ndarray, bool]:
    """Coerce array-like to a contiguous float64 (n, d, d) pose batch.

    Accepts a single (d, d) pose (n = 1) or a (n, d, d) batch with d in
    {3, 4}. Returns (batch, was_2d).
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a (d, d) or (n, d, d) array-like of floats with d in "
            f"{{3, 4}}: {exc}"
        ) from exc
    was_2d = arr.ndim == 2
    if was_2d:
        if arr.shape[0] != arr.shape[1] or arr.shape[0] not in (3, 4):
            raise ValueError(
                f"{name} must be square with d in {{3, 4}}, got shape {arr.shape!r}"
            )
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[1] != arr.shape[2] or arr.shape[1] not in (3, 4):
            raise ValueError(
                f"{name} must have shape (n, d, d) with d in {{3, 4}}, got "
                f"{arr.shape!r}"
            )
    else:
        raise ValueError(
            f"{name} must be a (d, d) or (n, d, d) array, got shape {arr.shape!r}"
        )
    return np.ascontiguousarray(arr), was_2d


def _as_points(name: str, value) -> tuple[np.ndarray, bool]:
    """Coerce array-like to a contiguous float64 (m, 3) point array.

    Accepts a single (3,) point or an (m, 3) array of row-major points.
    Returns (points, was_1d).
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a (3,) or (m, 3) array-like of floats: {exc}"
        ) from exc
    was_1d = arr.ndim == 1
    if was_1d:
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must have shape (3,) or (m, 3), got {arr.shape!r}")
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape (3,) or (m, 3), got {arr.shape!r}")
    else:
        raise ValueError(f"{name} must have shape (3,) or (m, 3), got {arr.shape!r}")
    return np.ascontiguousarray(arr), was_1d


def compose(a, b) -> np.ndarray:
    """Batch pose composition ``a * b`` (SE(3) for 4x4 input, SO(3) for 3x3).

    Both operands are (d, d) or (n, d, d) float64 batches with the same d.
    A singleton operand broadcasts against the other's batch (the toolbox's
    ``1 * M`` / ``N * 1`` rules); two non-singleton batches must have equal
    length. Returns an (n, d, d) float64 array — or (d, d) when both
    operands were single poses.

    Oracle equivalents: ``SE3(A) * SE3(B)``, ``[x * y for x, y in
    zip(SE3(A), SE3(B))]`` (and the SO3 analogues).
    """
    a_arr, a_2d = _as_pose_batch("a", a)
    b_arr, b_2d = _as_pose_batch("b", b)
    if a_arr.shape[1] != b_arr.shape[1]:
        raise ValueError(
            f"a and b must have the same pose dimension, got {a_arr.shape[1]} and "
            f"{b_arr.shape[1]}"
        )
    na, nb = a_arr.shape[0], b_arr.shape[0]
    if na == 1:
        n = nb
    elif nb == 1 or na == nb:
        n = na
    else:
        raise ValueError(
            f"cannot broadcast batches of {na} and {nb} poses: lengths must match "
            "or one side must be a singleton"
        )
    so3 = a_arr.shape[1] == 3
    if n == 0:
        out = np.empty((0, a_arr.shape[1], a_arr.shape[2]), dtype=np.float64)
    else:
        try:
            out = _native.compose(a_arr, b_arr, n, so3)
        except _native.NativeUnavailable:
            out = _reference.compose(a_arr, b_arr, n, so3)
    if a_2d and b_2d:
        return out[0]
    return out


def inverse(t) -> np.ndarray:
    """Batch pose inverse: SE(3) ``[R^T, -R^T t; 0 0 0 1]``, SO(3) transpose.

    Accepts a (d, d) pose or an (n, d, d) batch and returns the same shape
    in float64. For SE(3) the input's bottom row is never read (the
    structured inverse, exactly like the oracle's ``trinv``); the input
    must be a valid SE(3) matrix for the result to be meaningful.

    Oracle equivalents: ``SE3(T).inv()`` / ``SO3(R).inv()``.
    """
    t_arr, was_2d = _as_pose_batch("t", t)
    so3 = t_arr.shape[1] == 3
    try:
        out = _native.inverse(t_arr, so3)
    except _native.NativeUnavailable:
        out = _reference.inverse(t_arr, so3)
    if was_2d:
        return out[0]
    return out


def transform(t, points) -> np.ndarray:
    """Batch point transform: apply each pose to each point.

    ``t`` is a (d, d) pose or (n, d, d) batch (SE(3) for d = 4, SO(3) for
    d = 3); ``points`` is a single (3,) point or an (m, 3) row-major point
    array. Returns float64:

    * (3,)      — single pose, single point
    * (m, 3)    — single pose, m points (out[j] = T applied to points[j])
    * (n, 3)    — n poses, single point (out[i] = T[i] applied to point)
    * (n, m, 3) — n poses, m points (out[i, j] = T[i] applied to points[j])

    Oracle equivalents: ``SE3(T) * p`` (vector or 3xM matrix), ``SE3
    pose-list * vector``, and the SO3 analogues.
    """
    t_arr, t_2d = _as_pose_batch("t", t)
    p_arr, p_1d = _as_points("points", points)
    so3 = t_arr.shape[1] == 3
    try:
        out = _native.transform(t_arr, p_arr, so3)
    except _native.NativeUnavailable:
        out = _reference.transform(t_arr, p_arr, so3)
    if t_2d and p_1d:
        return out[0, 0]
    if t_2d:
        return out[0]
    if p_1d:
        return out[:, 0, :]
    return out
