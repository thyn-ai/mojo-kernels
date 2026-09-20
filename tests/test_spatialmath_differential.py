"""Differential tests: spatialmath_mojo must match spatialmath-python pose-for-pose.

Run twice by `scripts/test_all_spatialmath.sh`: once against the native Mojo
kernel and once with SPATIALMATH_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle within 1e-10 everywhere
(in practice the agreement is bit-exact or within a few ulps, ~1e-15).

The oracle is the published PyPI package `spatialmath-python` 1.1.18 (Peter
Corke's robotics toolbox, rai-opensource/spatialmath-python), exercised
through its object API: per-pose ``x * y`` dispatch in
``baseposelist._binop``, ``SE3.inv`` / ``SO3.inv`` (``trinv`` / transpose),
and the ``pose * vector`` / ``pose * matrix`` transformation paths of
``baseposematrix.__mul__``. This package accelerates exactly those per-pose
numpy operations over whole batches.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible on any
machine.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

sm = pytest.importorskip("spatialmath", reason="differential oracle")
smb = pytest.importorskip("spatialmath.base", reason="differential oracle")
from spatialmath import SE3, SO3  # noqa: E402

import spatialmath_mojo  # noqa: E402

ORACLE_VERSION = "1.1.18"
ATOL = 1e-10  # documented tolerance; measured agreement is ~1e-15


# ---------------------------------------------------------------- fixtures


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix from a unit axis and angle (clean-room fixture math)."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(theta)
        + (1.0 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * k
    )


def random_so3(seed: int, n: int) -> np.ndarray:
    """Deterministic batch of proper rotation matrices, shape (n, 3, 3)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3))
    for i in range(n):
        out[i] = _rodrigues(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
    return out


def random_se3(seed: int, n: int) -> np.ndarray:
    """Deterministic batch of proper SE(3) matrices, shape (n, 4, 4)."""
    rng = np.random.default_rng(seed + 1)
    rotations = random_so3(seed, n)
    out = np.repeat(np.eye(4)[np.newaxis, :, :], n, axis=0)
    out[:, :3, :3] = rotations
    out[:, :3, 3] = rng.normal(0.0, 5.0, size=(n, 3))
    return out


def random_points(seed: int, m: int) -> np.ndarray:
    """Deterministic (m, 3) row-major point cloud."""
    rng = np.random.default_rng(seed + 2)
    return rng.normal(0.0, 10.0, size=(m, 3))


def se3_list(batch: np.ndarray) -> SE3:
    return SE3([batch[i] for i in range(batch.shape[0])])


def so3_list(batch: np.ndarray) -> SO3:
    return SO3([batch[i] for i in range(batch.shape[0])])


def assert_parity(ours: np.ndarray, ref: np.ndarray) -> None:
    """Shape and dtype checks plus values within the documented tolerance."""
    ref_arr = np.asarray(ref, dtype=np.float64)
    assert isinstance(ours, np.ndarray), f"expected np.ndarray, got {type(ours)}"
    assert ours.dtype == np.float64, f"expected float64, got {ours.dtype}"
    assert ours.shape == ref_arr.shape, f"shape {ours.shape} != oracle {ref_arr.shape}"
    np.testing.assert_allclose(ours, ref_arr, rtol=0, atol=ATOL)


# ---------------------------------------------------------------- guards


def test_oracle_version_pinned():
    assert sm.__version__ == ORACLE_VERSION


def test_expected_backend():
    info = spatialmath_mojo.backend_info()
    if os.environ.get("SPATIALMATH_MOJO_DISABLE_NATIVE") == "1":
        assert info["native_available"] is False
    else:
        # The native run requires a built kernel (kernels/spatialmath/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == 1


# ---------------------------------------------------------------- compose


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_se3_compose_batch_parity(n):
    a, b = random_se3(101, n), random_se3(202, n)
    ours = spatialmath_mojo.compose(a, b)
    ref = se3_list(a) * se3_list(b)
    assert_parity(ours, np.stack([x.A for x in ref]))


@pytest.mark.parametrize("n", [2, 33])
def test_se3_compose_broadcast_parity(n):
    a, b = random_se3(103, n), random_se3(104, n)
    # singleton * batch and batch * singleton, oracle 1*M / N*1 rules
    left = spatialmath_mojo.compose(a[0], b)
    ref_left = SE3(a[0]) * se3_list(b)
    assert_parity(left, np.stack([x.A for x in ref_left]))
    right = spatialmath_mojo.compose(a, b[0])
    ref_right = se3_list(a) * SE3(b[0])
    assert_parity(right, np.stack([x.A for x in ref_right]))


def test_se3_compose_singleton_returns_2d():
    a, b = random_se3(105, 1), random_se3(106, 1)
    ours = spatialmath_mojo.compose(a[0], b[0])
    ref = SE3(a[0]) * SE3(b[0])
    assert ours.shape == (4, 4)
    assert_parity(ours, ref.A)


@pytest.mark.parametrize("n", [1, 3, 64])
def test_so3_compose_batch_parity(n):
    a, b = random_so3(107, n), random_so3(108, n)
    ours = spatialmath_mojo.compose(a, b)
    ref = so3_list(a) * so3_list(b)
    assert_parity(ours, np.stack([x.A for x in ref]))


def test_so3_compose_broadcast_and_singleton():
    a, b = random_so3(109, 9), random_so3(110, 9)
    assert_parity(
        spatialmath_mojo.compose(a[0], b),
        np.stack([x.A for x in SO3(a[0]) * so3_list(b)]),
    )
    ours = spatialmath_mojo.compose(a[0], b[0])
    assert ours.shape == (3, 3)
    assert_parity(ours, (SO3(a[0]) * SO3(b[0])).A)


# ---------------------------------------------------------------- inverse


@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_se3_inverse_parity(n):
    a = random_se3(111, n)
    ours = spatialmath_mojo.inverse(a)
    ref = se3_list(a).inv()
    assert_parity(ours, np.stack([x.A for x in ref]))


def test_se3_inverse_singleton_returns_2d():
    a = random_se3(113, 1)
    ours = spatialmath_mojo.inverse(a[0])
    assert ours.shape == (4, 4)
    assert_parity(ours, SE3(a[0]).inv().A)


@pytest.mark.parametrize("n", [1, 3, 64])
def test_so3_inverse_parity(n):
    r = random_so3(115, n)
    ours = spatialmath_mojo.inverse(r)
    ref = so3_list(r).inv()
    assert_parity(ours, np.stack([x.A for x in ref]))


def test_se3_inverse_ignores_bottom_row():
    # trinv reads only R and t: a garbage bottom row must not leak into the
    # result (the oracle's trinv builds from a zero matrix). spatialmath SE3
    # cannot hold such a matrix, so compare against base.trinv semantics.
    a = random_se3(117, 3)
    a[:, 3, :] = np.array([[9.0, 8.0, 7.0, 6.0]] * 3)
    ours = spatialmath_mojo.inverse(a)
    ref = np.stack([smb.trinv(_proper(a[i])) for i in range(3)])
    assert_parity(ours, ref)
    np.testing.assert_allclose(
        ours[:, 3, :],
        np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (3, 4)),
        rtol=0,
        atol=0.0,
    )


def _proper(t: np.ndarray) -> np.ndarray:
    out = t.copy()
    out[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


# ---------------------------------------------------------------- transform


@pytest.mark.parametrize("n", [1, 2, 17, 64])
def test_se3_transform_pose_batch_times_vector_parity(n):
    a = random_se3(121, n)
    v = np.array([1.5, -2.0, 3.25])
    ours = spatialmath_mojo.transform(a, v)
    # oracle pose-list * vector -> (3, n); our rows are per-pose points
    ref = se3_list(a) * v
    assert ours.shape == (n, 3)
    assert_parity(ours, ref.T)


@pytest.mark.parametrize("m", [1, 2, 50])
def test_se3_transform_singleton_times_points_parity(m):
    a = random_se3(123, 1)[0]
    p = random_points(124, m)
    ours = spatialmath_mojo.transform(a, p)
    # oracle singleton * (3, m) matrix -> (3, m)
    ref = SE3(a) * p.T
    assert ours.shape == (m, 3)
    assert_parity(ours, ref.T)


def test_se3_transform_singleton_times_vector():
    a = random_se3(125, 1)[0]
    v = np.array([0.5, 0.6, -0.7])
    ours = spatialmath_mojo.transform(a, v)
    ref = smb.h2e(SE3(a).A @ smb.e2h(smb.getvector(v, out="col"))).flatten()
    assert ours.shape == (3,)
    assert_parity(ours, ref)


def test_se3_transform_batch_times_points_parity():
    # The oracle has no single (n, m) call; the expected value is the
    # per-pose oracle result stacked — still the real oracle per pose.
    n, m = 6, 11
    a = random_se3(127, n)
    p = random_points(128, m)
    ours = spatialmath_mojo.transform(a, p)
    assert ours.shape == (n, m, 3)
    ref = np.stack([(SE3(a[i]) * p.T).T for i in range(n)])
    assert_parity(ours, ref)


@pytest.mark.parametrize("n", [1, 3, 64])
def test_so3_transform_pose_batch_times_vector_parity(n):
    r = random_so3(131, n)
    v = np.array([-1.0, 0.25, 2.0])
    ours = spatialmath_mojo.transform(r, v)
    ref = so3_list(r) * v
    assert ours.shape == (n, 3)
    assert_parity(ours, ref.T)


def test_so3_transform_singleton_times_points_and_vector():
    r = random_so3(133, 1)[0]
    p = random_points(134, 23)
    assert_parity(spatialmath_mojo.transform(r, p), (SO3(r) * p.T).T)
    v = np.array([3.0, -1.0, 0.5])
    ours = spatialmath_mojo.transform(r, v)
    assert ours.shape == (3,)
    assert_parity(ours, (SO3(r).A @ smb.getvector(v, out="col")).flatten())


def test_so3_transform_batch_times_points_parity():
    n, m = 5, 9
    r = random_so3(135, n)
    p = random_points(136, m)
    ours = spatialmath_mojo.transform(r, p)
    assert ours.shape == (n, m, 3)
    ref = np.stack([(SO3(r[i]) * p.T).T for i in range(n)])
    assert_parity(ours, ref)


def test_transform_homogeneous_division_is_applied():
    # h2e divides by the last homogeneous component. A (valid-structured)
    # pose gives w3 == 1, but a caller-supplied matrix with a scaled bottom
    # row must still divide — matching the oracle's h2e on raw matrices.
    a = random_se3(137, 2)
    a[:, 3, :] = np.array([[0.0, 0.0, 0.0, 2.0]] * 2)  # w3 == 2
    v = np.array([1.0, 1.0, 1.0])
    ours = spatialmath_mojo.transform(a, v)
    vh = smb.e2h(smb.getvector(v, out="col"))
    ref = np.stack([smb.h2e(a[i] @ vh).flatten() for i in range(2)])
    assert_parity(ours, ref)


# ------------------------------------------------------------ edge cases


def test_empty_batches():
    assert spatialmath_mojo.compose(np.empty((0, 4, 4)), np.empty((0, 4, 4))).shape == (0, 4, 4)
    assert spatialmath_mojo.inverse(np.empty((0, 3, 3))).shape == (0, 3, 3)
    assert spatialmath_mojo.transform(np.empty((0, 4, 4)), np.zeros((5, 3))).shape == (0, 5, 3)
    assert spatialmath_mojo.transform(np.eye(4), np.empty((0, 3))).shape == (0, 3)


def test_array_like_inputs():
    a = random_se3(141, 1)[0]
    as_lists = [[float(x) for x in row] for row in a]
    ours = spatialmath_mojo.inverse(as_lists)
    assert ours.shape == (4, 4) and ours.dtype == np.float64
    assert_parity(ours, SE3(a).inv().A)
    v = [1.0, 2.0, 3.0]
    assert spatialmath_mojo.transform(as_lists, v).shape == (3,)


def test_input_validation():
    a = random_se3(143, 2)
    r = random_so3(144, 2)
    with pytest.raises(ValueError):
        spatialmath_mojo.compose(a, r)  # mixed SE3/SO3
    with pytest.raises(ValueError):
        spatialmath_mojo.compose(a, random_se3(145, 3))  # 2 vs 3, no singleton
    with pytest.raises(ValueError):
        spatialmath_mojo.inverse(np.zeros((2, 5, 5)))  # bad dimension
    with pytest.raises(ValueError):
        spatialmath_mojo.inverse(np.zeros((4, 4, 4, 4)))  # not 2-D/3-D
    with pytest.raises(ValueError):
        spatialmath_mojo.transform(a, np.zeros((3, 2)))  # points not (m, 3)
    with pytest.raises(ValueError):
        spatialmath_mojo.transform(a, np.zeros(4))  # point not (3,)
    with pytest.raises(ValueError):
        spatialmath_mojo.compose(object(), a)  # not array-like floats
    with pytest.raises(ValueError):
        spatialmath_mojo.inverse(np.zeros((2, 4, 3)))  # not square


# ------------------------------------------------------- cross-backend unity


def test_native_and_fallback_agree(monkeypatch):
    # The Mojo kernel and the vendored fallback execute the same IEEE-754
    # float64 operations in the same order, but compiled code (the Mojo
    # kernel, like the oracle's own BLAS) may contract a*b+c into a fused
    # multiply-add while the fallback's per-pose numpy calls follow the
    # oracle's BLAS paths — so the backends can differ by ~1 ulp (~1e-15).
    # They must still agree far below the documented 1e-10 tolerance.
    a, b = random_se3(151, 40), random_se3(152, 40)
    r, s = random_so3(153, 40), random_so3(154, 40)
    p = random_points(155, 30)
    results = {}
    for disabled in (None, "1"):
        if disabled is None:
            monkeypatch.delenv("SPATIALMATH_MOJO_DISABLE_NATIVE", raising=False)
        else:
            monkeypatch.setenv("SPATIALMATH_MOJO_DISABLE_NATIVE", disabled)
        results[disabled] = (
            spatialmath_mojo.compose(a, b),
            spatialmath_mojo.compose(a[0], b),
            spatialmath_mojo.inverse(a),
            spatialmath_mojo.transform(a, p),
            spatialmath_mojo.compose(r, s),
            spatialmath_mojo.inverse(r),
            spatialmath_mojo.transform(r, p),
        )
    for native_arr, fallback_arr in zip(results[None], results["1"]):
        np.testing.assert_allclose(native_arr, fallback_arr, rtol=0, atol=1e-12)


# ------------------------------------------------------ agreement measurement


def test_measured_agreement_well_below_tolerance(capsys):
    """Report the real max abs diff over a battery; it must sit far under 1e-10."""
    worst = 0.0

    def track(ours, ref):
        nonlocal worst
        worst = max(worst, float(np.max(np.abs(ours - np.asarray(ref, dtype=np.float64)))))

    for seed in (161, 162, 163):
        a, b = random_se3(seed, 128), random_se3(seed + 10, 128)
        r, s = random_so3(seed + 20, 128), random_so3(seed + 30, 128)
        p = random_points(seed + 40, 64)
        v = np.array([1.0, -2.0, 0.5])
        track(
            spatialmath_mojo.compose(a, b),
            np.stack([x.A for x in se3_list(a) * se3_list(b)]),
        )
        track(spatialmath_mojo.inverse(a), np.stack([x.A for x in se3_list(a).inv()]))
        track(spatialmath_mojo.transform(a, v), (se3_list(a) * v).T)
        track(
            spatialmath_mojo.transform(a, p),
            np.stack([(SE3(a[i]) * p.T).T for i in range(a.shape[0])]),
        )
        track(
            spatialmath_mojo.compose(r, s),
            np.stack([x.A for x in so3_list(r) * so3_list(s)]),
        )
        track(spatialmath_mojo.inverse(r), np.stack([x.A for x in so3_list(r).inv()]))
        track(spatialmath_mojo.transform(r, v), (so3_list(r) * v).T)
        track(
            spatialmath_mojo.transform(r, p),
            np.stack([(SO3(r[i]) * p.T).T for i in range(r.shape[0])]),
        )
    with capsys.disabled():
        print(f"\nmeasured max|diff| over agreement battery: {worst:.3e}")
    assert worst <= 1e-12, f"agreement {worst:.3e} worse than the expected ulp level"
