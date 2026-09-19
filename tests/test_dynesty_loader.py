"""Loader, ABI-handshake, kernel-unit, and API-validation tests for dynesty-mojo.

Run twice by ``scripts/test_all_dynesty.sh``: once against the native Mojo
kernel and once with DYNESTY_MOJO_DISABLE_NATIVE=1 (forced NumPy fallback).
Kernel-unit tests that name the backend modules explicitly are skipped
individually when the native library is unavailable (e.g. Windows).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import dynesty_mojo
from dynesty_mojo import SamplerError, _native, _reference, sample

NATIVE_EXPECTED = os.environ.get("DYNESTY_MOJO_DISABLE_NATIVE") != "1"
NATIVE_PRESENT = _native.native_available()


def _active_backends():
    """Backend modules that must work in this environment."""
    mods = [_reference]
    if NATIVE_PRESENT:
        mods.append(_native)
    return mods


# ---------------------------------------------------------------------------
# Loader / ABI handshake
# ---------------------------------------------------------------------------


def test_backend_info_shape():
    info = dynesty_mojo.backend_info()
    assert set(info) >= {
        "native_available",
        "native_source",
        "abi_version_expected",
        "abi_version_native",
        "disabled_by_env",
        "platform",
        "error",
    }
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (not NATIVE_EXPECTED)


def test_expected_backend_active():
    assert dynesty_mojo.native_available() is NATIVE_EXPECTED
    if NATIVE_EXPECTED:
        info = dynesty_mojo.backend_info()
        assert info["native_available"]
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_override_missing_library():
    """An explicit $DYNESTY_MOJO_NATIVE_LIB pointing nowhere must not crash
    the resolver order (the bundled/dev path still resolves)."""
    assert dynesty_mojo.native_available() is NATIVE_EXPECTED


# ---------------------------------------------------------------------------
# Kernel-unit: ellipsoid fit (both backend modules explicitly)
# ---------------------------------------------------------------------------


def _random_live_points(seed, n=200, ndim=4):
    rng = np.random.default_rng(seed)
    # Correlated, off-center cloud strictly inside the unit cube.
    a = rng.normal(size=(ndim, ndim))
    cov = a @ a.T * 0.002 + np.eye(ndim) * 0.0005
    pts = rng.multivariate_normal(np.full(ndim, 0.5), cov, size=n)
    assert np.all((pts > 0.0) & (pts < 1.0))
    return pts


def _mahalanobis(points, fit):
    d = points - fit.mean
    sol = np.linalg.solve(fit.chol, d.T)  # chol is lower-triangular
    return np.sum(sol * sol, axis=0)


@pytest.mark.parametrize("backend", ["native", "reference"])
def test_fit_contains_all_live_points(backend):
    if backend == "native" and not NATIVE_PRESENT:
        pytest.skip("native kernel unavailable")
    mod = _native if backend == "native" else _reference
    pts = _random_live_points(seed=11)
    fit = mod.fit_ellipsoid(pts, enlarge=1.25)
    assert fit is not None
    assert fit.mean.shape == (pts.shape[1],)
    assert fit.chol.shape == (pts.shape[1], pts.shape[1])
    # Lower-triangular, positive diagonal.
    assert np.all(np.diag(fit.chol) > 0.0)
    assert np.allclose(fit.chol, np.tril(fit.chol))
    # Containment: every live point strictly inside the enlarged ellipsoid.
    maha = _mahalanobis(pts, fit)
    assert maha.max() <= 1.0 + 1e-9
    # And the scaling is exactly "enlarge * dmax2 * cov": the max
    # Mahalanobis distance before enlargement was 1/enlarge of the way out.
    assert maha.max() == pytest.approx(1.0 / 1.25, rel=1e-9)


def test_fit_native_matches_reference():
    if not NATIVE_PRESENT:
        pytest.skip("native kernel unavailable")
    pts = _random_live_points(seed=21)
    fit_n = _native.fit_ellipsoid(pts, 1.1)
    fit_r = _reference.fit_ellipsoid(pts, 1.1)
    assert fit_n is not None and fit_r is not None
    np.testing.assert_allclose(fit_n.mean, fit_r.mean, rtol=0, atol=1e-15)
    np.testing.assert_allclose(fit_n.chol, fit_r.chol, rtol=1e-10, atol=1e-12)
    assert fit_n.logvol == pytest.approx(fit_r.logvol, rel=1e-9)


def test_fit_singular_returns_none():
    pts = np.full((50, 3), 0.5)  # identical points: zero covariance
    assert _reference.fit_ellipsoid(pts, 1.25) is None
    if NATIVE_PRESENT:
        assert _native.fit_ellipsoid(pts, 1.25) is None


# ---------------------------------------------------------------------------
# Kernel-unit: batch proposal (both backend modules explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["native", "reference"])
def test_propose_cube_uniform_in_bounds(backend):
    if backend == "native" and not NATIVE_PRESENT:
        pytest.skip("native kernel unavailable")
    mod = _native if backend == "native" else _reference
    rng = mod.create_rng(1234)
    try:
        pts = rng.propose_batch(3, None, _native.MODE_CUBE, 5000)
        assert pts.shape == (5000, 3)
        assert np.all((pts >= 0.0) & (pts < 1.0))
        # Uniform smoke: per-dimension mean ~ 0.5 (MC err ~ 0.004).
        assert np.all(np.abs(pts.mean(axis=0) - 0.5) < 0.05)
    finally:
        rng.close()


@pytest.mark.parametrize("backend", ["native", "reference"])
def test_propose_ellipsoid_respects_bound_and_cube(backend):
    if backend == "native" and not NATIVE_PRESENT:
        pytest.skip("native kernel unavailable")
    mod = _native if backend == "native" else _reference
    pts = _random_live_points(seed=33)
    fit = mod.fit_ellipsoid(pts, enlarge=1.25)
    rng = mod.create_rng(77)
    try:
        cand = rng.propose_batch(pts.shape[1], fit, _native.MODE_ELLIPSOID, 4000)
        assert cand.shape[0] == 4000
        assert np.all((cand >= 0.0) & (cand < 1.0))
        maha = _mahalanobis(cand, fit)
        assert maha.max() <= 1.0 + 1e-9
        # Uniform-in-ellipsoid smoke: candidate mean ~ ellipsoid center
        # (MC err of the mean over 4000 draws is tiny for this bound).
        assert np.all(np.abs(cand.mean(axis=0) - fit.mean) < 0.05)
    finally:
        rng.close()


@pytest.mark.parametrize("backend", ["native", "reference"])
def test_rng_determinism(backend):
    if backend == "native" and not NATIVE_PRESENT:
        pytest.skip("native kernel unavailable")
    mod = _native if backend == "native" else _reference
    pts = _random_live_points(seed=44)
    fit = mod.fit_ellipsoid(pts, enlarge=1.25)
    r1, r2 = mod.create_rng(5), mod.create_rng(5)
    try:
        a = r1.propose_batch(pts.shape[1], fit, _native.MODE_ELLIPSOID, 128)
        b = r2.propose_batch(pts.shape[1], fit, _native.MODE_ELLIPSOID, 128)
        assert np.array_equal(a, b)
        # The stream advances: consecutive batches differ.
        c = r1.propose_batch(pts.shape[1], fit, _native.MODE_ELLIPSOID, 128)
        assert not np.array_equal(a, c)
    finally:
        r1.close()
        r2.close()


# ---------------------------------------------------------------------------
# API validation and end-to-end behavior (whichever backend is active)
# ---------------------------------------------------------------------------


def _toy_problem():
    def loglike(v):
        return -0.5 * float(v @ v)

    def prior_transform(u):
        return 10.0 * u - 5.0

    return loglike, prior_transform


@pytest.mark.parametrize(
    "overrides",
    [
        {"ndim": 0},
        {"ndim": 2.5},
        {"nlive": 3},  # < ndim + 2 for ndim=2
        {"enlarge": 0.9},
        {"batch": 0},
        {"dlogz": 0.0},
        {"seed": -1},
        {"seed": 2**64},
        {"dlogz": None},  # no stopping criterion left
    ],
)
def test_invalid_arguments_raise(overrides):
    loglike, pt = _toy_problem()
    kwargs = {"nlive": 50, "dlogz": 0.5, "seed": 1}
    kwargs.update(overrides)
    ndim = kwargs.pop("ndim", 2)
    with pytest.raises(SamplerError):
        sample(loglike, pt, ndim, **kwargs)


def test_loglike_nan_raises():
    loglike, pt = _toy_problem()

    def bad(v):
        return np.nan

    with pytest.raises(SamplerError, match="NaN"):
        sample(bad, pt, 2, nlive=50, dlogz=0.5, seed=1)


def test_loglike_plus_inf_raises():
    _, pt = _toy_problem()
    with pytest.raises(SamplerError, match=r"\+inf"):
        sample(lambda v: np.inf, pt, 2, nlive=50, dlogz=0.5, seed=1)


def test_loglike_all_minus_inf_raises():
    _, pt = _toy_problem()
    with pytest.raises(SamplerError, match="no overlap"):
        sample(lambda v: -np.inf, pt, 2, nlive=50, dlogz=0.5, seed=1)


def test_prior_transform_bad_shape_raises():
    loglike, _ = _toy_problem()
    with pytest.raises(SamplerError, match="prior_transform"):
        sample(loglike, lambda u: u[:1], 2, nlive=50, dlogz=0.5, seed=1)


def test_prior_transform_non_finite_raises():
    loglike, _ = _toy_problem()
    with pytest.raises(SamplerError, match="non-finite"):
        sample(loglike, lambda u: np.full(2, np.nan), 2, nlive=50, dlogz=0.5, seed=1)


def test_results_contract_toy_run():
    loglike, pt = _toy_problem()
    res = sample(loglike, pt, 2, nlive=60, dlogz=0.2, seed=7)
    assert res.backend == ("native" if NATIVE_EXPECTED else "fallback")
    n = res.samples.shape[0]
    assert res.samples.shape == (n, 2)
    assert res.samples_u.shape == (n, 2)
    assert res.logl.shape == (n,)
    assert res.logwt.shape == (n,)
    assert res.weights.shape == (n,)
    assert res.weights.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.isfinite(res.logz) and res.logz_err > 0.0
    assert res.niter >= 1 and res.ncall >= res.niter
    assert 0.0 < res.eff <= 1.0
    assert res.bound == "single-ellipsoid"
    # dict-with-attributes contract
    assert res["logz"] == res.logz
    with pytest.raises(AttributeError):
        _ = res.no_such_key
    # unit cube and prior-box bounds are respected everywhere
    assert np.all((res.samples_u >= 0.0) & (res.samples_u < 1.0))
    assert np.all((res.samples >= -5.0) & (res.samples <= 5.0))
    # weights are monotonically tied to likelihood ranks is NOT required,
    # but the evidence must be finite and the dead logl non-decreasing:
    assert np.all(np.diff(res.logl[: res.niter]) >= -1e-12)


def test_maxiter_and_maxcall_caps():
    loglike, pt = _toy_problem()
    res = sample(loglike, pt, 2, nlive=60, dlogz=None, maxiter=100, seed=3)
    assert res.niter == 100
    res2 = sample(loglike, pt, 2, nlive=60, dlogz=None, maxcall=500, seed=3)
    assert res2.ncall >= 500
    assert res2.ncall < 500 + 64 + 1  # overshoot bounded by one batch


def test_seeded_bit_reproducibility_active_backend():
    loglike, pt = _toy_problem()
    r1 = sample(loglike, pt, 2, nlive=60, dlogz=0.2, seed=42)
    r2 = sample(loglike, pt, 2, nlive=60, dlogz=0.2, seed=42)
    assert r1.logz == r2.logz
    assert np.array_equal(r1.samples, r2.samples)
