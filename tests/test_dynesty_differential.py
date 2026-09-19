"""Differential tests: dynesty_mojo must agree, statistically, with the real
dynesty sampler (PyPI oracle) and with analytic ground truth.

Run twice by ``scripts/test_all_dynesty.sh``: once against the native Mojo
kernel and once with DYNESTY_MOJO_DISABLE_NATIVE=1 (forced NumPy fallback).
The oracle is the pip-installable ``dynesty`` package (skipped cleanly when
not installed); both test problems also have analytic/quadrature ground
truth, so the gates never rest on the oracle alone.

Statistical parity, not trajectory parity: nested sampling is a Monte Carlo
algorithm, so runs are compared on POSTERIOR statistics — the evidence
logz within combined MC errors, and weighted-sample means/covariances
within their Monte Carlo standard errors (estimated per run from the
effective sample size ess = 1/sum(w^2)):

  |logz - logz_ref|      <= 4*sqrt(err1^2 + err2^2) + 0.15
  |mean_j - mu_j|        <= 5*sqrt(C_jj / ess)
  |cov_jk - C_jk|        <= 6*sqrt((C_jj*C_kk + C_jk^2) / ess) + 1e-4

All runs are seeded, so every gate is deterministic (it either always or
never holds for these seeds; measured margins are printed with -v and are
typically well inside the tolerances above).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import dynesty_mojo
from dynesty_mojo import sample

dynesty = pytest.importorskip(
    "dynesty", reason="differential oracle: pip install dynesty"
)

# ---------------------------------------------------------------------------
# Test problems (defined fresh from the literature; see README provenance).
# ---------------------------------------------------------------------------

NDIM_GAUSS = 5
GAUSS_B = 10.0  # uniform prior box [-B, B]^ndim, covers the likelihood mass


def _gauss_problem():
    """Correlated Gaussian likelihood, uniform box prior. Analytic logz."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(NDIM_GAUSS, NDIM_GAUSS))
    cov = (a @ a.T) * 0.005 + np.eye(NDIM_GAUSS) * 0.002
    cov_inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    logz_true = (
        0.5 * NDIM_GAUSS * np.log(2.0 * np.pi)
        + 0.5 * logdet
        - NDIM_GAUSS * np.log(2.0 * GAUSS_B)
    )

    def loglike(v):
        return -0.5 * float(v @ cov_inv @ v)

    def prior_transform(u):
        return 2.0 * GAUSS_B * u - GAUSS_B

    mean_true = np.zeros(NDIM_GAUSS)
    return loglike, prior_transform, mean_true, cov, logz_true


SHELL_R = 2.0
SHELL_W = 0.1
SHELL_B = 6.0  # uniform prior box [-B, B]^2
SHELL_C1 = np.array([-3.0, 0.0])
SHELL_C2 = np.array([3.0, 0.0])


def _shell_radial_grid():
    """Deterministic quadrature grid for the shells' exact radial integrals."""
    rho = np.linspace(1e-12, SHELL_R + 10.0 * SHELL_W, 2_000_001)
    weight = rho * np.exp(-0.5 * ((rho - SHELL_R) / SHELL_W) ** 2)
    return rho, weight


def _shells_problem():
    """Two Gaussian shells (Feroz & Hobson 2008 style). Quadrature logz."""

    def loglike(v):
        # log(exp(a) + exp(b)) in stable form: finite everywhere, exactly
        # the shell-sum likelihood (the shared 1/(sqrt(2 pi) w) constant is
        # folded back in analytically below).
        a = -0.5 * ((np.linalg.norm(v - SHELL_C1) - SHELL_R) / SHELL_W) ** 2
        b = -0.5 * ((np.linalg.norm(v - SHELL_C2) - SHELL_R) / SHELL_W) ** 2
        return float(np.logaddexp(a, b))

    def prior_transform(u):
        return 2.0 * SHELL_B * u - SHELL_B

    rho, weight = _shell_radial_grid()
    # Per-shell integral: (2 pi / (sqrt(2 pi) w)) ∫ rho exp(-(rho-r)^2/2w^2) drho.
    radial = np.trapezoid(weight, rho)
    shell_integral = (2.0 * np.pi) / (np.sqrt(2.0 * np.pi) * SHELL_W) * radial
    # Our loglike drops the 1/(sqrt(2 pi) w) normalization, so rescale:
    # Z = (1/V) * 2 shells * [sqrt(2 pi) w] * shell_integral... spelled out:
    prior_vol = (2.0 * SHELL_B) ** 2
    z_true = 2.0 * (np.sqrt(2.0 * np.pi) * SHELL_W) * shell_integral / prior_vol
    logz_true = float(np.log(z_true))

    # Posterior moments: mean is (0, 0) by symmetry; covariance has the
    # exact ring second moment m2 = ∫ rho^3 exp / ∫ rho exp.
    m2 = np.trapezoid(rho**2 * weight, rho) / np.trapezoid(weight, rho)
    mean_true = np.zeros(2)
    cov_true = np.diag([9.0 + 0.5 * m2, 0.5 * m2])
    return loglike, prior_transform, mean_true, cov_true, logz_true


# ---------------------------------------------------------------------------
# Tolerances (documented in the module header) and shared assertions.
# ---------------------------------------------------------------------------

LOGZ_SIGMA = 4.0
LOGZ_SLACK = 0.15
MEAN_SIGMA = 5.0
COV_SIGMA = 6.0
COV_SLACK = 1e-4

# Sampler configuration for this suite: quick but non-trivial.
NLIVE = 150
DLOGZ = 0.5
SEED_OURS = 20260919
SEED_ORACLE = 123


def _expected_backend() -> str:
    # scripts/test_all_dynesty.sh runs the suite once per backend.
    return "fallback" if os.environ.get("DYNESTY_MOJO_DISABLE_NATIVE") == "1" else "native"


def _ess(weights: np.ndarray) -> float:
    """Effective sample size of a normalized weight vector."""
    return float(1.0 / np.sum(weights**2))


def assert_posterior_close(res_samples, res_weights, res_logz, res_logz_err,
                           ref_samples, ref_weights, ref_logz, ref_logz_err,
                           mean_true, cov_true, logz_true, label):
    """Statistical-parity gates vs the oracle AND vs analytic truth."""
    ess = _ess(res_weights)
    mean = res_weights @ res_samples
    cov = np.cov(res_samples.T, aweights=res_weights)

    # logz vs oracle (combined MC errors) and vs analytic truth.
    combined = np.hypot(res_logz_err, ref_logz_err)
    assert abs(res_logz - ref_logz) <= LOGZ_SIGMA * combined + LOGZ_SLACK, (
        f"{label}: logz {res_logz:.4f} vs oracle {ref_logz:.4f} "
        f"(combined err {combined:.4f})"
    )
    assert abs(res_logz - logz_true) <= LOGZ_SIGMA * res_logz_err + LOGZ_SLACK, (
        f"{label}: logz {res_logz:.4f} vs truth {logz_true:.4f} "
        f"(err {res_logz_err:.4f})"
    )
    # The oracle itself must agree with truth (sanity-checks the gates).
    assert abs(ref_logz - logz_true) <= LOGZ_SIGMA * ref_logz_err + LOGZ_SLACK, (
        f"{label}: ORACLE logz {ref_logz:.4f} vs truth {logz_true:.4f}"
    )

    # Weighted mean/covariance vs analytic truth (MC standard errors).
    mean_tol = MEAN_SIGMA * np.sqrt(np.diag(cov_true) / ess)
    assert np.all(np.abs(mean - mean_true) <= mean_tol), (
        f"{label}: posterior mean {mean} vs truth {mean_true} (tol {mean_tol})"
    )
    cov_mc = np.sqrt(
        (np.outer(np.diag(cov_true), np.diag(cov_true)) + cov_true**2) / ess
    )
    assert np.all(np.abs(cov - cov_true) <= COV_SIGMA * cov_mc + COV_SLACK), (
        f"{label}: posterior cov max err "
        f"{np.abs(cov - cov_true).max():.5g} (tol {(COV_SIGMA * cov_mc + COV_SLACK).max():.5g})"
    )

    # And the weighted moments must agree with the oracle's own estimate.
    ref_mean = ref_weights @ ref_samples
    ess_pair = min(ess, _ess(ref_weights))
    pair_mean_tol = MEAN_SIGMA * np.sqrt(np.diag(cov_true) / ess_pair)
    assert np.all(np.abs(mean - ref_mean) <= pair_mean_tol), (
        f"{label}: posterior mean vs oracle max diff "
        f"{np.abs(mean - ref_mean).max():.5g} (tol {pair_mean_tol.max():.5g})"
    )


def _run_ours(problem, seed=SEED_OURS):
    loglike, prior_transform, _, _, _ = problem
    return sample(
        loglike,
        prior_transform,
        _ndim_of(problem),
        nlive=NLIVE,
        dlogz=DLOGZ,
        seed=seed,
    )


def _ndim_of(problem) -> int:
    return len(problem[2])


def _run_oracle(problem, seed=SEED_ORACLE, bootstrap=None):
    loglike, prior_transform, _, _, _ = problem
    rstate = np.random.default_rng(seed)
    kwargs = {}
    if bootstrap is not None:
        # Multi-modal bounds trip dynesty's bootstrap over-enlargement (its
        # own UserWarning prescribes bootstrap=0 and a ~20x speedup
        # results); still the oracle's documented standard options.
        kwargs["bootstrap"] = bootstrap
    sampler = dynesty.DynamicNestedSampler(
        loglike, prior_transform, _ndim_of(problem), nlive=NLIVE,
        rstate=rstate, **kwargs,
    )
    sampler.run_nested(dlogz_init=DLOGZ, print_progress=False)
    return sampler.results


@pytest.fixture(scope="module")
def gauss_oracle():
    return _run_oracle(_gauss_problem())


@pytest.fixture(scope="module")
def shells_oracle():
    return _run_oracle(_shells_problem(), bootstrap=0)


# ---------------------------------------------------------------------------
# Tests: correlated Gaussian (5-D) and Gaussian shells (2-D, bimodal).
# ---------------------------------------------------------------------------


def test_backend_is_the_expected_one():
    assert dynesty_mojo.native_available() == (_expected_backend() == "native")


def test_gaussian_shells_vs_oracle_and_truth(shells_oracle):
    problem = _shells_problem()
    _, _, mean_true, cov_true, logz_true = problem
    res = _run_ours(problem)
    assert_posterior_close(
        res.samples,
        res.weights,
        res.logz,
        res.logz_err,
        shells_oracle.samples,
        shells_oracle.importance_weights(),
        float(shells_oracle.logz[-1]),
        float(shells_oracle.logzerr[-1]),
        mean_true,
        cov_true,
        logz_true,
        f"shells[{_expected_backend()}]",
    )


def test_correlated_gaussian_vs_oracle_and_truth(gauss_oracle):
    problem = _gauss_problem()
    _, _, mean_true, cov_true, logz_true = problem
    res = _run_ours(problem)
    assert_posterior_close(
        res.samples,
        res.weights,
        res.logz,
        res.logz_err,
        gauss_oracle.samples,
        gauss_oracle.importance_weights(),
        float(gauss_oracle.logz[-1]),
        float(gauss_oracle.logzerr[-1]),
        mean_true,
        cov_true,
        logz_true,
        f"gauss[{_expected_backend()}]",
    )


def test_seeded_runs_are_bit_reproducible():
    """Same seed, same backend: identical results, bitwise."""
    problem = _gauss_problem()
    res1 = _run_ours(problem, seed=99)
    res2 = _run_ours(problem, seed=99)
    assert res1.logz == res2.logz
    assert np.array_equal(res1.samples, res2.samples)
    assert np.array_equal(res1.weights, res2.weights)


def test_vectorized_matches_scalar_path_statistically():
    """The vectorized evaluation path must land within MC error of the
    scalar path on the same problem (different call streams, same truth)."""
    problem = _gauss_problem()
    _, _, mean_true, cov_true, logz_true = problem

    rng = np.random.default_rng(0)
    a = rng.normal(size=(NDIM_GAUSS, NDIM_GAUSS))
    cov = (a @ a.T) * 0.005 + np.eye(NDIM_GAUSS) * 0.002
    cov_inv = np.linalg.inv(cov)

    def loglike_v(v):
        return -0.5 * np.einsum("ij,jk,ik->i", v, cov_inv, v)

    def pt_v(u):
        return 2.0 * GAUSS_B * u - GAUSS_B

    res = sample(
        loglike_v, pt_v, NDIM_GAUSS, nlive=NLIVE, dlogz=DLOGZ,
        seed=SEED_OURS, vectorized=True,
    )
    assert abs(res.logz - logz_true) <= LOGZ_SIGMA * res.logz_err + LOGZ_SLACK
    ess = _ess(res.weights)
    mean = res.weights @ res.samples
    assert np.all(np.abs(mean - mean_true) <= MEAN_SIGMA * np.sqrt(np.diag(cov_true) / ess))
