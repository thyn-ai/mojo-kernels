"""Differential tests: pykalman_mojo must match pykalman element-for-element.

Run twice by `scripts/test_all_pykalman.sh`: once against the native Mojo
kernel and once with PYKALMAN_MOJO_DISABLE_NATIVE=1 (forced pure-NumPy
fallback). Both backends must agree with pykalman within 1e-10 everywhere
(measured agreement is at the 1e-14 level on these seeded systems).

The oracle is the published PyPI package (pykalman==0.11.2, pinned in
pixi.toml [pypi-dependencies]; it works unmodified on numpy 2.x). Systems
are random but seeded and stable (transition spectral radius < 1) with SPD,
well-conditioned covariances, transition/observation offsets, and masked
observations.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pykalman = pytest.importorskip(
    "pykalman",
    reason="pykalman oracle not installed (pixi.toml pins pykalman==0.11.2)",
)

import pykalman_mojo
from pykalman_mojo import KalmanFilter

ATOL = 1e-10  # documented parity tolerance; measured agreement is ~1e-14

# (n_dim_state, n_dim_obs, n_timesteps) coverage grid.
DIMS = [
    pytest.param(1, 1, 50, id="scalar"),
    pytest.param(2, 1, 300, id="2x1"),
    pytest.param(3, 2, 300, id="3x2"),
    pytest.param(5, 3, 500, id="5x3"),
    pytest.param(8, 4, 200, id="8x4"),
    pytest.param(4, 6, 200, id="4x6-more-obs-than-state"),
]


def make_system(seed: int, n_s: int, n_o: int, T: int):
    """Random stable linear-Gaussian system + simulated observations (seeded)."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n_s, n_s)
    A *= 0.9 / max(abs(np.linalg.eigvals(A)))  # stable: spectral radius 0.9
    B = rng.randn(n_s, n_s)
    Q = B @ B.T + 0.2 * np.eye(n_s)  # SPD, well conditioned
    C = rng.randn(n_o, n_s)
    D = rng.randn(n_o, n_o)
    R = D @ D.T + 0.2 * np.eye(n_o)
    b = rng.randn(n_s)
    d = rng.randn(n_o)
    x0 = rng.randn(n_s)
    E = rng.randn(n_s, n_s)
    P0 = E @ E.T + np.eye(n_s)

    obs = np.zeros((T, n_o))
    x = x0.copy()
    for t in range(T):
        if t > 0:
            x = A @ x + b + rng.multivariate_normal(np.zeros(n_s), Q)
        obs[t] = C @ x + d + rng.multivariate_normal(np.zeros(n_o), R)

    params = dict(
        transition_matrices=A,
        observation_matrices=C,
        transition_covariance=Q,
        observation_covariance=R,
        transition_offsets=b,
        observation_offsets=d,
        initial_state_mean=x0,
        initial_state_covariance=P0,
    )
    return params, obs


def mask_observations(obs: np.ndarray, seed: int, frac: float = 0.1):
    """Masked-array observations with ~frac of timesteps fully masked."""
    rng = np.random.RandomState(seed)
    m = rng.rand(obs.shape[0]) < frac
    return np.ma.masked_array(obs, mask=np.repeat(m[:, None], obs.shape[1], axis=1))


def _expected_backend() -> str:
    # scripts/test_all_pykalman.sh runs the suite once per backend.
    return "fallback" if os.environ.get("PYKALMAN_MOJO_DISABLE_NATIVE") == "1" else "native"


def assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    assert isinstance(actual, np.ndarray), f"{label}: expected np.ndarray, got {type(actual)}"
    assert actual.dtype == np.float64, f"{label}: expected float64, got {actual.dtype}"
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=ATOL,
        err_msg=f"{label}: max|diff| = {diff:.3e}",
    )


@pytest.mark.parametrize("n_s,n_o,T", DIMS)
def test_filter_parity(n_s, n_o, T):
    params, obs = make_system(100 + n_s * 17 + n_o, n_s, n_o, T)
    ref_fm, ref_fc = pykalman.KalmanFilter(**params).filter(obs)
    kf = KalmanFilter(**params)
    fm, fc = kf.filter(obs)
    assert kf.backend == _expected_backend()
    assert_close(fm, ref_fm, "filtered_state_means")
    assert_close(fc, ref_fc, "filtered_state_covariances")


@pytest.mark.parametrize("n_s,n_o,T", DIMS)
def test_smooth_parity(n_s, n_o, T):
    params, obs = make_system(200 + n_s * 17 + n_o, n_s, n_o, T)
    ref_sm, ref_sc = pykalman.KalmanFilter(**params).smooth(obs)
    kf = KalmanFilter(**params)
    sm, sc = kf.smooth(obs)
    assert kf.backend == _expected_backend()
    assert_close(sm, ref_sm, "smoothed_state_means")
    assert_close(sc, ref_sc, "smoothed_state_covariances")


@pytest.mark.parametrize("n_s,n_o,T", DIMS[:4])
def test_masked_observations_parity(n_s, n_o, T):
    params, obs = make_system(300 + n_s * 17 + n_o, n_s, n_o, T)
    Z = mask_observations(obs, seed=5)
    ref = pykalman.KalmanFilter(**params)
    ours = KalmanFilter(**params)
    fm, fc = ours.filter(Z)
    ref_fm, ref_fc = ref.filter(Z)
    sm, sc = ours.smooth(Z)
    ref_sm, ref_sc = ref.smooth(Z)
    assert ours.backend == _expected_backend()
    assert_close(fm, ref_fm, "masked filtered means")
    assert_close(fc, ref_fc, "masked filtered covs")
    assert_close(sm, ref_sm, "masked smoothed means")
    assert_close(sc, ref_sc, "masked smoothed covs")


def test_partial_component_masking_skips_whole_timestep():
    # pykalman's rule: ANY masked component => the whole timestep is missing
    # (gain zero, filtered == predicted). Mirror it exactly.
    params, obs = make_system(401, 4, 3, 120)
    Z = np.ma.masked_array(obs.copy(), mask=np.zeros_like(obs, dtype=bool))
    Z.mask[10, 0] = True  # one component only
    Z.mask[11, 2] = True
    Z.mask[12] = True  # whole timestep
    ref = pykalman.KalmanFilter(**params)
    ours = KalmanFilter(**params)
    fm, fc = ours.filter(Z)
    ref_fm, ref_fc = ref.filter(Z)
    sm, sc = ours.smooth(Z)
    ref_sm, ref_sc = ref.smooth(Z)
    assert_close(fm, ref_fm, "partial-mask filtered means")
    assert_close(fc, ref_fc, "partial-mask filtered covs")
    assert_close(sm, ref_sm, "partial-mask smoothed means")
    assert_close(sc, ref_sc, "partial-mask smoothed covs")


def test_all_timesteps_masked():
    params, obs = make_system(402, 3, 2, 60)
    Z = np.ma.masked_array(obs, mask=np.ones_like(obs, dtype=bool))
    ref = pykalman.KalmanFilter(**params)
    ours = KalmanFilter(**params)
    fm, fc = ours.filter(Z)
    ref_fm, ref_fc = ref.filter(Z)
    sm, sc = ours.smooth(Z)
    ref_sm, ref_sc = ref.smooth(Z)
    assert_close(fm, ref_fm, "all-masked filtered means")
    assert_close(fc, ref_fc, "all-masked filtered covs")
    assert_close(sm, ref_sm, "all-masked smoothed means")
    assert_close(sc, ref_sc, "all-masked smoothed covs")


def test_defaults_and_dimension_inference():
    # Give only a transition matrix and an observation offset: pykalman
    # infers n_dim_state=3, n_dim_obs=2 and defaults everything else.
    given = dict(
        transition_matrices=[[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.7]],
        observation_offsets=[1.0, -2.0],
    )
    obs = np.random.RandomState(7).randn(80, 2)
    ref = pykalman.KalmanFilter(**given)
    ours = KalmanFilter(**given)
    fm, fc = ours.filter(obs)
    ref_fm, ref_fc = ref.filter(obs)
    sm, sc = ours.smooth(obs)
    ref_sm, ref_sc = ref.smooth(obs)
    assert ours.n_dim_state == ref.n_dim_state == 3
    assert ours.n_dim_obs == ref.n_dim_obs == 2
    assert_close(fm, ref_fm, "default-params filtered means")
    assert_close(fc, ref_fc, "default-params filtered covs")
    assert_close(sm, ref_sm, "default-params smoothed means")
    assert_close(sc, ref_sc, "default-params smoothed covs")


def test_no_params_at_all_defaults_to_scalar_random_walk():
    obs = np.random.RandomState(11).randn(40, 1)
    ref_fm, ref_fc = pykalman.KalmanFilter().filter(obs)
    fm, fc = KalmanFilter().filter(obs)
    assert_close(fm, ref_fm, "empty-params filtered means")
    assert_close(fc, ref_fc, "empty-params filtered covs")


def test_module_level_functions():
    params, obs = make_system(403, 3, 2, 150)
    fm, fc = pykalman_mojo.filter(obs, **params)
    sm, sc = pykalman_mojo.smooth(obs, **params)
    ref = pykalman.KalmanFilter(**params)
    assert_close(fm, ref.filter(obs)[0], "module filter means")
    assert_close(fc, ref.filter(obs)[1], "module filter covs")
    assert_close(sm, ref.smooth(obs)[0], "module smooth means")
    assert_close(sc, ref.smooth(obs)[1], "module smooth covs")


def test_1d_observations_are_univariate_series():
    # pykalman: a 1-D input becomes a (T, 1) observation series.
    params, obs = make_system(404, 2, 1, 90)
    series = obs[:, 0].copy()
    ref = pykalman.KalmanFilter(**params)
    ours = KalmanFilter(**params)
    fm, _ = ours.filter(series)
    ref_fm, _ = ref.filter(series)
    assert_close(fm, ref_fm, "1-D observations filtered means")


def test_single_timestep():
    # pykalman's observation parser transposes a (1, n_o>1) input and then
    # fails broadcasting inside its own filter — a single timestep is only
    # well-defined for univariate observations (both packages raise otherwise).
    params, obs = make_system(405, 3, 1, 1)
    ref = pykalman.KalmanFilter(**params)
    ours = KalmanFilter(**params)
    fm, fc = ours.filter(obs)
    ref_fm, ref_fc = ref.filter(obs)
    sm, sc = ours.smooth(obs)
    ref_sm, ref_sc = ref.smooth(obs)
    assert_close(fm, ref_fm, "T=1 filtered means")
    assert_close(fc, ref_fc, "T=1 filtered covs")
    assert_close(sm, ref_sm, "T=1 smoothed means")
    assert_close(sc, ref_sc, "T=1 smoothed covs")


def test_single_timestep_multivariate_raises_like_reference():
    params, obs = make_system(409, 3, 2, 1)
    with pytest.raises(ValueError):
        KalmanFilter(**params).filter(obs)
    with pytest.raises(ValueError):
        pykalman.KalmanFilter(**params).filter(obs)


def test_nan_observation_propagates_like_reference():
    # A plain (unmasked) array with NaN is NOT missing data for pykalman:
    # NaN propagates into the outputs. Both backends must reproduce the same
    # NaN pattern (garbage-in, garbage-out parity).
    params, obs = make_system(406, 3, 2, 80)
    obs_nan = obs.copy()
    obs_nan[20, 0] = np.nan
    ref_fm, _ = pykalman.KalmanFilter(**params).filter(obs_nan)
    fm, _ = KalmanFilter(**params).filter(obs_nan)
    assert np.array_equal(np.isnan(fm), np.isnan(ref_fm))
    both = ~np.isnan(fm)
    np.testing.assert_allclose(fm[both], ref_fm[both], rtol=0, atol=ATOL)


def test_parameter_mutation_between_calls():
    # Users may mutate matrices between calls (pykalman re-reads them each
    # call); the native handle must never serve stale matrices.
    params, obs = make_system(407, 3, 2, 100)
    ours = KalmanFilter(**params)
    ours.filter(obs)
    A2 = params["transition_matrices"] * 0.5  # still stable
    ours.transition_matrices = A2
    mutated = dict(params, transition_matrices=A2)
    fm, fc = ours.filter(obs)
    ref_fm, ref_fc = pykalman.KalmanFilter(**mutated).filter(obs)
    assert_close(fm, ref_fm, "mutated-params filtered means")
    assert_close(fc, ref_fc, "mutated-params filtered covs")


def test_inconsistent_parameters_raise_like_reference():
    bad = dict(transition_matrices=np.eye(3), observation_matrices=np.eye(2, 4))
    with pytest.raises(ValueError):
        KalmanFilter(**bad).filter(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        pykalman.KalmanFilter(**bad)


def test_wrong_matrix_shape_raises():
    # pykalman's dimension-consistency check fires on this too (generic
    # message); ours names the offending parameter when it gets that far.
    with pytest.raises(ValueError, match="not consistent"):
        KalmanFilter(
            transition_matrices=np.eye(3), transition_covariance=np.eye(2)
        ).filter(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="observation_covariance"):
        KalmanFilter(
            transition_matrices=np.eye(3),
            observation_matrices=np.eye(2, 3),
            observation_covariance=np.zeros((2, 3)),
        ).filter(np.zeros((10, 2)))


def test_time_varying_parameters_rejected():
    tv = np.stack([np.eye(2), np.eye(2) * 0.5])
    with pytest.raises(ValueError, match="time-invariant"):
        KalmanFilter(transition_matrices=tv).filter(np.zeros((10, 2)))


def test_wrong_observation_width_raises():
    params, _ = make_system(408, 3, 2, 20)
    with pytest.raises(ValueError, match="width"):
        KalmanFilter(**params).filter(np.zeros((20, 5)))
