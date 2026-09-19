"""Kalman filter and smoother, API-compatible with `pykalman`'s core surface.

`KalmanFilter` accepts the same constructor parameters as
`pykalman.KalmanFilter` (same defaults, same dimensionality inference, same
observation parsing) and its `.filter(X)` / `.smooth(X)` return the same
arrays. The module-level `filter(observations, **params)` /
`smooth(observations, **params)` are one-call conveniences.

Computation runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back
to the vendored NumPy reference otherwise. Parameter normalization and
observation parsing are shared by both backends, so results agree either
way; the differential suite asserts both against pykalman within 1e-10.

Scope: time-invariant systems. pykalman additionally supports time-varying
matrices (3-D arrays), EM, online `filter_update`, sampling, and likelihood
computation — those are out of scope here and rejected with a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pykalman_mojo import _reference
from pykalman_mojo._native import NativeModel, NativeUnavailable

__all__ = ["KalmanFilter", "filter", "smooth"]


@dataclass(frozen=True)
class Params:
    """Normalized, validated, float64 C-contiguous system matrices."""

    A: np.ndarray  # (n_s, n_s) transition matrix
    b: np.ndarray  # (n_s,) transition offset
    Q: np.ndarray  # (n_s, n_s) transition covariance
    C: np.ndarray  # (n_o, n_s) observation matrix
    d: np.ndarray  # (n_o,) observation offset
    R: np.ndarray  # (n_o, n_o) observation covariance
    x0: np.ndarray  # (n_s,) initial state mean
    P0: np.ndarray  # (n_s, n_s) initial state covariance
    n_dim_state: int
    n_dim_obs: int


def _as_float_array(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 2:
        raise ValueError(
            f"{name} is time-varying ({arr.ndim} dimensions); pykalman_mojo "
            "supports time-invariant systems only"
        )
    return arr


def _determine_dimensionality(candidates: list[int], explicit: int | None, label: str) -> int:
    """Mirror pykalman's rule: all non-None candidates must agree; default 1."""
    values = list(candidates)
    if explicit is not None:
        values.append(int(explicit))
    if not values:
        return 1
    if any(v != values[0] for v in values):
        raise ValueError(
            "The shape of all parameters is not consistent.  "
            "Please re-check their values."
        )
    if values[0] < 1:
        raise ValueError(f"{label} must be >= 1, got {values[0]}")
    return values[0]


def _normalize_params(
    transition_matrices,
    observation_matrices,
    transition_covariance,
    observation_covariance,
    transition_offsets,
    observation_offsets,
    initial_state_mean,
    initial_state_covariance,
    n_dim_state,
    n_dim_obs,
) -> Params:
    """Apply pykalman's defaults and dimensionality inference; validate shapes."""
    given = {
        "transition_matrices": transition_matrices,
        "observation_matrices": observation_matrices,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "transition_offsets": transition_offsets,
        "observation_offsets": observation_offsets,
        "initial_state_mean": initial_state_mean,
        "initial_state_covariance": initial_state_covariance,
    }
    arrays = {
        name: (_as_float_array(name, value) if value is not None else None)
        for name, value in given.items()
    }

    def dim_of(name: str, index: int) -> int | None:
        arr = arrays[name]
        return None if arr is None else int(arr.shape[index])

    n_s = _determine_dimensionality(
        [
            d
            for d in (
                dim_of("transition_matrices", -2),
                dim_of("transition_offsets", -1),
                dim_of("transition_covariance", -2),
                dim_of("initial_state_mean", -1),
                dim_of("initial_state_covariance", -2),
                dim_of("observation_matrices", -1),
            )
            if d is not None
        ],
        n_dim_state,
        "n_dim_state",
    )
    n_o = _determine_dimensionality(
        [
            d
            for d in (
                dim_of("observation_matrices", -2),
                dim_of("observation_offsets", -1),
                dim_of("observation_covariance", -2),
            )
            if d is not None
        ],
        n_dim_obs,
        "n_dim_obs",
    )

    defaults = {
        "transition_matrices": np.eye(n_s),
        "transition_offsets": np.zeros(n_s),
        "transition_covariance": np.eye(n_s),
        "observation_matrices": np.eye(n_o, n_s),
        "observation_offsets": np.zeros(n_o),
        "observation_covariance": np.eye(n_o),
        "initial_state_mean": np.zeros(n_s),
        "initial_state_covariance": np.eye(n_s),
    }
    expected_shapes = {
        "transition_matrices": (n_s, n_s),
        "transition_offsets": (n_s,),
        "transition_covariance": (n_s, n_s),
        "observation_matrices": (n_o, n_s),
        "observation_offsets": (n_o,),
        "observation_covariance": (n_o, n_o),
        "initial_state_mean": (n_s,),
        "initial_state_covariance": (n_s, n_s),
    }
    full: dict[str, np.ndarray] = {}
    for name, default in defaults.items():
        arr = arrays[name] if arrays[name] is not None else default
        if arr.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {arr.shape}, expected {expected_shapes[name]}"
            )
        full[name] = np.ascontiguousarray(arr, dtype=np.float64)

    return Params(
        A=full["transition_matrices"],
        b=full["transition_offsets"],
        Q=full["transition_covariance"],
        C=full["observation_matrices"],
        d=full["observation_offsets"],
        R=full["observation_covariance"],
        x0=full["initial_state_mean"],
        P0=full["initial_state_covariance"],
        n_dim_state=n_s,
        n_dim_obs=n_o,
    )


def _parse_observations(obs, n_dim_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror pykalman's `_parse_observations`; also derive the missing mask.

    Returns (values, mask): values is float64 (T, n_dim_obs); mask is int32
    (T,) with 1 wherever any component of the timestep is masked — pykalman
    treats such a timestep as missing in its entirety.
    """
    Z = np.ma.atleast_2d(obs)
    if Z.shape[0] == 1 and Z.shape[1] > 1:
        Z = Z.T
    if Z.ndim != 2 or Z.shape[1] != n_dim_obs:
        got = Z.shape[1] if Z.ndim == 2 else "?"
        raise ValueError(
            f"observations have width {got}, expected n_dim_obs = {n_dim_obs}"
        )
    mask = np.ma.getmaskarray(Z).any(axis=1).astype(np.int32)
    values = np.ascontiguousarray(np.asarray(Z), dtype=np.float64)
    return values, mask


def _params_equal(a: Params, b: Params) -> bool:
    return all(
        np.array_equal(getattr(a, name), getattr(b, name))
        for name in ("A", "b", "Q", "C", "d", "R", "x0", "P0")
    )


class KalmanFilter:
    """Kalman Filter/Smoother for time-invariant linear-Gaussian systems.

    Same constructor signature (minus EM/sampling knobs) and the same
    `.filter(X)` / `.smooth(X)` return values as `pykalman.KalmanFilter`.
    `X` may be a plain array (NaN propagates, as in pykalman) or a masked
    array (masked timesteps are treated as missing, as in pykalman).
    """

    def __init__(
        self,
        transition_matrices=None,
        observation_matrices=None,
        transition_covariance=None,
        observation_covariance=None,
        transition_offsets=None,
        observation_offsets=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        n_dim_state=None,
        n_dim_obs=None,
    ):
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self._native_model: NativeModel | None = None
        self._backend = "fallback"
        # Resolve dimensionality and validate parameters at construction
        # (pykalman does the same), storing the resolved dims as attributes.
        params = self._normalize()
        self.n_dim_state = params.n_dim_state
        self.n_dim_obs = params.n_dim_obs
        self._params: Params | None = params

    def _normalize(self) -> Params:
        return _normalize_params(
            self.transition_matrices,
            self.observation_matrices,
            self.transition_covariance,
            self.observation_covariance,
            self.transition_offsets,
            self.observation_offsets,
            self.initial_state_mean,
            self.initial_state_covariance,
            self.n_dim_state,
            self.n_dim_obs,
        )

    def _current_params(self) -> Params:
        """Re-normalize from the (possibly user-mutated) attributes, caching."""
        params = self._normalize()
        if self._params is None or not _params_equal(params, self._params):
            # Parameters changed (or first use): rebuild the native model so
            # the handle can never serve stale matrices.
            if self._native_model is not None:
                self._native_model.close()
                self._native_model = None
            self._params = params
            self._backend = "fallback"
        return self._params

    def _native(self, params: Params) -> NativeModel | None:
        if self._native_model is None:
            try:
                self._native_model = NativeModel(params)
            except NativeUnavailable:
                return None
            self._backend = "native"
        return self._native_model

    @property
    def backend(self) -> str:
        """Which backend serves this instance: "native" or "fallback"."""
        return self._backend

    def filter(self, X):
        """Filtered state means and covariances for observations X.

        Returns (filtered_state_means, filtered_state_covariances) with
        shapes (n_timesteps, n_dim_state) and
        (n_timesteps, n_dim_state, n_dim_state) — pykalman's contract.
        """
        params = self._current_params()
        obs, mask = _parse_observations(X, params.n_dim_obs)
        model = self._native(params)
        if model is not None:
            try:
                return model.filter(obs, mask)
            except NativeUnavailable:
                # e.g. singular innovation covariance (status 3): the
                # reference's pseudo-inverse handles it — serve this call
                # from the fallback, identically to an unsupported platform.
                pass
        return _reference.filter(params, obs, mask)

    def smooth(self, X):
        """Smoothed state means and covariances for observations X.

        Returns (smoothed_state_means, smoothed_state_covariances) with
        shapes (n_timesteps, n_dim_state) and
        (n_timesteps, n_dim_state, n_dim_state) — pykalman's contract.
        """
        params = self._current_params()
        obs, mask = _parse_observations(X, params.n_dim_obs)
        model = self._native(params)
        if model is not None:
            try:
                return model.smooth(obs, mask)
            except NativeUnavailable:
                pass
        return _reference.smooth(params, obs, mask)


def filter(observations, **params):  # noqa: A001
    """One-call Kalman filter: `filter(X, transition_matrices=..., ...)`.

    Accepts any `KalmanFilter` constructor parameter. Returns
    (filtered_state_means, filtered_state_covariances).
    """
    return KalmanFilter(**params).filter(observations)


def smooth(observations, **params):
    """One-call Kalman smoother: `smooth(X, transition_matrices=..., ...)`.

    Accepts any `KalmanFilter` constructor parameter. Returns
    (smoothed_state_means, smoothed_state_covariances).
    """
    return KalmanFilter(**params).smooth(observations)
