"""Vendored pure-NumPy Kalman filter + RTS smoother (the fallback backend).

Clean-room implementation of the textbook linear-Gaussian recursions
(Kalman 1960; Rauch, Tung & Striebel 1965), written to the published
pykalman package's conventions and operation order so both backends agree
with pykalman element-wise:

    predict:  mean = A @ x + b,          cov = A @ (P @ A.T) + Q
    correct:  S = C @ (P @ C.T) + R,     K = P @ (C.T @ pinv(S))
              mean += K @ (z - (C @ mean + d)),  cov -= K @ (C @ P)
    smooth:   J = P_f[t] @ (A.T @ pinv(P_pred[t+1]))
              mean_s[t] = mean_f[t] + J @ (mean_s[t+1] - mean_pred[t+1])
              cov_s[t]  = P_f[t] + J @ ((cov_s[t+1] - P_pred[t+1]) @ J.T)

pykalman uses ``scipy.linalg.pinv``; this module uses ``np.linalg.pinv``.
Both are SVD-based with an effective relative cutoff of eps * max(shape),
so for the well-conditioned covariances of stable systems the difference is
at the 1e-15 level — far inside the documented 1e-10 parity tolerance (the
differential suite measures the actual agreement on every run). Using the
NumPy pseudo-inverse keeps numpy the wrapper's only hard dependency.

Masking follows pykalman's all-or-nothing rule: a timestep whose observation
has any masked component is skipped whole (gain zero, filtered = predicted).
"""

from __future__ import annotations

import numpy as np


def _filter_full(params, obs: np.ndarray, mask: np.ndarray):
    """Forward pass; returns predicted and filtered means/covariances."""
    n_timesteps = obs.shape[0]
    n_s = params.n_dim_state

    A, b, Q = params.A, params.b, params.Q
    C, d, R = params.C, params.d, params.R

    pred_mean = np.empty((n_timesteps, n_s))
    pred_cov = np.empty((n_timesteps, n_s, n_s))
    filt_mean = np.empty((n_timesteps, n_s))
    filt_cov = np.empty((n_timesteps, n_s, n_s))

    for t in range(n_timesteps):
        if t == 0:
            pred_mean[t] = params.x0
            pred_cov[t] = params.P0
        else:
            pred_mean[t] = A @ filt_mean[t - 1] + b
            pred_cov[t] = A @ (filt_cov[t - 1] @ A.T) + Q
        if mask[t]:
            filt_mean[t] = pred_mean[t]
            filt_cov[t] = pred_cov[t]
        else:
            pred_obs_mean = C @ pred_mean[t] + d
            S = C @ (pred_cov[t] @ C.T) + R
            K = pred_cov[t] @ (C.T @ np.linalg.pinv(S))
            filt_mean[t] = pred_mean[t] + K @ (obs[t] - pred_obs_mean)
            filt_cov[t] = pred_cov[t] - K @ (C @ pred_cov[t])

    return pred_mean, pred_cov, filt_mean, filt_cov


def filter(params, obs: np.ndarray, mask: np.ndarray):  # noqa: A001
    """Filtered state means/covariances: (T, n_s), (T, n_s, n_s) float64."""
    _, _, filt_mean, filt_cov = _filter_full(params, obs, mask)
    return filt_mean, filt_cov


def smooth(params, obs: np.ndarray, mask: np.ndarray):
    """Rauch-Tung-Striebel smoothed means/covariances: (T, n_s), (T, n_s, n_s)."""
    pred_mean, pred_cov, filt_mean, filt_cov = _filter_full(params, obs, mask)

    A = params.A
    n_timesteps = obs.shape[0]
    n_s = params.n_dim_state

    smooth_mean = np.empty((n_timesteps, n_s))
    smooth_cov = np.empty((n_timesteps, n_s, n_s))
    smooth_mean[-1] = filt_mean[-1]
    smooth_cov[-1] = filt_cov[-1]

    for t in range(n_timesteps - 2, -1, -1):
        J = filt_cov[t] @ (A.T @ np.linalg.pinv(pred_cov[t + 1]))
        smooth_mean[t] = filt_mean[t] + J @ (smooth_mean[t + 1] - pred_mean[t + 1])
        smooth_cov[t] = filt_cov[t] + J @ (
            (smooth_cov[t + 1] - pred_cov[t + 1]) @ J.T
        )

    return smooth_mean, smooth_cov
