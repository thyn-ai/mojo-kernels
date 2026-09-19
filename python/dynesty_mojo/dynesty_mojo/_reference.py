"""Vendored pure-NumPy reference for the dynestymojo bounding/proposal kernel.

Clean-room, NumPy-only implementation of the exact semantics the Mojo kernel
exports (see ``kernels/dynesty/src/dynestymojo.mojo`` for the algorithm
provenance):

- :func:`fit_ellipsoid` — sample mean/covariance (ddof=1) of the live points,
  scaled so the ellipsoid contains every live point (max Mahalanobis
  distance), multiplied by the enlargement factor; returns the Cholesky
  factor and log-volume, or None when the covariance is numerically
  singular even after diagonal jitter (caller falls back to cube sampling).
- :class:`RefRng` — seeded (``numpy.random.Generator(PCG64)``) batch
  proposal: uniform over the ellipsoid intersected with the unit hypercube
  (mode 0, Gaussian-direction construction with cube rejection) or uniform
  over the whole cube (mode 1). Candidates within one batch are iid, so the
  wrapper's first-passing-candidate rule is an exact constrained-prior
  draw, identically to the native path.

The two backends are NOT bit-identical (different RNG algorithms by design)
but are statistically equivalent; each is bit-reproducible for a fixed seed
on its own. Used on every platform without a Mojo toolchain (e.g. Windows)
and whenever the native library fails to load — silently and correctly.
"""

from __future__ import annotations

import math

import numpy as np

from dynesty_mojo._native import MODE_CUBE, MODE_ELLIPSOID, EllipsoidFit

# Same internal attempt cap as the kernel (per propose_batch call, mode 0).
MAX_ATTEMPTS_FACTOR = 4096


def _log_unit_ball_volume(ndim: int) -> float:
    """log of the ndim-dimensional unit-ball volume pi^(n/2)/Gamma(n/2+1)."""
    half = ndim * 0.5
    return half * math.log(math.pi) - math.lgamma(half + 1.0)


def fit_ellipsoid(u_points: np.ndarray, enlarge: float) -> EllipsoidFit | None:
    """Fit the enlarged bounding ellipsoid of the live points (NumPy)."""
    u = np.ascontiguousarray(u_points, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"u_points must be 2-D (n, ndim); got shape {u.shape}")
    n, ndim = u.shape
    if n <= 1 or enlarge < 1.0:
        raise ValueError(f"invalid fit arguments: n={n}, enlarge={enlarge}")

    mean = u.mean(axis=0)
    centered = u - mean
    cov = (centered.T @ centered) / (n - 1)

    trace = float(np.trace(cov))
    jitter_base = 1e-12 * trace / ndim if trace > 0.0 else 1e-300

    chol = None
    jitter = 0.0
    for _ in range(7):
        try:
            chol = np.linalg.cholesky(cov + jitter * np.eye(ndim))
            break
        except np.linalg.LinAlgError:
            jitter = jitter_base if jitter == 0.0 else jitter * 10.0
    if chol is None:
        return None

    # Max squared Mahalanobis distance of any live point: solve L z = d
    # (plain solve; ndim is small and numpy has no public triangular solve).
    z = np.linalg.solve(chol, centered.T)  # (ndim, n)
    dmax2 = float(np.max(np.sum(z * z, axis=0)))
    if not dmax2 > 0.0:  # all live points identical, or NaN upstream
        return None

    scale = math.sqrt(enlarge * dmax2)
    chol_final = chol * scale
    half_logdet = float(np.sum(np.log(np.diag(chol_final))))
    logvol = _log_unit_ball_volume(ndim) + half_logdet
    return EllipsoidFit(mean=mean, chol=chol_final, logvol=logvol)


class RefRng:
    """Seeded NumPy batch proposer with the kernel's exact semantics."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(np.random.PCG64(seed))

    def propose_batch(
        self,
        ndim: int,
        fit: EllipsoidFit | None,
        mode: int,
        max_out: int,
    ) -> np.ndarray:
        """Up to max_out candidate unit-cube points, shape (k, ndim), 0 <= k."""
        if max_out <= 0 or ndim <= 0:
            return np.empty((0, ndim), dtype=np.float64)
        if mode == MODE_CUBE or fit is None:
            return self._rng.random((max_out, ndim))

        chunks: list[np.ndarray] = []
        count = 0
        attempts = 0
        max_attempts = max_out * MAX_ATTEMPTS_FACTOR
        inv_d = 1.0 / ndim
        # Draw in chunks (a whole remaining batch at a time); each candidate
        # is iid exactly as in the kernel's sequential construction.
        while count < max_out and attempts < max_attempts:
            want = max_out - count
            attempts += want
            g = self._rng.standard_normal((want, ndim))
            norms = np.linalg.norm(g, axis=1)
            u = self._rng.random(want)
            ok = (norms > 0.0) & (u > 0.0)
            if not np.any(ok):
                continue
            g = g[ok]
            norms = norms[ok]
            r = u[ok] ** inv_d
            directions = (g / norms[:, None]) * r[:, None]
            cand = fit.mean[None, :] + directions @ fit.chol.T
            in_cube = np.all((cand >= 0.0) & (cand < 1.0), axis=1)
            if np.any(in_cube):
                take = cand[in_cube][:want]
                chunks.append(take)
                count += take.shape[0]
        if count == 0:
            return np.empty((0, ndim), dtype=np.float64)
        return np.vstack(chunks)

    def close(self) -> None:
        pass  # nothing to release; kept symmetric with the native handle


def create_rng(seed: int) -> RefRng:
    """Create a fallback RNG. Never raises."""
    return RefRng(seed)
