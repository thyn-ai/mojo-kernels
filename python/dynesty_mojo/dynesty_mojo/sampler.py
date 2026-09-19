"""Nested-sampling driver: ellipsoidal bounding + batched proposals.

Implements the static-core nested sampling algorithm (Skilling 2006) with
single-ellipsoid bounding and uniform-in-ellipsoid rejection proposals
(Mukherjee, Parkinson & Liddle 2006; Feroz, Hobson & Bridges 2009), driven
through a dynesty-shaped API (``dynesty_mojo.sample`` mirrors the
essentials of ``DynamicNestedSampler.run_nested``). The bounding/proposal
machinery — the per-iteration numerical work that dominates sampler
overhead for cheap likelihoods (dynesty issue #432) — runs in the Mojo
kernel (or the vendored NumPy fallback); the user's ``loglike`` and
``prior_transform`` stay ordinary Python callables, exactly as in dynesty.

Algorithm (constant ``nlive``; deterministic mean-shrinkage prior volume):

  1. Draw ``nlive`` live points uniformly from the unit hypercube, map
     through ``prior_transform``, evaluate ``loglike``.
  2. Each iteration: the lowest-likelihood live point becomes a dead point
     with weight ``L_i * (X_{i-1} - X_i)``, ``log X_k = -k / nlive``. The
     remaining live points are bounded by the enlarged single ellipsoid
     and a replacement is drawn uniformly from the bound intersected with
     the unit cube, batched (candidates within a batch are iid, so the
     first candidate above the likelihood threshold is an exact
     constrained-prior draw — batching changes interpreter overhead, not
     the sampling distribution).
  3. Stop when the estimated remaining evidence, ``loglmax_live -
     i/nlive`` relative to the accumulated ``logZ``, is below ``dlogz``
     (or at ``maxiter`` / ``maxcall`` / ``logl_max``). The final live
     points are added with weight ``L * X_i / nlive`` each.

Evidence error: ``logz_err = sqrt(H / nlive)`` with ``H`` the information
(KL divergence) estimated from the same weights (Skilling 2006). This is a
run-internal estimate; it does not include the shrinkage-variance term —
same convention as the common textbook estimator.

Determinism: for a fixed integer ``seed`` each backend is bit-reproducible
(native xoshiro256** / fallback PCG64 — the backends are statistically
equivalent, not bit-identical to each other). ``seed=None`` draws a fresh
64-bit seed from OS entropy.
"""

from __future__ import annotations

import math
import secrets
from typing import Callable

import numpy as np

from dynesty_mojo import _native, _reference
from dynesty_mojo._native import MODE_CUBE, MODE_ELLIPSOID

__all__ = ["NSResults", "SamplerError", "sample"]

# Hard safety cap so a pathological likelihood cannot hang the sampler:
# proposal rounds per iteration (each round evaluates up to `batch` points).
_MAX_ROUNDS_PER_ITER = 10_000


class SamplerError(RuntimeError):  # noqa: N818
    """The user callbacks violated the sampler contract, no stopping
    criterion was given, or the sampler could make no progress."""


class NSResults(dict):
    """Sampling results: a dict with attribute access (dynesty convention).

    Keys: ``samples`` (posterior-weighted physical samples, dead+final
    live), ``samples_u`` (same points in the unit cube), ``logl``,
    ``logwt``, ``weights`` (normalized posterior weights),
    ``logz`` / ``logz_err`` (evidence estimate and its MC error),
    ``information`` (H), ``niter``, ``ncall``, ``eff``, plus the run
    configuration (``nlive``, ``ndim``, ``enlarge``, ``batch``, ``seed``,
    ``backend``, ``bound``).
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _validate_args(
    loglike,
    prior_transform,
    ndim: int,
    nlive: int,
    dlogz: float | None,
    logl_max: float,
    maxiter: int | None,
    maxcall: int | None,
    enlarge: float,
    batch: int,
    seed: int | None,
) -> int:
    """Fail fast on invalid configuration; returns the concrete 64-bit seed."""
    if not callable(loglike):
        raise SamplerError("loglike must be callable")
    if not callable(prior_transform):
        raise SamplerError("prior_transform must be callable")
    if not isinstance(ndim, (int, np.integer)) or isinstance(ndim, bool) or ndim < 1:
        raise SamplerError(f"ndim must be a positive integer; got {ndim!r}")
    if not isinstance(nlive, (int, np.integer)) or isinstance(nlive, bool):
        raise SamplerError(f"nlive must be an integer; got {nlive!r}")
    if nlive < ndim + 2:
        raise SamplerError(
            f"nlive must be at least ndim + 2 = {ndim + 2} for a full-rank "
            f"bounding covariance; got nlive={nlive}"
        )
    if not math.isfinite(enlarge) or enlarge < 1.0:
        raise SamplerError(f"enlarge must be a finite float >= 1.0; got {enlarge!r}")
    if not isinstance(batch, (int, np.integer)) or batch < 1:
        raise SamplerError(f"batch must be a positive integer; got {batch!r}")
    if dlogz is not None and (not math.isfinite(dlogz) or dlogz <= 0.0):
        raise SamplerError(f"dlogz must be a positive float or None; got {dlogz!r}")
    if math.isnan(logl_max):
        raise SamplerError("logl_max must not be NaN")
    if maxiter is not None and maxiter <= 0:
        raise SamplerError(f"maxiter must be positive or None; got {maxiter!r}")
    if maxcall is not None and maxcall <= 0:
        raise SamplerError(f"maxcall must be positive or None; got {maxcall!r}")
    if dlogz is None and maxiter is None and maxcall is None and math.isinf(logl_max):
        raise SamplerError(
            "no stopping criterion: set at least one of dlogz, maxiter, "
            "maxcall, or a finite logl_max"
        )
    if seed is None:
        return secrets.randbits(64)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise SamplerError(f"seed must be a non-negative integer or None; got {seed!r}")
    if not 0 <= int(seed) < 2**64:
        raise SamplerError(f"seed must fit in uint64; got {seed!r}")
    return int(seed)


def _check_logl(value, where: str) -> float:
    """One likelihood value must be a finite float or -inf."""
    try:
        logl = float(value)
    except (TypeError, ValueError) as exc:
        raise SamplerError(f"loglike returned a non-scalar at {where}: {value!r}") from exc
    if math.isnan(logl):
        raise SamplerError(f"loglike returned NaN at {where}")
    if logl == math.inf:
        raise SamplerError(f"loglike returned +inf at {where}; evidence is undefined")
    return logl


def _check_point(value, ndim: int, where: str) -> np.ndarray:
    """One prior_transform output must be a finite length-ndim vector."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (ndim,):
        raise SamplerError(
            f"prior_transform returned shape {np.asarray(value).shape} at {where}; "
            f"expected ({ndim},)"
        )
    if not np.all(np.isfinite(arr)):
        raise SamplerError(f"prior_transform returned non-finite values at {where}")
    return arr


def _eval_batch(loglike, prior_transform, cand, vectorized, ndim, where):
    """(V, logl) for a (k, ndim) unit-cube candidate batch, validated."""
    if vectorized:
        v = np.asarray(prior_transform(cand), dtype=np.float64)
        if v.shape != cand.shape or not np.all(np.isfinite(v)):
            raise SamplerError(
                f"vectorized prior_transform must return shape {cand.shape} of "
                f"finite values at {where}; got {v.shape}"
            )
        logl = np.asarray(loglike(v), dtype=np.float64).reshape(-1)
        if logl.shape != (cand.shape[0],):
            raise SamplerError(
                f"vectorized loglike must return ({cand.shape[0]},) at {where}; "
                f"got {logl.shape}"
            )
        if np.any(np.isnan(logl)):
            raise SamplerError(f"loglike returned NaN at {where}")
        if np.any(logl == math.inf):
            raise SamplerError(f"loglike returned +inf at {where}; evidence is undefined")
        return v, logl
    v_rows = np.empty_like(cand)
    logl = np.empty(cand.shape[0], dtype=np.float64)
    for j in range(cand.shape[0]):
        v_rows[j] = _check_point(prior_transform(cand[j]), ndim, f"{where} candidate {j}")
        logl[j] = _check_logl(loglike(v_rows[j]), f"{where} candidate {j}")
    return v_rows, logl


def sample(
    loglike: Callable,
    prior_transform: Callable,
    ndim: int,
    nlive: int = 500,
    dlogz: float | None = 0.01,
    logl_max: float = math.inf,
    maxiter: int | None = None,
    maxcall: int | None = None,
    enlarge: float = 1.25,
    batch: int = 64,
    seed: int | None = None,
    vectorized: bool = False,
) -> NSResults:
    """Run nested sampling. Mirrors the essentials of dynesty's
    ``DynamicNestedSampler`` for the single-ellipsoid bounding mode.

    Parameters
    ----------
    loglike : callable
        ``loglike(v) -> float`` for a physical parameter vector ``v`` of
        length ``ndim`` (or, with ``vectorized=True``, ``loglike(V)`` for a
        ``(k, ndim)`` array returning ``(k,)``). Must return finite values
        or ``-inf``; NaN/+inf raise :class:`SamplerError`.
    prior_transform : callable
        ``prior_transform(u) -> v`` mapping a unit-cube vector to physical
        parameters (vectorized: ``(k, ndim) -> (k, ndim)``).
    ndim : int
        Dimension of the parameter space.
    nlive : int
        Number of live points (>= ndim + 2). Default 500.
    dlogz : float or None
        Stop when the estimated remaining evidence, relative to the
        accumulated log-evidence, drops below ``dlogz``. Default 0.01.
    logl_max : float
        Stop when the dead-point likelihood reaches this ceiling.
    maxiter, maxcall : int or None
        Iteration / likelihood-call caps.
    enlarge : float
        Bounding-ellipsoid enlargement factor (multiplies the covariance
        of the live points, on top of the max-Mahalanobis containment
        scaling). >= 1.0. Default 1.25.
    batch : int
        Candidates proposed per kernel/NumPy call. Statistical results are
        independent of ``batch`` (first-passing-candidate rule); larger
        batches trade wasted likelihood evaluations for less interpreter
        overhead. Default 64.
    seed : int or None
        uint64 seed for bit-reproducible runs; None draws OS entropy.
    vectorized : bool
        Whether both callbacks accept ``(k, ndim)`` batches.

    Returns
    -------
    NSResults
        Dict with attribute access; see the class docstring.
    """
    concrete_seed = _validate_args(
        loglike,
        prior_transform,
        ndim,
        nlive,
        dlogz,
        logl_max,
        maxiter,
        maxcall,
        enlarge,
        batch,
        seed,
    )
    ndim = int(ndim)
    nlive = int(nlive)
    batch = int(batch)

    use_native = _native.native_available()
    backend = _native if use_native else _reference
    backend_name = "native" if use_native else "fallback"
    rng = backend.create_rng(concrete_seed)

    dead_u: list[np.ndarray] = []
    dead_v: list[np.ndarray] = []
    dead_logl: list[float] = []
    # log(X_{j-1} - X_j) = -(j-1)/nlive + log(1 - exp(-1/nlive)).
    log_shrink = math.log(-math.expm1(-1.0 / nlive))
    logz = -math.inf  # running log-evidence over dead points (stopping rule)
    niter = 0
    ncall = 0

    try:
        # --- initial live points: uniform over the whole unit cube ---
        u_live = rng.propose_batch(ndim, None, MODE_CUBE, nlive)
        v_live, logl_live = _eval_batch(
            loglike, prior_transform, u_live, vectorized, ndim, "initialization"
        )
        ncall += nlive
        if not np.any(np.isfinite(logl_live)):
            raise SamplerError(
                "loglike is -inf at every initial live point; the prior and "
                "likelihood have no overlap"
            )

        while True:
            worst = int(np.argmin(logl_live))
            loglstar = float(logl_live[worst])

            # Commit the dead point: weight L_j (X_{j-1} - X_j), and remove
            # it from the live set (nlive >= ndim + 2 >= 3, so >= 2 remain).
            dead_u.append(u_live[worst].copy())
            dead_v.append(v_live[worst].copy())
            dead_logl.append(loglstar)
            logz = float(np.logaddexp(logz, loglstar - niter / nlive + log_shrink))
            niter += 1
            keep = np.ones(nlive, dtype=bool)
            keep[worst] = False
            u_live = u_live[keep]
            v_live = v_live[keep]
            logl_live = logl_live[keep]

            # Stopping criteria, evaluated on the remaining live set at the
            # current prior volume X_j = exp(-j/nlive).
            if loglstar >= logl_max:
                break
            if maxiter is not None and niter >= maxiter:
                break
            if maxcall is not None and ncall >= maxcall:
                break
            if dlogz is not None:
                remaining = float(logl_live.max()) - niter / nlive
                if float(np.logaddexp(logz, remaining)) - logz < dlogz:
                    break

            # Fit the bound on the remaining live points and draw a
            # replacement above the likelihood threshold.
            fit = backend.fit_ellipsoid(u_live, enlarge)
            mode = MODE_ELLIPSOID if fit is not None else MODE_CUBE

            accepted = False
            rounds = 0
            while not accepted:
                cand = rng.propose_batch(ndim, fit, mode, batch)
                rounds += 1
                if cand.shape[0] == 0:
                    # Ellipsoid/cube intersection too small to hit: widen to
                    # the whole cube for this iteration (always valid).
                    if mode != MODE_CUBE:
                        mode = MODE_CUBE
                        continue
                    raise SamplerError("cube proposal returned no candidates")
                if vectorized:
                    v_cand, logl_cand = _eval_batch(
                        loglike,
                        prior_transform,
                        cand,
                        True,
                        ndim,
                        f"iteration {niter}",
                    )
                    ncall += cand.shape[0]
                    above = np.flatnonzero(logl_cand > loglstar)
                    if above.size:
                        j = int(above[0])
                        u_new = cand[j]
                        v_new = v_cand[j]
                        logl_new = float(logl_cand[j])
                        accepted = True
                else:
                    # Lazy evaluation: stop at the first candidate above the
                    # threshold, so cheap likelihoods are never over-called.
                    for j in range(cand.shape[0]):
                        v_j = _check_point(
                            prior_transform(cand[j]), ndim, f"iteration {niter}"
                        )
                        logl_j = _check_logl(loglike(v_j), f"iteration {niter}")
                        ncall += 1
                        if logl_j > loglstar:
                            u_new = cand[j]
                            v_new = v_j
                            logl_new = logl_j
                            accepted = True
                            break
                if not accepted and rounds > _MAX_ROUNDS_PER_ITER:
                    raise SamplerError(
                        f"no candidate above the likelihood threshold after "
                        f"{rounds} batches at iteration {niter}; the "
                        f"likelihood likely has a plateau at logl={loglstar}"
                    )

            u_live = np.vstack([u_live, u_new[None, :]])
            v_live = np.vstack([v_live, v_new[None, :]])
            logl_live = np.concatenate([logl_live, [logl_new]])
    finally:
        rng.close()

    # --- epilogue: final live points, weights, evidence, information ---
    dead_u_arr = np.asarray(dead_u, dtype=np.float64).reshape(niter, ndim)
    dead_v_arr = np.asarray(dead_v, dtype=np.float64).reshape(niter, ndim)
    dead_logl_arr = np.asarray(dead_logl, dtype=np.float64)

    all_u = np.vstack([dead_u_arr, u_live])
    all_v = np.vstack([dead_v_arr, v_live])
    all_logl = np.concatenate([dead_logl_arr, logl_live])

    logwt_dead = dead_logl_arr - np.arange(niter) / nlive + log_shrink
    logwt_live = logl_live - niter / nlive - math.log(nlive)
    all_logwt = np.concatenate([logwt_dead, logwt_live])

    logz_final = float(np.logaddexp.reduce(all_logwt))
    weights = np.exp(all_logwt - logz_final)
    finite = np.isfinite(all_logl)
    information = float(np.sum(weights[finite] * (all_logl[finite] - logz_final)))
    logz_err = math.sqrt(information / nlive)

    return NSResults(
        samples=all_v,
        samples_u=all_u,
        logl=all_logl,
        logwt=all_logwt,
        weights=weights,
        logz=logz_final,
        logz_err=logz_err,
        information=information,
        niter=niter,
        ncall=ncall,
        eff=niter / ncall if ncall else math.nan,
        nlive=nlive,
        ndim=ndim,
        enlarge=float(enlarge),
        batch=batch,
        seed=concrete_seed,
        backend=backend_name,
        bound="single-ellipsoid",
        vectorized=bool(vectorized),
    )
