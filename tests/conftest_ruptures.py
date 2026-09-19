"""Shared fixtures for the ruptures-mojo differential suite.

Deterministic seeded signals, generated locally from explicit seeds — no
network, no randomness without a fixed seed — so the suite is reproducible on
any machine. The oracle is the published PyPI package ruptures==1.1.10.
"""

from __future__ import annotations

import numpy as np


def make_signal(kind: str, seed: int, n: int) -> np.ndarray:
    """One synthetic 1-D signal, deterministic for (kind, seed, n)."""
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        return rng.standard_normal(n)
    if kind == "mean_shifts":
        # Piecewise-constant mean with Gaussian noise: the CPD textbook case.
        s = np.zeros(n)
        n_cps = int(rng.integers(2, 5))
        pts = np.sort(rng.choice(np.arange(n // 8, n - n // 8), size=n_cps, replace=False))
        val, prev = 0.0, 0
        for p in pts:
            s[prev:p] = val
            val += rng.standard_normal() * 3.0
            prev = p
        s[prev:] = val
        return s + rng.standard_normal(n) * 0.5
    if kind == "variance_shifts":
        s = rng.standard_normal(n)
        s[n // 2 :] *= 4.0
        s[: n // 4] *= 0.25
        return s
    if kind == "ints":
        # Integer-valued: exact cost ties are common — stresses tie-breaking.
        return rng.integers(0, 4, size=n).astype(np.float64)
    if kind == "const_plateau":
        # Constant stretches: every sub-segment of a plateau has cost 0.
        s = np.zeros(n)
        s[n // 3 : 2 * n // 3] = 2.5
        return s
    if kind == "trend":
        return np.linspace(0.0, 5.0, n) + rng.standard_normal(n) * 0.3
    raise ValueError(f"unknown signal kind {kind!r}")


SIGNAL_KINDS = ["gauss", "mean_shifts", "variance_shifts", "ints", "const_plateau", "trend"]


def expected_backend() -> str:
    import os

    return "fallback" if os.environ.get("RUPTURES_MOJO_DISABLE_NATIVE") == "1" else "native"
