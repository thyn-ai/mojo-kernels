#!/usr/bin/env python3
"""Reproducible benchmark: pykalman vs pykalman_mojo (filter + smooth).

Systems are generated locally from fixed seeds (no network, no datasets):
random stable linear-Gaussian systems (transition spectral radius 0.9, SPD
covariances, transition/observation offsets) with simulated observation
series. Timings: cold = fresh instance construction + first call; warm =
median of 5 subsequent calls on the same instance. Correctness is asserted
(element-wise vs pykalman within 1e-10) before any timing happens, so the
numbers below always come from a verified-correct build.

Run from the repository root (explicit env; no pixi task registration):

    PYTHONPATH=python/pykalman_mojo pixi run python benchmarks/bench_pykalman.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import numpy as np

N_RUNS = 5  # warm runs; median reported
ATOL = 1e-10
SEED = 20260919

# Main sweep: series length, fixed mid-size system.
T_SIZES = [1_000, 10_000, 100_000]
MAIN_DIMS = (5, 2)
# Dimension sweep at fixed series length.
DIMS_SIZES = [(2, 1), (5, 2), (10, 4), (20, 8)]
DIMS_T = 10_000


def make_system(seed: int, n_s: int, n_o: int, T: int):
    """Random stable linear-Gaussian system + simulated observations."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n_s, n_s)
    A *= 0.9 / max(abs(np.linalg.eigvals(A)))
    B = rng.randn(n_s, n_s)
    Q = B @ B.T + 0.2 * np.eye(n_s)
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


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    import numpy

    lines.append(f"- python: {platform.python_version()}, numpy: {numpy.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def bench_cell(kf_cls, params, obs, method: str):
    """(cold_s, warm_s) for one implementation: cold = construct + first call,
    warm = median of N_RUNS subsequent calls."""
    inst = kf_cls(**params)
    t0 = time.perf_counter()
    getattr(inst, method)(obs)
    cold = time.perf_counter() - t0
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        getattr(inst, method)(obs)
        samples.append(time.perf_counter() - t0)
    return cold, statistics.median(samples)


def main() -> None:
    from pykalman import KalmanFilter as RefKF

    import pykalman_mojo
    from pykalman_mojo import KalmanFilter

    info = pykalman_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- pykalman_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import pykalman

    print(f"- pykalman: {pykalman.__version__} (PyPI)")
    print(f"- seeds: SEED={SEED} (+dims/length per cell); warm: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'pykalman_mojo'"
        )

    cells = [("T", MAIN_DIMS[0], MAIN_DIMS[1], T) for T in T_SIZES] + [
        ("dims", n_s, n_o, DIMS_T) for (n_s, n_o) in DIMS_SIZES
    ]

    print("\n== correctness gate (element-wise vs pykalman, max abs diff) ==")
    systems = {}
    worst_all = 0.0
    for kind, n_s, n_o, T in cells:
        key = (n_s, n_o, T)
        if key not in systems:
            systems[key] = make_system(SEED + n_s * 31 + n_o * 7 + T, n_s, n_o, T)
        params, obs = systems[key]
        ref = RefKF(**params)
        ours = KalmanFilter(**params)
        ref_fm, ref_fc = ref.filter(obs)
        ref_sm, ref_sc = ref.smooth(obs)
        fm, fc = ours.filter(obs)
        sm, sc = ours.smooth(obs)
        fm_d = float(np.max(np.abs(fm - ref_fm)))
        fc_d = float(np.max(np.abs(fc - ref_fc)))
        sm_d = float(np.max(np.abs(sm - ref_sm)))
        sc_d = float(np.max(np.abs(sc - ref_sc)))
        worst = max(fm_d, fc_d, sm_d, sc_d)
        worst_all = max(worst_all, worst)
        status = "OK" if worst <= ATOL else "FAIL"
        print(
            f"  n_s={n_s:>2} n_o={n_o:>2} T={T:>7,}: filter {max(fm_d, fc_d):.2e}, "
            f"smooth {max(sm_d, sc_d):.2e}  [{status}]"
        )
        if worst > ATOL:
            sys.exit(f"correctness gate failed at n_s={n_s} n_o={n_o} T={T}")
    print(f"  worst over all cells: {worst_all:.2e} (tolerance {ATOL:.0e})")

    rows = []
    for method in ("filter", "smooth"):
        print(f"\n== {method}: cold (construct + first call) ==")
        print(
            f"{'n_s':>4} | {'n_o':>4} | {'T':>9} | {'pykalman':>10} | "
            f"{'pykalman_mojo':>13} | {'speedup':>8}"
        )
        print(f"{'-' * 4}-+-{'-' * 4}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 13}-+-{'-' * 8}")
        for kind, n_s, n_o, T in cells:
            params, obs = systems[(n_s, n_o, T)]
            cold_ref, warm_ref = bench_cell(RefKF, params, obs, method)
            cold_ours, warm_ours = bench_cell(KalmanFilter, params, obs, method)
            rows.append((method, kind, n_s, n_o, T, cold_ref, warm_ref, cold_ours, warm_ours))
            print(
                f"{n_s:>4} | {n_o:>4} | {T:>9,} | {cold_ref * 1e3:>9.2f}m | "
                f"{cold_ours * 1e3:>12.2f}m | {cold_ref / cold_ours:>7.1f}x"
            )
        print(f"\n== {method}: warm (median of {N_RUNS}) ==")
        print(
            f"{'n_s':>4} | {'n_o':>4} | {'T':>9} | {'pykalman ms':>12} | "
            f"{'pykalman_mojo ms':>16} | {'speedup':>8}"
        )
        print(f"{'-' * 4}-+-{'-' * 4}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 16}-+-{'-' * 8}")
        for method_r, kind, n_s, n_o, T, cold_ref, warm_ref, cold_ours, warm_ours in rows:
            if method_r != method:
                continue
            print(
                f"{n_s:>4} | {n_o:>4} | {T:>9,} | {warm_ref * 1e3:>12.2f} | "
                f"{warm_ours * 1e3:>16.3f} | {warm_ref / warm_ours:>7.1f}x"
            )

    print("\n== README paste block ==")
    for method in ("filter", "smooth"):
        print(
            f"\n| {method} T | n_s | n_o | pykalman cold (ms) | pykalman_mojo cold (ms) "
            f"| pykalman warm (ms) | pykalman_mojo warm (ms) | warm speedup |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method_r, kind, n_s, n_o, T, cold_ref, warm_ref, cold_ours, warm_ours in rows:
            if method_r != method:
                continue
            print(
                f"| {T:,} | {n_s} | {n_o} | {cold_ref * 1e3:.2f} | {cold_ours * 1e3:.2f} "
                f"| {warm_ref * 1e3:.2f} | {warm_ours * 1e3:.3f} | {warm_ref / warm_ours:.1f}x |"
            )


if __name__ == "__main__":
    main()
