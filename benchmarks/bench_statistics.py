#!/usr/bin/env python3
"""Reproducible benchmark: CPython stdlib `statistics` (oracle) vs statistics_mojo.

Data is generated locally from fixed seeds (no network, no datasets): float
and int columns of 1k / 10k / 100k / 1M values. Two measurements per
function:

* cold first-call — median of 5 fresh-interpreter first calls at n=100k
  (imports happen before the timer; for statistics_mojo this includes
  dlopen + ABI handshake + the CPython-layout self-test of the native
  kernel),
* warm steady-state — per-call latency, median of 5 batches of repeated
  calls at each size.

Correctness is asserted (exact equality with the oracle) before any timing
happens, so the numbers below always come from a verified-correct build.

Run from the repository root:

    PYTHONPATH=python/statistics_mojo pixi run python benchmarks/bench_statistics.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import numpy as np

SIZES = [1_000, 10_000, 100_000, 1_000_000]
COLD_SIZE = 100_000
N_RUNS = 5  # median over this many batches / fresh processes
BATCH = 20  # calls per warm batch
SEED = 20260920

SCALAR_FNS = ["mean", "fmean", "median", "variance", "stdev", "quantiles"]
INT_FNS = ["mean", "variance", "median"]
BATCH_COLS = 32
BATCH_ROWS = 32_000


def make_floats(seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).tolist()


def make_ints(seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(-10_000, 10_001, n).tolist()


def call_ours(mod, fn: str, data) -> None:
    if fn == "quantiles":
        mod.quantiles(data, n=100)
    else:
        getattr(mod, fn)(data)


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
    lines.append(f"- python: {platform.python_version()}, numpy: {np.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def cold_first_call(fn: str, ours: bool) -> float:
    """Median first-call latency (ms) over N_RUNS fresh interpreter processes."""
    call = (
        f"m.quantiles(data, n=100)" if fn == "quantiles" else f"m.{fn}(data)"
    )
    if ours:
        snippet = (
            "import time,sys;sys.path.insert(0,'python/statistics_mojo');"
            "import numpy as np, statistics_mojo as m;"
            f"data=np.random.default_rng({SEED}).standard_normal({COLD_SIZE}).tolist();"
            "t0=time.perf_counter();"
            f"{call};"
            "print(1e3*(time.perf_counter()-t0))"
        )
    else:
        snippet = (
            "import time,statistics as m;"
            "import numpy as np;"
            f"data=np.random.default_rng({SEED}).standard_normal({COLD_SIZE}).tolist();"
            "t0=time.perf_counter();"
            f"{call};"
            "print(1e3*(time.perf_counter()-t0))"
        )
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


def warm_latency(call, n_calls: int = BATCH) -> float:
    """Median per-call latency (ms) over N_RUNS batches of n_calls calls."""
    call()  # one warmup call outside the timer
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            call()
        samples.append((time.perf_counter() - t0) / n_calls)
    return 1e3 * statistics.median(samples)


def main() -> None:
    import statistics_mojo

    info = statistics_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- oracle: CPython stdlib statistics ({platform.python_version()})")
    print(
        f"- statistics_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(
        f"- seed: {SEED}; warm: median of {N_RUNS} batches x {BATCH} calls; "
        f"cold: median of {N_RUNS} fresh-process first calls at n={COLD_SIZE:,}"
    )
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'statistics_mojo'")

    print("\n== correctness gate (exact equality with stdlib statistics) ==")
    gate_data = make_floats(SEED + 1, 20_000)
    gate_ints = make_ints(SEED + 2, 20_000)
    for fn in SCALAR_FNS:
        ours = getattr(statistics_mojo, fn)(gate_data, n=100) if fn == "quantiles" else getattr(statistics_mojo, fn)(gate_data)
        ref = getattr(statistics, fn)(gate_data, n=100) if fn == "quantiles" else getattr(statistics, fn)(gate_data)
        assert repr(ours) == repr(ref), f"{fn}: {ours!r} != {ref!r}"
        print(f"  {fn:>10}: exact match  [OK]")
    for fn in INT_FNS:
        ours = getattr(statistics_mojo, fn)(gate_ints)
        ref = getattr(statistics, fn)(gate_ints)
        assert repr(ours) == repr(ref), f"int {fn}: {ours!r} != {ref!r}"
        print(f"  int {fn:>7}: exact match  [OK]")
    for fn in ["variance", "stdev", "pvariance", "pstdev"]:
        ours = getattr(statistics_mojo, fn)(gate_data)
        ref = getattr(statistics, fn)(gate_data)
        assert repr(ours) == repr(ref), f"{fn}: {ours!r} != {ref!r}"
        print(f"  {fn:>10}: exact match  [OK]")

    print("\n== cold first-call latency (ms), n=100,000 floats, median of 5 fresh processes ==")
    print(f"{'function':>10} | {'stdlib statistics':>18} | {'statistics_mojo':>16} | {'speedup':>8}")
    print(f"{'-' * 10}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    cold_rows = []
    for fn in SCALAR_FNS:
        c_ref = cold_first_call(fn, ours=False)
        c_ours = cold_first_call(fn, ours=True)
        cold_rows.append((fn, c_ref, c_ours, c_ref / c_ours))
        print(f"{fn:>10} | {c_ref:>18.3f} | {c_ours:>16.4f} | {c_ref / c_ours:>7.1f}x")

    print("\n== warm steady-state latency (ms/call), float lists, median of 5 batches ==")
    header = f"{'function':>10} | {'n':>10} | {'stdlib statistics':>18} | {'statistics_mojo':>16} | {'speedup':>8}"
    print(header)
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    warm_rows = []
    for fn in SCALAR_FNS:
        for n in SIZES:
            data = make_floats(SEED + n, n)
            t_ref = warm_latency(lambda: call_ours(statistics, fn, data))
            t_ours = warm_latency(lambda: call_ours(statistics_mojo, fn, data))
            warm_rows.append((fn, n, t_ref, t_ours, t_ref / t_ours))
            print(f"{fn:>10} | {n:>10,} | {t_ref:>18.3f} | {t_ours:>16.4f} | {t_ref / t_ours:>7.1f}x")

    print("\n== warm steady-state latency (ms/call), int lists, median of 5 batches ==")
    print(header)
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    int_rows = []
    for fn in INT_FNS:
        for n in SIZES:
            data = make_ints(SEED + n, n)
            t_ref = warm_latency(lambda: call_ours(statistics, fn, data))
            t_ours = warm_latency(lambda: call_ours(statistics_mojo, fn, data))
            int_rows.append((fn, n, t_ref, t_ours, t_ref / t_ours))
            print(f"{fn:>10} | {n:>10,} | {t_ref:>18.3f} | {t_ours:>16.4f} | {t_ref / t_ours:>7.1f}x")

    print("\n== warm steady-state latency (ms/call), float64 ndarray input ==")
    print(header)
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 18}-+-{'-' * 16}-+-{'-' * 8}")
    arr_rows = []
    for fn in ["mean", "fmean", "variance", "median"]:
        for n in SIZES:
            data = np.random.default_rng(SEED + n).standard_normal(n)
            t_ref = warm_latency(lambda: call_ours(statistics, fn, data))
            t_ours = warm_latency(lambda: call_ours(statistics_mojo, fn, data))
            arr_rows.append((fn, n, t_ref, t_ours, t_ref / t_ours))
            print(f"{fn:>10} | {n:>10,} | {t_ref:>18.3f} | {t_ours:>16.4f} | {t_ref / t_ours:>7.1f}x")

    print(f"\n== warm column-batch latency (ms/batch of {BATCH_COLS} columns x {BATCH_ROWS:,} rows) ==")
    print(f"{'function':>16} | {'stdlib loop':>12} | {'statistics_mojo':>16} | {'speedup':>8}")
    print(f"{'-' * 16}-+-{'-' * 12}-+-{'-' * 16}-+-{'-' * 8}")
    batch_rows = []
    cols = [make_floats(SEED + 100 + i, BATCH_ROWS) for i in range(BATCH_COLS)]
    for fn in ["mean_batch", "variance_batch", "stdev_batch", "median_batch", "fmean_batch"]:
        scalar = fn.removesuffix("_batch")
        t_ref = warm_latency(
            lambda: [getattr(statistics, scalar)(c) for c in cols], n_calls=5
        )
        t_ours = warm_latency(lambda: getattr(statistics_mojo, fn)(cols), n_calls=5)
        batch_rows.append((fn, t_ref, t_ours, t_ref / t_ours))
        print(f"{fn:>16} | {t_ref:>12.3f} | {t_ours:>16.4f} | {t_ref / t_ours:>7.1f}x")

    print("\n== README paste block ==")
    print("Cold first call (n=100,000 floats, median of 5 fresh processes):")
    print()
    print("| function | stdlib statistics (ms) | statistics-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|")
    for fn, c_ref, c_ours, speedup in cold_rows:
        print(f"| {fn} | {c_ref:.3f} | {c_ours:.4f} | {speedup:.1f}x |")
    print()
    print("Warm steady state, float lists (ms per call, median of 5 batches of 20 calls):")
    print()
    print("| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for fn, n, t_ref, t_ours, speedup in warm_rows:
        print(f"| {fn} | {n:,} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")
    print()
    print("Warm steady state, int lists (ms per call):")
    print()
    print("| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for fn, n, t_ref, t_ours, speedup in int_rows:
        print(f"| {fn} | {n:,} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")
    print()
    print("Warm steady state, float64 ndarray input (ms per call):")
    print()
    print("| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for fn, n, t_ref, t_ours, speedup in arr_rows:
        print(f"| {fn} | {n:,} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")
    print()
    print(f"Warm column batch ({BATCH_COLS} columns x {BATCH_ROWS:,} rows, ms per batch):")
    print()
    print("| function | stdlib loop (ms) | statistics-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|")
    for fn, t_ref, t_ours, speedup in batch_rows:
        print(f"| {fn} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")


if __name__ == "__main__":
    main()
