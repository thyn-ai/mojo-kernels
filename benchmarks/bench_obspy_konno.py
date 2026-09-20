#!/usr/bin/env python3
"""Reproducible benchmark: obspy vs obspy_mojo (Konno-Ohmachi smoothing).

Spectra are generated locally from fixed seeds (no network, no datasets):
random spectra on log-spaced frequency grids of 512 / 2048 / 8192 bins.
Timings are the median of 5 runs; each cell reports mean per-call latency.
Correctness is asserted (element-wise vs obspy within the documented
tolerance) before any timing happens, so the numbers below always come from
a verified-correct build.

Two timing modes, both median of 5:

  * cold first call: a fresh Python process imports the package and times
    its very first smoothing call (includes dlopen/ABI handshake for the
    native backend; subprocess wall time, median of 5 launches);
  * warm steady state: repeated calls in one process, median of 5 batches.

Run from the repository root (the oracle lives in .oracle-python):

    PYTHONPATH=".oracle-python:python/obspy_mojo" PYTHONNOUSERSITE=1 \\
        ~/.pixi/bin/pixi run python benchmarks/bench_obspy_konno.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

N_RUNS = 5  # median over this many runs
LOOP_SIZES = [512, 2048, 8192]  # default single-spectrum path
MATRIX_SIZES = [512, 2048]  # 2-D batch path
N_MATRIX_SPECTRA = 32
SEED = 20260919
RTOL, ATOL = 1e-9, 1e-12


def make_freqs(n: int) -> np.ndarray:
    return np.logspace(-1, 2, n)


def make_spectra(seed: int, shape) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) * 100.0


def time_warm(call, n_runs: int = N_RUNS) -> float:
    """Median seconds per call in steady state (warmed up first)."""
    call()  # warm-up
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        call()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_cold(package: str, n: int) -> float:
    """Median wall seconds from process start to the end of the first call.

    `package` is "obspy" or "obspy_mojo"; imports happen inside the timed
    region, so module import and (for obspy_mojo) dlopen + ABI handshake are
    included.
    """
    if package == "obspy":
        import_line = (
            "from obspy.signal.konnoohmachismoothing import "
            "konno_ohmachi_smoothing as smooth"
        )
    else:
        import_line = "from obspy_mojo import konno_ohmachi_smoothing as smooth"
    code = (
        "import time, numpy as np\n"
        "t0 = time.perf_counter()\n"
        f"{import_line}\n"
        f"freqs = np.logspace(-1, 2, {n})\n"
        f"spec = np.random.default_rng(1).random({n}) * 100.0\n"
        "smooth(spec, freqs, bandwidth=40, normalize=True)\n"
        "print(f'{time.perf_counter() - t0:.6f}')\n"
    )
    env = dict(os.environ)
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        samples.append(float(out.stdout.strip().splitlines()[-1]))
    return statistics.median(samples)


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


def main() -> None:
    import obspy
    from obspy.signal.konnoohmachismoothing import (
        konno_ohmachi_smoothing as oracle,
    )

    import obspy_mojo

    ours = obspy_mojo.konno_ohmachi_smoothing

    info = obspy_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- obspy (oracle): {obspy.__version__}")
    print(
        f"- obspy_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- runs: median of {N_RUNS}; seeds: SEED={SEED}+size")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'obspy_mojo'"
        )

    print("\n== correctness gate (element-wise vs obspy, float64) ==")
    worst = 0.0
    for n in LOOP_SIZES:
        freqs = make_freqs(n)
        spectra = make_spectra(SEED + n, n)
        for kw in (dict(normalize=True), dict(normalize=False), dict(count=2, normalize=True)):
            o = oracle(spectra, freqs, **kw)
            m = ours(spectra, freqs, **kw)
            rel = float(np.max(np.abs(m - o) / np.maximum(np.abs(o), 1e-300)))
            worst = max(worst, rel)
    for n in MATRIX_SIZES:
        freqs = make_freqs(n)
        spectra = make_spectra(SEED + n + 7, (4, n))
        for kw in (dict(normalize=True), dict(normalize=False), dict(count=2, normalize=True)):
            o = oracle(spectra, freqs, **kw)
            m = ours(spectra, freqs, **kw)
            rel = float(np.max(np.abs(m - o) / np.maximum(np.abs(o), 1e-300)))
            worst = max(worst, rel)
    status = "OK" if worst <= RTOL else "FAIL"
    print(f"  worst relative deviation across all gate cases: {worst:.3e}  [{status}]")
    if worst > RTOL:
        sys.exit("correctness gate failed")

    # --- loop path: the default single-spectrum hot path -------------------
    print("\n== loop path: single spectrum, bandwidth=40 (default) ==")
    print("== warm steady-state latency (ms/call, median of 5) ==")
    header = (
        f"{'freq bins':>10} | {'normalize':>10} | {'obspy':>10} | "
        f"{'obspy_mojo':>10} | {'fallback':>10} | {'speedup':>8}"
    )
    print(header)
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
    loop_rows = []
    for n in LOOP_SIZES:
        freqs = make_freqs(n)
        spectra = make_spectra(SEED + n, n)
        for normalize in (False, True):
            kw = dict(normalize=normalize)
            t_oracle = time_warm(lambda: oracle(spectra, freqs, **kw))
            t_ours = time_warm(lambda: ours(spectra, freqs, **kw))
            os.environ["OBSPY_MOJO_DISABLE_NATIVE"] = "1"
            try:
                t_fb = time_warm(lambda: ours(spectra, freqs, **kw))
            finally:
                del os.environ["OBSPY_MOJO_DISABLE_NATIVE"]
            label = "True" if normalize else "False"
            loop_rows.append((n, label, t_oracle, t_ours, t_fb))
            print(
                f"{n:>10,} | {label:>10} | {1e3 * t_oracle:>10.2f} | "
                f"{1e3 * t_ours:>10.3f} | {1e3 * t_fb:>10.2f} | {t_oracle / t_ours:>7.1f}x"
            )

    print("\n== loop path: cold first call (fresh process, import + first call, median of 5) ==")
    print(f"{'freq bins':>10} | {'obspy (s)':>10} | {'obspy_mojo (s)':>14} | {'speedup':>8}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 8}")
    cold_rows = []
    for n in LOOP_SIZES:
        c_oracle = time_cold("obspy", n)
        c_ours = time_cold("obspy_mojo", n)
        cold_rows.append((n, c_oracle, c_ours))
        print(f"{n:>10,} | {c_oracle:>10.3f} | {c_ours:>14.3f} | {c_oracle / c_ours:>7.1f}x")

    # --- matrix path: 2-D batch --------------------------------------------
    print(f"\n== matrix path: {N_MATRIX_SPECTRA} spectra x n bins, bandwidth=40, normalize=True ==")
    print(f"{'freq bins':>10} | {'obspy (ms)':>10} | {'obspy_mojo (ms)':>15} | {'speedup':>8}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 15}-+-{'-' * 8}")
    matrix_rows = []
    for n in MATRIX_SIZES:
        freqs = make_freqs(n)
        spectra = make_spectra(SEED + n + 13, (N_MATRIX_SPECTRA, n))
        kw = dict(normalize=True)
        t_oracle = time_warm(lambda: oracle(spectra, freqs, **kw))
        t_ours = time_warm(lambda: ours(spectra, freqs, **kw))
        matrix_rows.append((n, t_oracle, t_ours))
        print(
            f"{n:>10,} | {1e3 * t_oracle:>10.2f} | {1e3 * t_ours:>15.2f} "
            f"| {t_oracle / t_ours:>7.1f}x"
        )

    print("\n== README paste block ==")
    print("Loop path (default single-spectrum, b=40), warm steady-state, median of 5:")
    print()
    print("| freq bins | normalize | obspy ms/call | obspy-mojo ms/call | speedup |")
    print("|---:|---|---:|---:|---:|")
    for n, label, t_oracle, t_ours, _t_fb in loop_rows:
        print(f"| {n:,} | {label} | {1e3 * t_oracle:.2f} | {1e3 * t_ours:.3f} | {t_oracle / t_ours:.1f}x |")
    print()
    print("Cold first call (fresh process: import + first smoothing), median of 5:")
    print()
    print("| freq bins | obspy (s) | obspy-mojo (s) | speedup |")
    print("|---:|---:|---:|---:|")
    for n, c_oracle, c_ours in cold_rows:
        print(f"| {n:,} | {c_oracle:.3f} | {c_ours:.3f} | {c_oracle / c_ours:.1f}x |")
    print()
    print(f"Matrix path ({N_MATRIX_SPECTRA} spectra, normalize=True), warm, median of 5:")
    print()
    print("| freq bins | obspy ms/call | obspy-mojo ms/call | speedup |")
    print("|---:|---:|---:|---:|")
    for n, t_oracle, t_ours in matrix_rows:
        print(f"| {n:,} | {1e3 * t_oracle:.2f} | {1e3 * t_ours:.2f} | {t_oracle / t_ours:.1f}x |")
    print()
    print(f"Correctness gate worst relative deviation (float64): {worst:.3e}")


if __name__ == "__main__":
    main()
