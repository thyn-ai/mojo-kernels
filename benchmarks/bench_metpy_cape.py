#!/usr/bin/env python3
"""Reproducible benchmark: metpy.calc vs metpy_mojo (CAPE/CIN).

Soundings are generated locally from fixed seeds (no network, no datasets):
a 91-level analytic convective sounding for the single-column case, and a
(nlev, 25, 25) = 625-column gridded field for the issue-#480 use case.
Timings are the median of 5 runs; cold first-call is the first invocation
in a fresh interpreter process (median of 5 processes). Correctness is
asserted against the metpy oracle before any timing happens, so the
numbers always come from a verified-correct build.

Run from the repository root:

    PYTHONPATH=python/metpy_mojo pixi run python benchmarks/bench_metpy_cape.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

N_RUNS = 5
SEED = 20260919
# Same parity tolerances as the differential suite.
CAPE_RTOL, CAPE_ATOL = 2e-2, 1.0
CIN_RTOL, CIN_ATOL = 1e-2, 0.5

HERE = os.path.dirname(os.path.abspath(__file__))


def sounding():
    p = np.linspace(1000.0, 100.0, 91)
    z = -np.log(p / 1000.0) * 7.5
    T = np.maximum(300.0 - 6.5 * z, 222.0)
    Td = np.maximum(294.0 - 5.0 * z, 200.0)
    return p, T, Td


def grid(ny=25, nx=25):
    rng = np.random.RandomState(SEED)
    n = 91
    p = np.linspace(1000.0, 100.0, n)
    z = -np.log(p / 1000.0) * 7.5
    base_T = np.maximum(300.0 - 6.5 * z, 222.0)
    base_Td = np.maximum(294.0 - 5.0 * z, 200.0)
    T3 = np.empty((n, ny, nx))
    Td3 = np.empty((n, ny, nx))
    for y in range(ny):
        for x in range(nx):
            warm = rng.uniform(-1.5, 3.5)
            moist = rng.uniform(-3.0, 4.0)
            T3[:, y, x] = np.maximum(base_T + warm * np.exp(-z / 3.5), 222.0)
            Td3[:, y, x] = np.minimum(T3[:, y, x] - 0.5,
                                      base_Td + moist * np.exp(-z / 4.0))
    return p, T3, Td3


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
    for cmd, label in [(["mojo", "--version"], "mojo")]:
        try:
            v = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            lines.append(f"- {label}: {v}")
        except OSError:
            pass
    try:
        import metpy
        lines.append(f"- metpy (oracle): {metpy.__version__}")
    except ImportError:
        pass
    return "\n".join(lines)


def cold_first_call(script: str) -> float:
    """First-call wall time in a fresh interpreter (imports done first)."""
    code = (
        "import time, numpy as np\n"
        f"{script}\n"
    )
    # The subprocess runs from benchmarks/ (so `from bench_metpy_cape import
    # sounding` resolves) and inherits PYTHONPATH with relative entries made
    # absolute (the package under test and, locally, the metpy oracle).
    pkg = os.path.abspath(os.path.join(HERE, "..", "python", "metpy_mojo"))
    inherited = [
        os.path.abspath(e) if e and not os.path.isabs(e) else e
        for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([pkg] + inherited)
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env=env, cwd=HERE,
        ).stdout.strip()
        samples.append(float(out))
    return statistics.median(samples)


COLD_SCRIPT_OURS = """\
from bench_metpy_cape import sounding
import metpy_mojo
p, T, Td = sounding()
t0 = time.perf_counter()
metpy_mojo.cape_cin(p, T, Td)
print(time.perf_counter() - t0)
"""

COLD_SCRIPT_METPY = """\
from bench_metpy_cape import sounding
import metpy.calc as mpcalc
from metpy.units import units
p, T, Td = sounding()
pq, Tq, Tdq = p * units.hPa, T * units.kelvin, Td * units.kelvin
t0 = time.perf_counter()
mpcalc.surface_based_cape_cin(pq, Tq, Tdq)
print(time.perf_counter() - t0)
"""


def main() -> None:
    import metpy.calc as mpcalc
    from metpy.units import units

    import metpy_mojo

    info = metpy_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- metpy_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    print(f"- runs: median of {N_RUNS}; cold = first call in a fresh process")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'metpy_mojo'")

    p, T, Td = sounding()
    pq, Tq, Tdq = p * units.hPa, T * units.kelvin, Td * units.kelvin

    print("\n== correctness gate (surface + most-unstable, vs metpy oracle) ==")
    for which, oracle_fn in [("surface", mpcalc.surface_based_cape_cin),
                             ("most_unstable", mpcalc.most_unstable_cape_cin)]:
        cape, cin = metpy_mojo.cape_cin(p, T, Td, which=which)
        cape_o, cin_o = oracle_fn(pq, Tq, Tdq)
        dc = abs(cape - float(cape_o.m))
        di = abs(cin - float(cin_o.m))
        ok = (dc <= CAPE_ATOL + CAPE_RTOL * abs(cape_o.m)
              and di <= CIN_ATOL + CIN_RTOL * abs(cin_o.m))
        print(f"  {which:>14}: CAPE {cape:.2f} vs {cape_o.m:.2f} (|d|={dc:.3f}), "
              f"CIN {cin:.2f} vs {cin_o.m:.2f} (|d|={di:.3f})  [{'OK' if ok else 'FAIL'}]")
        if not ok:
            sys.exit("correctness gate failed")

    # --- single column ---
    print("\n== single column (91 levels): cold first call (fresh process) ==")
    t_cold_ours = cold_first_call(COLD_SCRIPT_OURS)
    t_cold_ref = cold_first_call(COLD_SCRIPT_METPY)
    print(f"  metpy.calc surface_based_cape_cin: {1e3 * t_cold_ref:9.3f} ms")
    print(f"  metpy_mojo cape_cin (native):      {1e3 * t_cold_ours:9.3f} ms")

    print("\n== single column (91 levels): warm steady-state ==")
    n_calls = 200
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            mpcalc.surface_based_cape_cin(pq, Tq, Tdq)
        samples.append((time.perf_counter() - t0) / n_calls)
    warm_ref = statistics.median(samples)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            metpy_mojo.cape_cin(p, T, Td)
        samples.append((time.perf_counter() - t0) / n_calls)
    warm_ours = statistics.median(samples)
    print(f"  metpy.calc: {1e3 * warm_ref:9.3f} ms/call")
    print(f"  metpy_mojo: {1e3 * warm_ours:9.3f} ms/call  ({warm_ref / warm_ours:7.1f}x)")

    # --- gridded field ---
    print("\n== gridded field (91, 25, 25) = 625 columns ==")
    p3, T3, Td3 = grid()
    # correctness gate on a few columns first
    for y, x in [(0, 0), (12, 12), (24, 24)]:
        cape_g, cin_g = metpy_mojo.cape_cin_grid(p3, T3, Td3)
        cape_o, cin_o = mpcalc.surface_based_cape_cin(
            p3 * units.hPa, T3[:, y, x] * units.kelvin, Td3[:, y, x] * units.kelvin)
        assert abs(float(cape_g[y, x]) - float(cape_o.m)) <= CAPE_ATOL + CAPE_RTOL * abs(cape_o.m)
        assert abs(float(cin_g[y, x]) - float(cin_o.m)) <= CIN_ATOL + CIN_RTOL * abs(cin_o.m)
    print("  correctness gate: 3/625 columns oracle-checked OK")

    def metpy_grid():
        ny, nx = T3.shape[1], T3.shape[2]
        out = np.empty((ny, nx))
        for y in range(ny):
            for x in range(nx):
                c, _ = mpcalc.surface_based_cape_cin(
                    p3 * units.hPa, T3[:, y, x] * units.kelvin,
                    Td3[:, y, x] * units.kelvin)
                out[y, x] = float(c.m)
        return out

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        metpy_grid()
        samples.append(time.perf_counter() - t0)
    grid_ref = statistics.median(samples)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        metpy_mojo.cape_cin_grid(p3, T3, Td3)
        samples.append(time.perf_counter() - t0)
    grid_ours = statistics.median(samples)
    n_cols = T3.shape[1] * T3.shape[2]
    print(f"  metpy.calc loop: {grid_ref:9.3f} s  ({1e3 * grid_ref / n_cols:8.3f} ms/column)")
    print(f"  metpy_mojo grid: {grid_ours:9.4f} s  ({1e3 * grid_ours / n_cols:8.4f} ms/column)"
          f"  ({grid_ref / grid_ours:7.1f}x)")

    # --- big grid, ours only (the issue-#480 use case at scale) ---
    print("\n== big grid (91, 100, 100) = 10,000 columns, metpy_mojo only ==")
    p_big, T_big, Td_big = grid(100, 100)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        metpy_mojo.cape_cin_grid(p_big, T_big, Td_big)
        samples.append(time.perf_counter() - t0)
    big = statistics.median(samples)
    print(f"  metpy_mojo grid: {big:9.4f} s  ({1e3 * big / 10000:8.4f} ms/column)")

    print("\n== README paste block ==")
    print("| workload | metpy.calc | metpy_mojo (native) | speedup |")
    print("|---|---:|---:|---:|")
    print(f"| single column, cold first call | {1e3 * t_cold_ref:.3f} ms "
          f"| {1e3 * t_cold_ours:.3f} ms | {t_cold_ref / t_cold_ours:.1f}x |")
    print(f"| single column, warm | {1e3 * warm_ref:.3f} ms | {1e3 * warm_ours:.3f} ms "
          f"| {warm_ref / warm_ours:.1f}x |")
    print(f"| grid 25x25 (625 cols) | {grid_ref:.3f} s | {grid_ours:.4f} s "
          f"| {grid_ref / grid_ours:.1f}x |")
    print(f"| grid 100x100 (10k cols) | — (est. {grid_ref * 16 / 60:.0f} min by rate) "
          f"| {big:.4f} s | — |")


if __name__ == "__main__":
    main()
