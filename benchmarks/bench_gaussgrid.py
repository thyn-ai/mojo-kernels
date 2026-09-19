#!/usr/bin/env python3
"""Reproducible benchmark: PyQuante1-path vs cclib_mojo (native kernel).

Workload: benzene, 6-31G* basis (12 atoms, 102 contracted Cartesian basis
functions, 192 primitives; basis data from PyQuante 1.6.5's basis library —
see tests/gaussgrid_fixtures.py for provenance). MO coefficients are drawn
from a fixed seed — they only drive linear combinations, so no physics is
required for a timing benchmark.

The "PyQuante1 path" is cclib's actual hot loop with a PyQuante-1 backend:
`cclib.method.volume.wavefunction` calls `pyamp()`, which loops over every
grid point in pure Python calling `CGBF.amp(x, y, z)`. PyQuante 1.6.5 is
Python-2 only and cannot run here, so the benchmark executes
`tests/pyquante1_oracle.py` — a verbatim Python-3 transcription of that
amplitude path (same formulas, same operation order, same per-point
interpreter cost profile), driven exactly the way cclib drives it
(getbfs -> gridpoints -> per-point amp loop -> resize). This is the loop
the Mojo kernel replaces.

Correctness is asserted (native vs the PyQuante1 oracle within 1e-10
relative) before any timing happens, so the numbers below always come from
a verified-correct build. Timings are the median of 5 runs. Setup (basis
object construction) is timed separately: cclib rebuilds its CGBF list on
every wavefunction() call, and we rebuild our flat arrays on every call —
the comparison is per-call vs per-call.

Run from the repository root:

    pixi run bench-cclib
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import gaussgrid_fixtures as fx  # noqa: E402
import pyquante1_oracle as oracle  # noqa: E402

N_RUNS = 5
SEED = 20260919
RTOL = 1e-10
ATOL = 1e-12

# Benchmark cells: (label, kind, n_mo, shape). The 100^3 wavefunction cell
# is the cclib cube-file workload at a typical production resolution.
CELLS = [
    ("wavefunction 1 MO, 50^3", "wavefunction", 1, (50, 50, 50)),
    ("wavefunction 1 MO, 100^3", "wavefunction", 1, (100, 100, 100)),
    ("density 3 MOs, 50^3", "density", 3, (50, 50, 50)),
]


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


def grid_spec(shape):
    side = 7.0  # Angstrom; covers the 4.98 A benzene diameter with margin
    origin = (-side, -side, -side)
    step = tuple(2.0 * side / (n - 1) for n in shape)
    return origin, step, shape


def time_median(fn, n_runs=N_RUNS) -> float:
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    import cclib_mojo
    from cclib_mojo import density_on_grid, wavefunction_on_grid
    from cclib_mojo._basis import flatten_gbasis

    info = cclib_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- cclib_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- seed: {SEED}; runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'cclib_mojo'"
        )

    gbasis, atomcoords = fx.benzene_6_31g_star()
    basis = flatten_gbasis(gbasis, atomcoords)
    n_bf = basis.n_bf
    print(
        f"- molecule: benzene, 6-31G* ({len(gbasis)} atoms, {n_bf} basis "
        f"functions, {basis.n_prims} primitives)"
    )

    print("\n== correctness gate (native vs PyQuante1 oracle, 16^3) ==")
    origin, step, shape = grid_spec((16, 16, 16))
    coeff_gate = fx.seeded_coeffs(SEED, 2, n_bf)
    ours_wf = wavefunction_on_grid(gbasis, atomcoords, coeff_gate[0], origin, step, shape)
    ref_wf = oracle.wavefunction(gbasis, atomcoords, coeff_gate[0], origin, step, shape)
    np.testing.assert_allclose(ours_wf, ref_wf, rtol=RTOL, atol=ATOL)
    ours_rho = density_on_grid(gbasis, atomcoords, coeff_gate, origin, step, shape)
    ref_rho = oracle.electrondensity_spin(gbasis, atomcoords, coeff_gate, origin, step, shape)
    np.testing.assert_allclose(ours_rho, ref_rho, rtol=RTOL, atol=ATOL)
    worst = max(
        float(np.max(np.abs(ours_wf - ref_wf))), float(np.max(np.abs(ours_rho - ref_rho)))
    )
    print(f"  max abs diff = {worst:.3e}  [OK]")

    print("\n== setup overhead (per-call basis construction, median of 5, ms) ==")
    t_getbfs = time_median(lambda: oracle.getbfs(gbasis, atomcoords))
    t_flatten = time_median(lambda: flatten_gbasis(gbasis, atomcoords))
    print(f"  PyQuante1-path getbfs: {1e3 * t_getbfs:>10.3f} ms")
    print(f"  cclib_mojo flatten:    {1e3 * t_flatten:>10.3f} ms")

    rows = []
    print("\n== evaluation (median of 5 runs, seconds) ==")
    print(f"{'cell':>26} | {'PyQuante1 path':>15} | {'cclib_mojo':>11} | {'speedup':>8}")
    print(f"{'-' * 26}-+-{'-' * 15}-+-{'-' * 11}-+-{'-' * 8}")
    for label, kind, n_mo, shape in CELLS:
        origin, step, shape = grid_spec(shape)
        coeff = fx.seeded_coeffs(SEED + n_mo, n_mo, n_bf)
        if kind == "wavefunction":
            ref_fn = lambda: oracle.wavefunction(
                gbasis, atomcoords, coeff[0], origin, step, shape
            )
            our_fn = lambda: wavefunction_on_grid(
                gbasis, atomcoords, coeff[0], origin, step, shape
            )
        else:
            ref_fn = lambda: oracle.electrondensity_spin(
                gbasis, atomcoords, coeff, origin, step, shape
            )
            our_fn = lambda: density_on_grid(
                gbasis, atomcoords, coeff, origin, step, shape
            )
        # Warm both paths once (also re-verifies the cell before timing).
        np.testing.assert_allclose(our_fn(), ref_fn(), rtol=RTOL, atol=ATOL)
        t_ref = time_median(ref_fn)
        t_ours = time_median(our_fn)
        rows.append((label, t_ref, t_ours))
        print(f"{label:>26} | {t_ref:>15.3f} | {t_ours:>11.4f} | {t_ref / t_ours:>7.1f}x")

    # Fallback context row (smallest cell only): what non-native platforms get.
    os.environ["CCLIB_MOJO_DISABLE_NATIVE"] = "1"
    try:
        import cclib_mojo._native as nat

        nat._LIB, nat._LIB_SOURCE = None, None
        origin, step, shape = grid_spec(CELLS[0][3])
        coeff = fx.seeded_coeffs(SEED + CELLS[0][2], CELLS[0][2], n_bf)
        fb_fn = lambda: wavefunction_on_grid(
            gbasis, atomcoords, coeff[0], origin, step, shape
        )
        np.testing.assert_allclose(
            fb_fn(),
            oracle.wavefunction(gbasis, atomcoords, coeff[0], origin, step, shape),
            rtol=RTOL,
            atol=ATOL,
        )
        t_fb = time_median(fb_fn)
    finally:
        del os.environ["CCLIB_MOJO_DISABLE_NATIVE"]
        nat._LIB, nat._LIB_SOURCE = None, None
    print(f"{'wavefunction 1 MO, 50^3 (fallback)':>26} | {rows[0][1]:>15.3f} | "
          f"{t_fb:>11.4f} | {rows[0][1] / t_fb:>7.1f}x")

    print("\n== README paste block ==")
    print("| workload | PyQuante1 path (s) | cclib-mojo (s) | speedup |")
    print("|---|---:|---:|---:|")
    for label, t_ref, t_ours in rows:
        print(f"| {label} | {t_ref:.2f} | {t_ours:.4f} | {t_ref / t_ours:.0f}x |")
    print(f"| wavefunction 1 MO, 50^3 — NumPy fallback | {rows[0][1]:.2f} | "
          f"{t_fb:.4f} | {rows[0][1] / t_fb:.0f}x |")
    print("\n| setup (per call) | PyQuante1 getbfs (ms) | cclib-mojo flatten (ms) |")
    print("|---|---:|---:|")
    print(f"| basis construction | {1e3 * t_getbfs:.2f} | {1e3 * t_flatten:.2f} |")


if __name__ == "__main__":
    main()
