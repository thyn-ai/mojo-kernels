# metpy-mojo

Fast, MetPy-compatible parcel **CAPE/CIN** for standard atmospheric
soundings — powered by a clean-room Mojo kernel, with a vendored
pure-Python fallback for platforms without a native build (including
Windows).

```python
import numpy as np
from metpy_mojo import cape_cin, cape_cin_grid, parcel_diagnostics

cape, cin = cape_cin(p, T, Td)                       # surface parcel
cape, cin = cape_cin(p, T, Td, which="most_unstable")
diag = parcel_diagnostics(p, T, Td)                  # + LCL/LFC/EL, parcel level

cape2d, cin2d = cape_cin_grid(p_lev, T3d, Td3d)      # (nlev, y, x) fields,
                                                     # one batched call
```

Units: pressure in hPa (strictly decreasing along each column),
temperature/dewpoint in K. CAPE/CIN in J/kg; CIN is <= 0; a column with
no LFC returns `(0.0, 0.0)`, and a missing EL is NaN with CAPE integrated
to the profile top — mirroring `metpy.calc.cape_cin`.

- **Same answers as MetPy**: differential-tested against
  `metpy.calc.surface_based_cape_cin` / `most_unstable_cape_cin` (metpy
  1.7.1) on a 31-sounding suite (classic, tropical, inversion, saturated,
  stable, moist-aloft, plus 24 seeded variants) for both parcel choices —
  see "Parity" below.
- **Much faster on grids**: `cape_cin_grid` computes every column of a
  3-D pressure-level field in one batched kernel call — the
  gridded-CAPE use case of MetPy issue #480 — instead of a Python loop
  around per-column LSODA with per-step unit-wrapped callbacks.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `METPY_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `metpy_mojo.backend_info()`.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

## Benchmark

Measured with `benchmarks/bench_metpy_cape.py` (correctness gate vs the
metpy oracle asserted before every timing run; median of 5 runs; cold =
first call in a fresh interpreter process). Environment: **Apple M4 Max,
macOS 26.6.2 arm64, Python 3.12.14, numpy 2.5.3, Mojo 1.1.0, metpy
1.7.1**, 2026-09-19.

| workload | metpy.calc | metpy_mojo (native) | speedup |
|---|---:|---:|---:|
| single column, cold first call | 7.331 ms | 7.631 ms | 1.0x |
| single column, warm | 6.461 ms | 0.083 ms | 78.0x |
| grid 25x25 (625 cols) | 4.618 s | 0.0295 s | 156.6x |
| grid 100x100 (10k cols) | — (est. 1 min by rate) | 0.4816 s | — |

Correctness gate (asserted before every timing run, vs the oracle on the
benchmark sounding): CAPE 567.02 vs 569.50 J/kg (0.44%), CIN −130.59 vs
−130.73 J/kg (0.11%) — the same numbers as the differential suite.

Cold first call is dominated by interpreter/dyld warmup on both sides
(metpy's pint registry init vs our dlopen + ABI handshake); the warm
number is the steady-state difference. The oracle's per-column cost is
the documented `parcel_profile` LSODA integration with unit-wrapped
Python thermodynamic callbacks; ours is one compiled-kernel call with all
thermodynamics inside.

## Parity with MetPy

The kernel is a clean-room implementation of the textbook
pseudoadiabatic parcel method (Bolton 1980; Emanuel 1994), not a port of
MetPy code. Parity was established **black-box** against the published
metpy package: RK45 vs the oracle's LSODA is not bit-exact by
construction, so tolerances are stated honestly:

| quantity | documented tolerance | measured on the 31-sounding suite (SB + MU = 62 cells) |
|---|---|---|
| CAPE | 2% relative **or** 1.0 J/kg absolute | <= 0.9% for CAPE >= 100 J/kg; <= 1.6% for 50-100 J/kg; worst marginal cell 0.43 J/kg absolute |
| CIN | 1% relative **or** 0.5 J/kg absolute | <= 0.35% relative; worst cell 0.32 J/kg absolute |
| LCL pressure | 0.5 hPa | <= 0.05 hPa |

The absolute floors cover *marginal* soundings (CAPE of a few J/kg),
where a sub-hPa LFC-placement difference legitimately flips a few J/kg
either way — the oracle itself is grid-resolution-sensitive there. The
multi-blip gating (which positive area counts) was decoded from the
oracle and is mirrored exactly: LFC = the buoyancy −→+ crossing paired
with the last +→− crossing (EL); CAPE = positive-only area LFC→EL (or
the profile top); CIN = signed area from the parcel start to the LFC,
floored at 0.

Two deliberate, documented differences from `metpy.calc.lfc` / `el`
standalone helpers: our `parcel_diagnostics` LFC/EL are the
virtual-temperature-buoyancy crossings that gate the integrals (the
oracle's standalone helpers use the plain temperature difference and may
sit ~20 hPa lower); and RK45 (deterministic Dormand-Prince, rtol 1e-9)
replaces LSODA. The native kernel and the pure-Python fallback agree to
~1e-7 on every output (asserted in the test suite).

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored pure-Python
reference implementation** (`metpy_mojo/_reference.py`, clean-room,
NumPy-only, same algorithm):

- Resolution order: `$METPY_MOJO_NATIVE_LIB` → the library bundled in
  the wheel → the repo development build output.
- `METPY_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Inspect what's active: `metpy_mojo.backend_info()` and
  `metpy_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only** — no sdist, because a
  source tarball cannot rebuild the native library. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries into the wheel and rewrite the kernel
  library's load paths to be wheel-relative, so no Mojo toolchain is
  needed at install time. (Redistribution terms for Modular's runtime
  binaries should be confirmed with Modular before any public release.)
  A pure `py3-none-any` fallback wheel can be produced with
  `METPY_MOJO_ALLOW_PURE_WHEEL=1` (e.g. for Windows).

## Scope

Standard atmospheric profiles: CAPE/CIN for surface-based and
most-unstable parcels on pressure-level soundings and gridded fields.
Deliberately out of scope today: mixed-layer parcels, downdraft CAPE,
unit-aware (pint/xarray) objects (plain arrays in hPa/K instead), and
profiles where saturation vapor pressure exceeds ambient pressure at some
level (validated and rejected). `metpy` itself is **not** a runtime
dependency — it is only the differential-test oracle.

## Differential tests

```
PYTHONPATH=python/metpy_mojo pixi run bash scripts/test_all_metpy_cape.sh
# builds nothing; run twice: once native, once with
# METPY_MOJO_DISABLE_NATIVE=1 (the kernel build is
# `bash kernels/metpy-cape/build.sh`)
```

`tests/` compares `metpy_mojo` against metpy on the deterministic seeded
suite: per-sounding CAPE/CIN for both parcel choices, native-vs-fallback
agreement, diagnostics self-consistency, MU-parcel-level and LCL
equality with the oracle, the 3-D grid API against per-column and
oracle-loop results, a 48-column synthetic field, NaN semantics for
missing LFC/EL, and loader/validation behaviour. 138 tests on the native
backend, 107 + 31 skips on the forced fallback.

License: Apache-2.0, © 2026 Algenta
