# cclib-mojo

**Fast electron-density and wavefunction-on-grid evaluation for
[cclib](https://cclib.github.io/)** — the computational-chemistry parsing
library — powered by a Mojo kernel, with a vendored NumPy fallback. It
accelerates the `cclib.method.volume` path (`Volume.wavefunction()` /
`electrondensity()`), whose PyQuante backend evaluates contracted Gaussian
basis functions one grid point at a time in pure Python. Prebuilt
per-platform binaries mean **no Mojo toolchain is ever required** on an end
user's machine, and results match the PyQuante reference to 1e-10 relative
(measured agreement is at the 1e-13–1e-16 level).

It works two ways:

1. **Alongside cclib** — `cclib_mojo.cclib_integration` mirrors
   `cclib.method.volume`'s functions (no monkeypatching): parse with cclib,
   evaluate with cclib-mojo, write cube files with cclib.
2. **Standalone** — `density_on_grid()` / `wavefunction_on_grid()` take
   cclib-style `gbasis` + geometry directly and return NumPy arrays in
   cclib's `Volume.data` layout.

## Install

The wheel for your platform is a signed asset of the matching
[GitHub Release](https://github.com/thyn-ai/mojo-kernels/releases) (bash or
zsh; PyPI distribution follows once the registry is enabled, see
[RELEASING.md § Registries](../../RELEASING.md#registries)):

```bash
V=0.1.3  # x-release-please-version
case "$(uname -sm)" in
  "Darwin arm64") WHEEL="cclib_mojo-$V-py3-none-macosx_14_0_arm64.whl" ;;
  "Linux x86_64") WHEEL="cclib_mojo-$V-py3-none-manylinux_2_35_x86_64.whl" ;;
esac
pip install "https://github.com/thyn-ai/mojo-kernels/releases/download/v$V/${WHEEL:?no prebuilt wheel for this platform}"
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows, where CI runs this package's fallback suite
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)) —
the same wheel API runs on the vendored NumPy fallback, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import numpy as np
from cclib_mojo import density_on_grid, wavefunction_on_grid

# Inputs exactly as cclib parses them from a logfile:
#   gbasis     -- per-atom list of (shell, [(exponent, coefficient), ...])
#   atomcoords -- (n_atoms, 3) in Angstrom
#   mocoeffs   -- (n_mo, n_bf) MO coefficients, e.g. ccdata.mocoeffs[0]
gbasis = [...]
atomcoords = np.array([...])

# One molecular orbital on a grid (cclib's wavefunction() equivalent):
psi = wavefunction_on_grid(gbasis, atomcoords, mocoeffs[3],
                           origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                           shape=(51, 51, 51))

# Electron density, sum over occupied MOs of |psi|^2
# (cclib's electrondensity_spin(); multiply by 2 for a closed shell):
nocc = 5
rho = density_on_grid(gbasis, atomcoords, mocoeffs[:nocc],
                      origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                      shape=(51, 51, 51))
# rho[i, j, k] is the density at
# (origin[0]+i*step[0], origin[1]+j*step[1], origin[2]+k*step[2]) — the
# exact layout cclib's Volume.data and its cube writer use.
```

A runnable version is `quickstart.py` at the repository root (water,
STO-3G):

```
python quickstart.py
```

## Using it from cclib

`cclib_mojo.cclib_integration` provides drop-in equivalents of cclib's
volume functions — same call shapes, same return convention (a copy of the
`Volume` with `data` replaced), no monkeypatching:

```python
import cclib
from cclib.method.volume import Volume
from cclib_mojo import cclib_integration   # instead of cclib.method.volume

data = cclib.io.ccread("benzene.log")
vol = Volume(origin=(-7, -7, -7), topcorner=(7, 7, 7), spacing=(0.2, 0.2, 0.2))

# cclib:  from cclib.method.volume import wavefunction
wf = cclib_integration.wavefunction(data, vol, data.mocoeffs[0][3])
wf.writeascube("mo4.cube")

# cclib:  from cclib.method.volume import electrondensity
nocc = data.homos[0] + 1
dens = cclib_integration.electrondensity(data, vol, [data.mocoeffs[0][:nocc]])
dens.writeascube("density.cube")
```

Equivalence with cclib's own `Volume.wavefunction()` / `electrondensity()`
(on its pyquante2 backend) is asserted in the differential test suite —
same grid points, same normalization, same cube-file ordering.

## Benchmark

Measured with `benchmarks/bench_gaussgrid.py` in this repository (run
`pixi run bench-cclib` to reproduce). Workload: **benzene, 6-31G\*** (12
atoms, 102 contracted Cartesian basis functions, 192 primitives — basis
data from PyQuante 1.6.5's basis library), seeded MO coefficients, median
of 5 runs, single-threaded. Environment: **Apple M4 Max, macOS 26.6.2
arm64, Python 3.12.14, numpy 2.5.3, Mojo 1.1.0**, 2026-09-19.

| workload | PyQuante1 path (s) | cclib-mojo (s) | speedup |
|---|---:|---:|---:|
| wavefunction 1 MO, 50³ | 32.23 | 0.0126 | 2567x |
| wavefunction 1 MO, 100³ | 345.37 | 0.0491 | 7033x |
| density 3 MOs, 50³ | 63.59 | 0.0335 | 1898x |
| wavefunction 1 MO, 50³ — NumPy fallback | 32.23 | 0.1691 | 191x |

Setup is per-call on both sides (cclib rebuilds its PyQuante basis objects
on every `wavefunction()` call; we rebuild our flat arrays on every call):

| setup (per call) | PyQuante1 getbfs (ms) | cclib-mojo flatten (ms) |
|---|---:|---:|
| basis construction | 2.09 | 1.48 |

Correctness gate (asserted before every timing run, element-wise vs the
PyQuante1 reference): max abs diff **1.5e-12** — the values are the same
numbers, not approximations.

Honest framing, two baselines (both measured, same machine and workload):

- **vs the PyQuante1 path** (the table above): ~1,900–7,000×. cclib's hot
  loop with a PyQuante-1 backend (`pyamp()` → per-point `CGBF.amp(x, y, z)`
  in pure Python) is Python-2 era code that cannot execute on Python 3, so
  the benchmark drives a verbatim Python-3 transcription of that amplitude
  path (`tests/pyquante1_oracle.py`, same formulas, same operation order,
  same per-point interpreter cost profile) exactly the way cclib drives it.
- **vs cclib's pyquante2 backend** (NumPy-vectorized `cgbf.mesh`, the
  fastest backend cclib ships today): cclib's real
  `Volume.wavefunction()` takes **0.689 s** for the same 50³ single-MO
  workload vs **0.0095 s** for cclib-mojo — **72×**, with max abs diff
  8.4e-13. The remaining gap over the PyQuante1 table is the numpy
  vectorization in pyquante2, which still pays per-(basis-function, grid)
  temporaries; the Mojo kernel pays one SIMD FMA per (point, primitive).

Why the gap is so large: the PyQuante path interprets ~5 Python operations
per (grid point × basis function × primitive) — hundreds of millions of
interpreter steps for a production cube grid. The Mojo kernel precomputes
the exactly-separable axis factors `exp(-a·dx²)·exp(-a·dy²)·exp(-a·dz²)`
once per call, then runs one SIMD fused multiply-add per (point, primitive)
in compiled float64, with the grid row under update kept cache-resident.

Why the gap is so large: the PyQuante path interprets ~5 Python operations
per (grid point × basis function × primitive) — hundreds of millions of
interpreter steps for a production cube grid. The Mojo kernel precomputes
the exactly-separable axis factors `exp(-a·dx²)·exp(-a·dy²)·exp(-a·dz²)`
once per call, then runs one SIMD fused multiply-add per (point, primitive)
in compiled float64, with the grid row under update kept cache-resident.

## How it works

```
pip install <cclib_mojo wheel from the GitHub Release>
        │
        ▼
cclib_mojo (thin Python wrapper)
        │  validates gbasis/geometry, computes THO primitive norms and
        │  contracted norms (shared by both backends), flattens to typed arrays
        ▼
libgaussgridmojo.dylib / .so        (Mojo kernel, C ABI v1)
        │  gaussgridmojo_eval: whole grid per call, SIMD over grid points
        ▼
float64 grid in cclib's Volume.data layout
```

- **Batch-shaped C ABI**: one call evaluates the whole 3-D grid for one MO
  (or accumulates the density over all MOs); FFI overhead is per-call, not
  per-point.
- **Reference-matching arithmetic**: normalization constants are computed
  in Python with PyQuante's exact formulas and operation order (THO
  eq. 2.2 and 2.12); the kernel evaluates the textbook contracted
  Cartesian Gaussian in IEEE-754 float64 and skips exactly-zero MO
  coefficients, the same rule cclib uses.
- **ABI handshake**: the wrapper checks `gaussgridmojo_abi_version()`
  before evaluating; a mismatch falls back cleanly.
- **Conventions verified against cclib**: centers converted Å→bohr with
  cclib's `convertor` constant (×1.8897261245), grid axes divided by
  cclib's bohr→Å constant (÷0.5291772109 — the two are not exact
  reciprocals and we mirror cclib exactly), shell expansion order of
  cclib's `sym2powerlist` (S/P/D/F), `Volume.data` C-ordering (x outer,
  z inner), and the `abs(coeff) > 0.0` inclusion rule.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored NumPy reference**
(`cclib_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$CCLIB_MOJO_NATIVE_LIB` → the library bundled in the
  wheel → the repo development build output.
- `CCLIB_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Both backends share basis flattening and normalization in
  `cclib_mojo/_basis.py`, so they cannot disagree about constants; the
  differential suite asserts both against PyQuante.
- Inspect what's active: `cclib_mojo.backend_info()` and
  `cclib_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `CCLIB_MOJO_ALLOW_PURE_WHEEL=1` (the wheel CI builds and tests on Windows: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

## Differential tests

```
pixi run test-cclib    # builds the kernel, then runs the suite twice:
                       # once native, once with CCLIB_MOJO_DISABLE_NATIVE=1
```

The suite (`tests/test_gaussgrid_*.py`) compares cclib-mojo against the
PyQuante reference within 1e-10 relative on deterministic fixtures: s/p/d/f
shells (STO-3G H2O, 6-31G\* d on O, cc-pVTZ f on C), multiple atoms, MO
amplitude vs summed density, non-cubic anisotropic grids, grids offset from
the origin and from all atoms, exact-zero coefficients, grid-point
ordering, and validation errors — plus end-to-end equivalence with cclib's
actual `Volume.wavefunction()` / `electrondensity()` on their pyquante2
backend. Basis-set provenance is documented in
`tests/gaussgrid_fixtures.py` (all values from PyQuante 1.6.5's published
basis library). 19 tests pass per backend (38 per full run), and the
benchmark asserts the same 1e-10 gate before every timing run.

## Scope and limitations

- Shells S/P/D/F (Cartesian), exactly matching cclib's `sym2powerlist`.
  G shells and spherical-harmonic (5d/7f) functions are not supported —
  cclib's volume path does not handle them either.
- Single-threaded kernel (determinism first); multithreading is future
  work and would multiply the speedup.
- cclib is *not* a dependency: `cclib_mojo.cclib_integration` duck-types
  `ccdata`/`Volume` objects. numpy is the only hard dependency.

## Citing

If you use cclib-mojo in academic work, please cite both cclib and this
package:

> cclib-mojo: Mojo-accelerated electron-density and wavefunction-on-grid
> evaluation for cclib. thyn-ai, 2026. https://github.com/thyn-ai/mojo-kernels
> (citation file with DOI forthcoming)

cclib itself: N. M. O'Boyle, A. L. Tenderholt, K. M. Langner, *cclib: A
library for package-independent computational chemistry algorithms*,
J. Comput. Chem. 29 (2008) 839–845. PyQuante: R. P. Muller, *PyQuante:
Python Quantum Chemistry*.

## License

Apache-2.0, © 2026 Algenta The kernel and wrapper are clean-room
implementations of the textbook contracted-Gaussian formulas (Taketa,
Huzinaga, O-ohata, J. Phys. Soc. Jap. 21, 2313 (1966)). cclib and PyQuante
are used only as test/benchmark references, never as runtime dependencies.
