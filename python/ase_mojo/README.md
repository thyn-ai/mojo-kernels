# ase-mojo

**Fast periodic neighbor lists with an [ASE](https://gitlab.com/ase/ase)-compatible
API** — the atomistic-simulation toolkit used ecosystem-wide in
computational chemistry and materials science — powered by a Mojo kernel,
with a vendored NumPy fallback. It accelerates
`ase.neighborlist.primitive_neighbor_list` / `neighbor_list`, the
linked-cell neighbor search every geometry optimization and MD run
performs. Prebuilt per-platform binaries mean **no Mojo toolchain is ever
required** on an end user's machine, and results are the **identical
(i, j, S) pair sets** the ASE oracle produces (strict `<` distance test;
documented 1e-9 cutoff-boundary band, measured zero boundary pairs across
the whole seeded differential corpus).

ASE is *not* a dependency: `Atoms` objects are duck-typed (`pbc`, `cell`,
`positions`, `numbers`), and the package is fully usable standalone on
plain NumPy arrays.

## Install

```
pip install ase-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows —
the same wheel API runs on the vendored NumPy fallback, silently and
correctly (and still ~10× faster than ASE's own search — see the benchmark
table). There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import numpy as np
from ase_mojo import primitive_neighbor_list, neighbor_list

cell = np.diag([5.0, 5.0, 5.0])
pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [3.0, 3.0, 3.0]])

# Same call shape as ase.neighborlist.primitive_neighbor_list:
i, j, S = primitive_neighbor_list("ijS", [True, True, True], cell, pos, 1.5)

# With an ASE Atoms object (ASE itself is not imported):
#   i, j, d = neighbor_list("ijd", atoms, 2.6)

# D = positions[j] - positions[i] + S.dot(cell), as in ASE.
```

A runnable, brute-force-verified version is `quickstart.py` next to this
README.

## Autopsy (why ASE's search is slow, measured)

`ase.neighborlist.primitive_neighbor_list` (3.26.0, current pip release)
computes scaled coordinates and a bin sort in NumPy, then runs the actual
neighbor search as a **Python loop over occupied bins with per-bin NumPy
slices** — interpreter dispatch and temporary allocation per (bin, offset)
pair dominate. Measured on this machine (Apple M4 Max), orthorhombic box,
~16 Å³/atom, cutoff 2.0 Å:

| atoms | pairs | ASE 3.26.0 (warm, ms) |
|---:|---:|---:|
| 500 | ~4.2k | 22.8 |
| 2,000 | ~16.2k | 69.4 |
| 8,000 | ~65.4k | 356.4 |

Cost class: *unfused compiled kernels + Python glue* — good NumPy pieces
(bin sort, wrapped coordinates) separated by a per-bin interpreter loop.
That is the 10–50× class, and the measured result below lands in it: the
Mojo kernel fuses the whole search (binning, offset walk, distance test,
emission filter) into one compiled pass over flat arrays, single
allocation per call, no Python on the hot path.

## Benchmark

Measured with `benchmarks/bench_ase_neighborlist.py` in this repository
(run it with `ASE_ORACLE_SITE=<dir with ase 3.26.0> pixi run python
benchmarks/bench_ase_neighborlist.py`). Workloads: random configurations
at typical MD density (~16 Å³/atom), cutoff 2.0 Å, full PBC; median of 5
warm calls; cold = first call in a fresh process (dlopen, ctypes binding,
allocator warm-up). Environment: **Apple M4 Max, macOS 26.6.2 arm64,
Python 3.12.14, numpy 2.5.3, Mojo 1.1.0, ASE 3.26.0**, measured
2026-09-19.

| workload | pairs | native cold (ms) | native warm (ms) | fallback cold (ms) | fallback warm (ms) | ASE cold (ms) | ASE warm (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ortho-2k | 4,196 | 6.465 | 1.421 | 6.909 | 3.915 | 42.282 | 33.194 |
| ortho-8k | 16,328 | 11.932 | 5.484 | 12.715 | 12.755 | 148.538 | 155.226 |
| triclinic-8k | 38,168 | 16.061 | 11.154 | 22.632 | 21.875 | 209.984 | 226.159 |

Speedups (ASE warm / ase-mojo warm):

| workload | native | NumPy fallback |
|---|---:|---:|
| ortho-2k | 23.4× | 8.5× |
| ortho-8k | 28.3× | 12.2× |
| triclinic-8k | 20.3× | 10.3× |

Correctness gate (asserted before every timing run, on a separate
200-atom triclinic validation case): oracle, native and fallback pair
sets are **identical** (12,312 pairs each).

Honest framing: ASE 3.26.0's search is already NumPy-assisted, so the win
is ~20–28×, not the hundreds-fold available against pure-Python targets.
The remaining gap over the native kernel is the per-bin Python glue the
kernel eliminates; the vendored NumPy fallback recovers most of it with a
vectorized equi-join (27 joins per call instead of a Python loop per
(bin, offset)).

## API and semantics

Drop-in-shaped after ASE 3.26.0, pinned black-box against the pip
package:

```python
primitive_neighbor_list(quantities, pbc, cell, positions, cutoff,
                        numbers=None, self_interaction=False,
                        use_scaled_positions=False, max_nbins=1000000.0,
                        *, bothways=True)
neighbor_list(quantities, a, cutoff, self_interaction=False,
              max_nbins=1000000.0, *, bothways=True)
```

- `quantities`: any subset of `"ijSdD"` (returned in the requested order);
  anything else raises `ValueError("Unsupported quantity specified.")`.
- `cutoff`: float (global), dict keyed by element **pairs** (atomic
  numbers or symbols, either order — `{(1, 6): 1.1, ("C", "C"): 1.85}`;
  pairs missing from the dict get cutoff 0), or a per-atom radius array
  (pair cutoff = rᵢ + rⱼ, `natural_cutoffs`-style). Dict cutoffs require
  `numbers`.
- Strict `<`: pairs at exactly the cutoff are excluded.
- Self-images `(i, i, S ≠ 0)` are always included; `(i, i, 0)` is
  included iff `self_interaction=True`.
- `bothways=True` is ASE's module-level behavior (every ordered
  `(i, j, S)`; ASE removed the `bothways` argument). `bothways=False`
  applies the classic `ase.neighborlist.NeighborList` reduction — keep
  `(i, j, S)` iff lex(S) > 0 or (S == 0 and i < j); `(i, i, 0)` iff
  `self_interaction` — verified against that class (direction-insensitive:
  the class's kept representative is an implementation detail of its
  internal half-stencil in multi-image cells; canonical pair content is
  identical, exactly once per unordered pair+image).
- Shift vectors refer to the **original** positions:
  `D = positions[j] - positions[i] + S.dot(cell)`; atoms outside the
  periodic box are mapped in with S recording the crossings.
- `max_nbins` caps bin-grid memory; results are unchanged (per-axis search
  ranges grow to compensate).
- Output is fully sorted by `(i, j, S)` (ASE sorts by `i` only and
  documents pair order as not guaranteed — compare order-insensitively).
- `i`, `j`, `S` are int64; `d`, `D` are float64.

## How it works

```
pip install ase-mojo
        │
        ▼
ase_mojo (thin Python wrapper)
        │  validates pbc/cell/positions, resolves cutoff (float | pair-dict
        │  | per-atom radii) into a typed per-pair form, owns D/d assembly
        ▼
libaseneighborlistmojo.dylib / .so     (Mojo kernel, C ABI v1)
        │  linked-cell search over fractional bins; per-axis search ranges
        │  from perpendicular lattice-plane heights (exact for triclinic);
        │  small cells yield multiple shift images per pair
        ▼
(i, j, S) int64 — sorted, deterministic
```

- **Batch-shaped C ABI**: one call builds the whole list (two-call
  capacity protocol: the kernel reports the exact buffer size on
  overflow, so at most one retry). FFI overhead is per-call, not per-pair.
- **Completeness by construction**: bin widths are chosen so each bin's
  *perpendicular lattice-plane height* is ≥ the largest pair cutoff;
  since |Δf_d| ≤ |Δr|/h_d for any displacement, searching
  `m_d ∈ [-K_d, K_d]`, `K_d = floor(cmax·nbins_d/h_d)+1`, is complete for
  arbitrary triclinic cells — and conservative over-search is filtered by
  the exact distance test, so correctness never depends on bin aspect
  ratios.
- **Determinism**: stable counting sort (atoms in index order), fixed
  offset iteration, IEEE-754 float64; the fallback evaluates the
  squared-distance predicate with the same operations in the same
  association order, so native and fallback emit **bit-identical** sets.
- **ABI handshake**: the wrapper checks
  `aseneighborlistmojo_abi_version()` before building; a mismatch falls
  back cleanly.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored NumPy reference**
(`ase_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$ASE_MOJO_NATIVE_LIB` → the library bundled in the
  wheel → the repo development build output.
- `ASE_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Both backends share the wrapper's validation and cutoff resolution, so
  they cannot disagree about inputs; the differential suite asserts both
  against ASE.
- Inspect what's active: `ase_mojo.backend_info()` and
  `ase_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `ASE_MOJO_ALLOW_PURE_WHEEL=1` (e.g. for Windows).

## Differential tests

```
bash kernels/ase_neighborlist/build.sh          # compile the Mojo kernel
pixi run bash scripts/test_all_ase_neighborlist.sh
        # installs the ASE oracle into a scratch dir, then runs the suite
        # twice: once native, once with ASE_MOJO_DISABLE_NATIVE=1
```

The suite (`tests/test_ase_neighborlist_*.py`) compares ase-mojo against
ASE 3.26.0 on a seeded corpus: orthorhombic + triclinic cells, full and
partial PBC, atoms inside/outside the box, skewed small cells with
multiple shift images per pair (cutoff > lattice-plane height), float /
pair-dict (number, symbol, mixed keys; missing pairs) / per-atom-radii
cutoffs, `self_interaction` on/off, `bothways` True/False (against ASE's
`NeighborList` class as a second oracle, direction-insensitively),
`use_scaled_positions`, tiny `max_nbins`, empty and single-atom systems,
singular cells with and without PBC, `neighbor_list` on real `Atoms`
objects, and error parity. **96 tests pass per backend (192 per full
run); measured agreement: identical (i, j, S) sets on every case, zero
pairs inside the 1e-9 cutoff-boundary band across the whole corpus.**

Parity tolerance (documented): the distance test is strict `<` in
float64; implementations differ in summation order at the last ulp, so a
pair within 1e-9 of its pair cutoff could flip. The suite treats such
pairs as an excluded band and asserts none occur — random configurations
essentially never produce them.

## Scope and limitations

- Quantities `i`, `j`, `S`, `d`, `D` only; ASE 3.26.0 supports exactly
  these five (its old `first` / `abs_max_d` quantities are gone upstream
  too).
- **Singular cell with any periodic axis**: ASE produces undefined output
  there (observed: garbage shift vectors); ase-mojo raises `ValueError`
  instead. Singular cell with *no* periodic axis works and matches ASE
  (plain Cartesian search, S = 0).
- `bothways=False` keeps a deterministic representative (rule above);
  ASE's `NeighborList` class may keep the mirror representative in
  multi-image cells — pair content is identical, direction can differ
  (compare canonically if you diff against the class directly).
- Plain Python lists for `positions` are accepted (ASE 3.26.0's primitive
  function requires ndarray-likes with `.T` in some paths — we take the
  superset).
- Single-threaded kernel (determinism first); multithreading is future
  work and would multiply the speedup.
- No `NeighborList`-class object (skin, connectivity reuse) — the
  module-level functions are the drop-in surface.
- macOS arm64 and Linux x86_64 native; everything else (including
  Windows) runs the NumPy fallback — silently, bit-identically, ~10×
  faster than ASE's own search.

## Citing

If you use ase-mojo in academic work, please cite both ASE and this
package:

> ase-mojo: Mojo-accelerated periodic neighbor lists with an
> ASE-compatible API. thyn-ai, 2026.
> https://github.com/thyn-ai/mojo-kernels (citation file with DOI
> forthcoming)

ASE itself: A. H. Larsen et al., *The Atomic Simulation Environment — A
Python library for working with atoms*, J. Phys.: Condens. Matter 29
(2017) 273002.

## License

Apache-2.0, © 2026 Algenta The kernel and wrapper are clean-room
implementations of the textbook linked-cell neighbor-search algorithm
(Allen & Tildesley 1987, §5.3.2). ASE is used only as a test/benchmark
reference, never as a runtime dependency, and no ASE source was read or
adapted.
