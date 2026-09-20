# pykalman-mojo

**Drop-in faster Kalman filter and smoother for
[pykalman](https://pykalman.github.io/) workloads** — powered by a Mojo
kernel, with a vendored NumPy fallback. pykalman is pure Python: its filter
and smoother run an interpreter loop over timesteps with ~11 `np.dot` calls
and one SVD (`scipy.linalg.pinv`) per step, plus masked-array bookkeeping on
every observation. pykalman-mojo runs the same recurrences, in the same
operation order, as one compiled pass over the whole series. Prebuilt
per-platform binaries mean **no Mojo toolchain is ever required** on an end
user's machine, and outputs match pykalman to 1e-10 (measured agreement is
at the 1e-14 level).

## Install

```
pip install pykalman-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows —
the same wheel API runs on the vendored NumPy fallback, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import numpy as np
from pykalman_mojo import KalmanFilter

# Same constructor, same call shapes, same outputs as pykalman.KalmanFilter:
kf = KalmanFilter(
    transition_matrices=[[1.0, 0.1], [0.0, 1.0]],
    observation_matrices=[[1.0, 0.0]],
    transition_covariance=[[0.03, 0.0], [0.0, 0.03]],
    observation_covariance=[[0.5]],
)

observations = np.array([[1.2], [0.9], [1.1], [0.8], [1.0]])
filtered_means, filtered_covs = kf.filter(observations)
smoothed_means, smoothed_covs = kf.smooth(observations)

# One-call form, exactly like the class:
from pykalman_mojo import filter, smooth
filtered_means, filtered_covs = filter(observations, transition_matrices=[[1.0]])

# Missing observations: pass a masked array (pykalman's convention — a
# timestep with ANY masked component is treated as missing in full).
masked = np.ma.masked_invalid(observations)
smoothed_means, smoothed_covs = kf.smooth(masked)
```

## Autopsy (why pykalman is slow — measured, not guessed)

Profiled `KalmanFilter.filter()` on a 10,000-step series (5-d state, 2-d
observations, Apple M4 Max, Python 3.12.14, numpy 2.5.3, scipy 1.18.1,
pykalman 0.11.2):

- **101.4 µs per timestep** for `filter()`, **288.3 µs** for `smooth()`
- **317 primitive function calls per timestep**, including **11 `np.dot`
  calls and 1 `scipy.linalg.pinv`** (a full SVD, on a 2×2 matrix)
- the top cumulative costs are not the math: numpy **masked-array
  bookkeeping** (`ma.core._update_from`, `__array_finalize__`) runs on every
  timestep even when nothing is masked, ahead of `_filter_correct` itself
  and `pinv`→`svd`

Classification: a pure-interpreter loop with per-timestep dispatch, per-step
small-matrix SVD, and heavy per-step temporaries — the 10–100× class. The
Mojo kernel keeps the identical recurrence and operation order but executes
the whole series as one compiled pass: small dense float64 matmuls with
sequential accumulation, and the per-step inverse computed by Gauss-Jordan
elimination instead of a dispatched SVD.

## Benchmark

Measured with `benchmarks/bench_pykalman.py` in this repository (reproduce
with `PYTHONPATH=python/pykalman_mojo pixi run python
benchmarks/bench_pykalman.py`). Random stable systems (seeded, transition
spectral radius 0.9, SPD covariances, offsets), simulated observations;
**cold** = fresh instance construction + first call, **warm** = median of 5
subsequent calls. Environment: **Apple M4 Max, macOS 26.6.2 arm64, Python
3.12.14, numpy 2.5.3, pykalman 0.11.2, Mojo 1.1.0**, 2026-09-19.

**filter** (per full series call):

| filter T | n_s | n_o | pykalman cold (ms) | pykalman_mojo cold (ms) | pykalman warm (ms) | pykalman_mojo warm (ms) | warm speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 5 | 2 | 50.02 | 0.52 | 83.53 | 0.296 | 282.4x |
| 10,000 | 5 | 2 | 760.31 | 5.44 | 955.58 | 2.992 | 319.4x |
| 100,000 | 5 | 2 | 11042.89 | 36.65 | 7208.01 | 39.559 | 182.2x |
| 10,000 | 2 | 1 | 665.90 | 1.36 | 616.82 | 0.746 | 827.1x |
| 10,000 | 5 | 2 | 558.49 | 4.06 | 715.16 | 5.109 | 140.0x |
| 10,000 | 10 | 4 | 798.14 | 23.71 | 761.12 | 22.750 | 33.5x |
| 10,000 | 20 | 8 | 828.65 | 167.11 | 914.04 | 182.436 | 5.0x |

**smooth** (per full series call):

| smooth T | n_s | n_o | pykalman cold (ms) | pykalman_mojo cold (ms) | pykalman warm (ms) | pykalman_mojo warm (ms) | warm speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 5 | 2 | 258.81 | 1.99 | 95.30 | 0.912 | 104.4x |
| 10,000 | 5 | 2 | 1132.85 | 9.85 | 1201.53 | 9.270 | 129.6x |
| 100,000 | 5 | 2 | 9523.87 | 123.94 | 9732.00 | 106.455 | 91.4x |
| 10,000 | 2 | 1 | 752.62 | 1.61 | 741.72 | 1.259 | 589.1x |
| 10,000 | 5 | 2 | 778.03 | 9.41 | 823.46 | 9.074 | 90.8x |
| 10,000 | 10 | 4 | 1217.21 | 51.72 | 922.08 | 55.411 | 16.6x |
| 10,000 | 20 | 8 | 1535.01 | 374.77 | 1360.73 | 346.405 | 3.9x |

Correctness gate (asserted before every timing run, element-wise vs
pykalman, documented tolerance 1e-10): worst max-abs-diff over all cells
**1.25e-13** — the values are the same numbers, not approximations.

The warm path is the honest steady state: same instance, same observations,
median of 5. The cold path includes everything a first call pays (native
library load once per process, model upload, output allocation) — and still
lands within ~2× of warm, because all of it is sub-millisecond next to
pykalman's per-timestep interpreter loop.

Honest scaling note: the speedup is largest for small/mid systems (the
common case — pykalman's per-timestep Python dispatch dominates there) and
narrows to ~4–5× at 20-d state / 8-d observation, where per-step dense
matmuls get large enough that BLAS does real work on pykalman's side while
the kernel runs its (unblocked) O(n³) small-matrix loops. The NumPy fallback
(Windows, or `PYKALMAN_MOJO_DISABLE_NATIVE=1`) is itself 1.6–2.4× faster
than pykalman warm (measured): same recurrences without masked-array
bookkeeping.

## How it works

```
pip install pykalman-mojo
        │
        ▼
pykalman_mojo (thin Python wrapper)
        │  validates/normalizes parameters (pykalman's defaults and
        │  dimensionality inference), derives the missing-data mask
        ▼
libpykalmanmojo.dylib / .so          (Mojo kernel, C ABI v1)
        │  pykalmanmojo_filter / pykalmanmojo_smooth:
        │  whole series per call, compiled float64 recurrences
        ▼
(T, n_s) means + (T, n_s, n_s) covariances, pykalman's exact contract
```

- **Batch-shaped C ABI**: a model handle owns the system matrices; each
  `filter`/`smooth` call crosses FFI once per series, not once per timestep.
- **Reference-matching arithmetic**: predict `A@(P@A.T)+Q`, gain
  `P@(C.T@inv(S))`, update `P - K@(C@P)`, RTS gain `F@(A.T@inv(P_pred))` —
  the same operation order as the reference, IEEE-754 float64, sequential
  accumulation. The per-step inverse is Gauss-Jordan with partial pivoting;
  for the well-conditioned covariances of stable systems it agrees with the
  reference's SVD pseudo-inverse to ~1e-13, far inside the 1e-10 tolerance.
- **Missing data**: pykalman's all-or-nothing rule — a timestep with any
  masked component skips the update entirely (gain zero, filtered equals
  predicted). The wrapper derives the mask from a masked array exactly like
  pykalman; plain arrays with NaN propagate NaN, also like pykalman.
- **Singular systems**: if the kernel hits an exactly singular innovation or
  predicted covariance (status 3), the wrapper serves that call from the
  NumPy fallback, whose pseudo-inverse handles singularity — same result,
  no crash.
- **ABI handshake**: the wrapper checks `pykalmanmojo_abi_version()` before
  any call; a mismatch falls back cleanly.
- **Memory**: the smoother's backward pass recomputes predicted (t+1) from
  stored filtered (t) with the same routine the forward pass uses — bit-
  identical to stored predictions at half the peak memory.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so the wrapper **falls back to a vendored NumPy reference**
(`pykalman_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$PYKALMAN_MOJO_NATIVE_LIB` → the library bundled in the
  wheel → the repo development build output.
- `PYKALMAN_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- Both backends share parameter normalization and observation parsing in
  `pykalman_mojo/core.py`, so they cannot disagree about conventions; the
  differential suite asserts both against pykalman.
- Inspect what's active: `pykalman_mojo.backend_info()` and
  `pykalman_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `PYKALMAN_MOJO_ALLOW_PURE_WHEEL=1` (e.g. for Windows).

## Differential tests

```
PYTHONPATH=python/pykalman_mojo pixi run bash scripts/test_all_pykalman.sh
# runs the suite twice: once native, once with PYKALMAN_MOJO_DISABLE_NATIVE=1
```

The suite (`tests/test_pykalman_*.py`) compares pykalman-mojo against the
published pykalman package within 1e-10 absolute on deterministic seeded
systems: scalar through 8-d state / 6-d observation, offsets on both
equations, 10% masked timesteps, partial-component masking (pykalman skips
the whole timestep), all-masked series, T=1 edge cases, default-parameter
inference, 1-D observation series, NaN propagation parity, parameter
mutation between calls, and validation errors. 36 tests pass per backend
(72 per full run; the singular-system loader test is native-only and skips
on the fallback pass), and the benchmark asserts the same 1e-10 gate before
every timing run.

## Scope and limitations

- **Time-invariant systems only.** pykalman also accepts time-varying
  matrices (3-D arrays); pykalman-mojo rejects them with a clear error.
- **`filter()` and `smooth()` only.** pykalman's `em()`, `filter_update()`,
  `sample()`, `loglikelihood()`, and the unscented/square-root variants are
  out of scope.
- A single-timestep multivariate observation `(1, n_o>1)` raises
  `ValueError` in both packages (pykalman's own parser quirk); ours says
  why.
- Single-threaded kernel (determinism first); the recurrence is inherently
  sequential in t, so there is little to parallelize honestly.
- numpy is the only hard dependency; the fallback's pseudo-inverse is
  `np.linalg.pinv` (no scipy), matching scipy's SVD pseudo-inverse to ~1e-15
  on well-conditioned systems.

## License

Apache-2.0, © 2026 Algenta. The kernel and wrapper are clean-room
implementations of the textbook recurrences (Kalman 1960; Rauch, Tung &
Striebel 1965). pykalman is used only as a test/benchmark reference, never
as a runtime dependency.
