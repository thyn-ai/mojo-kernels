# spatialmath-mojo

Fast batched **SE(3)/SO(3) pose algebra** — **compose, inverse, point
transforms** — matching [`spatialmath-python`](https://github.com/petercorke/spatialmath-python)
(Peter Corke's robotics toolbox) pose-for-pose, powered by a clean-room Mojo
kernel, with a vendored pure-Python fallback for platforms without a native
build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import numpy as np
import spatialmath_mojo as smm

A = np.stack([...])   # (n, 4, 4) float64 SE(3) poses
B = np.stack([...])
P = np.random.randn(m, 3)  # (m, 3) points

smm.compose(A, B)      # (n, 4, 4): per-pose A[i] @ B[i]
smm.inverse(A)         # (n, 4, 4): per-pose [Rᵀ, -Rᵀt; 0 0 0 1]
smm.transform(A, P)    # (n, m, 3): h2e(T[i] @ e2h(P[j]))
# 3x3 inputs run the SO(3) variants; singletons broadcast (1×N, N×1).
```

- **Same values**: outputs match the oracle (`spatialmath-python` 1.1.18,
  exercised through its object API: `SE3 * SE3`, `SE3.inv()`, `SE3 * vector`,
  `SE3 * matrix`, and the SO3 analogues) element-wise within 1e-10 — measured
  agreement on the differential battery is 7.1e-15 on the native backend and
  **0.0 (bit-exact)** on the fallback backend. Both backends are tested
  against the oracle.
- **Same semantics**: full per-pose 4×4 products (bottom row included), the
  structured `trinv` inverse (the input's bottom row is never read), and
  `h2e`'s homogeneous division (applied even when it is exactly 1) — the same
  IEEE-754 float64 operation order as the toolbox's per-pose numpy dispatch.
- **Much faster**: 7.7x–843x warm on pose-batch operations, depending on the
  operation and batch size (the oracle dispatches per pose through
  `baseposelist._binop`; this package evaluates the whole batch in one kernel
  call). Full method and numbers below, including the cold-start cost.
- **No toolchain needed**: per-platform wheels ship the compiled kernel,
  self-contained (the Mojo runtime is vendored in). Everywhere else the same
  API transparently runs on the vendored fallback.
- Force the fallback with `SPATIALMATH_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `spatialmath_mojo.backend_info()`.

## Install

```
pip install spatialmath-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel. On
any other platform — including Windows, where CI runs this package's fallback suite
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)) — the same wheel API runs on the
vendored pure-Python fallback, silently and correctly. There is no sdist: a
source tarball cannot rebuild the native library. The only runtime dependency
is NumPy; spatialmath-python itself is *not* required.

## API

Poses are (d, d) arrays or (n, d, d) batches, float64 in/out — d = 4 selects
SE(3), d = 3 selects SO(3). Points are (3,) or (m, 3) row-major arrays.
Invalid shapes raise `ValueError` (the toolbox coerces or broadcasts in some
of these cases — this package fails early instead).

| function | signature | oracle equivalent |
|---|---|---|
| `compose` | `compose(a, b)` | `SE3(A) * SE3(B)`, pose-list `Ta * Tb` (and SO3) |
| `inverse` | `inverse(t)` | `SE3(T).inv()` / `SO3(R).inv()` |
| `transform` | `transform(t, points)` | `SE3(T) * v`, `SE3(T) * P`, pose-list `Ta * v` (and SO3) |

Shape conventions:

- `compose`: a singleton operand broadcasts against the other's batch (the
  toolbox's `1 * M` / `N * 1` rules); two non-singleton batches must have
  equal length. Two (d, d) inputs return (d, d); anything else returns
  (n, d, d).
- `inverse`: returns the input's shape.
- `transform`: (d, d)+(3,) → (3,); (d, d)+(m, 3) → (m, 3);
  (n, d, d)+(3,) → (n, 3) (the oracle's pose-list × vector returns this
  transposed, as (3, n)); (n, d, d)+(m, 3) → (n, m, 3).
- `inverse` on SE(3) uses the structured inverse: the input's bottom row is
  never read and the output's is exactly `[0, 0, 0, 1]` (identical to the
  oracle's `trinv` for valid SE(3) matrices; the input must be valid SE(3)
  for the result to be meaningful).

## Benchmarks

Measured on this machine (date: 2026-09-19; macOS-26.6.2 arm64, Apple M4 Max;
Python 3.12.14, NumPy 2.5.3, Mojo 1.1.0; oracle spatialmath-python 1.1.18)
with `benchmarks/bench_spatialmath.py` — correctness against the oracle is
asserted before timing. The oracle's pose objects are constructed outside
the timer; only the pose operation is timed (`Ta * Tb`, `Ta.inv()`, `Ta * v`,
`T0 * P` — the per-pose dispatch in `spatialmath.baseposelist._binop`). Our
side times the whole `spatialmath_mojo` call on plain numpy arrays,
validation included. "Cold" is the very first call in a fresh interpreter
(median of 5 process launches; for spatialmath-mojo this includes the
one-time `dlopen` + ABI handshake of the native kernel). "Warm" is the
steady-state per-call latency (median of 5 batches of 20 calls).

Cold first call (n=10,000 poses, median of 5 fresh processes):

| operation | spatialmath-python (ms) | spatialmath-mojo (ms) | speedup |
|---|---:|---:|---:|
| se3_compose | 6.170 | 5.1544 | 1.2x |
| se3_inverse | 20.463 | 5.1175 | 4.0x |
| se3_transform_batch_vector | 16.938 | 3.8405 | 4.4x |
| se3_transform_single_points | 0.128 | 4.2264 | 0.0x |
| so3_compose | 6.099 | 4.0532 | 1.5x |
| so3_inverse | 1.125 | 4.4003 | 0.3x |
| so3_transform_batch_vector | 7.415 | 4.2025 | 1.8x |

Two cold rows sit below 1x for structural reasons: a cold spatialmath-mojo
call carries the one-time `dlopen` + ABI handshake (~4 ms here), while the
oracle's `se3_transform_single_points` (one 4×4 BLAS call, 0.13 ms) and
`so3_inverse` (one transpose per pose, 1.1 ms) are already cheaper than that
handshake. From the second call on, every operation is at or above 1x (see
the warm table); the handshake amortizes over the process lifetime.

Warm steady state (ms per call, median of 5 batches of 20 calls):

| operation | poses | spatialmath-python (ms) | spatialmath-mojo (ms) | speedup |
|---|---:|---:|---:|---:|
| se3_compose | 100 | 0.056 | 0.0072 | 7.7x |
| se3_compose | 1,000 | 0.615 | 0.0283 | 21.8x |
| se3_compose | 10,000 | 5.974 | 0.2311 | 25.9x |
| se3_compose | 100,000 | 63.627 | 2.3482 | 27.1x |
| se3_inverse | 100 | 0.203 | 0.0038 | 53.4x |
| se3_inverse | 1,000 | 1.949 | 0.0094 | 208.4x |
| se3_inverse | 10,000 | 19.923 | 0.0256 | 779.0x |
| se3_inverse | 100,000 | 197.194 | 0.2772 | 711.4x |
| se3_transform_batch_vector | 100 | 0.189 | 0.0049 | 38.4x |
| se3_transform_batch_vector | 1,000 | 1.913 | 0.0079 | 242.7x |
| se3_transform_batch_vector | 10,000 | 18.812 | 0.0223 | 843.2x |
| se3_transform_batch_vector | 100,000 | 189.573 | 0.3187 | 594.8x |
| se3_transform_single_points | 100 | 0.005 | 0.0051 | 1.0x |
| se3_transform_single_points | 1,000 | 0.008 | 0.0062 | 1.3x |
| se3_transform_single_points | 10,000 | 0.057 | 0.0250 | 2.3x |
| se3_transform_single_points | 100,000 | 0.867 | 0.3760 | 2.3x |
| so3_compose | 100 | 0.071 | 0.0059 | 12.1x |
| so3_compose | 1,000 | 0.661 | 0.0164 | 40.2x |
| so3_compose | 10,000 | 6.878 | 0.1027 | 66.9x |
| so3_compose | 100,000 | 74.079 | 1.6406 | 45.2x |
| so3_inverse | 100 | 0.013 | 0.0035 | 3.6x |
| so3_inverse | 1,000 | 0.119 | 0.0064 | 18.6x |
| so3_inverse | 10,000 | 1.204 | 0.0308 | 39.1x |
| so3_inverse | 100,000 | 12.356 | 0.8892 | 13.9x |
| so3_transform_batch_vector | 100 | 0.074 | 0.0051 | 14.5x |
| so3_transform_batch_vector | 1,000 | 0.844 | 0.0076 | 110.7x |
| so3_transform_batch_vector | 10,000 | 8.244 | 0.0319 | 258.6x |
| so3_transform_batch_vector | 100,000 | 99.378 | 0.2879 | 345.2x |

`se3_transform_single_points` is the one workload with no per-pose dispatch
on the oracle side (a single 4×4 @ 4×m BLAS call), so the speedup there is
1.0x–2.3x; every batch-dispatched operation is 3.6x–843x.

## Correctness

The differential suite (`tests/test_spatialmath_differential.py`) compares
against the published `spatialmath-python` 1.1.18 package on seeded random
proper-SE(3)/SO(3) batches plus edge cases: singleton × batch and batch ×
singleton broadcasting, singleton (2-D) in/out shapes, batch × vector,
singleton × point-matrix, batch × point-matrix (stacked per-pose oracle
calls), homogeneous division by a non-unit bottom row, `trinv`'s ignored
bottom row, empty batches, array-like inputs, and invalid-input rejection.
Gate: `atol=1e-10`; measured agreement is 7.1e-15 on the native backend and
exactly 0.0 on the fallback (the fallback issues the same per-pose NumPy
calls as the oracle). The whole suite runs twice — native backend and
forced fallback (`SPATIALMATH_MOJO_DISABLE_NATIVE=1`). The two backends
agree with each other to ~1 ulp: compiled code (the Mojo kernel, like the
oracle's own BLAS) may contract `a*b+c` into a fused multiply-add, which
the ordered per-pose calls do not.

Out of scope (oracle features not exposed): pose *classes* and everything
hanging off them (plotting, trajectories, `SE3.Rx/Trans/...` constructors,
quaternion/Euler/RPY conversions, twists, `SO2`/`SE2`, `@` composition with
normalization, scalar multiplication, pose arithmetic other than
compose/inverse/transform), as well as validation/orthonormalization of
input matrices (inputs must be valid SE(3)/SO(3); `inverse` on SE(3)
nevertheless tolerates a garbage bottom row exactly like `trinv`). Matrices
only — no angle representations.

## Development

```
# build the Mojo kernel (needs the repo pixi environment)
pixi run bash kernels/spatialmath/build.sh

# oracle for the differential suite (pinned; installed into an unmanaged
# directory because `pixi run` prunes pip installs from the pixi env)
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --target .oracle-spatialmath "spatialmath-python==1.1.18"

# differential + loader tests, native then forced fallback
PYTHONPATH="python/spatialmath_mojo:.oracle-spatialmath" PYTHONNOUSERSITE=1 \
  pixi run bash scripts/test_all_spatialmath.sh

# benchmark (refuses to run on the fallback backend)
PYTHONPATH="python/spatialmath_mojo:.oracle-spatialmath" PYTHONNOUSERSITE=1 \
  pixi run python benchmarks/bench_spatialmath.py

# self-contained platform wheel (delocate/auditwheel repair)
PYTHONNOUSERSITE=1 pixi run bash python/spatialmath_mojo/build_wheel.sh
```

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
