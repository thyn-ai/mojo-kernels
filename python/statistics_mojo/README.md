# statistics-mojo

A drop-in faster replacement for the hot paths of the CPython standard
library [`statistics`](https://docs.python.org/3/library/statistics.html)
module — **mean, fmean, median, median_low, median_high, mode, multimode,
variance, stdev, pvariance, pstdev, quantiles** — powered by a clean-room
Mojo kernel, with a pure-Python fallback for platforms without a native
build (the fallback delegates to the stdlib module itself, so it is correct
by construction everywhere, including Windows).

```python
import statistics_mojo as stats

stats.mean([1, 2, 3, 4])                  # 2.5 (float, like the stdlib)
stats.mean([1, 2, 3])                     # 2 (int, like the stdlib)
stats.variance([2.75, 1.75, 1.25, 0.25])  # 1.3720238095238095 — bit-exact
stats.stdev(big_float_list)               # correctly rounded, ~20x faster
stats.quantiles(data, n=100)              # percentiles, ~12x faster
stats.median(data)                        # radix-sorted, ~11x faster

# Column batch API (one kernel pass over many columns):
stats.variance_batch([col_a, col_b, col_c])
stats.mean_batch([col_a, col_b, col_c])
```

- **Bit-exact**: outputs equal the CPython 3.12 stdlib oracle value-for-value,
  type-for-type, and error-for-error — including the documented exact-rational
  semantics (mean/variance are computed with exact arithmetic, not float64
  sums), `int` results where the stdlib returns `int` (`mean([1,2,3]) == 2`),
  `Fraction`/`Decimal` results for `Fraction`/`Decimal` inputs, and identical
  `StatisticsError` messages. The differential suite asserts exact equality
  (type, value, repr, exception type and message) on both the native and the
  forced-fallback backend.
- **Exact-rational, not approximate**: for float64 data the kernel accumulates
  sums and sums of squares in wide fixed-point superaccumulators (one bit
  position per possible binary exponent), reproducing the stdlib's exact
  rational `_sum`/`_ss` bit-for-bit; the single final division and float
  conversion is correctly rounded, exactly like the stdlib. `fmean` replicates
  CPython's `math.fsum` algorithm step for step — including its
  order-dependent intermediate-overflow behaviour.
- **Much faster**: 15–53x warm on the variance/stdev family, 14–44x on `mean`
  (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel,
  self-contained (the Mojo runtime is vendored in). Everywhere else the same
  API transparently runs on the fallback (which calls the stdlib module —
  always present, and the reference implementation being accelerated).
- Force the fallback with `STATISTICS_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `statistics_mojo.backend_info()`.

## Install

```
pip install statistics-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel. On
any other platform — including Windows — the same wheel API runs on the
pure-Python fallback, silently and correctly; the fallback is exercised in CI
on macOS and Linux via the forced-fallback pass of the differential suite
(`STATISTICS_MOJO_DISABLE_NATIVE=1`). There is no sdist: a source tarball
cannot rebuild the native library. The only runtime dependency is NumPy.

## API

All functions accept any iterable (lists, tuples, ranges, generators, NumPy
arrays) and return exactly what the stdlib returns for the same input —
builtin `int`/`float`/`Fraction`/`Decimal` for list input. Two documented
nuances:

- **ndarray input**: results are returned as builtin `float`/`int` where the
  stdlib returns NumPy scalar types (`np.float64`, `np.int64`); values agree
  bit-for-bit. (Integer ndarrays route to the stdlib path: the stdlib's
  numpy-scalar coercion for those truncates — `statistics.mean(np.arange(3))`
  is `np.int64(1)`, not `1.0` — and that behaviour is preserved, not
  accelerated.)
- The kernel accelerates **homogeneous float64 / int64-shaped data**:
  lists of floats, lists of ints (compact CPython ints, |v| < 2^60), tuples
  and generators of the same, mixed int/float data with ints |v| ≤ 2^53, and
  float64 ndarrays. Everything else — `Fraction`, `Decimal`, mixed-type data,
  huge ints, NaN/inf-bearing data for `median`/`quantiles`, both `+0.0` and
  `-0.0` present (sort order of signed zeros is identity-defined) — runs on
  the stdlib implementation, on both backends, with identical results.

| function | signature | notes |
|---|---|---|
| `mean` | `mean(data)` | exact-rational sum; int-or-float result per the stdlib |
| `fmean` | `fmean(data, weights=None)` | `math.fsum` semantics; `weights` runs the stdlib path |
| `median` / `median_low` / `median_high` | `median(data)` | radix sort replaces timsort |
| `mode` / `multimode` | `mode(data)` | Counter semantics; identical on both backends (not kernel-accelerated) |
| `variance` / `stdev` | `variance(data, xbar=None)` | exact sum of squared deviations |
| `pvariance` / `pstdev` | `pvariance(data, mu=None)` | population variants |
| `quantiles` | `quantiles(data, *, n=4, method="exclusive")` | n=4/10/100 and any n; exclusive + inclusive |
| `mean_batch` … `quantiles_batch` | `variance_batch(columns)` etc. | column batch: one kernel pass for uniform list/ndarray batches; per-column stdlib semantics |

`StatisticsError` is re-exported (`statistics_mojo.StatisticsError is
statistics.StatisticsError`), so `except statistics.StatisticsError` catches
errors from both. Error cases match the stdlib exactly: `mean([])` →
"mean requires at least one data point", `variance([1.0])` → "variance
requires at least two data points", `quantiles([1.0])` → "must have at least
two data points", `quantiles(..., n=0)` → "n must be at least 1",
`quantiles(..., method="bogus")` → `ValueError("Unknown method: 'bogus'")`,
`fmean` intermediate overflow → `OverflowError("intermediate overflow in
fsum")`, `fmean([inf, -inf])` → `ValueError("-inf + inf in fsum")`.

## Benchmarks

Measured on this machine (date: 2026-09-20; macOS-26.6.2 arm64, Apple M4 Max;
Python 3.12.14, NumPy 2.5.3, Mojo 1.1.0; oracle: CPython stdlib `statistics`
3.12.14) with `benchmarks/bench_statistics.py` — exact equality with the
oracle is asserted before timing. "Cold" is the very first call in a fresh
interpreter (median of 5 process launches; for statistics-mojo this includes
the one-time `dlopen` + ABI handshake + CPython-layout self-test of the
native kernel, which is why the otherwise fastest calls show cold speedups
below warm). "Warm" is the steady-state per-call latency (median of 5 batches
of 20 calls).

Cold first call (n=100,000 floats, median of 5 fresh processes):

| function | stdlib statistics (ms) | statistics-mojo (ms) | speedup |
|---|---:|---:|---:|
| mean | 19.849 | 8.2301 | 2.4x |
| fmean | 2.469 | 9.0576 | 0.3x |
| median | 10.095 | 8.8275 | 1.1x |
| variance | 47.366 | 30.6650 | 1.5x |
| stdev | 39.845 | 14.5522 | 2.7x |
| quantiles | 26.616 | 9.7312 | 2.7x |

Warm steady state, float lists (ms per call, median of 5 batches of 20 calls):

| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |
|---|---:|---:|---:|---:|
| mean | 1,000 | 0.208 | 0.0143 | 14.5x |
| mean | 10,000 | 1.949 | 0.0438 | 44.4x |
| mean | 100,000 | 20.469 | 0.7653 | 26.7x |
| mean | 1,000,000 | 202.935 | 8.5459 | 23.7x |
| fmean | 1,000 | 0.008 | 0.0145 | 0.5x |
| fmean | 10,000 | 0.116 | 0.1734 | 0.7x |
| fmean | 100,000 | 1.952 | 2.0427 | 1.0x |
| fmean | 1,000,000 | 21.087 | 19.8738 | 1.1x |
| median | 1,000 | 0.028 | 0.0215 | 1.3x |
| median | 10,000 | 0.671 | 0.1112 | 6.0x |
| median | 100,000 | 9.615 | 0.9670 | 9.9x |
| median | 1,000,000 | 127.026 | 11.8054 | 10.8x |
| variance | 1,000 | 0.328 | 0.0212 | 15.5x |
| variance | 10,000 | 3.219 | 0.0603 | 53.4x |
| variance | 100,000 | 27.946 | 1.3894 | 20.1x |
| variance | 1,000,000 | 286.655 | 14.6888 | 19.5x |
| stdev | 1,000 | 0.304 | 0.0193 | 15.7x |
| stdev | 10,000 | 2.929 | 0.0619 | 47.3x |
| stdev | 100,000 | 27.478 | 1.3501 | 20.4x |
| stdev | 1,000,000 | 283.712 | 15.5060 | 18.3x |
| quantiles | 1,000 | 0.038 | 0.0380 | 1.0x |
| quantiles | 10,000 | 0.677 | 0.1336 | 5.1x |
| quantiles | 100,000 | 10.333 | 1.1091 | 9.3x |
| quantiles | 1,000,000 | 133.897 | 11.2655 | 11.9x |

Warm steady state, int lists (ms per call):

| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |
|---|---:|---:|---:|---:|
| mean | 1,000 | 0.067 | 0.0092 | 7.3x |
| mean | 10,000 | 0.677 | 0.0351 | 19.3x |
| mean | 100,000 | 6.889 | 0.5490 | 12.5x |
| mean | 1,000,000 | 72.417 | 7.1146 | 10.2x |
| variance | 1,000 | 0.136 | 0.0136 | 10.0x |
| variance | 10,000 | 1.214 | 0.0450 | 27.0x |
| variance | 100,000 | 10.828 | 0.7142 | 15.2x |
| variance | 1,000,000 | 114.636 | 7.8817 | 14.5x |
| median | 1,000 | 0.029 | 0.0254 | 1.2x |
| median | 10,000 | 0.655 | 0.1880 | 3.5x |
| median | 100,000 | 9.484 | 1.6262 | 5.8x |
| median | 1,000,000 | 114.451 | 16.0331 | 7.1x |

Warm steady state, float64 ndarray input (ms per call):

| function | n | stdlib statistics (ms) | statistics-mojo (ms) | speedup |
|---|---:|---:|---:|---:|
| mean | 1,000 | 0.236 | 0.0130 | 18.2x |
| mean | 10,000 | 2.253 | 0.0430 | 52.4x |
| mean | 100,000 | 23.098 | 0.7062 | 32.7x |
| mean | 1,000,000 | 223.073 | 8.1562 | 27.4x |
| fmean | 1,000 | 0.029 | 0.0156 | 1.9x |
| fmean | 10,000 | 0.292 | 0.1736 | 1.7x |
| fmean | 100,000 | 2.930 | 1.9243 | 1.5x |
| fmean | 1,000,000 | 30.511 | 19.8158 | 1.5x |
| variance | 1,000 | 0.370 | 0.0197 | 18.8x |
| variance | 10,000 | 3.297 | 0.0634 | 52.0x |
| variance | 100,000 | 32.235 | 1.4111 | 22.8x |
| variance | 1,000,000 | 314.455 | 14.7303 | 21.3x |
| median | 1,000 | 0.116 | 0.0200 | 5.8x |
| median | 10,000 | 1.616 | 0.1232 | 13.1x |
| median | 100,000 | 20.440 | 0.9684 | 21.1x |
| median | 1,000,000 | 288.330 | 11.0768 | 26.0x |

Warm column batch (32 columns x 32,000 rows, ms per batch):

| function | stdlib loop (ms) | statistics-mojo (ms) | speedup |
|---|---:|---:|---:|
| mean_batch | 200.221 | 9.2813 | 21.6x |
| variance_batch | 288.771 | 15.4646 | 18.7x |
| stdev_batch | 292.281 | 15.8582 | 18.4x |
| median_batch | 92.461 | 11.9304 | 7.7x |
| fmean_batch | 20.256 | 20.6922 | 1.0x |

Honest notes on the numbers:

- **fmean is parity, not a win** (0.5–1.1x warm on lists, ~1.5x on ndarrays):
  the stdlib's `fmean` is a thin wrapper over C `math.fsum`, already C-speed;
  the kernel replicates that algorithm (bit-exact, quirks included) at
  roughly the same speed. The cold first call pays the one-time kernel load
  (0.3x). There is no honest 10x here.
- Small inputs (n ≤ 1,000) are latency-bound: `quantiles` at n=1,000 is 1.0x.
- The batch API wins by single-pass kernel execution per column with no
  per-element Python work; `fmean_batch` is again fsum-parity.

## Correctness

The differential suite (`tests/test_statistics_differential.py`) compares
against the CPython 3.12 stdlib `statistics` module — the reference
implementation — on seeded random data (floats, ints, mixed) plus adversarial
cases: empty and single-element inputs, huge ints (> 2^63), ints at the 2^53
exactness boundary, subnormals, 1e±308 magnitudes, inf/NaN data, signed
zeros, exactness cases from the stdlib docs (Decimal context division,
Fraction results), iterators, generators, strided/integer/uint64 ndarrays,
`xbar`/`mu` centres of every supported type, and every error path. Gate:
**exact equality** — same type, same value, same repr (catches `-0.0`), same
exception type and message. Parity tolerance: none (bit-for-bit). Measured
agreement: exact on every case in the battery. The whole suite runs twice —
native backend (862 tests) and forced fallback (860 tests + 2 backend-gated
skips) — via `scripts/test_all_statistics.sh`.

Out of scope (stdlib features not exposed): `geometric_mean`,
`harmonic_mean`, `median_grouped`, `covariance`, `correlation`,
`linear_regression`, `NormalDist`, and `fmean` with `weights` (runs the
stdlib path). Not kernel-accelerated (stdlib path on both backends, identical
results): `mode`/`multimode`, Decimal/Fraction/mixed-type data, ints with
|v| ≥ 2^60, integer ndarrays (see API nuances), NaN/inf-bearing
median/quantiles inputs, and mixed-signed-zero sorting.

## Development

```
# build the Mojo kernel (needs the repo pixi environment)
pixi run bash kernels/statistics/build.sh

# differential + loader tests, native then forced fallback
PYTHONPATH="python/statistics_mojo" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_statistics.sh

# benchmark (refuses to run on the fallback backend)
PYTHONPATH=python/statistics_mojo PYTHONNOUSERSITE=1 pixi run python benchmarks/bench_statistics.py

# self-contained platform wheel (delocate/auditwheel repair)
PYTHONNOUSERSITE=1 pixi run bash python/statistics_mojo/build_wheel.sh
```

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
