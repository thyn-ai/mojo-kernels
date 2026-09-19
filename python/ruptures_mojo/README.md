# ruptures-mojo

A drop-in faster replacement for the [`ruptures`](https://pypi.org/project/ruptures/)
change-point detection package, powered by a clean-room Mojo kernel — with a
vendored pure-Python fallback for platforms without a native build (including
Windows).

```python
import ruptures_mojo

bkps = ruptures_mojo.detect(signal, "dynp", n_bkps=3)      # == ruptures.Dynp(...).fit_predict(signal, 3)
bkps = ruptures_mojo.detect(signal, "pelt", pen=10)        # == ruptures.Pelt(...).fit_predict(signal, 10)
bkps = ruptures_mojo.detect(signal, "binseg", n_bkps=2)    # == ruptures.Binseg(...).fit_predict(signal, 2)
```

`detect(signal, method, *, model="l2", min_size=2, jump=5, n_bkps=None,
pen=None)` returns the same sorted breakpoint list (plain Python `int`s,
always ending with `n_samples`) as the matching `ruptures` estimator with the
same `model` / `min_size` / `jump`.

- **Same results**: breakpoint sets are *integer equal* to `ruptures`
  (PyPI ruptures==1.1.10) — not approximately, exactly. The differential
  suite asserts list equality against the oracle on both the native and the
  fallback backend across seeded Gaussian, mean-shift, variance-shift,
  integer-valued (tie-heavy), constant-plateau, and trend signals, for all
  three methods, both cost models, and a grid of `min_size` / `jump` values
  (792 oracle comparisons in the seeded grid alone; 242 tests per backend).
- **Much faster**: the dynamic-programming recurrences (Dynp), the pruned
  linear scan (PELT), and the greedy split search (Binseg) run in one
  compiled Mojo call over the whole signal — no per-segment Python/NumPy
  calls. Measured numbers below.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `RUPTURES_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `ruptures_mojo.backend_info()` and
  `ruptures_mojo.last_backend()`.

## Scope

- 1-D signals (shape `(n,)` or `(n, 1)`), finite `float64` values.
- Cost models: `l2` (least squared deviation) and `l1` (least absolute
  deviation). The effective `min_size` is `max(min_size, 1)` for `l2` and
  `max(min_size, 2)` for `l1`, as in ruptures.
- Methods: `dynp` (exact DP, requires `n_bkps`), `pelt` (requires `pen`),
  `binseg` (requires `n_bkps`).
- Not supported (raises `ValueError`): other cost models (`rbf`, `ar`,
  `normal`, ...), multi-dimensional signals, non-finite input, Binseg's
  `pen`/`epsilon` stopping rules.
- The native Dynp path holds a `G x G` segment-cost matrix for the
  `G = n // jump + 2` admissible grid and supports grids up to 6000
  (e.g. `n = 30,000` at `jump = 5`); larger grids transparently use the
  fallback. PELT and Binseg have no such limit.
- Input is converted to `float64`; integer and `float64` inputs give
  bit-faithful parity with ruptures. (`float32` input is accepted but
  ruptures would compute its costs in `float32`, so parity there is not
  bit-level.)
- Like the oracle, PELT's pruning can degenerate to its O(n^2) worst case on
  long signals with small penalties (both implementations visit the same
  candidate set); the native path stays 2x-50x faster there, not orders of
  magnitude. Dynp is the exact-DP method and is the workload this package
  accelerates the most.

## How exactness is achieved

`ruptures`' l2 segment cost is `signal[a:b].var() * m` computed with NumPy's
two-pass pairwise-summation variance; l1 is `|x - median(x)|` summed the same
way. The Mojo kernel re-implements NumPy's pairwise summation order
(8-accumulator blocks, binary recursion above 128 elements) bit-for-bit, so
segment costs are bit-identical to the oracle's; candidate enumeration, the
left-to-right summation of segment costs, and the tie-breaking rules (Dynp /
PELT: smallest candidate wins; Binseg: largest candidate within a segment,
leftmost segment across segments) replicate the oracle's observable behaviour
exactly. That is why the returned breakpoint sets are integer equal, and why
they stay equal on tie-heavy (integer-valued or constant) signals.

## Benchmark

Method: one whole detection call (`fit_predict` / `detect`) on a
piecewise-constant-mean + Gaussian-noise signal, seeded and reproducible
(`benchmarks/bench_ruptures.py`). Warm = median of 5 repeated calls in a live
process; cold = median of 5 first-call times in fresh interpreter processes.
Correctness gate: exact breakpoint-list equality with ruptures asserted
before timing. Measured on this machine:

- date: 2026-09-19
- machine: macOS-26.6.2-arm64 (Apple M4 Max)
- python: 3.12.14, numpy: 2.5.3; mojo: Mojo 1.1.0 (8189361e)
- oracle: PyPI ruptures==1.1.10; seed: 20260919+n; runs: median of 5

| workload | ruptures warm (ms) | ruptures_mojo warm (ms) | warm speedup | ruptures cold (ms) | ruptures_mojo cold (ms) | cold speedup |
|---|---:|---:|---:|---:|---:|---:|
| Dynp  n=1,000   K=3  jump=1 | 5467.90 | 120.635 | 45.3x | 7153.93 | 186.791 | 38.3x |
| Dynp  n=2,000   K=5  jump=2 | 9447.07 | 268.884 | 35.1x | 11094.76 | 296.890 | 37.4x |
| PELT  n=10,000  pen=10 jump=1 | 36466.59 | 972.047 | 37.5x | 44546.26 | 927.704 | 48.0x |
| PELT  n=20,000  pen=10 jump=5 | 9594.89 | 917.051 | 10.5x | 14910.95 | 935.189 | 15.9x |
| Binseg n=50,000  K=20 jump=1 | 12972.39 | 4455.929 | 2.9x | 10939.65 | 5166.216 | 2.1x |
| Binseg n=100,000 K=20 jump=5 | 11876.65 | 5324.330 | 2.2x | 11609.48 | 6041.528 | 1.9x |

Why the gap: ruptures evaluates every candidate segment cost as a separate
Python call into NumPy (`signal[a:b].var(axis=0).sum()`), and the Dynp
recursion builds a Python dict per candidate partition — the maintainer has
noted the DP "cannot be NumPy-vectorized". It does not need to be: the whole
recurrence is a single compiled kernel here, with each segment cost evaluated
by SIMD pairwise summation in one pass over a typed buffer. The honest
exception is Binseg at large n: there both sides are bound by the same
memory-bandwidth-limited passes over long segments (NumPy's variance is SIMD
too), so the win is 2x-3x rather than 30x-50x — that is the real number, and
it is why the table above reports it.

## Development

Source, tests, and benchmark: <https://github.com/thyn-ai/mojo-kernels>

```
bash kernels/ruptures/build.sh                  # compile the Mojo kernel
bash scripts/test_all_ruptures.sh               # differential suite, both backends
python benchmarks/bench_ruptures.py             # reproduce the numbers above
bash python/ruptures_mojo/build_wheel.sh        # build + repair this wheel
```

License: Apache-2.0, © 2026 Algenta
