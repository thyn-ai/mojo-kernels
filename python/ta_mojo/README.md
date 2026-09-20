# ta-mojo

Fast technical-analysis indicators — **EMA, RSI (Wilder), ATR (Wilder), MACD**
— matching [`pandas-ta-classic`](https://pypi.org/project/pandas-ta-classic/)
(the maintained successor to `twopirllc/pandas-ta`) value-for-value, powered by
a clean-room Mojo kernel, with a vendored pure-Python fallback for platforms
without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import numpy as np
import ta_mojo

close = np.array([...], dtype=float)

ta_mojo.ema(close, length=10)              # EMA_10
ta_mojo.rsi(close, length=14)             # RSI_14 (Wilder)
ta_mojo.atr(high, low, close, length=14)  # ATRr_14
res = ta_mojo.macd(close, 12, 26, 9)      # MACD_12_26_9
res.macd, res.signal, res.histogram       # float64 ndarrays
```

- **Same values**: outputs match the oracle (pandas code path, no TA-Lib)
  element-wise — exact NaN/warmup-prefix masks, values within 1e-10 (measured
  agreement on the differential battery: 8.5e-14; EMA/RSI/ATR are bit-exact on
  the native backend). Both the native and the pure-Python fallback backends
  are tested against the oracle.
- **Same warmup semantics**: SMA-seeded recursion (TA-Lib style), so EMA's
  first `length - 1` values are NaN, RSI/ATR's first `length` are NaN, and the
  MACD signal line starts `(slow - 1) + (signal - 1)` bars in — exactly like
  the oracle, including `adjust=True/False` EMA variants (`sma=False` runs the
  plain recursion with no NaN prefix).
- **Much faster**: 1.8x–93x warm, depending on indicator and series length
  (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel,
  self-contained (the Mojo runtime is vendored in). Everywhere else the same
  API transparently runs on the vendored fallback.
- Force the fallback with `TA_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `ta_mojo.backend_info()`.

## Install

```
pip install ta-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel. On
any other platform — including Windows, where CI runs this package's fallback suite
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)) — the same wheel API runs on the
vendored pure-Python fallback, silently and correctly. There is no sdist: a
source tarball cannot rebuild the native library. The only runtime dependency
is NumPy; pandas is *not* required.

## API

All functions accept any 1-D array-like (lists, NumPy arrays, pandas Series)
and return float64 `np.ndarray`s of the same length. `length`/`fast`/`slow`/
`signal`/`drift` must be positive ints (`None` selects the oracle default);
invalid values raise `ValueError` (the oracle silently substitutes defaults —
this package fails early instead). A series shorter than the required window
yields an all-NaN array (the oracle returns `None` there; an all-NaN array is
the array-valued equivalent).

| function | signature | oracle equivalent |
|---|---|---|
| `ema` | `ema(close, length=10, *, adjust=False, sma=True)` | `ta.ema(close, length, adjust=..., sma=...)` |
| `rsi` | `rsi(close, length=14, *, scalar=100.0, drift=1)` | `ta.rsi(close, length, scalar, drift=...)` |
| `atr` | `atr(high, low, close, length=14, *, drift=1)` | `ta.atr(high, low, close, length, mamode="rma", drift=...)` |
| `macd` | `macd(close, fast=12, slow=26, signal=9)` | `ta.macd(close, fast, slow, signal)` |

`macd` returns a `MacdResult(macd, signal, histogram)` named tuple of three
arrays (the oracle's `MACD_…`, `MACDs_…`, `MACDh_…` columns). If `slow < fast`
the two are swapped, like the oracle.

## Benchmarks

Measured on this machine (date: 2026-09-19; macOS-26.6.2 arm64, Apple M4 Max;
Python 3.12.14, NumPy 2.5.3, Mojo 1.1.0; oracle pandas-ta-classic 0.8.32 on
pandas 3.0.6, pandas code path) with `benchmarks/bench_ta.py` — correctness
against the oracle is asserted before timing. "Cold" is the very first call in
a fresh interpreter (median of 5 process launches; for ta-mojo this includes
the one-time `dlopen` + ABI handshake of the native kernel, which is why the
otherwise-fastest EMA call shows a cold speedup below 1x). "Warm" is the
steady-state per-call latency (median of 5 batches of 20 calls).

Cold first call (n=100,000 bars, median of 5 fresh processes):

| indicator | pandas-ta-classic (ms) | ta-mojo (ms) | speedup |
|---|---:|---:|---:|
| ema | 5.507 | 13.3780 | 0.4x |
| rsi | 13.251 | 9.7808 | 1.4x |
| atr | 21.978 | 9.7757 | 2.2x |
| macd | 64.462 | 22.3130 | 2.9x |

Warm steady state (ms per call, median of 5 batches of 20 calls):

| indicator | bars | pandas-ta-classic (ms) | ta-mojo (ms) | speedup |
|---|---:|---:|---:|---:|
| ema | 1,000 | 0.099 | 0.0080 | 12.3x |
| ema | 10,000 | 0.154 | 0.0494 | 3.1x |
| ema | 100,000 | 1.196 | 0.5193 | 2.3x |
| ema | 1,000,000 | 11.370 | 6.3202 | 1.8x |
| rsi | 1,000 | 1.208 | 0.0245 | 49.3x |
| rsi | 10,000 | 2.770 | 0.3834 | 7.2x |
| rsi | 100,000 | 6.355 | 3.1257 | 2.0x |
| rsi | 1,000,000 | 57.997 | 32.2361 | 1.8x |
| atr | 1,000 | 1.471 | 0.0159 | 92.8x |
| atr | 10,000 | 4.202 | 0.2789 | 15.1x |
| atr | 100,000 | 23.704 | 1.0347 | 22.9x |
| atr | 1,000,000 | 157.833 | 12.6551 | 12.5x |
| macd | 1,000 | 0.848 | 0.0149 | 56.9x |
| macd | 10,000 | 8.112 | 0.2251 | 36.0x |
| macd | 100,000 | 76.762 | 1.8880 | 40.7x |
| macd | 1,000,000 | 728.769 | 14.1052 | 51.7x |

## Correctness

The differential suite (`tests/test_ta_differential.py`) compares against the
published `pandas-ta-classic` 0.8.32 package (pandas 3.0.6, its default pandas
code path — TA-Lib absent) on seeded random-walk series plus adversarial
cases: leading/interior NaNs, constant series, flat (zero-range) bars,
`length=1`..`50`, `drift>1`, fast/slow swap, and series shorter than the
window. Gate: exact NaN-mask equality and `atol=1e-10`; measured agreement is
8.5e-14 (native EMA/RSI/ATR are bit-identical to the oracle; the residual MACD
difference is ulp-level seed-mean summation order). The whole suite runs twice
— native backend and forced fallback (`TA_MOJO_DISABLE_NATIVE=1`). The two
backends agree with each other to ~1 ulp: compiled code (the Mojo kernel, like
the oracle's own Cython wheels) may contract `a*b+c` into a fused
multiply-add, which CPython bytecode cannot.

Out of scope (oracle features not exposed): TA-Lib backends, `offset`,
`fillna`/`fill_method`, `mamode` variants of ATR other than `"rma"`, MACD
`asmode`, signal-line/crossover DataFrames, and every other indicator in the
library.

## Development

```
# build the Mojo kernel (needs the repo pixi environment)
pixi run bash kernels/ta/build.sh

# oracle for the differential suite (pinned; installed into an unmanaged
# directory because `pixi run` prunes pip installs from the pixi env)
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --target .oracle-ta "pandas==3.0.6" "pandas-ta-classic==0.8.32"

# differential + loader tests, native then forced fallback
PYTHONPATH="python/ta_mojo:.oracle-ta" PYTHONNOUSERSITE=1 pixi run bash scripts/test_all_ta.sh

# benchmark (refuses to run on the fallback backend)
PYTHONPATH="python/ta_mojo:.oracle-ta" PYTHONNOUSERSITE=1 pixi run python benchmarks/bench_ta.py

# self-contained platform wheel (delocate/auditwheel repair)
PYTHONNOUSERSITE=1 pixi run bash python/ta_mojo/build_wheel.sh
```

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
