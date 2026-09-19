# obspy-mojo

Drop-in faster [Konno-Ohmachi smoothing](https://doi.org/10.1785/BSSA0880010228)
for [ObsPy](https://pypi.org/project/obspy/) users, powered by a clean-room
Mojo kernel — with a vendored pure-NumPy fallback for platforms without a
native build (including Windows).

```python
import numpy as np
from obspy_mojo import konno_ohmachi_smoothing

smoothed = konno_ohmachi_smoothing(spectra, frequencies)          # b=40 default
smoothed = konno_ohmachi_smoothing(spectra, frequencies, bandwidth=40,
                                   count=1, enforce_no_matrix=False,
                                   max_memory_usage=512, normalize=False)
```

The signature, defaults, validation errors, loop-vs-matrix branch decision
(including the exact memory-estimate formula), and edge semantics (zero /
duplicate / negative frequencies, `count` applications, float32 vs float64)
are identical to
[`obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing`](https://docs.obspy.org/packages/autogen/obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing.html)
(obspy 1.5.1). The contract was established clean-room: the kernel and the
fallback were written fresh from the published algorithm (Konno & Ohmachi,
1998, BSSA 88(1):228-241) and the oracle's behavior was pinned down purely
by probing its inputs and outputs — never by reading its source.

- **Same results**: output matches obspy element-wise within
  **rtol=1e-9, atol=1e-12** on float64 (measured agreement across the
  differential suite and the benchmark gate: **<= 2.1e-14** relative on the
  native backend, **<= 5.4e-15** on the fallback) and **rtol=1e-4, atol=1e-6**
  on float32 (measured <= 5e-6). The residual ulps come from SIMD sin/log10
  evaluation and SIMD-lane accumulation (NumPy uses pairwise summation);
  everything else — branch decisions, edge semantics, dtypes, errors — is
  mirrored exactly.
- **Faster**: a fused window+reduction kernel (no O(n^2) window matrix is
  materialized on the default single-spectrum path; log10(f) is precomputed
  once per call) — 3.9x-6.7x faster steady-state smoothing, up to 45x faster
  cold first call (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-NumPy fallback —
  which is itself faster than obspy's per-center loop (chunked evaluation).
- Force the fallback with `OBSPY_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `obspy_mojo.backend_info()`.

## How it works

obspy's default single-spectrum path evaluates the Konno-Ohmachi window

```
W(f, fc) = (sin(x)/x)^4,  x = b * log10(f / fc)
```

in a Python-level loop over all center frequencies, allocating several
temporary arrays per center. The Mojo kernel instead fuses the whole
window+reduction into one pass over the samples per center (SIMD lanes,
no temporaries, log10 precomputed once), exported behind a small C ABI
(`konnomojo_smooth_loop_{f64,f32}` for the loop path,
`konnomojo_window_matrix_{f64,f32}` for the raw matrix of the matrix path —
row normalization, matrix powers, and the matmul itself stay with
NumPy/BLAS). The wrapper mirrors obspy's branch decision:

- **loop path** (1-D input with `count == 1`, `enforce_no_matrix=True`, or a
  window-matrix estimate above `max_memory_usage` MB): per-center fused
  window+reduction;
- **matrix path** (2-D+ input or `count > 1`, when the estimate fits):
  build W, optionally row-normalize, take the `count`-th matrix power, one
  matmul. The memory estimate is
  `(n_freqs**2 + n_freqs + 2 * spectra.shape[0]) * dtype.itemsize / 2**20 < max_memory_usage`
  (strict `<`), probed to the ulp against the oracle.

The two paths give measurably different results in obspy (the loop path
normalizes each center's window by its own sum; the matrix path
row-normalizes the sample-by-center matrix) — both are reproduced
faithfully, so results match the oracle on either branch.

## Benchmarks

Measured 2026-09-19 on an Apple M4 Max (macOS 26.6.2, arm64), python 3.12.14,
numpy 2.5.3, Mojo 1.1.0 (max 26.6), obspy 1.5.1. Median of 5 runs; random
spectra (seeded) on log-spaced grids; correctness against obspy asserted
before every timing run (worst relative deviation across all gate cases:
**1.35e-14**). Reproduce from the repository root:

```bash
PYTHONPATH=".oracle-python:python/obspy_mojo" PYTHONNOUSERSITE=1 \
    ~/.pixi/bin/pixi run python benchmarks/bench_obspy_konno.py
```

Loop path (default single-spectrum, b=40), warm steady-state, median of 5:

| freq bins | normalize | obspy ms/call | obspy-mojo ms/call | speedup |
|---:|---|---:|---:|---:|
| 512 | False | 13.31 | 2.588 | 5.1x |
| 512 | True | 8.18 | 2.092 | 3.9x |
| 2,048 | False | 90.89 | 22.566 | 4.0x |
| 2,048 | True | 151.57 | 32.840 | 4.6x |
| 8,192 | False | 2008.95 | 299.455 | 6.7x |
| 8,192 | True | 1418.49 | 293.054 | 4.8x |

Cold first call (fresh process: import + first smoothing), median of 5.
obspy's import pulls in its full dependency tree; obspy-mojo imports only
numpy and dlopens its kernel:

| freq bins | obspy (s) | obspy-mojo (s) | speedup |
|---:|---:|---:|---:|
| 512 | 2.281 | 0.050 | 45.4x |
| 2,048 | 2.042 | 0.087 | 23.5x |
| 8,192 | 3.620 | 0.366 | 9.9x |

Matrix path (32 spectra, normalize=True), warm, median of 5 (matmul time is
shared NumPy/BLAS in both columns; the kernel accelerates the O(n^2) window
build):

| freq bins | obspy ms/call | obspy-mojo ms/call | speedup |
|---:|---:|---:|---:|
| 512 | 8.65 | 1.85 | 4.7x |
| 2,048 | 87.71 | 28.90 | 3.0x |

The pure-NumPy fallback (used on Windows and other unsupported platforms)
measured in the same runs: 5.23 / 99.16 / 1283.87 ms (normalize=False) and
7.35 / 110.94 / 1444.76 ms (normalize=True) at 512 / 2,048 / 8,192 bins —
i.e. 0.9x-2.5x obspy's own loop, and always bit-faithful to it within the
same documented tolerance.

## Compatibility and scope

- **Supported**: `konno_ohmachi_smoothing` for float32/float64 ndarrays of
  any dimensionality (smoothing along the last axis), all documented
  options, native on macOS arm64 and Linux x86_64 (wheels tagged
  accordingly), fallback everywhere else.
- **Out of scope**: the rest of obspy (no Trace/Stream integration, no
  `calculate_smoothing_matrix` helper export); platforms without a Mojo
  toolchain get the fallback only.
- The differential suite (`tests/test_obspy_konno.py`,
  `tests/test_obspy_konno_loader.py`) runs the same 109 checks against the
  installed obspy package on both backends (native, then
  `OBSPY_MOJO_DISABLE_NATIVE=1`).

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
