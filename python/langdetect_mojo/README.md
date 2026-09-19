# langdetect-mojo

Drop-in faster replacement for the [`langdetect`](https://pypi.org/project/langdetect/)
language detector, powered by a Mojo kernel — with a vendored pure-Python
fallback that keeps the package correct everywhere else.

```python
import langdetect_mojo

langdetect_mojo.detect("War doesn't show who's right, just who's left.")
# 'en'
langdetect_mojo.detect_langs("War doesn't show who's right, just who's left.")
# [en:0.9999985859701489]
```

- **Same output, bit for bit.** For a given seed, `detect()` and
  `detect_langs()` return exactly the same strings as `langdetect` 1.0.9 with
  `DetectorFactory.seed` set — same language ranking, same probability values
  (0 ulp agreement on macOS and Linux, verified by the differential suite on
  every corpus text, on both backends).
- **Deterministic by default.** The reference is non-deterministic unless you
  set `DetectorFactory.seed`; this package seeds every call (default seed 0),
  exactly like the reference does for each fresh `Detector`.
- **55 languages**, using the reference package's own shipped profiles, read
  at runtime from the installed `langdetect` package (never re-derived).

## Install

```bash
pip install langdetect-mojo
```

Wheels are published for macOS arm64 and Linux x86_64 (with the Mojo kernel
inside). On any other platform the same wheel installs and works via the
pure-Python fallback — same results, no native code.

## Quickstart

```python
import langdetect_mojo

langdetect_mojo.detect("Bonjour le monde")
# 'fr'

langdetect_mojo.detect_langs("Bonjour le monde")
# [fr:0.9999949579736466]

langdetect_mojo.set_seed(42)   # mirrors langdetect.detector_factory.DetectorFactory.seed = 42
langdetect_mojo.detect("Bonjour le monde")
# 'fr'  (deterministic under the chosen seed)
```

Errors mirror the reference: texts with no usable features raise
`langdetect_mojo.LangDetectError` with the same message and numeric code (5)
as `langdetect.LangDetectException`:

```python
try:
    langdetect_mojo.detect("12345 !!!")
except langdetect_mojo.LangDetectError as exc:
    assert str(exc) == "No features in text."
    assert exc.code == 5
```

Check which engine is serving detection:

```python
langdetect_mojo.backend()        # 'native' or 'fallback'
langdetect_mojo.backend_info()   # resolver diagnostics (ABI handshake, source path)
```

Set `LANGDETECT_MOJO_DISABLE_NATIVE=1` to force the pure-Python fallback.

## How it works

The reference algorithm (Nakatani Shuyo's language-detection, as packaged on
PyPI): text is preprocessed (URL/e-mail masking, Vietnamese normalization,
space collapsing, non-Latin cleaning), 1–3-grams are extracted with
script-based normalization, and per-language probabilities are estimated by a
seeded sampling procedure — 7 trials of multiplicative updates with smoothing
`alpha = 0.5 + gauss()*0.05`, base frequency 10000, convergence at 0.99999.

- `kernels/langdetect/src/langdetectmojo.mojo` implements the full pipeline in
  one native call. The seeded sampler reproduces CPython's
  `random.Random(seed)` bit-for-bit (MT19937 with `init_by_array` seeding,
  res53 `random()`, `getrandbits`-based `_randbelow`, Box–Muller `gauss` with
  its cached second deviate, and CPython 3.12's Neumaier-compensated `sum()`),
  which is what makes the output bit-identical rather than merely close.
- `langdetect_mojo/_data.py` reads the profiles (`prob = freq / n_words[len-1]`
  per 1–3-gram) and normalization tables from the installed `langdetect`
  package and hands them to the kernel as a validated binary blob.
- `langdetect_mojo/_fallback.py` is a clean-room pure-Python implementation of
  the same procedure (stdlib `re` + `random`), used on platforms without a
  native wheel; it is bit-identical to the seeded oracle by construction.
- The ctypes loader resolves the bundled shared library, verifies an ABI
  version handshake, and falls back transparently if anything is off.

Supported scope: `detect(text)` and `detect_langs(text)` (module level, with
`set_seed`/`get_seed`). Out of scope, same as the reference module-level API:
custom `Detector` configuration (alpha, priors, max text length), profile
training, and non-integer seeds (the reference accepts any hashable; integer
seeds in `[-(2**64-1), 2**64-1]` are supported here, negative behaves like its
absolute value, as CPython does).

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14, Mojo
1.1.0) on 2026-09-19, median of 5, after the benchmark's correctness gate
verified bit-identical output on all 56 corpus texts for both backends
(`benchmarks/bench_langdetect.py`, 55-language corpus):

Warm steady-state `detect()` (median of 5):

| workload | PyPI langdetect | langdetect-mojo (native) | langdetect-mojo (fallback) | native speedup |
|---|---:|---:|---:|---:|
| 55-language corpus, per call | 621.8 µs | 90.6 µs | 813.4 µs | 6.9x |
| long text (12720 chars, truncated to 10000), per call | 6.894 ms | 1.609 ms | 12.290 ms | 4.3x |

Cold first-call (fresh process, median of 5):

| package | import + first detect |
|---|---:|
| PyPI langdetect (seeded) | 147.8 ms |
| langdetect-mojo (native) | 214.9 ms |
| langdetect-mojo (fallback) | 226.9 ms |

The native cold start is slower than the oracle's: profile JSON parsing plus
building and handing the 4.4 MB profile blob to the kernel happens once per
process. It is amortized over every subsequent call (~7x faster each). The
fallback exists for correctness, not speed — on platforms without a native
wheel you get bit-identical results at somewhat below oracle speed.

## Development

```bash
# compile the Mojo kernel (macOS .dylib / Linux .so)
~/.pixi/bin/pixi run bash kernels/langdetect/build.sh

# differential tests: native backend, then forced fallback
~/.pixi/bin/pixi run bash scripts/test_all_langdetect.sh

# benchmark (correctness-gated)
PYTHONPATH=python/langdetect_mojo python benchmarks/bench_langdetect.py

# build + repair the platform wheel (delocate/auditwheel self-containment)
bash python/langdetect_mojo/build_wheel.sh
```

The test oracle is the PyPI `langdetect` package itself (pinned to 1.0.9 by
the test scripts); the differential suite asserts bit-identical `detect()`
strings and `detect_langs()` probabilities against it on both backends.

---

Built by Algenta. Not affiliated with the upstream `langdetect` project;
`langdetect` remains the required source of the shipped language profiles.
