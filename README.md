# mojo-kernels

[![ci](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci.yml) [![All Contributors](https://img.shields.io/github/all-contributors/thyn-ai/mojo-kernels?color=ee8449)](#contributors)
[![ci-cclib](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-cclib.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-cclib.yml)
[![ci-fuse](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-fuse.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-fuse.yml)
[![CodeQL](https://github.com/thyn-ai/mojo-kernels/actions/workflows/codeql.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/thyn-ai/mojo-kernels/badge)](https://scorecard.dev/viewer/?uri=github.com/thyn-ai/mojo-kernels)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

**Clean-room [Mojo](https://www.modular.com/mojo) kernels as drop-in
accelerators for popular Python and TypeScript libraries.** Same API, same
results — measured **70x–8,700x** speedups with bit-exact-to-last-ulp parity
against the reference packages, asserted by differential test suites that run
on both backends. Prebuilt per-platform binaries mean **no Mojo toolchain is
ever required** on an end user's machine, and every package ships a vendored
pure-language (Python / JavaScript) fallback, so unsupported platforms —
including Windows — get silently correct behavior. Raw Mojo source lives open
in this repo; the reference libraries are used only as test and benchmark
oracles, never as runtime dependencies.

## The kernels

| package | accelerates | kernel | measured speedup | parity |
|---|---|---|---|---:|
| [`bm25-mojo`](python/bm25_mojo/README.md) | [`rank_bm25`](https://pypi.org/project/rank_bm25/) (Python) | [`kernels/bm25`](kernels/bm25) | 116x–8,769x | max abs diff 3.6e-15 |
| [`fuse-mojo`](typescript/fuse-mojo/README.md) | [Fuse.js](https://fusejs.io) fuzzy search (TypeScript/Node) | [`kernels/fuse`](kernels/fuse) | 8.8x–38.7x warm, 13x cold | bit-identical (0 diff) |
| [`cclib-mojo`](python/cclib_mojo/README.md) | [cclib](https://cclib.github.io/) electron-density grids (Python) | [`kernels/gaussgrid`](kernels/gaussgrid) | 72x–7,033x | max abs diff 1.5e-12 |

All numbers below are measured on this machine (Apple M4 Max), median of 5
runs, with a correctness gate asserted before every timing pass — reproduce
them with `pixi run bench`, `pixi run bench-fuse`, and `pixi run bench-cclib`.

## Benchmarks

### bm25-mojo — BM25 scoring (`rank_bm25` drop-in)

Measured with `benchmarks/bench_bm25.py` (run `pixi run bench` to reproduce).
Corpora are generated locally from fixed seeds: 1k / 10k / 100k documents of
30-60 tokens from a Zipf-ish 20k-term vocabulary; 20 queries per cell; median
of 5 runs. Environment: **Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14,
numpy 2.5.3, Mojo 1.1.0, rank_bm25 0.2.2 (PyPI)**, 2026-09-19.

| corpus | query terms | rank_bm25 ms/query | bm25_mojo ms/query | speedup |
|---:|---:|---:|---:|---:|
| 1,000 | 5 | 0.338 | 0.0029 | 116.1x |
| 1,000 | 10 | 0.640 | 0.0038 | 167.4x |
| 1,000 | 20 | 1.278 | 0.0041 | 315.1x |
| 10,000 | 5 | 4.038 | 0.0047 | 863.8x |
| 10,000 | 10 | 7.570 | 0.0050 | 1504.1x |
| 10,000 | 20 | 14.869 | 0.0066 | 2263.5x |
| 100,000 | 5 | 47.737 | 0.0104 | 4577.2x |
| 100,000 | 10 | 89.463 | 0.0117 | 7647.7x |
| 100,000 | 20 | 162.178 | 0.0185 | 8769.3x |

Correctness gate (asserted before every timing run, element-wise vs
`rank_bm25`): max abs diff **3.6e-15** at 100k docs — the scores are the same
numbers, not approximations.

### fuse-mojo — fuzzy search (Fuse.js drop-in)

Measured with `benchmarks/bench_fuse.mjs` (run `pixi run bench-fuse` to
reproduce; a correctness gate asserting identical `refIndex` order and scores
within 1e-9 runs before every timing pass — measured agreement here is
**exactly 0**, the scores are bit-identical, not approximations). Corpora are
generated locally from fixed seeds: 10k / 50k / 100k documents of 5-15
word-like tokens (~3000-word vocabulary, a unicode token in every 11th
document); patterns of length 3 / 8 / 16 code units drawn from corpus tokens,
half with 1-2 seeded typos; 15 patterns per cell; median of 5 runs after a
warmup round. **Cold** is the first-call experience: a fresh instance (index
build included) plus its first query, median of 5 runs × 3 patterns.
Environment: **Apple M4 Max (16 threads), macOS arm64, Node v23.10.0,
fuse.js 7.1.0 (npm), Mojo 1.1.0**, 2026-09-19.

Search — warm steady state (median of 5 runs):

| corpus | pattern len | Fuse.js ms/query | fuse-mojo ms/query | Fuse.js q/s | fuse-mojo q/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 3 | 14.813 | 1.629 | 67.5 | 613.9 | 9.1x |
| 10,000 | 8 | 34.502 | 1.584 | 29.0 | 631.2 | 21.8x |
| 10,000 | 16 | 75.916 | 3.080 | 13.2 | 324.7 | 24.6x |
| 50,000 | 3 | 72.545 | 8.270 | 13.8 | 120.9 | 8.8x |
| 50,000 | 8 | 184.715 | 5.731 | 5.4 | 174.5 | 32.2x |
| 50,000 | 16 | 414.061 | 13.077 | 2.4 | 76.5 | 31.7x |
| 100,000 | 3 | 168.540 | 18.851 | 5.9 | 53.0 | 8.9x |
| 100,000 | 8 | 419.301 | 16.232 | 2.4 | 61.6 | 25.8x |
| 100,000 | 16 | 1,239.508 | 32.003 | 0.8 | 31.2 | 38.7x |

Search — cold first call (fresh index build + first query, median of 5 runs ×
3 patterns):

| corpus | pattern len | Fuse.js cold (ms) | fuse-mojo cold (ms) | speedup |
|---:|---:|---:|---:|---:|
| 10,000 | 3 | 18.9 | 6.6 | 2.9x |
| 10,000 | 8 | 37.5 | 6.1 | 6.2x |
| 10,000 | 16 | 75.3 | 6.7 | 11.2x |
| 50,000 | 3 | 84.8 | 29.5 | 2.9x |
| 50,000 | 8 | 195.7 | 27.3 | 7.2x |
| 50,000 | 16 | 402.6 | 30.4 | 13.3x |
| 100,000 | 3 | 171.4 | 55.5 | 3.1x |
| 100,000 | 8 | 417.7 | 60.5 | 6.9x |
| 100,000 | 16 | 945.4 | 70.3 | 13.4x |

### cclib-mojo — electron-density / wavefunction grids (cclib `method.volume`)

Measured with `benchmarks/bench_gaussgrid.py` in this repository (run
`pixi run bench-cclib` to reproduce). Workload: **benzene, 6-31G\*** (12
atoms, 102 contracted Cartesian basis functions, 192 primitives — basis data
from PyQuante 1.6.5's basis library), seeded MO coefficients, median of 5
runs, single-threaded. Environment: **Apple M4 Max, macOS 26.6.2 arm64,
Python 3.12.14, numpy 2.5.3, Mojo 1.1.0**, 2026-09-19.

| workload | PyQuante1 path (s) | cclib-mojo (s) | speedup |
|---|---:|---:|---:|
| wavefunction 1 MO, 50³ | 32.23 | 0.0126 | 2567x |
| wavefunction 1 MO, 100³ | 345.37 | 0.0491 | 7033x |
| density 3 MOs, 50³ | 63.59 | 0.0335 | 1898x |
| wavefunction 1 MO, 50³ — NumPy fallback | 32.23 | 0.1691 | 191x |

Correctness gate (asserted before every timing run, element-wise vs the
PyQuante1 reference): max abs diff **1.5e-12** — the values are the same
numbers, not approximations. Against cclib's fastest shipping backend
(pyquante2, NumPy-vectorized): **72x** on the same 50³ single-MO workload
(0.689 s vs 0.0095 s, max abs diff 8.4e-13). Full methodology and the honest
two-baseline framing are in the
[package README](python/cclib_mojo/README.md).

## Install & quickstart

### bm25-mojo (Python)

```
pip install bm25-mojo
```

```python
# then use it exactly like rank_bm25
from bm25_mojo import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)                  # same call shape as rank_bm25
scores = bm25.get_scores(["windy", "in", "London"]) # np.ndarray, one score per doc
top = bm25.get_top_n(["windy", "in", "London"], corpus, n=2)
```

`BM25Okapi`, `BM25L`, and `BM25Plus` are all provided with the same
constructor arguments, index attributes, and methods as their `rank_bm25`
counterparts. Force the fallback with `BM25_MOJO_DISABLE_NATIVE=1`; inspect
the active backend with `bm25_mojo.backend_info()`. Full API parity notes:
[`python/bm25_mojo/README.md`](python/bm25_mojo/README.md).

### fuse-mojo (TypeScript / Node)

```
npm install @fuse-mojo/core
```

```js
// then use it exactly like fuse.js
import Fuse from '@fuse-mojo/core'

const books = [
  { title: "Old Man's War", author: { firstName: 'John', lastName: 'Scalzi' } },
  { title: 'The Lock Artist', author: { firstName: 'Steve', lastName: 'Hamilton' } },
]

const fuse = new Fuse(books, { keys: ['title', 'author.firstName'] })
fuse.search('lock')
// → [{ item: {...}, refIndex: 1 }]
```

CommonJS works too: `const Fuse = require('@fuse-mojo/core')`. The 90% option
surface (`keys`, `threshold`, `location`, `distance`, `minMatchCharLength`,
`includeScore`, `includeMatches`, ...) is bit-exact vs Fuse.js 7.1.0;
unsupported options throw `UnsupportedOptionError` on both backends. Force the
fallback with `FUSE_MOJO_DISABLE_NATIVE=1`; inspect with `Fuse.backendInfo()`.
Full option matrix: [`typescript/fuse-mojo/README.md`](typescript/fuse-mojo/README.md).

### cclib-mojo (Python, computational chemistry)

```
pip install cclib-mojo
```

```python
import numpy as np
from cclib_mojo import density_on_grid, wavefunction_on_grid

# Inputs exactly as cclib parses them from a logfile:
#   gbasis     -- per-atom list of (shell, [(exponent, coefficient), ...])
#   atomcoords -- (n_atoms, 3) in Angstrom
#   mocoeffs   -- (n_mo, n_bf) MO coefficients, e.g. ccdata.mocoeffs[0]
psi = wavefunction_on_grid(gbasis, atomcoords, mocoeffs[3],
                           origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                           shape=(51, 51, 51))
rho = density_on_grid(gbasis, atomcoords, mocoeffs[:nocc],
                      origin=(-5, -5, -5), step=(0.2, 0.2, 0.2),
                      shape=(51, 51, 51))
```

Also works alongside cclib (`cclib_mojo.cclib_integration` mirrors
`cclib.method.volume`'s functions — parse with cclib, evaluate with
cclib-mojo, write cube files with cclib). A runnable example (water, STO-3G)
is `quickstart.py` at the repository root. Force the fallback with
`CCLIB_MOJO_DISABLE_NATIVE=1`. Full integration guide:
[`python/cclib_mojo/README.md`](python/cclib_mojo/README.md).

## Fallback semantics (every package)

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so every package **falls back to a vendored pure-language reference
implementation**, silently and correctly:

- Resolution order: `$<NAME>_MOJO_NATIVE_LIB` → the library bundled in the
  wheel / platform package → the repo development build output.
- `<NAME>_MOJO_DISABLE_NATIVE=1` forces the fallback; each differential suite
  runs twice — once native, once forced-fallback — and asserts both against
  the real reference package.
- The wrapper checks an `<name>mojo_abi_version()` handshake before any
  native call; a mismatch falls back cleanly.
- Wheels and platform packages are **per-platform and self-contained**:
  `delocate` (macOS) / `auditwheel repair` / `patchelf` (Linux) vendor the
  Mojo runtime libraries and rewrite load paths to be package-relative.
  (Redistribution terms for Modular's runtime binaries should be confirmed
  with Modular before any public release.)

## The kernel factory

Every kernel in this repo is laid out the same way — the factory template:

```
kernels/<name>/src/<name>.mojo   # clean-room Mojo kernel, exported batch C ABI
kernels/<name>/build.sh          # mojo build --emit shared-lib → build/
python/<name>_mojo/              # Python wrapper (pyproject + hatch build hook)
python/<name>_mojo/<name>_mojo/  #   __init__ / core / _native.py / _reference.py
typescript/<name>-mojo/          # TS wrapper workspace (npm core + platform
                                 #   optionalDependencies + koffi + vendored fallback)
tests/                           # differential suite vs the reference library,
                                 #   run on both backends (native + forced fallback)
benchmarks/                      # seeded, reproducible benchmark scripts with a
                                 #   correctness gate before every timing pass
pixi.toml                        # pinned Mojo + Python toolchain, all tasks
.github/workflows/ci-*.yml       # per-kernel: build → test → wheel/pack repair
                                 #   → hermetic install smoke, ubuntu + macOS
```

Adding a new kernel means: write the kernel with the same batch-shaped ABI
(`<name>mojo_abi_version` / create / score / destroy — FFI cost per call, not
per item), copy the wrapper template (`_native.py` loader with ABI handshake +
`_reference.py` vendored fallback, or the koffi + optionalDependencies TS
equivalent), point the build hook at the new library, and add differential
tests against the real reference package plus a seeded benchmark with cold and
warm numbers. CI and packaging follow automatically.

### Build from source

```
curl -fsSL https://pixi.sh/install.sh | bash   # if you don't have pixi
pixi install
pixi run build-kernel-bm25   # → kernels/bm25/build/libbm25mojo.{dylib,so}
pixi run test                # differential suite, both backends
pixi run bench               # reproduce the bm25 numbers above
pixi run wheel-bm25          # → python/bm25_mojo/dist/*.whl (platform wheel)
```

Equivalents for the other kernels: `build-kernel-fuse / test-fuse /
bench-fuse / pack-fuse / smoke-fuse` and `build-kernel-gaussgrid /
test-cclib / bench-cclib / wheel-cclib`.

## Roadmap

This repo now covers three domains — text retrieval (bm25), fuzzy search
(Fuse.js), and computational chemistry (cclib grids) — and the factory is
built for repetition: more clean-room kernels targeting popular pure-Python
and pure-JavaScript hot loops are landing on `main`, each with the same
guarantees (differential parity, measured cold + warm benchmarks, self-
contained packages, pure-language fallback everywhere). Watch the repo or
check back here — the kernel table above grows as each one lands.

## Contributing and security

Contributions are welcome — [CONTRIBUTING.md](./CONTRIBUTING.md) maps each CI
workflow to the `pixi run` task that reproduces it locally and describes what
a kernel change needs (differential parity on both backends, clean-room only).
Security reports are never public: see [SECURITY.md](./SECURITY.md) for the
private channels and what is in scope. This project follows the
[Contributor Covenant](./CODE_OF_CONDUCT.md).


## Related repositories

Open-source repositories from the Algenta team. The Algenta engine itself is proprietary; everything listed here is Apache-2.0. Issues and discussions are welcome in whichever repository owns the code.

- [thyn-ai/algenta-sdk](https://github.com/thyn-ai/algenta-sdk) — Python & TypeScript SDKs for the Algenta decision engine: governed tool profiles, execution receipts, approvals.
- [thyn-ai/algenta-integrations](https://github.com/thyn-ai/algenta-integrations) — Framework integrations for Algenta: LangChain, LlamaIndex, pydantic-ai, MAF, Haystack, LiteLLM, Ray Serve, vLLM, Vercel AI SDK and n8n.
- [thyn-ai/mojo-kernels](https://github.com/thyn-ai/mojo-kernels) (this repository) — Clean-room Mojo kernels as drop-in accelerators for popular Python/TypeScript libraries, with bit-exact parity and pure-language fallbacks.
- [thyn-ai/security-toolchain](https://github.com/thyn-ai/security-toolchain) — The pinned, checksum-verified security toolchain (Gitleaks, Opengrep, OSV-Scanner, Trivy config, actionlint) that every thyn-ai repository runs locally and in CI.

## Contributors

Thanks go to these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome.

## License

Apache-2.0, © 2026 Algenta. All kernels in this repo are clean-room
implementations of published textbook algorithms. The reference packages
(`rank_bm25`, Fuse.js, cclib/PyQuante) are used only as test and benchmark
oracles, never as runtime dependencies; Fuse.js is additionally vendored as
fuse-mojo's fallback backend under its own Apache-2.0 license (see
[`typescript/fuse-mojo/packages/core/NOTICE`](typescript/fuse-mojo/packages/core/NOTICE)).
