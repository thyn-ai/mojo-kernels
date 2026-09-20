# mojo-kernels

[![ci](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci.yml) [![All Contributors](https://img.shields.io/github/all-contributors/thyn-ai/mojo-kernels?color=ee8449)](#contributors)
[![ci-cclib](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-cclib.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-cclib.yml)
[![ci-fuse](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-fuse.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/ci-fuse.yml)
[![CodeQL](https://github.com/thyn-ai/mojo-kernels/actions/workflows/codeql.yml/badge.svg)](https://github.com/thyn-ai/mojo-kernels/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/thyn-ai/mojo-kernels/badge)](https://scorecard.dev/viewer/?uri=github.com/thyn-ai/mojo-kernels)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

**Clean-room [Mojo](https://www.modular.com/mojo) kernels as drop-in
accelerators for popular Python and TypeScript libraries.** Same API, same
results — measured **70x–13,137x** speedups with bit-exact-to-last-ulp parity
against the reference packages, asserted by differential test suites that run
on both backends. Prebuilt per-platform binaries mean **no Mojo toolchain is
ever required** on an end user's machine, and every package ships a vendored
pure-language (Python / JavaScript) fallback, so unsupported platforms get
silently correct behavior. Windows has no Mojo toolchain, so the fallback is
what runs there; CI runs it on Windows for every package but one
([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)). Raw Mojo source lives open
in this repo; the reference libraries are used only as test and benchmark
oracles, never as runtime dependencies.

Custom Mojo kernels give Algenta its speed. The same team publishes these kernels in the
open, under Apache-2.0: the same API and the same results as the libraries they
accelerate, prebuilt binaries so no Mojo toolchain is ever required, and a pure-language
fallback everywhere. The engine's own kernels are proprietary and are not in this
repository.

## The kernels

| package | accelerates | kernel | measured speedup | parity |
|---|---|---|---|---:|
| [`bm25-mojo`](python/bm25_mojo/README.md) | [`rank_bm25`](https://pypi.org/project/rank_bm25/) (Python) | [`kernels/bm25`](kernels/bm25) | 149x–13,137x vs rank_bm25; 1.10x–1.95x vs bm25s | max abs diff 3.6e-15 |
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
numpy 2.5.3, Mojo 1.1.0, rank_bm25 0.2.2 (PyPI)**, 2026-09-20.

| corpus | query terms | rank_bm25 ms/query | bm25_mojo ms/query | speedup |
|---:|---:|---:|---:|---:|
| 1,000 | 5 | 0.292 | 0.0020 | 148.7x |
| 1,000 | 10 | 0.566 | 0.0027 | 209.9x |
| 1,000 | 20 | 1.069 | 0.0029 | 375.1x |
| 10,000 | 5 | 2.853 | 0.0037 | 780.6x |
| 10,000 | 10 | 5.447 | 0.0039 | 1396.8x |
| 10,000 | 20 | 11.201 | 0.0047 | 2389.5x |
| 100,000 | 5 | 45.313 | 0.0092 | 4942.1x |
| 100,000 | 10 | 87.818 | 0.0125 | 7024.2x |
| 100,000 | 20 | 159.890 | 0.0122 | 13137.2x |

Correctness gate (asserted before every timing run, element-wise vs
`rank_bm25`): max abs diff **3.6e-15** at 100k docs — the scores are the same
numbers, not approximations.

Index build is one-time (v2: CSR postings are built inside the kernel and the
idf-weighted scores are baked there, so the Python side only scans the corpus
— the same scan rank_bm25 does):

| corpus | rank_bm25 build (s) | bm25_mojo build (s) |
|---:|---:|---:|
| 1,000 | 0.006 | 0.011 |
| 10,000 | 0.047 | 0.075 |
| 100,000 | 0.423 | 0.688 |

#### vs bm25s (the numba incumbent) — bm25-mojo wins every cell

[`bm25s`](https://github.com/xhluca/bm25s) is the established fast BM25
implementation (numba-JIT scorer over a precomputed sparse score matrix). It
is **not** a `rank_bm25` drop-in — different API, and its idf flooring
differs from `rank_bm25`'s by design — so scores are not element-wise
comparable; this is a workload-identical speed comparison on the same harness
(`benchmarks/bench_bm25s.py`, same seeds/corpora/queries as above; median of
5 runs, 20 queries per cell; correctness gate vs rank_bm25 asserted before
timing). Same machine and environment as above, plus **bm25s 0.3.11, numba
0.67.0**, bm25s timed in its recommended configuration (`method='lucene'`,
float32, `compile()` numba scorer). Sanity check: mean top-10 ranking overlap
between bm25s and rank_bm25 is **1.00** (10 queries, 10k docs). bm25_mojo is
the faster of its single-query loop and its `get_scores_batch` batch API per
cell; the full per-variant grid is in the bench script output.

Warm steady-state (ms/query, lower is better):

| corpus | query terms | bm25s (numba, best variant) | bm25_mojo (best path) | speedup |
|---:|---:|---:|---:|---:|
| 1,000 | 5 | 0.0023 | 0.0014 | **1.71x** |
| 1,000 | 10 | 0.0026 | 0.0016 | **1.61x** |
| 1,000 | 20 | 0.0031 | 0.0020 | **1.60x** |
| 10,000 | 5 | 0.0032 | 0.0017 | **1.95x** |
| 10,000 | 10 | 0.0035 | 0.0019 | **1.84x** |
| 10,000 | 20 | 0.0048 | 0.0029 | **1.66x** |
| 100,000 | 5 | 0.0079 | 0.0055 | **1.43x** |
| 100,000 | 10 | 0.0090 | 0.0068 | **1.33x** |
| 100,000 | 20 | 0.0115 | 0.0105 | **1.10x** |

An earlier revision of this table (v1 kernel) honestly reported bm25s winning
most cells (0.45x-1.02x). The counterattack is documented layer by layer in
[`benchmarks/AUTOPSY-bm25s.md`](benchmarks/AUTOPSY-bm25s.md): baking the
idf-weighted scores into the CSR at index time (the bm25s trick, in float64
with bit-exact op order), eliminating per-query buffer zeroing, query
batching with malloc-friendly panel sizes, and hot-path slimming — all
verified against the 1e-8 parity gate after each step (measured agreement
stayed 0-3.6e-15). float32 value storage was measured and **rejected** (9.4e-7
max error, 94x over the parity gate); threading was not needed — all cells
win single-threaded.

The remaining honest differences:

- **Cold start**: bm25s pays a one-time numba JIT cost (**1.1-4.1 s** for
  float32, ~0.2 s float64, measured once per process) plus lazy-binding cost
  on some freshly built indexes; bm25-mojo's kernel is AOT-compiled, first
  query on a fresh index is **0.003-0.015 ms** (batch) — no JIT, no warmup.
- **Index build (one-time)**: at 100k docs bm25s took **1.58 s** vs
  bm25-mojo's **0.81 s** and rank_bm25's **0.47 s** (1k: 0.016 / 0.010 /
  0.006 s; 10k: 0.142 / 0.085 / 0.053 s, respectively — same run as the bm25s
  table above).
- **Semantics**: bm25-mojo is bit-faithful to `rank_bm25` (max abs diff
  3.6e-15 above); bm25s implements its own well-documented variants.
- **Dependencies**: bm25s needs numba/llvmlite for its fast path; bm25-mojo
  is a self-contained per-platform wheel with a pure-Python fallback.

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

# and a batch addition rank_bm25 doesn't have: one FFI call, many queries
panel = bm25.get_scores_batch([["windy", "London"], ["weather", "today"]])
# panel[i] is bit-identical to bm25.get_scores(queries[i])
```

`BM25Okapi`, `BM25L`, and `BM25Plus` are all provided with the same
constructor arguments, index attributes, and methods as their `rank_bm25`
counterparts, plus `get_scores_batch` for batched scoring. Force the fallback
with `BM25_MOJO_DISABLE_NATIVE=1`; inspect
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

### Verifying a download

Every asset on a [GitHub Release](https://github.com/thyn-ai/mojo-kernels/releases)
ships with a keyless Sigstore signature bundle (`<asset>.sigstore.json`) and
is covered by SLSA build provenance (`multiple.intoto.jsonl`), both produced
by the release workflow itself. [`RELEASING.md`](RELEASING.md#verify-a-release)
has the `cosign verify-blob` and `slsa-verifier` commands.

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
- Windows: no Mojo toolchain exists, so the pure fallback is what runs
  there. [`windows-fallback.yml`](.github/workflows/windows-fallback.yml)
  builds every Python package's `py3-none-any` wheel with its
  `<NAME>_MOJO_ALLOW_PURE_WHEEL=1` escape hatch, installs it into a fresh
  venv beside the pinned oracle and runs the forced-fallback suite on
  `windows-latest`; every TypeScript package runs its `test:fallback`
  script there. The one exception is `dynesty-mojo`: its suite's sanity check on the
  *oracle* does not hold on Windows, so it makes no Windows claim (its
  README says why).
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
fuzz/                            # differential fuzz harness (atheris + seed-corpus
                                 #   replay) comparing kernel, fallback and reference
benchmarks/                      # seeded, reproducible benchmark scripts with a
                                 #   correctness gate before every timing pass
pixi.toml                        # pinned Mojo + Python toolchain, all tasks
.github/workflows/ci-*.yml       # per-kernel: build → test → wheel/pack repair
                                 #   → hermetic install smoke, ubuntu + macOS
.github/workflows/fuzz.yml       # bounded fuzzing per PR, longer nightly
```

Adding a new kernel means: write the kernel with the same batch-shaped ABI
(`<name>mojo_abi_version` / create / score / destroy — FFI cost per call, not
per item), copy the wrapper template (`_native.py` loader with ABI handshake +
`_reference.py` vendored fallback, or the koffi + optionalDependencies TS
equivalent), point the build hook at the new library, and add differential
tests against the real reference package, a differential fuzz harness (see
[`fuzz/README.md`](./fuzz/README.md)) and a seeded benchmark with cold and
warm numbers. CI and packaging follow automatically.

### Build from source

```
curl -fsSL https://pixi.sh/install.sh | bash   # if you don't have pixi
pixi install
pixi run build-kernel-bm25   # → kernels/bm25/build/libbm25mojo.{dylib,so}
pixi run test                # differential suite, both backends
pixi run fuzz-regression     # replay the fuzzing seed corpora (fuzz/README.md)
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

Open-source tooling around Algenta, from the Algenta team. The Algenta engine itself is proprietary; everything listed here is Apache-2.0. Issues and discussions are welcome in whichever repository owns the code.

- [thyn-ai/algenta-sdk](https://github.com/thyn-ai/algenta-sdk) — Python and TypeScript SDKs for Algenta: governed data queries, simulations, decision memory with execution receipts, agent runs with approvals.
- [thyn-ai/algenta-integrations](https://github.com/thyn-ai/algenta-integrations) — Framework integrations for Algenta: LangChain, LlamaIndex, pydantic-ai, MAF, Haystack, LiteLLM, Ray Serve, vLLM, Vercel AI SDK and n8n.
- [thyn-ai/mojo-kernels](https://github.com/thyn-ai/mojo-kernels) (this repository) — Clean-room Mojo kernels as drop-in accelerators for popular Python/TypeScript libraries, with bit-exact parity and pure-language fallbacks.
- [thyn-ai/security-toolchain](https://github.com/thyn-ai/security-toolchain) — The pinned, checksum-verified security toolchain (Gitleaks, Opengrep, OSV-Scanner, Trivy config, actionlint) that every thyn-ai repository runs locally and in CI.
- [thyn-ai/feedback](https://github.com/thyn-ai/feedback) — Public issue intake for the open-source tooling around Algenta and for the Codna GitHub App.
- [thyn-ai/codna-action](https://github.com/thyn-ai/codna-action) — GitHub Action for Codna: fix, review or secure a repository in CI through the same packaged local runtime the CLI uses.

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
