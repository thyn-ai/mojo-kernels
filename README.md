# mojo-kernels

**Mojo kernels as drop-in accelerators for popular Python libraries** — prebuilt
per-platform binaries, a thin `ctypes` wrapper, and a vendored pure-Python
fallback. Users `pip install` a faster library and never see a Mojo toolchain.
Raw Mojo source lives open in this repo.

First flagship kernel: **`bm25-mojo`**, a drop-in faster replacement for the
[`rank_bm25`](https://pypi.org/project/rank_bm25/) package.

```python
# pip install bm25-mojo   →   then use it exactly like rank_bm25
from bm25_mojo import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)                 # same call shape as rank_bm25
scores = bm25.get_scores(["windy", "in", "London"]) # np.ndarray, one score per doc
top = bm25.get_top_n(["windy", "in", "London"], corpus, n=2)
```

`BM25Okapi`, `BM25L`, and `BM25Plus` are all provided with the same constructor
arguments, index attributes (`corpus_size`, `avgdl`, `doc_freqs`, `idf`,
`doc_len`, `average_idf`, ...), and methods (`get_scores`, `get_batch_scores`,
`get_top_n`) as their `rank_bm25` counterparts. Like `rank_bm25`, the
tokenizer defaults to none and **no lowercasing** is applied.

## Benchmark

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

Index build is one-time; ours is currently ~3x slower than `rank_bm25`'s
(postings construction in Python — a candidate for a future kernel), and pays
for itself after a handful of queries:

| corpus | rank_bm25 build (s) | bm25_mojo build (s) |
|---:|---:|---:|
| 1,000 | 0.006 | 0.016 |
| 10,000 | 0.045 | 0.123 |
| 100,000 | 0.499 | 1.540 |

Why the gap is so large: `rank_bm25.get_scores` walks a Python dict
(`doc.get(q)`) for **every** (query term × document) pair — O(|Q|·N) of
interpreted work. `bm25-mojo` walks only each query term's postings list in a
compiled Mojo kernel with SIMD arithmetic — work proportional to the number of
matching documents, not the corpus.

## How it works

```
pip install bm25-mojo
        │
        ▼
bm25_mojo (thin Python wrapper, ~500 lines)
        │  builds vocab, idf, doc lengths, CSR postings — once, in Python
        ▼
libbm25mojo.dylib / .so          (Mojo kernel, C ABI v1)
        │  bm25mojo_index_create / bm25mojo_score / bm25mojo_index_destroy
        ▼
postings lists + SIMD float64 scoring, one lane per posting
```

- **Batch-shaped C ABI**: the index is built once (`bm25mojo_index_create`,
  CSR postings copied into native memory), then each `get_scores` call passes
  one whole token-id query and receives all N document scores. FFI overhead
  is per-call, not per-document.
- **Bit-matching arithmetic**: idf values are computed in Python with
  `math.log` exactly as the reference does, and the kernel evaluates the
  per-term contribution in the same IEEE-754 float64 operation order as the
  reference's NumPy expressions, one SIMD lane per posting. The differential
  suite asserts element-wise agreement within 1e-8; measured agreement is at
  the last-ulp level.
- **ABI handshake**: the wrapper checks `bm25mojo_abi_version()` against its
  own expected version before scoring; a mismatch falls back cleanly.
- **BM25Plus delta floor**: the reference gives every document a constant
  `idf·delta` per query term (even documents without the term); the kernel
  applies that floor with a dense SIMD pass and adds only the excess for
  posted documents.

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always go
missing — so the wrapper **falls back to a vendored pure-Python reference
implementation** (`bm25_mojo/_reference.py`, clean-room, NumPy-only):

- Resolution order: `$BM25_MOJO_NATIVE_LIB` → the library bundled in the
  wheel → the repo development build output.
- `BM25_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs this
  way as its second pass).
- Both backends share index construction and idf calculation in
  `bm25_mojo/core.py`, so they cannot disagree about the index; the
  differential suite asserts both against `rank_bm25`.
- Inspect what's active: `bm25_mojo.backend_info()` and
  `bm25_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_2_35_x86_64`) and **wheel-only** — no sdist, because a
  source tarball cannot rebuild the native library. A pure `py3-none-any`
  fallback wheel can be produced with `BM25_MOJO_ALLOW_PURE_WHEEL=1` (e.g. for
  Windows).

## API parity notes

- Tokenizer path: `rank_bm25` maps the tokenizer with a
  `multiprocessing.Pool`; `bm25-mojo` uses a sequential map, which returns
  element-for-element identical results without spawning processes.
- `rank_bm25` does not lowercase tokens; neither does `bm25-mojo`.
- **BM25L follows the published PyPI `rank_bm25` 0.2.2** (the oracle is pinned
  in `pixi.toml`). Upstream's GitHub master changed the BM25L formula after
  0.2.2 (dropping a leading `q_freq` factor); if a future PyPI release adopts
  it, `BM25L` will be revisited. `BM25Okapi` and `BM25Plus` are identical in
  both.

## Differential tests

```
pixi run test    # builds the kernel, then runs the suite twice:
                 # once native, once with BM25_MOJO_DISABLE_NATIVE=1
```

`tests/` compares `bm25_mojo` against `rank_bm25` (pinned `==0.2.2`) on
deterministic seeded corpora: medium and 10k-document corpora, empty
documents, unseen query terms, repeated terms, single-document corpora, custom
`k1`/`b`/`epsilon`/`delta` parameters, the tokenizer path, `get_batch_scores`
(including negative indices), and `get_top_n` ordering including exact ties —
for all three variants, on both backends. 34 tests per backend pass.

## Repository layout (the kernel factory)

```
kernels/<name>/src/<name>.mojo   # clean-room Mojo kernel, exported C ABI
kernels/<name>/build.sh          # mojo build --emit shared-lib → build/
python/<name>_mojo/              # wrapper package (pyproject + hatch hook)
python/<name>_mojo/<name>_mojo/  #   __init__ / core / _native.py / _reference.py
tests/                           # differential suite vs the reference library
benchmarks/                      # seeded, reproducible benchmark scripts
pixi.toml                        # pinned Mojo + Python toolchain, all tasks
.github/workflows/ci.yml         # ubuntu + macOS: build → test → wheel → smoke (+ advisory auditwheel)
```

Adding a new kernel means: write the kernel with the same ABI shape
(`<name>mojo_abi_version` / create / score / destroy), copy the wrapper
template (`_native.py` loader + `_reference.py` fallback), point the hatch
hook at the new library, add differential tests against the reference library
and a seeded benchmark. CI and packaging follow automatically.

### Build from source

```
curl -fsSL https://pixi.sh/install.sh | bash   # if you don't have pixi
pixi install
pixi run build-kernel-bm25   # → kernels/bm25/build/libbm25mojo.{dylib,so}
pixi run test                # differential suite, both backends
pixi run bench               # reproduce the numbers above
pixi run wheel-bm25          # → python/bm25_mojo/dist/*.whl (platform wheel)
```

## License

Apache-2.0, © 2026 Algenta, Inc. All kernels in this repo are clean-room
implementations of published textbook algorithms. `rank_bm25` itself is used
only as the test/benchmark oracle, never as a runtime dependency.
