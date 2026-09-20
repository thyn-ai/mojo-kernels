# bm25-mojo

A drop-in faster replacement for the [`rank_bm25`](https://pypi.org/project/rank_bm25/)
package, powered by a clean-room Mojo kernel — with a vendored pure-Python
fallback for platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
from bm25_mojo import BM25Okapi  # same API as rank_bm25.BM25Okapi

bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)        # np.ndarray, float64
top = bm25.get_top_n(tokenized_query, documents, n=5)
panel = bm25.get_scores_batch(list_of_queries)   # batch API: bit-identical rows
```

- **Same results**: scores match `rank_bm25` element-wise (last-ulp level;
  the differential suite asserts agreement within 1e-8 on both the native and
  fallback backends). `BM25Okapi`, `BM25L`, and `BM25Plus` are provided.
- **Much faster scoring**: the kernel walks postings lists with the
  idf-weighted scores baked in at index time — 149x-13,137x faster
  `get_scores` than rank_bm25, and **1.1x-1.95x faster than bm25s 0.3.11
  (numba, its recommended config) across the whole 1k-100k doc grid** (Apple
  M4 Max; full method and numbers in the repository README and
  `benchmarks/AUTOPSY-bm25s.md`).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `BM25_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `bm25_mojo.backend_info()`.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
