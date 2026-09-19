# bm25-mojo

A drop-in faster replacement for the [`rank_bm25`](https://pypi.org/project/rank_bm25/)
package, powered by a clean-room Mojo kernel — with a vendored pure-Python
fallback for platforms without a native build (including Windows).

```python
from bm25_mojo import BM25Okapi  # same API as rank_bm25.BM25Okapi

bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)        # np.ndarray, float64
top = bm25.get_top_n(tokenized_query, documents, n=5)
```

- **Same results**: scores match `rank_bm25` element-wise (last-ulp level;
  the differential suite asserts agreement within 1e-8 on both the native and
  fallback backends). `BM25Okapi`, `BM25L`, and `BM25Plus` are provided.
- **Much faster scoring**: postings lists + SIMD instead of a Python dict
  lookup for every (query term × document) pair — 100x-8000x faster
  `get_scores` on 1k-100k document corpora (Apple M4 Max; full method and
  numbers in the repository README).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `BM25_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `bm25_mojo.backend_info()`.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta, Inc.
