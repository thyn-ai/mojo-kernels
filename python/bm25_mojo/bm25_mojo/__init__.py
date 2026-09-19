"""bm25-mojo: a drop-in faster replacement for the `rank_bm25` package.

Same classes, same call shapes, same scores — powered by a Mojo kernel where
the platform supports it (macOS arm64, Linux x86_64), with a vendored
pure-Python fallback everywhere else (including Windows).

    from bm25_mojo import BM25Okapi

    bm25 = BM25Okapi(tokenized_corpus)          # same as rank_bm25.BM25Okapi
    scores = bm25.get_scores(tokenized_query)   # np.ndarray of float64
    top = bm25.get_top_n(tokenized_query, documents, n=5)

Set BM25_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from bm25_mojo._native import backend_info, native_available
from bm25_mojo.core import BM25, BM25L, BM25Okapi, BM25Plus

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "BM25Okapi",
    "BM25L",
    "BM25Plus",
    "backend_info",
    "native_available",
    "__version__",
]
