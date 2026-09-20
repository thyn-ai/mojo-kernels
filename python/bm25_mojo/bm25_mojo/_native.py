"""ctypes loader for the bm25mojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$BM25_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``bm25_mojo/_native/``
    3. the repository development build output ``kernels/bm25/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``BM25_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  bm25mojo_abi_version(void)
    void*    bm25mojo_index_create(int64_t n_docs, const double* doc_len,
                                   double avgdl, double k1, double b,
                                   double delta, int32_t variant,
                                   int64_t n_terms, const int64_t* offsets,
                                   const int32_t* docs, const double* freqs,
                                   const double* idf)
    int32_t  bm25mojo_score(void* handle, const int32_t* qids,
                            int64_t n_query, double* out_scores)
    void     bm25mojo_index_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/bm25/src/bm25mojo.mojo. A mismatch means
# the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 2

# Score-formula variants, mirrored from the kernel.
VARIANT_OKAPI = 0
VARIANT_L = 1
VARIANT_PLUS = 2

_ENV_LIB = "BM25_MOJO_NATIVE_LIB"
_ENV_DISABLE = "BM25_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native bm25mojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libbm25mojo.dylib"
    if sys.platform.startswith("linux"):
        return "libbm25mojo.so"
    if sys.platform.startswith("win"):
        return "bm25mojo.dll"  # no Mojo toolchain builds this today
    return "libbm25mojo.so"


def _candidate_paths() -> list[tuple[str, str]]:
    """(source_label, path) candidates, in resolver order."""
    out: list[tuple[str, str]] = []
    override = os.environ.get(_ENV_LIB)
    if override:
        out.append((f"env {_ENV_LIB}", override))
    here = os.path.dirname(__file__)
    out.append(("bundled in wheel", os.path.join(here, "_native", _lib_basename())))
    out.append(
        (
            "repo-dev build output",
            os.path.abspath(
                os.path.join(here, "..", "..", "..", "kernels", "bm25", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.bm25mojo_abi_version.argtypes = []
    lib.bm25mojo_abi_version.restype = ctypes.c_int32
    # ABI v2: the kernel builds the CSR postings and bakes idf-weighted
    # scores from doc-major term-frequency streams.
    lib.bm25mojo_index_create.argtypes = [
        ctypes.c_int64,  # n_docs
        f64p,  # doc_len[n_docs]
        ctypes.c_double,  # avgdl
        ctypes.c_double,  # k1
        ctypes.c_double,  # b
        ctypes.c_double,  # delta
        ctypes.c_int32,  # variant
        ctypes.c_int64,  # n_terms
        i64p,  # doc_offsets[n_docs+1]
        i32p,  # doc_tids[ntok]
        f64p,  # doc_freqs[ntok]
        f64p,  # idf[n_terms]
    ]
    lib.bm25mojo_index_create.restype = ctypes.c_void_p
    lib.bm25mojo_score.argtypes = [ctypes.c_void_p, i32p, ctypes.c_int64, f64p]
    lib.bm25mojo_score.restype = ctypes.c_int32
    lib.bm25mojo_score_batch.argtypes = [
        ctypes.c_void_p,
        i32p,
        i64p,
        ctypes.c_int64,
        f64p,
    ]
    lib.bm25mojo_score_batch.restype = ctypes.c_int32
    lib.bm25mojo_index_destroy.argtypes = [ctypes.c_void_p]
    lib.bm25mojo_index_destroy.restype = None


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_SOURCE, _LOAD_ERROR
    if os.environ.get(_ENV_DISABLE) == "1":
        raise NativeUnavailable(f"native kernel disabled by {_ENV_DISABLE}=1")
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        errors: list[str] = []
        for label, path in _candidate_paths():
            try:
                if not path or not os.path.exists(path):
                    continue
                try:
                    lib = ctypes.CDLL(path)
                except OSError as exc:
                    errors.append(f"{label} ({path}): {exc}")
                    continue
                try:
                    _bind_abi(lib)
                    abi = int(lib.bm25mojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                _LIB, _LIB_SOURCE = lib, f"{label} ({path})"
                _LOAD_ERROR = None
                return lib
            except OSError as exc:
                errors.append(f"{label}: {exc}")
        _LOAD_ERROR = "; ".join(errors) or "no native kernel found on any resolver path"
        raise NativeUnavailable(_LOAD_ERROR)


def native_available() -> bool:
    """True if the native kernel can score right now. Never raises."""
    try:
        _load()
        return True
    except NativeUnavailable:
        return False


def backend_info() -> dict:
    """Diagnostics for the active backend. Never raises."""
    info = {
        "native_available": False,
        "native_source": None,
        "abi_version_expected": ABI_VERSION,
        "abi_version_native": None,
        "disabled_by_env": os.environ.get(_ENV_DISABLE) == "1",
        "platform": sys.platform,
        "error": None,
    }
    try:
        lib = _load()
    except NativeUnavailable as exc:
        info["error"] = str(exc)
        return info
    info["native_available"] = True
    info["native_source"] = _LIB_SOURCE
    info["abi_version_native"] = int(lib.bm25mojo_abi_version())
    return info


class NativeIndex:
    """Owned handle to a native corpus index. Not thread-safe to close twice."""

    def __init__(
        self,
        doc_len: np.ndarray,
        avgdl: float,
        k1: float,
        b: float,
        delta: float,
        variant: int,
        doc_offsets: np.ndarray,
        doc_tids: np.ndarray,
        doc_freqs: np.ndarray,
        idf: np.ndarray,
    ) -> None:
        lib = _load()  # raises NativeUnavailable
        f64p = ctypes.POINTER(ctypes.c_double)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        doc_len = np.ascontiguousarray(doc_len, dtype=np.float64)
        doc_offsets = np.ascontiguousarray(doc_offsets, dtype=np.int64)
        doc_tids = np.ascontiguousarray(doc_tids, dtype=np.int32)
        doc_freqs = np.ascontiguousarray(doc_freqs, dtype=np.float64)
        idf = np.ascontiguousarray(idf, dtype=np.float64)
        n_docs = np.int64(doc_len.shape[0])
        n_terms = np.int64(idf.shape[0])
        handle = lib.bm25mojo_index_create(
            n_docs,
            doc_len.ctypes.data_as(f64p),
            ctypes.c_double(avgdl),
            ctypes.c_double(k1),
            ctypes.c_double(b),
            ctypes.c_double(delta),
            ctypes.c_int32(variant),
            n_terms,
            doc_offsets.ctypes.data_as(i64p),
            doc_tids.ctypes.data_as(i32p),
            doc_freqs.ctypes.data_as(f64p),
            idf.ctypes.data_as(f64p),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the index (invalid sizes, streams, or "
                "variant); falling back to the pure-Python reference"
            )
        # The kernel copies every buffer; these arrays may be freed by the GC.
        self._lib = lib
        self._handle = handle
        self._n_docs = int(n_docs)
        # Hot-path caches: avoid per-call ctypes constructor overhead.
        self._score_fn = lib.bm25mojo_score
        self._batch_fn = lib.bm25mojo_score_batch
        self._f64p = ctypes.POINTER(ctypes.c_double)
        self._i32p = ctypes.POINTER(ctypes.c_int32)
        self._i64p = ctypes.POINTER(ctypes.c_int64)

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def score(self, query_term_ids: np.ndarray) -> np.ndarray:
        """Score one token-id query against every document. Returns float64[n_docs].

        The output buffer comes from np.zeros (calloc): the kernel accumulates
        without zeroing, so per-query cost is O(postings), not O(n_docs).
        """
        if self._handle is None:
            raise NativeUnavailable("native index is closed")
        if query_term_ids.dtype != np.int32 or not query_term_ids.flags.c_contiguous:
            query_term_ids = np.ascontiguousarray(query_term_ids, dtype=np.int32)
        out = np.zeros(self._n_docs, dtype=np.float64)
        rc = self._score_fn(
            self._handle,
            query_term_ids.ctypes.data_as(self._i32p),
            len(query_term_ids),
            out.ctypes.data_as(self._f64p),
        )
        if rc != 0:
            raise NativeUnavailable(f"native scoring failed with status {rc}")
        return out

    def score_batch(self, query_term_ids_list: list[np.ndarray]) -> np.ndarray:
        """Score a batch of token-id queries in one FFI call.

        Returns a float64 (n_queries, n_docs) panel whose row i is
        bit-identical to `score(query_term_ids_list[i])`.
        """
        if self._handle is None:
            raise NativeUnavailable("native index is closed")
        n_queries = len(query_term_ids_list)
        if n_queries == 0:
            return np.zeros((0, self._n_docs), dtype=np.float64)
        counts = np.fromiter((len(q) for q in query_term_ids_list), dtype=np.int64,
                             count=n_queries)
        qids_flat = np.concatenate(
            [np.ascontiguousarray(q, dtype=np.int32) for q in query_term_ids_list]
        )
        offsets = np.zeros(n_queries + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        panel = np.zeros((n_queries, self._n_docs), dtype=np.float64)
        rc = self._batch_fn(
            self._handle,
            qids_flat.ctypes.data_as(self._i32p),
            offsets.ctypes.data_as(self._i64p),
            n_queries,
            panel.ctypes.data_as(self._f64p),
        )
        if rc != 0:
            raise NativeUnavailable(f"native batch scoring failed with status {rc}")
        return panel

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.bm25mojo_index_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
