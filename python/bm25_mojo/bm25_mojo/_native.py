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

Stable C ABI::

    int32_t  bm25mojo_abi_version(void)              -> 4
    void*    bm25mojo_build_index_corpus(...)
    void     bm25mojo_build_result_free(result)
    void*    bm25mojo_index_create(...)
    int32_t  bm25mojo_vocab_attach(handle, ...)
    int32_t  bm25mojo_score(handle, ...)
    int32_t  bm25mojo_score_batch(handle, ...)
    int32_t  bm25mojo_score_batch_str(handle, ...)
    void     bm25mojo_index_destroy(handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/bm25/src/bm25mojo.mojo. A mismatch means
# the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 4
# Older libraries back to ABI v3 are still usable for scoring and for the
# Python-side build path, so the handshake accepts [ABI_VERSION_MIN, ABI_VERSION].
ABI_VERSION_MIN = 3

# Score-formula variants, mirrored from the kernel.
VARIANT_OKAPI = 0
VARIANT_L = 1
VARIANT_PLUS = 2

_ENV_LIB = "BM25_MOJO_NATIVE_LIB"
_ENV_DISABLE = "BM25_MOJO_DISABLE_NATIVE"

# Widest fixed-width token slot the kernel accepts (MAX_TOKEN_WIDTH there).
MAX_TOKEN_WIDTH = 4096


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native bm25mojo kernel could not be found, loaded, or verified."""


class UnsafeTokenError(ValueError):
    """A token the native string path cannot take (non-`str`, embedded NUL,
    or over-wide slot); the caller falls back to id-mapping for that batch."""


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


def _bind_abi(lib: ctypes.CDLL, abi: int) -> None:
    # Every pointer argument is declared c_void_p and fed a plain integer
    # address (numpy's `arr.ctypes.data`) at call time: ctypes converts a
    # Python int to void* without constructing any per-call pointer object,
    # the cheapest marshal path there is. The C side sees the same ABI.
    vp = ctypes.c_void_p
    lib.bm25mojo_abi_version.argtypes = []
    lib.bm25mojo_abi_version.restype = ctypes.c_int32
    # ABI v2+: the kernel builds the CSR postings and bakes idf-weighted
    # scores from doc-major term-frequency streams.
    lib.bm25mojo_index_create.argtypes = [
        ctypes.c_int64,  # n_docs
        vp,  # doc_len[n_docs]
        ctypes.c_double,  # avgdl
        ctypes.c_double,  # k1
        ctypes.c_double,  # b
        ctypes.c_double,  # delta
        ctypes.c_int32,  # variant
        ctypes.c_int64,  # n_terms
        vp,  # doc_offsets[n_docs+1]
        vp,  # doc_tids[ntok]
        vp,  # doc_freqs[ntok]
        vp,  # idf[n_terms]
    ]
    lib.bm25mojo_index_create.restype = ctypes.c_void_p
    lib.bm25mojo_vocab_attach.argtypes = [vp, vp, ctypes.c_int64, ctypes.c_int64]
    lib.bm25mojo_vocab_attach.restype = ctypes.c_int32
    lib.bm25mojo_score.argtypes = [vp, vp, ctypes.c_int64, vp]
    lib.bm25mojo_score.restype = ctypes.c_int32
    lib.bm25mojo_score_batch.argtypes = [vp, vp, vp, ctypes.c_int64, vp]
    lib.bm25mojo_score_batch.restype = ctypes.c_int32
    lib.bm25mojo_score_batch_str.argtypes = [
        vp,
        vp,
        ctypes.c_int64,
        vp,
        ctypes.c_int64,
        vp,
    ]
    lib.bm25mojo_score_batch_str.restype = ctypes.c_int32
    lib.bm25mojo_index_destroy.argtypes = [vp]
    lib.bm25mojo_index_destroy.restype = None
    if abi >= 4:
        lib.bm25mojo_build_index_corpus.argtypes = [
            ctypes.c_int64,  # n_docs
            vp,  # tokens[total_bytes]
            vp,  # doc_offsets[n_docs + 1]
            ctypes.c_int32,  # variant
            ctypes.c_double,  # k1
            ctypes.c_double,  # b
            ctypes.c_double,  # delta
            ctypes.c_double,  # epsilon
        ]
        lib.bm25mojo_build_index_corpus.restype = ctypes.c_void_p
        lib.bm25mojo_build_index_tids.argtypes = [
            ctypes.c_int64,  # n_docs
            vp,  # token_tids[total_tokens]
            vp,  # doc_offsets[n_docs + 1]
            ctypes.c_int32,  # variant
            ctypes.c_double,  # k1
            ctypes.c_double,  # b
            ctypes.c_double,  # delta
            ctypes.c_double,  # epsilon
        ]
        lib.bm25mojo_build_index_tids.restype = ctypes.c_void_p
        lib.bm25mojo_build_result_free.argtypes = [vp]
        lib.bm25mojo_build_result_free.restype = None


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_ABI: int | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_ABI, _LIB_SOURCE, _LOAD_ERROR
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
                    lib.bm25mojo_abi_version.argtypes = []
                    lib.bm25mojo_abi_version.restype = ctypes.c_int32
                    abi = int(lib.bm25mojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi < ABI_VERSION_MIN or abi > ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} not in supported range "
                        f"v{ABI_VERSION_MIN}..v{ABI_VERSION}"
                    )
                    continue
                try:
                    _bind_abi(lib, abi)
                except Exception as exc:
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                _LIB, _LIB_ABI, _LIB_SOURCE = lib, abi, f"{label} ({path})"
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


def native_build_available() -> bool:
    """True if the loaded native library supports the ABI v4 corpus builder."""
    try:
        _load()
    except NativeUnavailable:
        return False
    return _LIB_ABI is not None and _LIB_ABI >= 4


class _BuildResultC(ctypes.Structure):
    """C layout of the Mojo BuildResult struct."""

    _fields_ = [
        ("status", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("n_docs", ctypes.c_int64),
        ("n_terms", ctypes.c_int64),
        ("nnz", ctypes.c_int64),
        ("avgdl", ctypes.c_double),
        ("doc_len", ctypes.c_void_p),
        ("doc_offsets", ctypes.c_void_p),
        ("doc_tids", ctypes.c_void_p),
        ("doc_freqs", ctypes.c_void_p),
        ("df", ctypes.c_void_p),
        ("vocab_buf", ctypes.c_void_p),
        ("vocab_lens", ctypes.c_void_p),
    ]


_CTYPE_FOR_DTYPE = {
    np.float64: ctypes.c_double,
    np.int64: ctypes.c_int64,
    np.int32: ctypes.c_int32,
    np.uint8: ctypes.c_uint8,
}


def _array_from_ptr(ptr: int, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Wrap a C pointer as a temporary numpy array (caller must copy to own)."""
    c_type = _CTYPE_FOR_DTYPE[dtype]
    c_ptr = ctypes.cast(ptr, ctypes.POINTER(c_type))
    return np.ctypeslib.as_array(c_ptr, shape=shape)


def _build_result_to_dict(result_ptr: int) -> dict:
    """Copy a native BuildResult into Python-owned numpy arrays and free it."""
    lib = _load()
    result = ctypes.cast(result_ptr, ctypes.POINTER(_BuildResultC)).contents
    if result.status != 0:
        lib.bm25mojo_build_result_free(result_ptr)
        raise NativeUnavailable(f"native corpus build failed with status {result.status}")
    n_terms = int(result.n_terms)
    n_docs_out = int(result.n_docs)
    nnz = int(result.nnz)
    try:
        doc_len = _array_from_ptr(result.doc_len, np.float64, (n_docs_out,)).copy()
        doc_offsets_out = _array_from_ptr(result.doc_offsets, np.int64, (n_docs_out + 1,)).copy()
        doc_tids = _array_from_ptr(result.doc_tids, np.int32, (nnz,)).copy()
        doc_freqs = _array_from_ptr(result.doc_freqs, np.float64, (nnz,)).copy()
        df = _array_from_ptr(result.df, np.int64, (n_terms,)).copy()
        vocab_lens = _array_from_ptr(result.vocab_lens, np.int32, (n_terms,)).copy()
        vocab_buf_size = int(vocab_lens.sum())
        vocab_buf = _array_from_ptr(result.vocab_buf, np.uint8, (vocab_buf_size,)).copy()
    finally:
        lib.bm25mojo_build_result_free(result_ptr)
    return {
        "n_docs": n_docs_out,
        "n_terms": n_terms,
        "nnz": nnz,
        "avgdl": float(result.avgdl),
        "doc_len": doc_len,
        "doc_offsets": doc_offsets_out,
        "doc_tids": doc_tids,
        "doc_freqs": doc_freqs,
        "df": df,
        "vocab_buf": vocab_buf,
        "vocab_lens": vocab_lens,
    }


def build_index_corpus(
    n_docs: int,
    tokens: bytes,
    doc_offsets: np.ndarray,
    variant: int,
    k1: float,
    b: float,
    delta: float,
    epsilon: float,
) -> dict:
    """Native corpus ingestion: length-prefixed UTF-8 token stream -> index data.

    The returned arrays are freshly allocated numpy copies; the Mojo-side
    result object is freed before returning. Raises :class:`NativeUnavailable`
    on any build error so the caller can fall back to the Python build path.
    """
    lib = _load()
    if _LIB_ABI is None or _LIB_ABI < 4:
        raise NativeUnavailable("native library does not expose ABI v4 corpus builder")
    tokens_arr = np.frombuffer(tokens, dtype=np.uint8)
    doc_offsets = np.ascontiguousarray(doc_offsets, dtype=np.int64)
    result_ptr = lib.bm25mojo_build_index_corpus(
        ctypes.c_int64(n_docs),
        tokens_arr.ctypes.data,
        doc_offsets.ctypes.data,
        ctypes.c_int32(variant),
        ctypes.c_double(k1),
        ctypes.c_double(b),
        ctypes.c_double(delta),
        ctypes.c_double(epsilon),
    )
    if not result_ptr:
        raise NativeUnavailable("native corpus build returned NULL")
    return _build_result_to_dict(result_ptr)


def build_index_tids(
    n_docs: int,
    token_tids: np.ndarray,
    doc_offsets: np.ndarray,
    variant: int,
    k1: float,
    b: float,
    delta: float,
    epsilon: float,
) -> dict:
    """Native corpus ingestion: term-id stream -> index data.

    `token_tids` must be a contiguous int32 array of all term ids in document
    order; `doc_offsets` has n_docs + 1 token offsets. Term ids are assigned by
    the caller in first-appearance order and must form a dense 0..n_terms-1
    range. The returned arrays are freshly allocated numpy copies; the Mojo-side
    result object is freed before returning. Raises :class:`NativeUnavailable`
    on any build error so the caller can fall back to another build path.
    """
    lib = _load()
    if _LIB_ABI is None or _LIB_ABI < 4:
        raise NativeUnavailable("native library does not expose ABI v4 term-id builder")
    token_tids = np.ascontiguousarray(token_tids, dtype=np.int32)
    doc_offsets = np.ascontiguousarray(doc_offsets, dtype=np.int64)
    result_ptr = lib.bm25mojo_build_index_tids(
        ctypes.c_int64(n_docs),
        token_tids.ctypes.data,
        doc_offsets.ctypes.data,
        ctypes.c_int32(variant),
        ctypes.c_double(k1),
        ctypes.c_double(b),
        ctypes.c_double(delta),
        ctypes.c_double(epsilon),
    )
    if not result_ptr:
        raise NativeUnavailable("native term-id build returned NULL")
    return _build_result_to_dict(result_ptr)


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
        vocab_terms: np.ndarray | None = None,
    ) -> None:
        lib = _load()  # raises NativeUnavailable
        doc_len = np.ascontiguousarray(doc_len, dtype=np.float64)
        doc_offsets = np.ascontiguousarray(doc_offsets, dtype=np.int64)
        doc_tids = np.ascontiguousarray(doc_tids, dtype=np.int32)
        doc_freqs = np.ascontiguousarray(doc_freqs, dtype=np.float64)
        idf = np.ascontiguousarray(idf, dtype=np.float64)
        n_docs = np.int64(doc_len.shape[0])
        n_terms = np.int64(idf.shape[0])
        handle = lib.bm25mojo_index_create(
            n_docs,
            doc_len.ctypes.data,
            ctypes.c_double(avgdl),
            ctypes.c_double(k1),
            ctypes.c_double(b),
            ctypes.c_double(delta),
            ctypes.c_int32(variant),
            n_terms,
            doc_offsets.ctypes.data,
            doc_tids.ctypes.data,
            doc_freqs.ctypes.data,
            idf.ctypes.data,
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
        self._batch_str_fn = lib.bm25mojo_score_batch_str
        # Attach the vocabulary for the native string-token batch path. On any
        # refusal (mismatched sizes, embedded-NUL term) the index simply keeps
        # using the id-mapping path — results are identical, mapping is slower.
        self._str_ok = False
        if vocab_terms is not None:
            vocab_terms = np.ascontiguousarray(vocab_terms)
            if (
                vocab_terms.dtype.kind == "S"
                and vocab_terms.shape[0] == int(n_terms)
                and 0 < vocab_terms.dtype.itemsize <= MAX_TOKEN_WIDTH
            ):
                rc = lib.bm25mojo_vocab_attach(
                    handle,
                    vocab_terms.ctypes.data,
                    vocab_terms.dtype.itemsize,
                    int(n_terms),
                )
                self._str_ok = rc == 0

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @property
    def str_ok(self) -> bool:
        """Whether the native string-token batch path is armed for this index."""
        return self._str_ok

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
            query_term_ids.ctypes.data,
            len(query_term_ids),
            out.ctypes.data,
        )
        if rc != 0:
            raise NativeUnavailable(f"native scoring failed with status {rc}")
        return out

    def score_batch_flat(
        self, qids_flat: np.ndarray, offsets: np.ndarray, n_queries: int
    ) -> np.ndarray:
        """Score a pre-packed batch of queries in one FFI call.

        `qids_flat`/`offsets` are the concatenated int32 term ids and their
        int64 per-query offsets (n_queries + 1 entries) — the exact C ABI
        layout, so callers that pack once pay no per-query marshalling.
        Returns a float64 (n_queries, n_docs) panel whose row i is
        bit-identical to scoring query i alone.
        """
        if self._handle is None:
            raise NativeUnavailable("native index is closed")
        if n_queries == 0:
            return np.zeros((0, self._n_docs), dtype=np.float64)
        if qids_flat.dtype != np.int32 or not qids_flat.flags.c_contiguous:
            qids_flat = np.ascontiguousarray(qids_flat, dtype=np.int32)
        if offsets.dtype != np.int64 or not offsets.flags.c_contiguous:
            offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        panel = np.zeros((n_queries, self._n_docs), dtype=np.float64)
        rc = self._batch_fn(
            self._handle,
            qids_flat.ctypes.data,
            offsets.ctypes.data,
            n_queries,
            panel.ctypes.data,
        )
        if rc != 0:
            raise NativeUnavailable(f"native batch scoring failed with status {rc}")
        return panel

    def score_batch_str(
        self, tokens: np.ndarray, counts: np.ndarray, n_queries: int
    ) -> np.ndarray:
        """Score a batch of string-token queries in one FFI call.

        `tokens` is the concatenated batch as a fixed-width ``'S'`` array
        (NUL-padded slots); `counts[q]` is query q's token count. Mapping,
        offset accumulation and scoring all run natively; each row of the
        returned float64 (n_queries, n_docs) panel is bit-identical to
        `score_batch_flat` over the mapped ids. Raises UnsafeTokenError when a
        slot cannot be a NUL-padded string (embedded NUL) or is over-wide —
        the caller then falls back to id-mapping for that batch.
        """
        if self._handle is None:
            raise NativeUnavailable("native index is closed")
        if n_queries == 0:
            return np.zeros((0, self._n_docs), dtype=np.float64)
        if not self._str_ok:
            raise NativeUnavailable("native string path is not armed (no vocab map)")
        tokens = np.ascontiguousarray(tokens)
        counts = np.ascontiguousarray(counts, dtype=np.int32)
        width = tokens.dtype.itemsize
        if tokens.dtype.kind != "S" or width <= 0 or width > MAX_TOKEN_WIDTH:
            raise UnsafeTokenError(f"unsupported token slot width {width!r}")
        panel = np.zeros((n_queries, self._n_docs), dtype=np.float64)
        rc = self._batch_str_fn(
            self._handle,
            tokens.ctypes.data,
            width,
            counts.ctypes.data,
            n_queries,
            panel.ctypes.data,
        )
        if rc == 3:
            raise UnsafeTokenError("query token with an embedded NUL byte")
        if rc != 0:
            raise NativeUnavailable(f"native string batch scoring failed with status {rc}")
        return panel

    def score_batch(self, query_term_ids_list: list[np.ndarray]) -> np.ndarray:
        """Score a batch of token-id queries in one FFI call.

        Returns a float64 (n_queries, n_docs) panel whose row i is
        bit-identical to `score(query_term_ids_list[i])`.
        """
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
        return self.score_batch_flat(qids_flat, offsets, n_queries)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.bm25mojo_index_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
