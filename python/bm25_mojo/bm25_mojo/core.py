"""Drop-in BM25 ranking, API-compatible with the `rank_bm25` package.

`BM25Okapi`, `BM25L`, and `BM25Plus` accept the same constructor arguments,
expose the same index attributes (`corpus_size`, `avgdl`, `doc_freqs`, `idf`,
`doc_len`, ...), and return the same types from `get_scores`, `get_batch_scores`,
and `get_top_n` as their `rank_bm25` counterparts.

Scoring runs on the native Mojo kernel when its shared library is available
(macOS arm64 / Linux x86_64 wheels) and transparently falls back to the
vendored pure-Python reference implementation otherwise. Both backends share
index construction and idf calculation in this module, so results are
identical either way; the differential test suite asserts element-wise
agreement with `rank_bm25` on both paths.
"""

from __future__ import annotations

import math

import numpy as np

from bm25_mojo import _reference
from bm25_mojo._native import (
    VARIANT_L,
    VARIANT_OKAPI,
    VARIANT_PLUS,
    NativeIndex,
    NativeUnavailable,
    UnsafeTokenError,
)

__all__ = ["BM25", "BM25Okapi", "BM25L", "BM25Plus"]


class BM25:
    """Shared index construction and the `rank_bm25` query API surface."""

    # Overridden by subclasses: kernel variant id and fallback scorer.
    _variant: int = VARIANT_OKAPI
    _delta: float = 0.0

    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)
        self._backend = self._build_native_index(nd)

    def _tokenize_corpus(self, corpus):
        # rank_bm25 maps the tokenizer with multiprocessing.Pool; a sequential
        # map returns element-for-element identical results without spawning
        # processes (deterministic, and safe in test/CI sandboxes).
        return [self.tokenizer(document) for document in corpus]

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents containing the word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def _build_native_index(self, nd) -> str:
        """Hand the numeric index to the native kernel; return the backend name.

        Falls back to the vendored reference scorer when the native library is
        unavailable, and in the degenerate avgdl == 0 case (an entirely empty
        corpus), where the reference's IEEE behaviour (NaN propagation) is the
        contract we mirror.

        The kernel builds the CSR postings and bakes the idf-weighted scores
        itself; this method only assembles doc-major term streams. The loops
        below are deliberately pushed into C-level iteration (`map` over a
        dict's `.get`, `list.extend` over `dict.values()`, one numpy
        conversion per stream) because this is the build-time hot path.
        """
        if self.avgdl == 0:
            self._native_index = None
            return "fallback"
        # Vocabulary in first-appearance order (same iteration order as `nd`).
        self._vocab = {word: tid for tid, word in enumerate(nd)}
        vocab_get = self._vocab.get
        doc_offsets = [0]
        tids: list[int] = []
        freqs: list[float] = []
        for frequencies in self.doc_freqs:
            tids.extend(map(vocab_get, frequencies))
            freqs.extend(frequencies.values())
            doc_offsets.append(len(tids))
        idf_values = np.array([self.idf[word] for word in self._vocab], dtype=np.float64)
        # Vocabulary as fixed-width 'S' slots in id order (dict iteration
        # order), for the kernel's native token->id map. The native map keys
        # are byte strings, so it may only be used when every term is a `str`
        # — numpy 'S' conversion would silently stringify non-str keys and
        # could conflate distinct dict keys (e.g. int 1 with "1"). Non-ASCII
        # terms cannot encode to 'S' either. Any refusal simply leaves the
        # id-mapping path in charge (identical results, slower mapping).
        vocab_terms = None
        if all(map(str.__instancecheck__, self._vocab)):
            try:
                vocab_terms = np.asarray(list(self._vocab), dtype="S")
            except (UnicodeEncodeError, TypeError, ValueError):
                vocab_terms = None
        try:
            self._native_index = NativeIndex(
                doc_len=np.array(self.doc_len, dtype=np.float64),
                avgdl=float(self.avgdl),
                k1=float(self.k1),
                b=float(self.b),
                delta=float(self._delta),
                variant=self._variant,
                doc_offsets=np.array(doc_offsets, dtype=np.int64),
                doc_tids=np.array(tids, dtype=np.int32),
                doc_freqs=np.array(freqs, dtype=np.float64),
                idf=idf_values,
                vocab_terms=vocab_terms,
            )
            return "native"
        except NativeUnavailable:
            self._native_index = None
            return "fallback"

    @property
    def backend(self) -> str:
        """Which scorer serves this instance: "native" or "fallback"."""
        return self._backend

    def get_scores(self, query):
        if self._native_index is not None:
            # Unseen terms contribute exactly 0 (their idf lookup is 0 and the
            # per-document factor is finite), so they are skipped — the same
            # result the reference computes by adding 0. map/filter in C for
            # speed: this is the per-query hot path.
            vocab_get = self._vocab.get
            qids = np.array(
                [v for v in map(vocab_get, query) if v is not None], dtype=np.int32
            )
            return self._native_index.score(qids)
        return self._reference_scores(query)

    def get_scores_batch(self, queries):
        """Score a batch of tokenized queries in one call.

        Returns a float64 array of shape (len(queries), corpus_size); row i is
        bit-identical to ``get_scores(queries[i])``. Batching amortizes FFI,
        allocation, and token-mapping overhead across the batch. On the
        fallback backend this simply loops the reference scorer.
        """
        if type(queries) is not list:
            queries = list(queries)
        if not queries:
            return np.zeros((0, self.corpus_size), dtype=np.float64)
        if self._native_index is not None:
            ni = self._native_index
            if ni.str_ok:
                # Native string-token path: numpy encodes the batch into
                # fixed-width byte slots in C, then one FFI call maps (in the
                # kernel's vocab hash), packs and scores. In the arena's
                # interleave pattern this avoids re-walking a megabyte-scale
                # Python dict per call (the eviction tax dominates the M
                # cell); the ~1us conversion premium is the tight-loop cost.
                # Exactness guards: only `str` tokens may take this path —
                # 'S' conversion would stringify anything else and could
                # conflate distinct dict keys — and non-ASCII strings cannot
                # encode to 'S'; both fall through (identical results).
                try:
                    tokens_flat = [t for q in queries for t in q]
                    if not all(map(str.__instancecheck__, tokens_flat)):
                        raise UnsafeTokenError("non-str token in batch")
                    counts = np.array([len(q) for q in queries], dtype=np.int32)
                    tokens = (
                        np.asarray(tokens_flat, dtype="S")
                        if tokens_flat
                        else np.zeros((0,), dtype="S1")
                    )
                    return ni.score_batch_str(tokens, counts, len(queries))
                except (UnsafeTokenError, UnicodeEncodeError):
                    pass  # fall through to the id-mapping path for this batch
            # Single pass: token ids and per-query offsets are accumulated
            # into flat Python lists (list.extend over a map/filter runs at C
            # speed), then converted with exactly two numpy calls — instead of
            # building one small numpy array per query plus a concatenate and
            # a cumsum, whose fixed dispatch costs dominate small batches.
            # Unseen terms map to None and are skipped: they contribute
            # exactly 0, the same result the reference computes by adding 0.
            vocab_get = self._vocab.get
            flat: list[int] = []
            offsets = [0]
            extend = flat.extend
            add_offset = offsets.append
            for query in queries:
                extend(v for v in map(vocab_get, query) if v is not None)
                add_offset(len(flat))
            return ni.score_batch_flat(
                np.array(flat, dtype=np.int32),
                np.array(offsets, dtype=np.int64),
                len(queries),
            )
        return np.array([self._reference_scores(query) for query in queries])

    def _reference_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        """Scores for a subset of documents. Per-document BM25 scores are
        independent, so this is bit-identical to scoring everything and
        selecting `doc_ids` (which also mirrors negative-index behaviour)."""
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        return self.get_scores(query)[doc_ids].tolist()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    _variant = VARIANT_OKAPI

    def _calc_idf(self, nd):
        # Collect the idf sum for the average used by the epsilon floor, and
        # the terms whose idf is negative (term in more than half the corpus).
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def _reference_scores(self, query):
        return _reference.okapi_scores(
            self.doc_freqs, self.idf, self.doc_len, self.avgdl, query, self.k1, self.b
        )


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    _variant = VARIANT_L

    @property
    def _delta(self):
        return self.delta

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def _reference_scores(self, query):
        return _reference.l_scores(
            self.doc_freqs,
            self.idf,
            self.doc_len,
            self.avgdl,
            query,
            self.k1,
            self.b,
            self.delta,
        )


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    _variant = VARIANT_PLUS

    @property
    def _delta(self):
        return self.delta

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            # log of the quotient (not a difference of logs) — matches the
            # reference's float result bit-for-bit.
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def _reference_scores(self, query):
        return _reference.plus_scores(
            self.doc_freqs,
            self.idf,
            self.doc_len,
            self.avgdl,
            query,
            self.k1,
            self.b,
            self.delta,
        )
