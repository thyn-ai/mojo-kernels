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
        """
        if self.avgdl == 0:
            self._native_index = None
            return "fallback"
        # Vocabulary in first-appearance order (same iteration order as `nd`).
        self._vocab = {word: tid for tid, word in enumerate(nd)}
        n_terms = len(self._vocab)
        postings_docs: list[list[int]] = [[] for _ in range(n_terms)]
        postings_freqs: list[list[float]] = [[] for _ in range(n_terms)]
        for doc_id, frequencies in enumerate(self.doc_freqs):
            for word, freq in frequencies.items():
                tid = self._vocab[word]
                postings_docs[tid].append(doc_id)
                postings_freqs[tid].append(float(freq))
        offsets = np.zeros(n_terms + 1, dtype=np.int64)
        for tid in range(n_terms):
            offsets[tid + 1] = offsets[tid] + len(postings_docs[tid])
        docs = np.array([d for tid_list in postings_docs for d in tid_list], dtype=np.int32)
        freqs = np.array(
            [f for tid_list in postings_freqs for f in tid_list], dtype=np.float64
        )
        idf_values = np.array([self.idf[word] for word in self._vocab], dtype=np.float64)
        try:
            self._native_index = NativeIndex(
                doc_len=np.array(self.doc_len, dtype=np.float64),
                avgdl=float(self.avgdl),
                k1=float(self.k1),
                b=float(self.b),
                delta=float(self._delta),
                variant=self._variant,
                offsets=offsets,
                docs=docs,
                freqs=freqs,
                idf=idf_values,
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
            # result the reference computes by adding 0.
            qids = np.array(
                [self._vocab[q] for q in query if q in self._vocab], dtype=np.int32
            )
            return self._native_index.score(qids)
        return self._reference_scores(query)

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
