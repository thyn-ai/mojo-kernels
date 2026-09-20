"""Differential tests: bm25_mojo must match rank_bm25 element-for-element.

Run twice by `scripts/test_all.sh`: once against the native Mojo kernel and
once with BM25_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback). Both
backends must agree with rank_bm25 within 1e-8 everywhere (in practice the
agreement is bit-exact or within a couple of ulps).

The oracle is the published PyPI package, pinned to rank_bm25==0.2.2 in
pixi.toml. Note that rank_bm25's GitHub master changed BM25L after 0.2.2;
this package matches the published release that `pip install rank_bm25`
delivers.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from rank_bm25 import BM25L as RefBM25L
from rank_bm25 import BM25Okapi as RefBM25Okapi
from rank_bm25 import BM25Plus as RefBM25Plus

from bm25_mojo import BM25L, BM25Okapi, BM25Plus

from conftest import (
    assert_scores_close,
    corpus_terms,
    make_corpus,
    make_corpus_with_empty_docs,
    make_queries,
)

VARIANTS = [
    pytest.param(BM25Okapi, RefBM25Okapi, {}, id="okapi"),
    pytest.param(BM25L, RefBM25L, {}, id="l"),
    pytest.param(BM25Plus, RefBM25Plus, {}, id="plus"),
]

CUSTOM_PARAMS = [
    pytest.param(BM25Okapi, RefBM25Okapi, {"k1": 1.2, "b": 0.5, "epsilon": 0.1}, id="okapi-custom"),
    pytest.param(BM25L, RefBM25L, {"k1": 2.0, "b": 0.9, "delta": 0.25}, id="l-custom"),
    pytest.param(BM25Plus, RefBM25Plus, {"k1": 0.8, "b": 0.3, "delta": 2.5}, id="plus-custom"),
]


def _expected_backend() -> str:
    # scripts/test_all.sh runs the suite once per backend.
    return "fallback" if os.environ.get("BM25_MOJO_DISABLE_NATIVE") == "1" else "native"


def _check_full_parity(ours, ref, queries, documents) -> None:
    # Index attribute parity.
    assert ours.corpus_size == ref.corpus_size
    assert ours.avgdl == ref.avgdl
    assert ours.doc_len == ref.doc_len
    assert ours.doc_freqs == ref.doc_freqs
    assert ours.idf == ref.idf
    # get_scores parity for every query, element-wise.
    for query in queries:
        assert_scores_close(ours.get_scores(query), ref.get_scores(query))
    # get_top_n parity, including ordering of tied scores.
    for query in queries:
        assert ours.get_top_n(query, documents, n=5) == ref.get_top_n(
            query, documents, n=5
        )


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_seeded_corpus_parity(ours_cls, ref_cls, params, medium_corpus):
    ours = ours_cls(medium_corpus, **params)
    ref = ref_cls(medium_corpus, **params)
    assert ours.backend == _expected_backend()
    queries = make_queries(seed=11, terms=corpus_terms(medium_corpus))
    documents = [f"doc-{i}" for i in range(len(medium_corpus))]
    _check_full_parity(ours, ref, queries, documents)


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_ten_k_doc_corpus(ours_cls, ref_cls, params, ten_k_corpus):
    ours = ours_cls(ten_k_corpus, **params)
    ref = ref_cls(ten_k_corpus, **params)
    queries = make_queries(seed=13, terms=corpus_terms(ten_k_corpus), n_queries=4)
    documents = [f"doc-{i}" for i in range(len(ten_k_corpus))]
    _check_full_parity(ours, ref, queries, documents)


@pytest.mark.parametrize("ours_cls,ref_cls,params", CUSTOM_PARAMS)
def test_custom_parameters(ours_cls, ref_cls, params, medium_corpus):
    ours = ours_cls(medium_corpus, **params)
    ref = ref_cls(medium_corpus, **params)
    queries = make_queries(seed=17, terms=corpus_terms(medium_corpus), n_queries=4)
    _check_full_parity(ours, ref, queries, medium_corpus)  # token lists as documents


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_empty_docs(ours_cls, ref_cls, params):
    corpus = make_corpus_with_empty_docs(seed=2, n_docs=210)
    ours = ours_cls(corpus, **params)
    ref = ref_cls(corpus, **params)
    queries = make_queries(seed=19, terms=corpus_terms(corpus), n_queries=4)
    _check_full_parity(ours, ref, queries, [f"doc-{i}" for i in range(len(corpus))])


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_unseen_query_terms(ours_cls, ref_cls, params, medium_corpus):
    ours = ours_cls(medium_corpus, **params)
    ref = ref_cls(medium_corpus, **params)
    terms = corpus_terms(medium_corpus)
    queries = [
        ["never-seen", "also-never"],
        [terms[0], "never-seen", terms[1]],
        ["never-seen"] * 3,
    ]
    _check_full_parity(ours, ref, queries, [f"d{i}" for i in range(len(medium_corpus))])


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_repeated_terms(ours_cls, ref_cls, params):
    corpus = [
        ["alpha"] * 5 + ["beta"],
        ["alpha", "beta", "beta"],
        ["gamma"],
        ["alpha", "gamma", "delta"] * 3,
    ]
    ours = ours_cls(corpus, **params)
    ref = ref_cls(corpus, **params)
    queries = [["alpha", "alpha", "alpha"], ["beta", "beta"], ["gamma", "gamma", "delta"]]
    _check_full_parity(ours, ref, queries, list("abcd"))


@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_single_doc_corpus(ours_cls, ref_cls, params):
    corpus = [["only", "doc", "with", "only", "terms"]]
    ours = ours_cls(corpus, **params)
    ref = ref_cls(corpus, **params)
    queries = [["only"], ["doc", "unseen"], ["terms", "terms"]]
    _check_full_parity(ours, ref, queries, ["the-doc"])


@pytest.mark.parametrize("delta", [-sys.float_info.max, sys.float_info.max], ids=["-max", "+max"])
def test_plus_overflowing_floor_matches_reference(delta):
    """BM25Plus whose per-term floor idf * delta overflows float64.

    rank_bm25 evaluates idf * (delta + ...) once per document, so every score
    is +-inf. The kernel adds the floor densely and gives posted documents only
    their excess over it, which is inf - inf = NaN unless such a term is
    evaluated in reference order (found by fuzz/fuzz_bm25.py; the seeds are
    fuzz/corpus/bm25/regression-plus-overflowing-floor-*.bin).
    """
    corpus = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "d"]]  # idf("c") = ln 5 > 1
    ours = BM25Plus(corpus, delta=delta)
    ref = RefBM25Plus(corpus, delta=delta)
    assert ours.backend == _expected_backend()
    with np.errstate(over="ignore"):
        expected = ref.get_scores(["a", "c"])
        actual = ours.get_scores(["a", "c"])
    assert np.isinf(expected).all() and (np.sign(expected) == np.sign(delta)).all()
    assert_scores_close(actual, expected)

@pytest.mark.parametrize("ours_cls,ref_cls,params", VARIANTS)
def test_top_n_tie_order(ours_cls, ref_cls, params):
    # docs 2 and 3 are token-identical => bit-identical scores on both
    # backends; the remaining docs tie at zero. np.argsort is deterministic
    # for identical input, so the full ordering must match exactly.
    corpus = [
        ["apple", "banana"],
        ["apple"],
        ["zebra", "zebra"],
        ["zebra", "zebra"],
        ["apple", "banana", "cherry", "date"],
    ]
    ours = ours_cls(corpus, **params)
    ref = ref_cls(corpus, **params)
    documents = ["d0", "d1", "d2", "d3", "d4"]
    for query in (["zebra"], ["apple"], ["zebra", "apple"], ["unseen"]):
        s_ours, s_ref = ours.get_scores(query), ref.get_scores(query)
        assert s_ours[2] == s_ours[3]  # tie preserved on our side
        assert_scores_close(s_ours, s_ref)
        assert ours.get_top_n(query, documents, n=3) == ref.get_top_n(query, documents, n=3)
        assert ours.get_top_n(query, documents, n=10) == ref.get_top_n(
            query, documents, n=10
        )  # n > corpus size


def test_tokenizer_path_and_case_sensitivity():
    # rank_bm25 does NOT lowercase: "Apple" and "apple" are distinct terms.
    raw = ["Apple apple APPLE", "banana Apple", "CHERRY cherry"]
    tok = lambda s: s.split(" ")  # noqa: E731
    ours = BM25Okapi(raw, tokenizer=tok)
    # rank_bm25 tokenizes with multiprocessing.Pool (needs a picklable
    # tokenizer); a pre-tokenized corpus is result-identical.
    ref = RefBM25Okapi([tok(doc) for doc in raw])
    assert ours.corpus_size == 3
    assert "Apple" in ours.idf and "apple" in ours.idf and "APPLE" in ours.idf
    queries = [["Apple"], ["apple", "cherry"], ["CHERRY", "unseen"]]
    _check_full_parity(ours, ref, queries, raw)


def test_average_idf_attribute_okapi(medium_corpus):
    ours = BM25Okapi(medium_corpus)
    ref = RefBM25Okapi(medium_corpus)
    assert ours.average_idf == ref.average_idf
    # Negative idfs (terms in more than half the docs) floored to eps*avg.
    tiny = [["x", "y"], ["x", "y"], ["x"]]
    ours_t = BM25Okapi(tiny)
    ref_t = RefBM25Okapi(tiny)
    assert ours_t.idf == ref_t.idf
    assert ours_t.average_idf == ref_t.average_idf
    for word, value in ref_t.idf.items():
        if word in ("x", "y"):  # df >= N/2 => negative before flooring
            assert value == ref_t.epsilon * ref_t.average_idf


def test_get_batch_scores_parity(medium_corpus):
    ours = BM25Okapi(medium_corpus)
    ref = RefBM25Okapi(medium_corpus)
    terms = corpus_terms(medium_corpus)
    query = [terms[0], terms[3], "unseen", terms[0]]
    for doc_ids in ([0, 1, 2], [5], [0, 5, -1], []):
        assert ours.get_batch_scores(query, doc_ids) == ref.get_batch_scores(
            query, doc_ids
        )


def test_top_n_asserts_on_document_mismatch(medium_corpus):
    ours = BM25Okapi(medium_corpus)
    with pytest.raises(AssertionError):
        ours.get_top_n(["anything"], ["too", "short"])


def test_empty_corpus_raises_like_reference():
    with pytest.raises(ZeroDivisionError):
        BM25Okapi([])
    with pytest.raises(ZeroDivisionError):
        RefBM25Okapi([])
