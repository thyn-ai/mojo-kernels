"""Shared fixtures: deterministic seeded corpora and score assertions.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the differential suite is bit-reproducible
on any machine.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# The fuzzing harnesses (fuzz/fuzz_bm25.py, fuzz/fuzz_cclib.py) are scripts,
# not a package; tests/test_fuzz_regression_*.py import them to replay the
# seed corpora, so fuzz/ goes on sys.path before those modules are collected.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuzz"))

SCORE_ATOL = 1e-8  # documented tolerance; results are typically bit-identical


def make_corpus(
    seed: int,
    n_docs: int,
    vocab_size: int = 2000,
    min_len: int = 5,
    max_len: int = 60,
) -> list[list[str]]:
    """Zipf-ish synthetic token corpus, deterministic for a given seed."""
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(vocab_size)]
    weights = [1.0 / (i + 1) for i in range(vocab_size)]
    return [
        rng.choices(vocab, weights, k=rng.randint(min_len, max_len))
        for _ in range(n_docs)
    ]


def make_corpus_with_empty_docs(seed: int, n_docs: int) -> list[list[str]]:
    """Corpus with every 7th document empty (the rest 1-40 tokens)."""
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(500)]
    corpus = []
    for i in range(n_docs):
        if i % 7 == 3:
            corpus.append([])
        else:
            corpus.append(rng.choices(vocab, k=rng.randint(1, 40)))
    return corpus


def corpus_terms(corpus: list[list[str]]) -> list[str]:
    """Distinct terms of a corpus in first-appearance order."""
    seen: dict[str, None] = {}
    for doc in corpus:
        for tok in doc:
            seen.setdefault(tok)
    return list(seen)


def make_queries(
    seed: int,
    terms: list[str],
    n_queries: int = 8,
    max_len: int = 8,
    unseen: tuple[str, ...] = ("zzz_unseen_1", "zzz_unseen_2"),
) -> list[list[str]]:
    """Queries mixing corpus terms, unseen terms, and repeated terms."""
    rng = random.Random(seed)
    queries = []
    for i in range(n_queries):
        k = rng.randint(1, max_len)
        q = [rng.choice(terms) for _ in range(k)]
        if i % 3 == 1:
            q.append(unseen[i % len(unseen)])
        if i % 3 == 2 and q:
            q.append(q[0])  # repeated term
        queries.append(q)
    queries.append([])  # empty query
    return queries


def assert_scores_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(actual, np.ndarray), f"expected np.ndarray, got {type(actual)}"
    assert actual.dtype == np.float64, f"expected float64 scores, got {actual.dtype}"
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=SCORE_ATOL)


@pytest.fixture(scope="session")
def medium_corpus() -> list[list[str]]:
    return make_corpus(seed=1, n_docs=500)


@pytest.fixture(scope="session")
def ten_k_corpus() -> list[list[str]]:
    return make_corpus(seed=3, n_docs=10_000, vocab_size=5000)
