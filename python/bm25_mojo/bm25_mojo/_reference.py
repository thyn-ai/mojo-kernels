"""Vendored pure-Python reference scorers for the BM25 family.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``BM25_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
textbook formulas (Trotman et al., "Improvements to BM25 and Language Models
Examined") written to be observably identical to the widely used `rank_bm25`
package: same parameter defaults, same NumPy operation order, same dtypes, so
scores agree bit-for-bit (both paths are IEEE-754 float64 element-wise).

Only scoring lives here; index construction and idf calculation are shared
with the native path in `bm25_mojo.core`, so the two backends can never
disagree about the index itself.
"""

from __future__ import annotations

import numpy as np


def okapi_scores(
    doc_freqs: list[dict],
    idf: dict,
    doc_len: list[int],
    avgdl: float,
    query: list[str],
    k1: float,
    b: float,
) -> np.ndarray:
    """BM25Okapi: idf * (qf * (k1 + 1) / (qf + k1 * (1 - b + b * dl / avgdl)))."""
    score = np.zeros(len(doc_freqs))
    doc_len_arr = np.array(doc_len)
    for q in query:
        q_freq = np.array([(doc.get(q) or 0) for doc in doc_freqs])
        score += (idf.get(q) or 0) * (
            q_freq * (k1 + 1) / (q_freq + k1 * (1 - b + b * doc_len_arr / avgdl))
        )
    return score


def l_scores(
    doc_freqs: list[dict],
    idf: dict,
    doc_len: list[int],
    avgdl: float,
    query: list[str],
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    """BM25L as published on PyPI (rank_bm25 0.2.2):
    idf * qf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta).

    Note: rank_bm25's GitHub master dropped the leading qf factor after the
    0.2.2 release; this package matches the published PyPI package, which is
    what `pip install rank_bm25` users run.
    """
    score = np.zeros(len(doc_freqs))
    doc_len_arr = np.array(doc_len)
    for q in query:
        q_freq = np.array([(doc.get(q) or 0) for doc in doc_freqs])
        ctd = q_freq / (1 - b + b * doc_len_arr / avgdl)
        score += (
            (idf.get(q) or 0) * q_freq * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)
        )
    return score


def plus_scores(
    doc_freqs: list[dict],
    idf: dict,
    doc_len: list[int],
    avgdl: float,
    query: list[str],
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    """BM25Plus: idf * (delta + qf * (k1 + 1) / (k1 * (1 - b + b * dl / avgdl) + qf))."""
    score = np.zeros(len(doc_freqs))
    doc_len_arr = np.array(doc_len)
    for q in query:
        q_freq = np.array([(doc.get(q) or 0) for doc in doc_freqs])
        score += (idf.get(q) or 0) * (
            delta
            + (q_freq * (k1 + 1)) / (k1 * (1 - b + b * doc_len_arr / avgdl) + q_freq)
        )
    return score
