#!/usr/bin/env python3
"""Reproducible benchmark: bm25s (numpy + numba) vs bm25_mojo, with rank_bm25.

Same harness, corpora, seeds, and query batches as ``bench_bm25.py`` (the
generators are imported from it): 1k / 10k / 100k documents of 30-60 tokens
from a Zipf-ish 20k-term vocabulary, 20 queries per cell, median of 5 runs.
bm25s is timed in three configurations:

- ``bm25s (numpy)``: as shipped — ``BM25()`` defaults (method='lucene',
  float32 scores, NumPy backend).
- ``bm25s (numba)``: same index, ``bm25.compile()`` numba scorer activated
  (the accelerated path bm25s recommends for large corpora).
- ``bm25s (numba f64)``: numba scorer with ``dtype='float64'``, for a
  precision-matched reference point (bm25_mojo / rank_bm25 are float64).

Per cell we report the cold first-call latency (first ``get_scores`` on a
freshly built index) and the warm steady-state per-query latency (median of
5 batches of 20 queries). The one-time numba JIT warmup is measured once at
startup and printed separately.

Correctness: bm25_mojo is gated element-wise against rank_bm25 (<=1e-8)
before any timing, exactly as in bench_bm25.py. bm25s is NOT element-wise
comparable: its methods floor the Robertson idf differently than rank_bm25's
``epsilon * average_idf`` rule (and 'lucene' uses a non-negative idf), which
is by design. We instead report top-10 ranking agreement vs rank_bm25 as a
sanity metric.

Run from the repository root (bm25s and numba are benchmark-only deps):

    PYTHONPATH=python/bm25_mojo pixi run python benchmarks/bench_bm25s.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_bm25 import (  # noqa: E402
    ATOL,
    N_QUERIES,
    N_RUNS,
    QUERY_LENS,
    SEED,
    SIZES,
    machine_info,
    make_corpus,
    make_queries,
)

RANK_SANITY_DOCS = 10_000  # corpus size for the top-10 ranking agreement check
RANK_SANITY_QUERIES = 10
TOP_K = 10


def build_bm25s(corpus, dtype, use_numba):
    """Build a bm25s index; returns (model, build_seconds, compile_seconds)."""
    import bm25s

    t0 = time.perf_counter()
    bm = bm25s.BM25(dtype=dtype)  # method='lucene' (their default)
    bm.index(corpus, show_progress=False)
    t_build = time.perf_counter() - t0
    t_compile = 0.0
    if use_numba:
        t0 = time.perf_counter()
        bm.compile()  # activate the numba scorer backend
        t_compile = time.perf_counter() - t0
    return bm, t_build, t_compile


def warm_and_cold(bm25, queries):
    """Cold first-call latency (s), then warm median per-batch time (s)."""
    t0 = time.perf_counter()
    bm25.get_scores(queries[0])
    cold = time.perf_counter() - t0
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for q in queries:
            bm25.get_scores(q)
        samples.append(time.perf_counter() - t0)
    return cold, statistics.median(samples)


def warm_and_cold_batch(bm25, queries):
    """Same measurement through the batch API: one call scores all queries.

    Cold latency is the first full-batch call; warm is the median full-batch
    call over N_RUNS. Both are reported per query (divided by len(queries)).

    Chunking rule (documented in benchmarks/AUTOPSY-bm25s.md): the batch is
    split into panels of BATCH_CHUNK_ROWS queries when that keeps each panel
    under ~1 MB — the threshold above which this platform's malloc stops
    recycling the allocation across calls and starts paying fresh page faults
    (measured: a 1.6 MB panel costs ~84 us/call while a recycled 800 KB panel
    costs ~4.5 us/row). Above ~1 MB rows, mid-size chunks are strictly worse
    than one full panel, so the whole batch goes in one call.
    """
    chunks = _batch_chunks(bm25.corpus_size, queries)

    def run_batch():
        for chunk in chunks:
            bm25.get_scores_batch(chunk)

    t0 = time.perf_counter()
    run_batch()
    cold = time.perf_counter() - t0
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        run_batch()
        samples.append(time.perf_counter() - t0)
    return cold, statistics.median(samples)


BATCH_CHUNK_ROWS = 10
BATCH_PANEL_MAX_BYTES = 1 << 20


def _batch_chunks(n_docs: int, queries: list) -> list:
    row_bytes = n_docs * np.dtype(np.float64).itemsize
    chunk = len(queries)
    if BATCH_CHUNK_ROWS * row_bytes <= BATCH_PANEL_MAX_BYTES:
        chunk = BATCH_CHUNK_ROWS
    return [queries[i : i + chunk] for i in range(0, len(queries), chunk)]


def topk_overlap(scores_a, scores_b, k: int) -> float:
    """Overlap fraction between the top-k doc id sets of two score vectors."""
    a = set(np.argsort(-scores_a)[:k].tolist())
    b = set(np.argsort(-scores_b)[:k].tolist())
    return len(a & b) / k


def main() -> None:
    from rank_bm25 import BM25Okapi as RefBM25Okapi

    import bm25_mojo
    from bm25_mojo import BM25Okapi

    info = bm25_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    import bm25s
    import numba

    print(f"- bm25s: {bm25s.__version__}, numba: {numba.__version__}")
    print(f"- bm25_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'bm25_mojo'")

    # One-time numba JIT warmup, measured once per score dtype so per-cell
    # cold numbers stay interpretable (numba compiles the scorer on its first
    # call; the compiled code is then reused across corpus sizes).
    tiny = make_corpus(SEED, 64)
    for dtype in ("float32", "float64"):
        t0 = time.perf_counter()
        warm_bm, _, _ = build_bm25s(tiny, dtype, use_numba=True)
        warm_bm.warmup_numba_scorer()
        warm_bm.get_scores(["w00000"])
        print(f"- one-time numba JIT warmup ({dtype}, tiny corpus, first-ever call): "
              f"{time.perf_counter() - t0:.2f}s")
    print(f"- seeds: corpus={SEED}, queries=SEED+size+qlen; runs: median of {N_RUNS} "
          f"over batches of {N_QUERIES} queries; bm25s config: method='lucene', f32 unless noted")

    print("\n== correctness gate (bm25_mojo vs rank_bm25, element-wise) ==")
    corpora = {size: make_corpus(SEED + size, size) for size in SIZES}
    for size in SIZES:
        corpus = corpora[size]
        terms = list({t for doc in corpus[:200] for t in doc})
        queries = make_queries(SEED + size, terms, 10)[:3]
        ours, ref = BM25Okapi(corpus), RefBM25Okapi(corpus)
        worst = max(
            float(np.max(np.abs(ours.get_scores(q) - ref.get_scores(q)))) for q in queries
        )
        status = "OK" if worst <= ATOL else "FAIL"
        print(f"  {size:>7,} docs: max|diff| = {worst:.3e}  [{status}]")
        if worst > ATOL:
            sys.exit(f"correctness gate failed at {size} docs")

    print("\n== ranking sanity (bm25s vs rank_bm25 top-10 overlap, 10k docs) ==")
    corpus = corpora[RANK_SANITY_DOCS]
    terms = list({t for doc in corpus for t in doc})
    sanity_queries = make_queries(SEED + RANK_SANITY_DOCS, terms, 10)[:RANK_SANITY_QUERIES]
    ref = RefBM25Okapi(corpus)
    bm_s, _, _ = build_bm25s(corpus, "float32", use_numba=False)
    overlaps = [
        topk_overlap(ref.get_scores(q), bm_s.get_scores(q), TOP_K) for q in sanity_queries
    ]
    print(f"  mean top-{TOP_K} overlap over {len(sanity_queries)} queries: "
          f"{statistics.mean(overlaps):.2f} (idf flooring differs by design; scores "
          f"are not element-wise comparable)")

    print("\n== index build (one-time, seconds) ==")
    print(f"{'corpus':>10} | {'rank_bm25':>10} | {'bm25_mojo':>10} | "
          f"{'bm25s idx':>10} | {'bm25s cmpl':>10}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    build_rows = []
    for size in SIZES:
        corpus = corpora[size]
        t0 = time.perf_counter()
        RefBM25Okapi(corpus)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        BM25Okapi(corpus)
        t_ours = time.perf_counter() - t0
        _, t_s, t_j = build_bm25s(corpus, "float32", use_numba=True)
        build_rows.append((size, t_ref, t_ours, t_s, t_j))
        print(f"{size:>10,} | {t_ref:>10.3f} | {t_ours:>10.3f} | {t_s:>10.3f} | {t_j:>10.3f}")

    variants = ["rank_bm25", "bm25s (numpy)", "bm25s (numba)", "bm25s (numba f64)",
                "bm25_mojo", "bm25_mojo (batch)"]
    print("\n== scoring: cold first-call latency (ms) ==")
    print(f"{'corpus':>10} | {'q terms':>7} | " + " | ".join(f"{v:>15}" for v in variants))
    print(f"{'-' * 10}-+-{'-' * 7}-+-" + "-+-".join("-" * 15 for _ in variants))
    cold_rows = []
    warm_rows = []
    for size in SIZES:
        corpus = corpora[size]
        ref = RefBM25Okapi(corpus)
        ours = BM25Okapi(corpus)
        bm_np, _, _ = build_bm25s(corpus, "float32", use_numba=False)
        bm_nb, _, _ = build_bm25s(corpus, "float32", use_numba=True)
        bm_nb64, _, _ = build_bm25s(corpus, "float64", use_numba=True)
        impls = [ref, bm_np, bm_nb, bm_nb64, ours]
        for q_len in QUERY_LENS:
            queries = make_queries(SEED + size + q_len, terms_for(corpus), q_len)
            colds, warms = [], []
            for impl in impls:
                cold, warm = warm_and_cold(impl, queries)
                colds.append(1e3 * cold)
                warms.append(1e3 * warm / N_QUERIES)
            cold_b, warm_b = warm_and_cold_batch(ours, queries)
            colds.append(1e3 * cold_b / N_QUERIES)
            warms.append(1e3 * warm_b / N_QUERIES)
            cold_rows.append((size, q_len, colds))
            warm_rows.append((size, q_len, warms))
            print(f"{size:>10,} | {q_len:>7} | " + " | ".join(f"{c:>15.4f}" for c in colds))

    print("\n== scoring: warm per-query latency (ms, median of 5 batches of 20) ==")
    print(f"{'corpus':>10} | {'q terms':>7} | " + " | ".join(f"{v:>15}" for v in variants) +
          f" | {'best vs bm25s':>14}")
    print(f"{'-' * 10}-+-{'-' * 7}-+-" + "-+-".join("-" * 15 for _ in variants) +
          f"-+-{'-' * 14}")
    for size, q_len, warms in warm_rows:
        # Compares the fastest bm25s variant with the fastest bm25_mojo path.
        best_bm25s = min(warms[1], warms[2], warms[3])
        best_mojo = min(warms[4], warms[5])
        ratio = best_bm25s / best_mojo
        print(f"{size:>10,} | {q_len:>7} | " + " | ".join(f"{w:>15.4f}" for w in warms) +
              f" | {ratio:>13.2f}x")

    print("\n== README paste block (bm25s = fastest of its variants per cell; "
          "bm25_mojo = fastest of single/batch per cell) ==")
    print("| corpus | query terms | bm25s ms/query (warm) | bm25_mojo ms/query (warm) | "
          "bm25_mojo speedup | bm25s cold (ms) | bm25_mojo cold (ms) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for (size, q_len, warms), (_, _, colds) in zip(warm_rows, cold_rows):
        best_warm = min(warms[1], warms[2], warms[3])
        best_cold = min(colds[1], colds[2], colds[3])
        mojo_warm = min(warms[4], warms[5])
        mojo_cold = min(colds[4], colds[5])
        ratio = best_warm / mojo_warm
        print(f"| {size:,} | {q_len} | {best_warm:.4f} | {mojo_warm:.4f} | {ratio:.2f}x | "
              f"{best_cold:.3f} | {mojo_cold:.3f} |")


def terms_for(corpus: list[list[str]]) -> list[str]:
    return list({t for doc in corpus for t in doc})


if __name__ == "__main__":
    main()
