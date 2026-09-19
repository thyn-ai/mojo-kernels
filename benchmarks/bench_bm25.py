#!/usr/bin/env python3
"""Reproducible benchmark: rank_bm25 vs bm25_mojo (BM25Okapi).

Corpora are generated locally from fixed seeds (no network, no datasets):
1k / 10k / 100k documents of 30-60 tokens from a Zipf-ish 20k-term
vocabulary, queried with 5 / 10 / 20-term queries. Timings are the median of
5 runs over a fixed batch of queries; each cell reports docs/sec and mean
per-query latency. Correctness is asserted (element-wise vs rank_bm25 within
1e-8) before any timing happens, so the numbers below always come from a
verified-correct build.

Run from the repository root:

    pixi run bench
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

N_QUERIES = 20  # queries per (corpus size, query length) cell
N_RUNS = 5  # median over this many runs of the query batch
SIZES = [1_000, 10_000, 100_000]
QUERY_LENS = [5, 10, 20]
VOCAB_SIZE = 20_000
SEED = 20260919
ATOL = 1e-8


def make_corpus(seed: int, n_docs: int) -> list[list[str]]:
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(VOCAB_SIZE)]
    weights = [1.0 / (i + 1) for i in range(VOCAB_SIZE)]  # Zipf-ish
    return [rng.choices(vocab, weights, k=rng.randint(30, 60)) for _ in range(n_docs)]


def make_queries(seed: int, terms: list[str], q_len: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [
        [rng.choice(terms) for _ in range(q_len)] + (["zzz_unseen"] if i % 4 == 0 else [])
        for i in range(N_QUERIES)
    ]


def time_batch(make_bm25, corpus, queries) -> float:
    """Median seconds to answer the whole query batch (get_scores per query)."""
    bm25 = make_bm25(corpus)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for q in queries:
            bm25.get_scores(q)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    import numpy

    lines.append(f"- python: {platform.python_version()}, numpy: {numpy.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def main() -> None:
    from rank_bm25 import BM25Okapi as RefBM25Okapi

    import bm25_mojo
    from bm25_mojo import BM25Okapi

    info = bm25_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- bm25_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    import rank_bm25

    print(f"- rank_bm25: {getattr(rank_bm25, '__version__', '0.2.2 (PyPI)')}")
    print(f"- seeds: corpus={SEED}, queries=SEED+size+qlen; runs: median of {N_RUNS} "
          f"over batches of {N_QUERIES} queries")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'bm25_mojo'")

    print("\n== index build (one-time, seconds) ==")
    print(f"{'corpus':>10} | {'rank_bm25':>10} | {'bm25_mojo':>10}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    build_rows = []
    corpora = {}
    for size in SIZES:
        corpus = make_corpus(SEED + size, size)
        corpora[size] = corpus
        t0 = time.perf_counter()
        RefBM25Okapi(corpus)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        BM25Okapi(corpus)
        t_ours = time.perf_counter() - t0
        build_rows.append((size, t_ref, t_ours))
        print(f"{size:>10,} | {t_ref:>10.3f} | {t_ours:>10.3f}")

    print("\n== correctness gate (BM25Okapi, element-wise vs rank_bm25) ==")
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

    print("\n== scoring: mean per-query latency (ms) ==")
    header = f"{'corpus':>10} | {'q terms':>7} | {'rank_bm25':>10} | {'bm25_mojo':>10} | {'speedup':>8}"
    print(header)
    print(f"{'-' * 10}-+-{'-' * 7}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
    lat_rows = []
    thr_rows = []
    for size in SIZES:
        corpus = corpora[size]
        terms = list({t for doc in corpus for t in doc})
        for q_len in QUERY_LENS:
            queries = make_queries(SEED + size + q_len, terms, q_len)
            t_ref = time_batch(RefBM25Okapi, corpus, queries)
            t_ours = time_batch(BM25Okapi, corpus, queries)
            ms_ref, ms_ours = 1e3 * t_ref / N_QUERIES, 1e3 * t_ours / N_QUERIES
            dps_ref, dps_ours = size * N_QUERIES / t_ref, size * N_QUERIES / t_ours
            lat_rows.append((size, q_len, ms_ref, ms_ours, ms_ref / ms_ours))
            thr_rows.append((size, q_len, dps_ref, dps_ours, dps_ours / dps_ref))
            print(f"{size:>10,} | {q_len:>7} | {ms_ref:>10.3f} | {ms_ours:>10.4f} "
                  f"| {ms_ref / ms_ours:>7.1f}x")

    print("\n== scoring: throughput (docs scored / sec) ==")
    print(header)
    print(f"{'-' * 10}-+-{'-' * 7}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")
    for size, q_len, dps_ref, dps_ours, speedup in thr_rows:
        print(f"{size:>10,} | {q_len:>7} | {dps_ref:>10,.0f} | {dps_ours:>10,.0f} "
              f"| {speedup:>7.1f}x")

    print("\n== README paste block ==")
    print("| corpus | query terms | rank_bm25 ms/query | bm25_mojo ms/query | speedup |")
    print("|---:|---:|---:|---:|---:|")
    for size, q_len, ms_ref, ms_ours, speedup in lat_rows:
        print(f"| {size:,} | {q_len} | {ms_ref:.3f} | {ms_ours:.4f} | {speedup:.1f}x |")
    print("\n| corpus | rank_bm25 build (s) | bm25_mojo build (s) |")
    print("|---:|---:|---:|")
    for size, t_ref, t_ours in build_rows:
        print(f"| {size:,} | {t_ref:.3f} | {t_ours:.3f} |")


if __name__ == "__main__":
    main()
