#!/usr/bin/env python3
"""Layer-by-layer autopsy: bm25_mojo query path vs bm25s (numba) query path.

NOTE (2026-09-20): this instruments the **v1** (pre-optimization) kernel's
query path — it produced the layer table in benchmarks/AUTOPSY-bm25s.md. With
the v2 kernel (baked weights, no kernel-side zeroing) the "(f) ffi+alloc+zero"
line measures the wrapper's calloc instead of a kernel memset, and the walk
no longer divides per posting.

Methodology notes: one corpus size is alive at a time (build -> measure ->
free -> gc) so measurements are not distorted by multi-GB RSS from other
cells; allocation micro-benchmarks run in tight loops identical in shape to
the benchmark harness (batches of 20 calls, median of 7 batches). All numbers
are measured on this machine, nothing is estimated.

Layers:  (a) query-term -> postings resolution, (b) per-term idf lookup,
(c) score-buffer zeroing, (d) scoring walk per posting, (e) bandwidth vs
compute per cell, (f) FFI/allocation overhead, (g) index/query-time split.

Run:  PYTHONPATH=python/bm25_mojo pixi run python benchmarks/autopsy_bm25s_layers.py
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_bm25 import SEED, make_corpus, make_queries

SIZES = [1_000, 10_000, 100_000]
QLENS = [5, 20]
REPS = 7
INNER = 20


def median_us(fn, reps=REPS, inner=INNER) -> float:
    """Median per-call microseconds over `reps` batches of `inner` calls."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / inner)
    return statistics.median(samples) / 1e3


def measure_size(size: int) -> dict:
    import bm25s

    from bm25_mojo import BM25Okapi

    corpus = make_corpus(SEED + size, size)

    # --- allocation micro-costs, measured before big models exist ---
    alloc = {
        "empty_f64": median_us(lambda: [np.empty(size, dtype=np.float64) for _ in range(INNER)]),
        "zeros_f64": median_us(lambda: [np.zeros(size, dtype=np.float64) for _ in range(INNER)]),
        "zeros_f32": median_us(lambda: [np.zeros(size, dtype=np.float32) for _ in range(INNER)]),
    }

    ours = BM25Okapi(corpus)
    idx = ours._native_index
    vocab = ours._vocab
    counts = np.zeros(len(vocab), dtype=np.int64)
    for df in ours.doc_freqs:
        for w in df:
            counts[vocab[w]] += 1
    nnz = int(counts.sum())

    bm = bm25s.BM25(dtype="float32")
    bm.index(corpus, show_progress=False)
    bm.compile()

    out = {"size": size, "n_terms": len(vocab), "nnz": nnz, "alloc": alloc}
    for qlen in QLENS:
        terms = list(vocab.keys())
        queries = make_queries(SEED + size + qlen, terms, qlen)
        qids_list = [
            np.array([vocab[t] for t in q if t in vocab], dtype=np.int32) for q in queries
        ]
        empty_qids = np.zeros(0, dtype=np.int32)
        postings = [
            int(counts[[vocab[t] for t in q if t in vocab]].sum()) for q in queries
        ]

        res = {
            "postings": statistics.mean(postings),
            # ours
            "o_full": median_us(lambda: [ours.get_scores(q) for q in queries]),
            "o_native": median_us(lambda: [idx.score(q) for q in qids_list]),
            "o_emptyq": median_us(lambda: [idx.score(empty_qids) for _ in queries]),
            # bm25s
            "s_full": median_us(lambda: [bm.get_scores(q) for q in queries]),
            "s_map": median_us(lambda: [bm.get_tokens_ids(q) for q in queries]),
            "s_fromids": median_us(lambda: [bm.get_scores_from_ids(list(q)) for q in qids_list]),
        }
        out[qlen] = res

    del corpus, ours, idx, bm, counts
    gc.collect()
    return out


def main() -> None:
    import bm25_mojo

    assert bm25_mojo.native_available(), "native kernel required for the autopsy"

    results = [measure_size(size) for size in SIZES]

    for row in results:
        a = row["alloc"]
        print(f"\n== {row['size']:,} docs | terms {row['n_terms']:,} | nnz {row['nnz']:,} | "
              f"avg df {row['nnz'] / row['n_terms']:.1f} ==")
        print(f"  alloc tight-loop: np.empty f64 {a['empty_f64']:.2f} us | "
              f"np.zeros f64 {a['zeros_f64']:.2f} us | np.zeros f32 {a['zeros_f32']:.2f} us")
        for qlen in QLENS:
            r = row[qlen]
            p = max(r["postings"], 1e-9)
            print(f"  -- {qlen} terms, {r['postings']:.0f} postings/query --")
            print(f"    ours  total {r['o_full']:7.2f} us | (a) map {r['o_full'] - r['o_native']:6.2f} | "
                  f"(f) ffi+alloc+zero {r['o_emptyq']:6.2f} | "
                  f"(d) walk {r['o_native'] - r['o_emptyq']:6.2f} "
                  f"({(r['o_native'] - r['o_emptyq']) / p * 1e3:6.2f} ns/post)")
            print(f"    bm25s total {r['s_full']:7.2f} us | (a) map {r['s_map']:6.2f} | "
                  f"fromids {r['s_fromids']:6.2f} | "
                  f"(d) walk~ {r['s_fromids'] - row['alloc']['zeros_f32']:6.2f} "
                  f"({(r['s_fromids'] - row['alloc']['zeros_f32']) / p * 1e3:6.2f} ns/post)")


if __name__ == "__main__":
    main()
