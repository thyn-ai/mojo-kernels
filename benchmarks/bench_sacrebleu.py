#!/usr/bin/env python3
"""Reproducible benchmark: sacrebleu vs sacrebleu_mojo (BLEU + chrF).

Corpora are generated locally from fixed seeds (no network, no datasets):
hypothesis/reference sentence pairs of 10-40 tokens from a Zipf-ish 12k-term
vocabulary with punctuation mixed in. Correctness is asserted (`.score`
agreement with the oracle within 1e-9) before any timing happens, so the
numbers below always come from a verified-correct build.

Two timing modes per cell:
  * cold: the first call after import (end-to-end, includes everything).
  * warm: median of 5 repeated calls (steady state).

Run from the repository root:

    PYTHONPATH=python/sacrebleu_mojo python benchmarks/bench_sacrebleu.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

N_RUNS = 5  # warm timing: median over this many runs
SIZES = [2_000, 10_000, 30_000]
VOCAB_SIZE = 12_000
SEED = 20260919
ATOL = 1e-9

WORDS_EXTRA = (
    "don't it's 3.14 2,500 100-200 state-of-the-art (x) [y] {z} <w> &amp; "
    "#hash @mention $price %percent ^caret *star pipe| tilde~ end. stop; go! "
)


def make_corpus(seed: int, n_pairs: int):
    rng = random.Random(seed)
    vocab = [f"w{i:05d}" for i in range(VOCAB_SIZE)] + WORDS_EXTRA.split()
    weights = [1.0 / (i + 1) for i in range(len(vocab))]

    def sent():
        return " ".join(rng.choices(vocab, weights, k=rng.randint(10, 40)))

    hyps, refs = [], []
    for _ in range(n_pairs):
        r = sent()
        # hypotheses are ref near-copies (drop/replace some tokens), like MT output
        w = r.split()
        for _ in range(rng.randint(0, 4)):
            if not w:
                break
            op = rng.random()
            i = rng.randrange(len(w))
            if op < 0.4:
                w.pop(i)
            elif op < 0.8:
                w[i] = rng.choice(vocab)
            else:
                w.insert(i, rng.choice(vocab))
        hyps.append(" ".join(w) if w and rng.random() < 0.95 else r)
        refs.append(r)
    return hyps, [refs]


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


def time_call(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def median_of_runs(fn, *args) -> float:
    samples = [time_call(fn, *args) for _ in range(N_RUNS)]
    return statistics.median(samples)


def main() -> None:
    import sacrebleu as oracle

    import sacrebleu_mojo

    info = sacrebleu_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- sacrebleu_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    print(f"- oracle: sacrebleu {oracle.__version__}")
    print(f"- seeds: corpus=SEED+size; warm runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'sacrebleu_mojo'")

    corpora = {size: make_corpus(SEED + size, size) for size in SIZES}

    print("\n== correctness gate (score agreement with the oracle) ==")
    worst_b = worst_c = 0.0
    for size in SIZES:
        hyps, refs = corpora[size]
        hyps, refs = hyps[:500], [refs[0][:500]]
        db = abs(oracle.corpus_bleu(hyps, refs).score - sacrebleu_mojo.corpus_bleu(hyps, refs).score)
        dc = abs(oracle.corpus_chrf(hyps, refs).score - sacrebleu_mojo.corpus_chrf(hyps, refs).score)
        worst_b, worst_c = max(worst_b, db), max(worst_c, dc)
        print(f"  {size:>7,} pairs (sampled 500): |dBLEU| = {db:.3e}, |dCHRF| = {dc:.3e}")
    if worst_b > ATOL or worst_c > ATOL:
        sys.exit("correctness gate failed")

    rows_bleu = []
    rows_chrf = []
    for size in SIZES:
        hyps, refs = corpora[size]
        # cold first call (fresh functions: oracle and ours each get one cold call)
        cold_o_b = time_call(oracle.corpus_bleu, hyps, refs)
        cold_m_b = time_call(sacrebleu_mojo.corpus_bleu, hyps, refs)
        warm_o_b = median_of_runs(oracle.corpus_bleu, hyps, refs)
        warm_m_b = median_of_runs(sacrebleu_mojo.corpus_bleu, hyps, refs)
        rows_bleu.append((size, cold_o_b, cold_m_b, warm_o_b, warm_m_b))
        cold_o_c = time_call(oracle.corpus_chrf, hyps, refs)
        cold_m_c = time_call(sacrebleu_mojo.corpus_chrf, hyps, refs)
        warm_o_c = median_of_runs(oracle.corpus_chrf, hyps, refs)
        warm_m_c = median_of_runs(sacrebleu_mojo.corpus_chrf, hyps, refs)
        rows_chrf.append((size, cold_o_c, cold_m_c, warm_o_c, warm_m_c))

    def report(name, rows):
        print(f"\n== {name}: seconds per call ==")
        print(f"{'pairs':>10} | {'oracle cold':>11} | {'mojo cold':>10} | "
              f"{'oracle warm':>11} | {'mojo warm':>10} | {'cold speedup':>12} | {'warm speedup':>12}")
        print(f"{'-' * 10}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}")
        for size, co, cm, wo, wm in rows:
            print(f"{size:>10,} | {co:>11.4f} | {cm:>10.4f} | {wo:>11.4f} | {wm:>10.4f} "
                  f"| {co / cm:>11.2f}x | {wo / wm:>11.2f}x")

    report("corpus_bleu (13a tokenizer)", rows_bleu)
    report("corpus_chrf (chars, word_order=0)", rows_chrf)

    # fallback timing for context (informational, not part of the gate)
    size = SIZES[-1]
    hyps, refs = corpora[size]
    os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"] = "1"
    try:
        warm_fb = median_of_runs(sacrebleu_mojo.corpus_bleu, hyps, refs)
        warm_fc = median_of_runs(sacrebleu_mojo.corpus_chrf, hyps, refs)
    finally:
        del os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"]
    print(f"\n== context: pure-Python fallback, warm (s) at {size:,} pairs ==")
    print(f"  corpus_bleu: {warm_fb:.4f}  corpus_chrf: {warm_fc:.4f}")

    print("\n== README paste block ==")
    print("BLEU (corpus_bleu, 13a tokenizer):")
    print("| pairs | sacrebleu cold (s) | sacrebleu_mojo cold (s) | sacrebleu warm (s) | sacrebleu_mojo warm (s) | cold speedup | warm speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for size, co, cm, wo, wm in rows_bleu:
        print(f"| {size:,} | {co:.4f} | {cm:.4f} | {wo:.4f} | {wm:.4f} | {co / cm:.2f}x | {wo / wm:.2f}x |")
    print("\nchrF (corpus_chrf):")
    print("| pairs | sacrebleu cold (s) | sacrebleu_mojo cold (s) | sacrebleu warm (s) | sacrebleu_mojo warm (s) | cold speedup | warm speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for size, co, cm, wo, wm in rows_chrf:
        print(f"| {size:,} | {co:.4f} | {cm:.4f} | {wo:.4f} | {wm:.4f} | {co / cm:.2f}x | {wo / wm:.2f}x |")


if __name__ == "__main__":
    main()
