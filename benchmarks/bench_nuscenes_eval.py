#!/usr/bin/env python3
"""Reproducible benchmark: nuscenes-devkit DetectionEval vs nuscenes_eval_mojo.

Synthetic box corpora are generated locally from fixed seeds (no network, no
dataset downloads): small / medium / large sample counts with realistic box
densities. Both sides run the full post-DB-load evaluation chain: box
construction, center-distance + points + bike-rack filtering, greedy matching
at all 10 classes x 4 distance thresholds, and AP/TP metric assembly (nuScenes
DB loading itself is out of scope for both). Correctness against the oracle is
asserted (full metrics-dict agreement within 1e-8) before any timing happens,
so the numbers below always come from a verified-correct build.

Run from the repository root:

    PYTHONPATH="python/nuscenes_eval_mojo:.oracle-nuscenes-eval:tests" PYTHONNOUSERSITE=1 \
        pixi run python benchmarks/bench_nuscenes_eval.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

N_WARM_RUNS = 5  # median over this many timed repetitions
N_COLD_PROBES = 5  # median over this many fresh-process launches
SIZES = {"small": 60, "medium": 300, "large": 1200}
SEED = 20260919
ATOL = 1e-8


def machine_info() -> str:
    import numpy

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
    lines.append(f"- python: {platform.python_version()}, numpy: {numpy.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def _corpus(n_samples: int):
    from test_nuscenes_eval_fixtures import make_corpus

    return make_corpus(
        seed=SEED + n_samples,
        n_samples=n_samples,
        gt_per_sample=(8, 20),
        pred_per_gt=1.35,
        fp_rate=1.0,
        ties=True,
    )


def _count_boxes(gt, predictions) -> tuple:
    n_gt = sum(len(s["annotations"]) for s in gt["samples"])
    n_pred = sum(len(v) for v in predictions["results"].values())
    return n_gt, n_pred


def _cold_probe(corpus_path: str, which: str) -> float:
    """Time the very first full evaluation in a fresh interpreter."""
    probe = f"""
import json, time
with open({corpus_path!r}) as f:
    gt, pred = json.load(f)
if {which!r} == "oracle":
    from test_nuscenes_eval_fixtures import reference_metrics as run
else:
    import nuscenes_eval_mojo as ne
    run = ne.evaluate
t0 = time.perf_counter()
run(gt, pred)
print(f"{{time.perf_counter() - t0:.6f}}")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "python/nuscenes_eval_mojo:.oracle-nuscenes-eval:tests"
    env["PYTHONNOUSERSITE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main() -> None:
    import nuscenes_eval_mojo as ne
    from test_nuscenes_eval_fixtures import compare_metrics, reference_metrics

    info = ne.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- nuscenes_eval_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import importlib.metadata

    print(f"- nuscenes-devkit: {importlib.metadata.version('nuscenes-devkit')}")
    print(
        f"- seeds: corpus=SEED+size; warm: median of {N_WARM_RUNS} runs; "
        f"cold: median of {N_COLD_PROBES} fresh processes (import done, first call timed)"
    )
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'nuscenes_eval_mojo'")

    print("\n== corpora (deterministic, seeded) ==")
    corpora = {}
    for name, n_samples in SIZES.items():
        gt, pred = _corpus(n_samples)
        n_gt, n_pred = _count_boxes(gt, pred)
        corpora[name] = (gt, pred)
        print(f"  {name:>7}: {n_samples:>5} samples, {n_gt:>6} gt annotations, {n_pred:>6} predictions")

    print("\n== correctness gate (full metrics dict vs nuscenes-devkit) ==")
    for name in ("small", "medium"):
        gt, pred = corpora[name]
        worst = compare_metrics(ne.evaluate(gt, pred), reference_metrics(gt, pred), atol=ATOL)
        status = "OK" if worst <= ATOL else "FAIL"
        print(f"  {name:>7}: max|diff| = {worst:.3e}  [{status}]")
        if worst > ATOL:
            sys.exit(f"correctness gate failed on {name}")

    print("\n== cold first evaluation (s, median of fresh processes) ==")
    header = f"{'corpus':>8} | {'DetectionEval path':>18} | {'nuscenes_eval_mojo':>18} | {'speedup':>8}"
    print(header)
    print(f"{'-' * 8}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 8}")
    cold_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in SIZES:
            gt, pred = corpora[name]
            corpus_path = os.path.join(tmp, f"corpus_{name}.json")
            with open(corpus_path, "w") as handle:
                json.dump((gt, pred), handle)
            t_oracle = statistics.median(
                _cold_probe(corpus_path, "oracle") for _ in range(N_COLD_PROBES)
            )
            t_ours = statistics.median(
                _cold_probe(corpus_path, "ours") for _ in range(N_COLD_PROBES)
            )
            cold_rows.append((name, t_oracle, t_ours))
            print(f"{name:>8} | {t_oracle:>18.3f} | {t_ours:>18.3f} | {t_oracle / t_ours:>7.1f}x")

    print("\n== warm steady state (s per full evaluation, median of runs) ==")
    print(header)
    print(f"{'-' * 8}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 8}")
    warm_rows = []
    for name in SIZES:
        gt, pred = corpora[name]
        oracle_samples, ours_samples = [], []
        for _ in range(N_WARM_RUNS):
            t0 = time.perf_counter()
            reference_metrics(gt, pred)
            oracle_samples.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ne.evaluate(gt, pred)
            ours_samples.append(time.perf_counter() - t0)
        t_oracle, t_ours = statistics.median(oracle_samples), statistics.median(ours_samples)
        warm_rows.append((name, t_oracle, t_ours))
        print(f"{name:>8} | {t_oracle:>18.3f} | {t_ours:>18.3f} | {t_oracle / t_ours:>7.1f}x")

    print("\n== README paste block ==")
    print("| corpus | samples | DetectionEval path cold (s) | nuscenes_eval_mojo cold (s) | cold speedup | DetectionEval path warm (s) | nuscenes_eval_mojo warm (s) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (name, t_cold_o, t_cold_u), (_, t_warm_o, t_warm_u) in zip(cold_rows, warm_rows):
        print(
            f"| {name} | {SIZES[name]:,} | {t_cold_o:.3f} | {t_cold_u:.3f} "
            f"| {t_cold_o / t_cold_u:.1f}x | {t_warm_o:.3f} | {t_warm_u:.3f} "
            f"| {t_warm_o / t_warm_u:.1f}x |"
        )


if __name__ == "__main__":
    main()
