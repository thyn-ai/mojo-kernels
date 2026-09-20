#!/usr/bin/env python3
"""Reproducible benchmark: py-motmetrics (oracle) vs motmetrics_mojo.

Streams are generated locally from fixed seeds (no network, no datasets):
synthetic MOTChallenge-style event streams of frames x objects x hypotheses
with births/deaths, ID switches, misses, false positives and NaN
(do-not-pair) distance entries. Two measurements:

* cold first-run — median of 5 fresh-interpreter full pipelines (accumulate
  the whole stream + compute the MOTChallenge metrics) at 1,000 frames x 40
  objects x 44 hypotheses (imports happen before the timer; for
  motmetrics_mojo this includes dlopen + ABI handshake of the native kernel),
* warm steady-state — per-pipeline latency, median of 5 batches of repeated
  full pipelines (fresh accumulator per pipeline) at each size.

Correctness is asserted (all MOTChallenge metrics vs the oracle, counts
exactly, MOTP within 1e-10) before any timing happens, so the numbers below
always come from a verified-correct build.

Run from the repository root:

    PYTHONPATH="python/motmetrics_mojo:.oracle-motmetrics" PYTHONNOUSERSITE=1 \
        pixi run python benchmarks/bench_motmetrics.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import numpy as np

# (n_frames, n_obj, n_hyp)
SIZES = [(100, 10, 11), (500, 25, 27), (1_000, 40, 44), (2_000, 80, 88)]
COLD_SIZE = (1_000, 40, 44)
N_RUNS = 5  # median over this many batches / fresh processes
BATCH = 3  # pipelines per warm batch (each pipeline is a whole stream)
SEED = 20260919
ATOL = 1e-10

REPO_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))

MOTCHALLENGE = [
    "idf1", "idp", "idr", "recall", "precision", "num_unique_objects",
    "mostly_tracked", "partially_tracked", "mostly_lost",
    "num_false_positives", "num_misses", "num_switches",
    "num_fragmentations", "mota", "motp", "num_transfer", "num_ascend",
    "num_migrate",
]


def make_stream(seed: int, n_frames: int, n_obj: int, n_hyp: int):
    """Deterministic synthetic MOT stream (same generator as the test suite,
    denser: objects/hyps are topped up every frame so matrices stay full)."""
    rng = np.random.default_rng(seed)
    frames = []
    alive_o: list[int] = []
    alive_h: list[int] = []
    next_o = 1
    next_h = 1000
    owner: dict[int, int] = {}
    for _ in range(n_frames):
        for o in list(alive_o):
            if rng.random() < 0.01:
                alive_o.remove(o)
                for h, oo in list(owner.items()):
                    if oo == o:
                        del owner[h]
        for h in list(alive_h):
            if rng.random() < 0.012:
                alive_h.remove(h)
                owner.pop(h, None)
        while len(alive_o) < n_obj:
            alive_o.append(next_o)
            next_o += 1
        while len(alive_h) < n_hyp:
            alive_h.append(next_h)
            next_h += 1
        for h in alive_h:
            if h not in owner and rng.random() < 0.9:
                owner[h] = alive_o[rng.integers(len(alive_o))]
        for h in list(owner):
            if rng.random() < 0.03:
                owner[h] = alive_o[rng.integers(len(alive_o))]
        pos_o = {o: rng.random() * 10 + o for o in alive_o}
        d = np.full((len(alive_o), len(alive_h)), np.nan)
        for i, o in enumerate(alive_o):
            for j, h in enumerate(alive_h):
                if rng.random() < 0.03:
                    continue
                base = abs(pos_o[o] - pos_o.get(owner.get(h, -1), 5.0))
                d[i, j] = base + rng.random() * 0.5
        frames.append((list(alive_o), list(alive_h), d))
    return frames


def pipeline_ours(M, frames):
    acc = M.MOTAccumulator(auto_id=True)
    for oids, hids, d in frames:
        acc.update(oids, hids, d)
    return M.compute(acc, metrics=MOTCHALLENGE)


def pipeline_oracle(mm, frames):
    acc = mm.MOTAccumulator(auto_id=True)
    for oids, hids, d in frames:
        acc.update(oids, hids, d)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=MOTCHALLENGE, name="x").iloc[0]


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
    lines.append(f"- python: {platform.python_version()}, numpy: {np.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def cold_first_run() -> tuple[float, float]:
    """Median full-pipeline latency (s) over N_RUNS fresh interpreter processes."""
    nf, no, nh = COLD_SIZE
    ours_snippet = (
        f"import time,sys;sys.path.insert(0,{REPO_ROOT + '/python/motmetrics_mojo'!r});"
        f"sys.path.insert(0,{REPO_ROOT + '/benchmarks'!r});"
        "import bench_motmetrics as b, motmetrics_mojo as M;"
        "f=b.make_stream(%d,%d,%d,%d);"
        "t0=time.perf_counter();"
        "b.pipeline_ours(M,f);"
        "print(time.perf_counter()-t0)"
    ) % (SEED, nf, no, nh)
    oracle_snippet = (
        f"import time,sys;sys.path.insert(0,{REPO_ROOT + '/.oracle-motmetrics'!r});"
        f"sys.path.insert(0,{REPO_ROOT + '/benchmarks'!r});"
        "import bench_motmetrics as b, motmetrics as mm;"
        "f=b.make_stream(%d,%d,%d,%d);"
        "t0=time.perf_counter();"
        "b.pipeline_oracle(mm,f);"
        "print(time.perf_counter()-t0)"
    ) % (SEED, nf, no, nh)
    ours_samples, oracle_samples = [], []
    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)  # the snippets set sys.path explicitly
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", ours_snippet],
            capture_output=True, text=True, check=True, env=env,
        )
        ours_samples.append(float(out.stdout.strip()))
        out = subprocess.run(
            [sys.executable, "-c", oracle_snippet],
            capture_output=True, text=True, check=True, env=env,
        )
        oracle_samples.append(float(out.stdout.strip()))
    return statistics.median(oracle_samples), statistics.median(ours_samples)


def warm_latency(call, n_calls: int = BATCH) -> float:
    """Median per-pipeline latency (s) over N_RUNS batches of n_calls calls."""
    call()  # one warmup pipeline outside the timer
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            call()
        samples.append((time.perf_counter() - t0) / n_calls)
    return statistics.median(samples)


def main() -> None:
    import motmetrics as mm

    import motmetrics_mojo as M

    info = M.backend_info()
    print("== environment ==")
    print(machine_info())
    import importlib.metadata

    print(f"- oracle: motmetrics {importlib.metadata.version('motmetrics')} "
          f"(scipy solver path)")
    print(f"- motmetrics_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    print(f"- seed: {SEED}; warm: median of {N_RUNS} batches x {BATCH} pipelines; "
          f"cold: median of {N_RUNS} fresh-process full pipelines at "
          f"{COLD_SIZE[0]:,} frames x {COLD_SIZE[1]} obj x {COLD_SIZE[2]} hyp")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'motmetrics_mojo'")

    print("\n== correctness gate (MOTChallenge metrics vs py-motmetrics) ==")
    frames = make_stream(SEED + 1, 200, 20, 22)
    ref = pipeline_oracle(mm, frames)
    got = pipeline_ours(M, frames)
    worst = 0.0
    for name in MOTCHALLENGE:
        rv, gv = float(ref[name]), float(got[name])
        if np.isnan(rv) or np.isnan(gv):
            if np.isnan(rv) and np.isnan(gv):
                continue
            sys.exit(f"correctness gate failed for {name}: oracle={rv} ours={gv}")
        diff = abs(rv - gv)
        worst = max(worst, diff)
        if diff > ATOL:
            sys.exit(f"correctness gate failed for {name}: oracle={rv} ours={gv}")
    print(f"  18 MOTChallenge metrics: max|diff| = {worst:.3e}  [OK]")

    print(f"\n== cold first full pipeline (s), {COLD_SIZE[0]:,} frames x "
          f"{COLD_SIZE[1]} obj x {COLD_SIZE[2]} hyp, median of {N_RUNS} fresh processes ==")
    c_ref, c_ours = cold_first_run()
    print(f"  py-motmetrics: {c_ref:.3f} | motmetrics_mojo: {c_ours:.4f} | "
          f"speedup: {c_ref / c_ours:.1f}x")

    print("\n== warm steady-state (s per full pipeline), median of "
          f"{N_RUNS} batches x {BATCH} ==")
    header = f"{'frames':>8} | {'obj':>4} | {'hyp':>4} | {'py-motmetrics':>14} | {'mojo':>9} | {'speedup':>8}"
    print(header)
    print(f"{'-' * 8}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 14}-+-{'-' * 9}-+-{'-' * 8}")
    warm_rows = []
    for nf, no, nh in SIZES:
        frames = make_stream(SEED + nf, nf, no, nh)
        t_ref = warm_latency(lambda: pipeline_oracle(mm, frames))
        t_ours = warm_latency(lambda: pipeline_ours(M, frames))
        warm_rows.append((nf, no, nh, t_ref, t_ours, t_ref / t_ours))
        print(f"{nf:>8,} | {no:>4} | {nh:>4} | {t_ref:>14.3f} | {t_ours:>9.5f} "
              f"| {t_ref / t_ours:>7.1f}x")

    print("\n== README paste block ==")
    print(f"Cold first full pipeline (accumulate + compute; {COLD_SIZE[0]:,} frames x "
          f"{COLD_SIZE[1]} objects x {COLD_SIZE[2]} hypotheses, median of {N_RUNS} fresh processes):\n")
    print("| pipeline | py-motmetrics (s) | motmetrics-mojo (s) | speedup |")
    print("|---|---:|---:|---:|")
    print(f"| accumulate + compute | {c_ref:.3f} | {c_ours:.4f} | {c_ref / c_ours:.1f}x |")
    print()
    print(f"Warm steady state (seconds per full pipeline, median of {N_RUNS} batches "
          f"of {BATCH}):\n")
    print("| frames | objects | hypotheses | py-motmetrics (s) | motmetrics-mojo (s) | speedup |")
    print("|---:|---:|---:|---:|---:|---:|")
    for nf, no, nh, t_ref, t_ours, speedup in warm_rows:
        print(f"| {nf:,} | {no} | {nh} | {t_ref:.3f} | {t_ours:.5f} | {speedup:.1f}x |")


if __name__ == "__main__":
    main()
