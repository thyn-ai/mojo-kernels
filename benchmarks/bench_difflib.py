#!/usr/bin/env python3
"""Reproducible benchmark: stdlib difflib vs difflib_mojo (native + fallback).

Correctness is gated first: every workload must produce bit-identical
output (ratios, matching blocks, ranked close-match lists) on the active
backend, or the benchmark aborts. Timings:

- cold first-call: wall time of import + first SequenceMatcher.ratio() and
  first get_close_matches() in a fresh process, median of N_RUNS launches.
- warm steady-state: per-call latency over the workload, median over runs,
  for the stdlib reference, the native kernel and the forced fallback.

Workloads (deterministic, seeded, no network):
  W1  ratio() on two ~12KB program-text revisions (scattered line edits)
  W2  ratio() on two 100KB strings, 5000-codepoint alphabet, 2000 edits
  W3  get_close_matches: 1 word vs 2000 candidates (12-char alphabet)
  W4  get_close_matches_batch: 48 words vs 2000 candidates, same pool

The fallback timings run in a subprocess with DIFFLIB_MOJO_DISABLE_NATIVE=1
(re-invoke this script with --fallback-only; the main run does that itself).

Run from the repository root, e.g.:

    PYTHONPATH=python/difflib_mojo pixi run python benchmarks/bench_difflib.py
"""

from __future__ import annotations

import difflib
import os
import platform
import random
import statistics
import subprocess
import sys
import time

N_RUNS = 5  # median over this many batches / launches
SEED = 20260920


# ---------------------------------------------------------------------------
# Deterministic workloads
# ---------------------------------------------------------------------------


def _rand_str(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_w1() -> tuple[str, str]:
    """~12KB program-text pair with scattered line-level edits."""
    rng = random.Random(SEED)
    words = [
        "def", "return", "import", "for", "while", "if", "else", "value",
        "index", "result", "data", "self", "none", "true", "false",
    ]
    lines = []
    while sum(len(line) for line in lines) < 12_000:
        lines.append(
            "    " * rng.randint(0, 3)
            + " ".join(rng.choice(words) for _ in range(rng.randint(2, 8)))
        )
    lines_b = list(lines)
    for _ in range(80):
        op = rng.randrange(3)
        i = rng.randrange(len(lines_b))
        if op == 0:
            lines_b.insert(i, "    " + " ".join(rng.choice(words) for _ in range(4)))
        elif op == 1 and len(lines_b) > 1:
            del lines_b[i]
        else:
            lines_b[i] = lines_b[i] + "  # edited"
    return "\n".join(lines), "\n".join(lines_b)


def make_w2() -> tuple[str, str]:
    """Two 100KB strings over a 5000-codepoint alphabet, 2000 point edits."""
    rng = random.Random(SEED + 1)
    alphabet = [chr(0x4E00 + i) for i in range(5000)]
    a = "".join(rng.choice(alphabet) for _ in range(100_000))
    b = list(a)
    for _ in range(2000):
        b[rng.randrange(len(b))] = rng.choice(alphabet)
    return a, "".join(b)


def make_pool() -> tuple[list[str], list[str]]:
    """48 query words + 2000 candidates over a 12-char alphabet, at
    identifier/path lengths (20-80 chars) — the shape fuzzy finders,
    spell-checkers and record linkers actually feed get_close_matches."""
    rng = random.Random(SEED + 2)
    alphabet = "abcdefghijkl"
    pool = [_rand_str(rng, rng.randint(20, 80), alphabet) for _ in range(2000)]
    words = [_rand_str(rng, rng.randint(20, 60), alphabet) for _ in range(48)]
    return words, pool


W1A, W1B = make_w1()
W2A, W2B = make_w2()
WORDS, POOL = make_pool()

COLD_SNIPPETS = {
    "oracle": (
        "import time; t0=time.perf_counter();"
        "import difflib;"
        "difflib.SequenceMatcher(None, 'private Thread currentThread;',"
        " 'private volatile Thread currentThread;').ratio();"
        "difflib.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy']);"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "native": (
        "import time; t0=time.perf_counter();"
        "import difflib_mojo;"
        "difflib_mojo.SequenceMatcher(None, 'private Thread currentThread;',"
        " 'private volatile Thread currentThread;').ratio();"
        "difflib_mojo.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy']);"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "fallback": (
        "import time; t0=time.perf_counter();"
        "import difflib_mojo;"
        "difflib_mojo.SequenceMatcher(None, 'private Thread currentThread;',"
        " 'private volatile Thread currentThread;').ratio();"
        "difflib_mojo.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy']);"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def cold_time(kind: str) -> float:
    env = dict(os.environ)
    if kind == "fallback":
        env["DIFFLIB_MOJO_DISABLE_NATIVE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", COLD_SNIPPETS[kind]],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def warm_time(fn, n_calls: int, n_runs: int = N_RUNS) -> float:
    """Median-of-n_runs total time for n_calls invocations, per call."""
    totals = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        totals.append((time.perf_counter() - t0) / n_calls)
    return statistics.median(totals)


def gate_correctness(difflib_mojo) -> None:
    """Bit-identical outputs on every workload, active backend, or abort."""
    for a, b in [(W1A, W1B), (W2A, W2B)]:
        want = difflib.SequenceMatcher(None, a, b)
        got = difflib_mojo.SequenceMatcher(None, a, b)
        assert got.ratio() == want.ratio()
        assert [tuple(t) for t in got.get_matching_blocks()] == [
            tuple(t) for t in want.get_matching_blocks()
        ]
        assert got.get_opcodes() == want.get_opcodes()
        assert got.quick_ratio() == want.quick_ratio()
    for w in WORDS[:8]:
        assert difflib_mojo.get_close_matches(w, POOL, 3, 0.6) == (
            difflib.get_close_matches(w, POOL, 3, 0.6)
        )
    assert difflib_mojo.get_close_matches_batch(WORDS[:8], POOL, 3, 0.6) == [
        difflib.get_close_matches(w, POOL, 3, 0.6) for w in WORDS[:8]
    ]


def run_backend(difflib_mojo, label: str) -> list[tuple[str, float, float]]:
    """Time all four workloads against the oracle on the active backend."""
    rows = []
    w0 = WORDS[0]

    def oracle_batch():
        return [difflib.get_close_matches(w, POOL, 3, 0.6) for w in WORDS]

    specs = [
        (
            f"W1 ratio() {len(W1A)}+{len(W1B)} chars program text",
            lambda: difflib.SequenceMatcher(None, W1A, W1B).ratio(),
            lambda: difflib_mojo.SequenceMatcher(None, W1A, W1B).ratio(),
            3,
            N_RUNS,
        ),
        (
            "W2 ratio() 100k+100k chars (5k-codepoint alphabet)",
            lambda: difflib.SequenceMatcher(None, W2A, W2B).ratio(),
            lambda: difflib_mojo.SequenceMatcher(None, W2A, W2B).ratio(),
            3,
            N_RUNS,
        ),
        (
            "W3 get_close_matches 1 word x 2000 candidates",
            lambda: difflib.get_close_matches(w0, POOL, 3, 0.6),
            lambda: difflib_mojo.get_close_matches(w0, POOL, 3, 0.6),
            3,
            N_RUNS,
        ),
        (
            "W4 batch 48 words x 2000 candidates",
            oracle_batch,
            lambda: difflib_mojo.get_close_matches_batch(WORDS, POOL, 3, 0.6),
            2,
            3,
        ),
        (
            "W5 find_longest_match() 100k x 100k (hot loop, isolated)",
            lambda: difflib.SequenceMatcher(None, W2A, W2B).find_longest_match(),
            lambda: difflib_mojo.SequenceMatcher(None, W2A, W2B).find_longest_match(),
            3,
            N_RUNS,
        ),
    ]
    for name, oracle_fn, ours_fn, n_calls, n_runs in specs:
        t_ref = warm_time(oracle_fn, n_calls, n_runs)
        t_ours = warm_time(ours_fn, n_calls, n_runs)
        rows.append((name, t_ref, t_ours))
        print(
            f"[{label}] {name:<52} {t_ref:>9.4f}s {t_ours:>10.4f}s "
            f"{t_ref / t_ours:>8.1f}x",
            flush=True,
        )
    return rows


def print_env(difflib_mojo) -> None:
    print(
        f"env: {platform.machine()} {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}, numpy "
        f"{__import__('numpy').__version__}, difflib-mojo {difflib_mojo.__version__}"
    )


def main() -> None:
    import difflib_mojo

    fallback_only = "--fallback-only" in sys.argv
    if fallback_only:
        gate_correctness(difflib_mojo)
        print("[fallback] correctness gate OK (bit-identical to stdlib)")
        run_backend(difflib_mojo, "fallback")
        return

    print("gating on bit-identical correctness (native backend) ...")
    gate_correctness(difflib_mojo)
    info = difflib_mojo.backend_info()
    assert info["native_available"], f"benchmark must run on the native kernel: {info}"
    print("gate OK — native backend, outputs bit-identical to stdlib\n")
    print_env(difflib_mojo)
    print(f"median of {N_RUNS} runs; warm = steady-state per call\n")

    print(f"{'workload':<64} {'stdlib':>10} {'mojo':>10} {'speedup':>9}")
    native_rows = run_backend(difflib_mojo, "native")

    # Fallback section in a subprocess (env toggle needs a fresh loader).
    env = dict(os.environ, DIFFLIB_MOJO_DISABLE_NATIVE="1")
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--fallback-only"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(out.stdout, end="")

    cold = {k: statistics.median([cold_time(k) for _ in range(N_RUNS)]) for k in COLD_SNIPPETS}
    print("\ncold first-call (fresh process: import + first ratio + first get_close_matches):")
    for k in ("oracle", "native", "fallback"):
        print(f"  {k:<9} {cold[k] * 1000:8.2f} ms")


if __name__ == "__main__":
    main()
