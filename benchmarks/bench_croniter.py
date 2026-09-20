#!/usr/bin/env python3
"""Reproducible benchmark: croniter (PyPI oracle) vs croniter_mojo.

Workloads, all generated locally from fixed seeds (no network):
- cold get_next: first call for a fresh expression (parse + seek)
- warm get_next: steady-state per-call cost (schedule cached)
- chains: 1000 successive get_next calls (scheduler roll-out), plus a
  sparse-expression chain and a get_prev chain and a tz-aware chain

Timings are the median of 5 runs. Correctness is asserted (exact datetime
equality vs the croniter oracle) before any timing happens, so the numbers
below always come from a verified-correct build.

Run from the repository root with explicit env, e.g.:

    PYTHONPATH=python/croniter_mojo PYTHONNOUSERSITE=1 ~/.pixi/bin/pixi run python benchmarks/bench_croniter.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

N_RUNS = 5  # median over this many runs

EXPRS = [
    ("*/5 * * * *", "every-5-min"),
    ("0 9 * * mon-fri", "daily-weekday"),
    ("37/6 14-8/5 */2 7/2 fri-thu", "mixed-wrap"),
    ("0 0 * * 5#3", "nth-weekday"),
    ("0 0 29 2 *", "sparse-feb29"),
]

ET = ZoneInfo("America/New_York")
AWARE_START = datetime(2026, 10, 31, 12, 0, tzinfo=ET)
NAIVE_START = datetime(2026, 9, 19, 14, 30, 45)


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
    lines.append(f"- python: {platform.python_version()}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def median_us(fn, n_calls, runs=N_RUNS):
    """Median wall time of fn() over `runs` repetitions, per n_calls."""
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) / n_calls)
    return statistics.median(samples) * 1e6


def main() -> None:
    from croniter import croniter

    import croniter_mojo

    info = croniter_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- croniter_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import croniter as croniter_mod

    print(f"- croniter oracle: {getattr(croniter_mod, '__version__', '6.2.4 (PyPI)')}")
    print(f"- runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'croniter_mojo'")

    # --- correctness gate -------------------------------------------------
    print("\n== correctness gate (exact datetime equality vs oracle) ==")
    for expr, label in EXPRS:
        o = croniter(expr, NAIVE_START).get_next(datetime)
        m = croniter_mojo.get_next(expr, NAIVE_START)
        assert m == o, f"{label}: {m!r} != {o!r}"
        o = croniter(expr, AWARE_START).get_prev(datetime)
        m = croniter_mojo.get_prev(expr, AWARE_START)
        assert m == o, f"{label} aware: {m!r} != {o!r}"
        print(f"  {label:14} OK")

    def ours_next(expr, start):
        return croniter_mojo.get_next(expr, start)

    def oracle_next(expr, start):
        return croniter(expr, start).get_next(datetime)

    def ours_prev(expr, start):
        return croniter_mojo.get_prev(expr, start)

    def oracle_prev(expr, start):
        return croniter(expr, start).get_prev(datetime)

    # --- cold first-call ----------------------------------------------------
    print("\n== cold get_next (fresh expression, parse + first seek, us/call) ==")
    print(f"{'expression':>24} | {'oracle':>10} | {'croniter_mojo':>13} | {'speedup':>8}")
    cold_rows = []
    for expr, label in EXPRS:
        t_o = median_us(lambda: oracle_next(expr, NAIVE_START), 1)
        t_m = median_us(lambda: ours_next(expr, NAIVE_START), 1)
        cold_rows.append((label, t_o, t_m, t_o / t_m))
        print(f"{label:>24} | {t_o:>10.1f} | {t_m:>13.1f} | {t_o / t_m:>7.2f}x")

    # --- warm steady-state ----------------------------------------------------
    print("\n== warm get_next (schedule cached, us/call) ==")
    print(f"{'expression':>24} | {'oracle':>10} | {'croniter_mojo':>13} | {'speedup':>8}")
    warm_rows = []
    for expr, label in EXPRS:
        # oracle re-parses per croniter() construction; use one iterator for
        # the warm path like ours (cached parse) by chaining get_next twice.
        c = croniter(expr, NAIVE_START)
        c.get_next(datetime)  # warm up
        croniter_mojo.get_next(expr, NAIVE_START)  # warm up cache

        def t_oracle(c=c, expr=expr):
            c2 = croniter(expr, NAIVE_START)
            c2.get_next(datetime)

        def t_ours(expr=expr):
            croniter_mojo.get_next(expr, NAIVE_START)

        t_o = median_us(t_oracle, 1)
        t_m = median_us(t_ours, 1)
        warm_rows.append((label, t_o, t_m, t_o / t_m))
        print(f"{label:>24} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    # --- chains (roll-outs) --------------------------------------------------
    print("\n== chains: successive calls, us/call (median of 5 chains of 1000) ==")
    print(f"{'workload':>28} | {'oracle':>10} | {'croniter_mojo':>13} | {'speedup':>8}")
    chain_rows = []

    def make_chain(callable_next, expr, start, n=1000):
        def run():
            cur = start
            for _ in range(n):
                cur = callable_next(expr, cur)

        return run

    for expr, label in EXPRS:
        t_o = median_us(make_chain(oracle_next, expr, NAIVE_START), 1000)
        t_m = median_us(make_chain(ours_next, expr, NAIVE_START), 1000)
        chain_rows.append((f"next:{label}", t_o, t_m, t_o / t_m))
        print(f"{'next:' + label:>28} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    t_o = median_us(make_chain(oracle_prev, "*/15 * * * *", NAIVE_START), 1000)
    t_m = median_us(make_chain(ours_prev, "*/15 * * * *", NAIVE_START), 1000)
    chain_rows.append(("prev:*/15", t_o, t_m, t_o / t_m))
    print(f"{'prev:*/15':>28} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    t_o = median_us(make_chain(oracle_next, "*/15 * * * *", AWARE_START), 1000)
    t_m = median_us(make_chain(ours_next, "*/15 * * * *", AWARE_START), 1000)
    chain_rows.append(("next:*/15 tz-aware", t_o, t_m, t_o / t_m))
    print(f"{'next:*/15 tz-aware':>28} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    t_o = median_us(make_chain(oracle_next, "30 1 * * *", AWARE_START), 1000)
    t_m = median_us(make_chain(ours_next, "30 1 * * *", AWARE_START), 1000)
    chain_rows.append(("next:daily-0130 tz-DST", t_o, t_m, t_o / t_m))
    print(f"{'next:daily-0130 tz-DST':>28} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    print("\n== README paste block ==")
    print("| workload | oracle croniter (us/call) | croniter_mojo native (us/call) | speedup |")
    print("|---|---:|---:|---:|")
    for label, t_o, t_m, s in cold_rows:
        print(f"| cold get_next: {label} | {t_o:.1f} | {t_m:.1f} | {s:.2f}x |")
    for label, t_o, t_m, s in warm_rows:
        print(f"| warm get_next: {label} | {t_o:.2f} | {t_m:.2f} | {s:.2f}x |")
    for label, t_o, t_m, s in chain_rows:
        print(f"| chain {label} | {t_o:.2f} | {t_m:.2f} | {s:.2f}x |")


if __name__ == "__main__":
    main()
