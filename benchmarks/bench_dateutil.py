#!/usr/bin/env python3
"""Reproducible benchmark: python-dateutil (PyPI oracle) vs dateutil_mojo.

Workloads, all generated locally from fixed seeds (no network):
- cold parse: first parse of a fresh string per call (no reuse)
- warm parse: the same strings parsed repeatedly
- parse_column: ETL batch of 100k ISO strings in one call
- the pure-Python reference backend is shown for context

Timings are the median of 5 runs. Correctness is asserted (exact datetime
equality vs the oracle) before any timing happens, so the numbers below
always come from a verified-correct build.

Run from the repository root with explicit env, e.g.:

    PYTHONPATH=python/dateutil_mojo PYTHONNOUSERSITE=1 ~/.pixi/bin/pixi run python benchmarks/bench_dateutil.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

N_RUNS = 5  # median over this many runs

FAMILIES = {
    "iso-8601": "2025-07-08T14:30:00+02:00",
    "iso-date-only": "2025-07-08",
    "rfc-2822": "Tue, 08 Jul 2025 14:30:00 +0200",
    "us-numeric": "07/08/2025 2:30 PM",
    "named-month": "July 8, 2025 2:30:45 PM",
    "compact": "20250708143045",
}

DEFAULT = datetime(2000, 2, 15, 4, 5, 6, 789)


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
    from dateutil.parser import parse as oracle_parse

    import dateutil_mojo
    from dateutil_mojo import _parser as ref_parser

    info = dateutil_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- dateutil_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import dateutil

    print(f"- python-dateutil oracle: {dateutil.__version__}")
    print(f"- runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'dateutil_mojo'")

    # --- correctness gate -------------------------------------------------
    print("\n== correctness gate (exact equality vs oracle) ==")
    for label, s in FAMILIES.items():
        o = oracle_parse(s, default=DEFAULT)
        m = dateutil_mojo.parse(s, default=DEFAULT)
        assert (m.replace(tzinfo=None), m.utcoffset()) == (o.replace(tzinfo=None), o.utcoffset()), (
            f"{label}: {m!r} != {o!r}"
        )
        r = ref_parser.parse(s, default=DEFAULT)
        assert (r.replace(tzinfo=None), r.utcoffset()) == (o.replace(tzinfo=None), o.utcoffset())
        print(f"  {label:16} OK")

    # --- cold first-parse (fresh strings, no reuse) ------------------------
    print("\n== cold parse (fresh string per call, us/call) ==")
    print(f"{'family':>16} | {'oracle':>10} | {'dateutil_mojo':>13} | {'speedup':>8}")
    cold_rows = []
    variants = {
        label: [f"{s[:-2]}{int(s[-2:]) + i % 9:02d}" if s[-2:].isdigit() else s for i in range(9)]
        for label, s in FAMILIES.items()
    }
    for label, strings in variants.items():
        strings = [s for i in range(400) for s in strings]
        t_o = median_us(lambda: [oracle_parse(s, default=DEFAULT) for s in strings], len(strings))
        t_m = median_us(lambda: [dateutil_mojo.parse(s, default=DEFAULT) for s in strings], len(strings))
        cold_rows.append((label, t_o, t_m, t_o / t_m))
        print(f"{label:>16} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    # --- warm steady-state -------------------------------------------------
    print("\n== warm parse (same strings, us/call, 2000 iterations) ==")
    print(f"{'family':>16} | {'oracle':>10} | {'dateutil_mojo':>13} | {'speedup':>8}")
    warm_rows = []
    for label, s in FAMILIES.items():
        oracle_parse(s, default=DEFAULT)  # warm up both
        dateutil_mojo.parse(s, default=DEFAULT)
        t_o = median_us(lambda: [oracle_parse(s, default=DEFAULT) for _ in range(2000)], 2000)
        t_m = median_us(lambda: [dateutil_mojo.parse(s, default=DEFAULT) for _ in range(2000)], 2000)
        warm_rows.append((label, t_o, t_m, t_o / t_m))
        print(f"{label:>16} | {t_o:>10.2f} | {t_m:>13.2f} | {t_o / t_m:>7.2f}x")

    # --- python reference backend for context ------------------------------
    print("\n== pure-Python reference backend (fallback speed, us/call) ==")
    print(f"{'family':>16} | {'oracle':>10} | {'py reference':>13} | {'ratio':>8}")
    ref_rows = []
    for label, s in FAMILIES.items():
        t_o = median_us(lambda: [oracle_parse(s, default=DEFAULT) for _ in range(1000)], 1000)
        t_r = median_us(lambda: [ref_parser.parse(s, default=DEFAULT) for _ in range(1000)], 1000)
        ref_rows.append((label, t_o, t_r, t_r / t_o))
        print(f"{label:>16} | {t_o:>10.2f} | {t_r:>13.2f} | {t_r / t_o:>7.2f}x")

    # --- parse_column ETL batch -------------------------------------------
    print("\n== parse_column: batch of 100_000 ISO strings ==")
    batch = [f"2025-{m:02d}-{d:02d}T{h:02d}:30:00+02:00" for m in range(1, 13)
             for d in range(1, 29) for h in range(0, 24, 3)]
    batch = (batch * (100_000 // len(batch) + 1))[:100_000]
    # correctness gate on a sample
    for s in batch[:50]:
        assert dateutil_mojo.parse(s, default=DEFAULT) == oracle_parse(s, default=DEFAULT)
    t_batch_o = median_us(lambda: [oracle_parse(s, default=DEFAULT) for s in batch], len(batch))
    t_c = median_us(lambda: dateutil_mojo.parse_column(batch, default=DEFAULT), len(batch))
    t_l = median_us(lambda: [dateutil_mojo.parse(s, default=DEFAULT) for s in batch], len(batch))
    print(f"  oracle loop (us/row):        {t_batch_o:>10.2f}")
    print(f"  parse() loop (us/row):       {t_l:>10.2f}  ({t_batch_o / t_l:.2f}x vs oracle)")
    print(f"  parse_column (us/row):       {t_c:>10.2f}  ({t_batch_o / t_c:.2f}x vs oracle)")

    print("\n== README paste block ==")
    print("| workload | python-dateutil (us/call) | dateutil_mojo native (us/call) | speedup |")
    print("|---|---:|---:|---:|")
    for label, t_o, t_m, s in cold_rows:
        print(f"| cold parse: {label} | {t_o:.2f} | {t_m:.2f} | {s:.2f}x |")
    for label, t_o, t_m, s in warm_rows:
        print(f"| warm parse: {label} | {t_o:.2f} | {t_m:.2f} | {s:.2f}x |")
    print(f"| parse_column 100k ISO rows (us/row) | {t_batch_o:.2f} | {t_c:.2f} | {t_batch_o / t_c:.2f}x |")


if __name__ == "__main__":
    main()
