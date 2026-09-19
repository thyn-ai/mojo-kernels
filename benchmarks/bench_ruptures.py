#!/usr/bin/env python3
"""Reproducible benchmark: ruptures vs ruptures_mojo (Dynp / Pelt / Binseg).

Signals are generated locally from fixed seeds (no network, no datasets):
piecewise-constant mean + Gaussian noise at several lengths. Each workload is
one whole detection call (fit_predict / detect), because that is the unit an
end user runs.

Both temperatures are reported, per the program's measurement rules:
- cold: median of 5 first-call times, each in a fresh interpreter process
  (includes dynamic-library load and ABI handshake on our side, estimator
  setup on the oracle side);
- warm: median of 5 repeated calls in a live process (steady state).

Correctness is asserted (exact breakpoint-list equality with ruptures) before
any timing happens, so the numbers below always come from a verified-correct
build. Timings are wall-clock medians; the oracle is PyPI ruptures==1.1.10.

Run from the repository root:

    PYTHONPATH=python/ruptures_mojo pixi run python benchmarks/bench_ruptures.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

N_RUNS = 5  # median over this many runs, both temperatures
SEED = 20260919

# Workloads: (method, n, call kwargs, min_size, jump, label).
WORKLOADS = [
    ("dynp", 1_000, {"n_bkps": 3}, 2, 1, "Dynp  n=1,000   K=3  jump=1"),
    ("dynp", 2_000, {"n_bkps": 5}, 2, 2, "Dynp  n=2,000   K=5  jump=2"),
    ("pelt", 10_000, {"pen": 10.0}, 2, 1, "PELT  n=10,000  pen=10 jump=1"),
    ("pelt", 20_000, {"pen": 10.0}, 2, 5, "PELT  n=20,000  pen=10 jump=5"),
    ("binseg", 50_000, {"n_bkps": 20}, 2, 1, "Binseg n=50,000  K=20 jump=1"),
    ("binseg", 100_000, {"n_bkps": 20}, 2, 5, "Binseg n=100,000 K=20 jump=5"),
]


def make_signal(n: int, seed: int) -> np.ndarray:
    """Piecewise-constant mean (6 random shifts) + unit Gaussian noise."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    val, prev = 0.0, 0
    for p in sorted(rng.choice(np.arange(n // 10, n - n // 10), size=6, replace=False)):
        s[prev:p] = val
        val += rng.standard_normal() * 3.0
        prev = p
    s[prev:] = val
    return s + rng.standard_normal(n)


def oracle_detect(rpt, method, sig, kw, min_size, jump):
    est_cls = {"dynp": rpt.Dynp, "pelt": rpt.Pelt, "binseg": rpt.Binseg}[method]
    est = est_cls(model="l2", min_size=min_size, jump=jump)
    if method == "pelt":
        return est.fit_predict(sig, kw["pen"])
    return est.fit_predict(sig, kw["n_bkps"])


def time_warm(fn, reps=N_RUNS) -> float:
    fn()  # one untimed warm-up call
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_cold(impl: str, method: str, n: int, kw: dict, min_size: int, jump: int) -> float:
    """Median first-call time in a fresh interpreter (import excluded)."""
    payload = json.dumps(
        {"impl": impl, "method": method, "n": n, "kw": kw, "min_size": min_size, "jump": jump, "seed": SEED + n}
    )
    runner = (
        "import json, sys, time\n"
        "import numpy as np\n"
        "sys.path.insert(0, 'benchmarks')\n"
        "from bench_ruptures import make_signal\n"
        "cfg = json.loads(sys.argv[1])\n"
        "sig = make_signal(cfg['n'], cfg['seed'])\n"
        "if cfg['impl'] == 'oracle':\n"
        "    import ruptures as rpt\n"
        "    from bench_ruptures import oracle_detect\n"
        "    t0 = time.perf_counter()\n"
        "    oracle_detect(rpt, cfg['method'], sig, cfg['kw'], cfg['min_size'], cfg['jump'])\n"
        "else:\n"
        "    import ruptures_mojo\n"
        "    t0 = time.perf_counter()\n"
        "    ruptures_mojo.detect(sig, cfg['method'], model='l2', min_size=cfg['min_size'],\n"
        "                         jump=cfg['jump'], **cfg['kw'])\n"
        "print(f'{time.perf_counter() - t0:.6f}')\n"
    )
    samples = []
    for _ in range(N_RUNS):
        # A fresh process per sample; retry once on transient failures
        # (shared-CI-machine hiccups), surfacing stderr if it persists.
        last_exc: subprocess.CalledProcessError | None = None
        for _attempt in range(2):
            try:
                out = subprocess.run(
                    [sys.executable, "-c", runner, payload],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                last_exc = None
                break
            except subprocess.CalledProcessError as exc:
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(
                f"cold-call subprocess failed twice: {last_exc}\n{last_exc.stderr}"
            )
        samples.append(float(out.stdout.strip()))
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
    import ruptures as rpt

    import ruptures_mojo

    info = ruptures_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- ruptures_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- ruptures (oracle): {rpt.__version__}")
    print(f"- seed: {SEED}+n; runs: median of {N_RUNS} (cold: fresh processes; warm: live process)")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'ruptures_mojo'")

    print("\n== correctness gate (exact breakpoint-list equality vs ruptures) ==")
    for method, n, kw, ms, jp, label in WORKLOADS:
        sig = make_signal(n, SEED + n)
        want = oracle_detect(rpt, method, sig, kw, ms, jp)
        got = ruptures_mojo.detect(sig, method, model="l2", min_size=ms, jump=jp, **kw)
        status = "OK" if got == list(want) else "FAIL"
        print(f"  {label}: {status}")
        if got != list(want):
            sys.exit(f"correctness gate failed for {label}: {got} != {want}")

    print("\n== warm steady-state (median of 5, live process) ==")
    header = f"{'workload':>28} | {'ruptures ms':>12} | {'ruptures_mojo ms':>16} | {'speedup':>8}"
    print(header)
    print(f"{'-' * 28}-+-{'-' * 12}-+-{'-' * 16}-+-{'-' * 8}")
    warm_rows = []
    for method, n, kw, ms, jp, label in WORKLOADS:
        sig = make_signal(n, SEED + n)
        t_ref = time_warm(lambda: oracle_detect(rpt, method, sig, kw, ms, jp))
        t_ours = time_warm(
            lambda: ruptures_mojo.detect(sig, method, model="l2", min_size=ms, jump=jp, **kw)
        )
        warm_rows.append((label, t_ref, t_ours))
        print(
            f"{label:>28} | {1e3 * t_ref:>12.2f} | {1e3 * t_ours:>16.3f} | {t_ref / t_ours:>7.1f}x"
        )

    print("\n== cold first call (median of 5 fresh processes) ==")
    print(header)
    print(f"{'-' * 28}-+-{'-' * 12}-+-{'-' * 16}-+-{'-' * 8}")
    cold_rows = []
    for method, n, kw, ms, jp, label in WORKLOADS:
        t_ref = time_cold("oracle", method, n, kw, ms, jp)
        t_ours = time_cold("mojo", method, n, kw, ms, jp)
        cold_rows.append((label, t_ref, t_ours))
        print(
            f"{label:>28} | {1e3 * t_ref:>12.2f} | {1e3 * t_ours:>16.3f} | {t_ref / t_ours:>7.1f}x"
        )

    print("\n== README paste block ==")
    print("| workload | ruptures warm (ms) | ruptures_mojo warm (ms) | warm speedup | ruptures cold (ms) | ruptures_mojo cold (ms) | cold speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for (label, t_ref_w, t_ours_w), (_, t_ref_c, t_ours_c) in zip(warm_rows, cold_rows):
        print(
            f"| {label.strip()} | {1e3 * t_ref_w:.2f} | {1e3 * t_ours_w:.3f} | {t_ref_w / t_ours_w:.1f}x "
            f"| {1e3 * t_ref_c:.2f} | {1e3 * t_ours_c:.3f} | {t_ref_c / t_ours_c:.1f}x |"
        )


if __name__ == "__main__":
    main()
