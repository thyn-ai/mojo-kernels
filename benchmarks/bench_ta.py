#!/usr/bin/env python3
"""Reproducible benchmark: pandas-ta-classic (oracle) vs ta_mojo.

Series are generated locally from fixed seeds (no network, no datasets):
random-walk closes (plus consistent high/low for ATR) of 1k / 10k / 100k / 1M
bars. Two measurements per indicator:

* cold first-call — median of 5 fresh-interpreter first calls at n=100k
  (imports happen before the timer; for ta_mojo this includes dlopen +
  ABI handshake of the native kernel),
* warm steady-state — per-call latency, median of 5 batches of repeated
  calls at each size.

Correctness is asserted (element-wise vs the oracle within 1e-10, exact NaN
masks) before any timing happens, so the numbers below always come from a
verified-correct build.

Run from the repository root:

    PYTHONPATH=python/ta_mojo pixi run python benchmarks/bench_ta.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time

import numpy as np

SIZES = [1_000, 10_000, 100_000, 1_000_000]
COLD_SIZE = 100_000
N_RUNS = 5  # median over this many batches / fresh processes
BATCH = 20  # calls per warm batch
SEED = 20260919
ATOL = 1e-10

INDICATORS = ["ema", "rsi", "atr", "macd"]


def make_inputs(seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    high = close + rng.uniform(0.0, 1.5, n)
    low = close - rng.uniform(0.0, 1.5, n)
    return {"close": close, "high": high, "low": low}


def call_ours(mod, name: str, inputs: dict) -> None:
    if name == "ema":
        mod.ema(inputs["close"], 10)
    elif name == "rsi":
        mod.rsi(inputs["close"], 14)
    elif name == "atr":
        mod.atr(inputs["high"], inputs["low"], inputs["close"], 14)
    else:
        mod.macd(inputs["close"], 12, 26, 9)


def call_oracle(ta, pd, name: str, series: dict) -> None:
    if name == "ema":
        ta.ema(series["close"], 10)
    elif name == "rsi":
        ta.rsi(series["close"], 14)
    elif name == "atr":
        ta.atr(series["high"], series["low"], series["close"], 14)
    else:
        ta.macd(series["close"], fast=12, slow=26, signal=9)


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


def cold_first_call(name: str, ours: bool) -> float:
    """Median first-call latency (ms) over N_RUNS fresh interpreter processes."""
    if ours:
        snippet = (
            "import time,numpy as np,sys;sys.path.insert(0,'python/ta_mojo');"
            "import ta_mojo as m;"
            "r=np.random.default_rng(%d);c=100+np.cumsum(r.normal(0,1,%d));"
            "h=c+r.uniform(0,1.5,%d);l=c-r.uniform(0,1.5,%d);"
            "i=dict(close=c,high=h,low=l);"
            "t0=time.perf_counter();"
            "getattr(m,'%s')(c,10) if '%s'=='ema' else "
            "(m.rsi(c,14) if '%s'=='rsi' else "
            "(m.atr(h,l,c,14) if '%s'=='atr' else m.macd(c,12,26,9)));"
            "print(1e3*(time.perf_counter()-t0))"
        ) % (SEED, COLD_SIZE, COLD_SIZE, COLD_SIZE, name, name, name, name)
    else:
        snippet = (
            "import time,numpy as np,pandas as pd,pandas_ta_classic as ta;"
            "r=np.random.default_rng(%d);c=100+np.cumsum(r.normal(0,1,%d));"
            "h=c+r.uniform(0,1.5,%d);l=c-r.uniform(0,1.5,%d);"
            "cs,hs,ls=pd.Series(c),pd.Series(h),pd.Series(l);"
            "t0=time.perf_counter();"
            "ta.ema(cs,10) if '%s'=='ema' else "
            "(ta.rsi(cs,14) if '%s'=='rsi' else "
            "(ta.atr(hs,ls,cs,14) if '%s'=='atr' else ta.macd(cs,fast=12,slow=26,signal=9)));"
            "print(1e3*(time.perf_counter()-t0))"
        ) % (SEED, COLD_SIZE, COLD_SIZE, COLD_SIZE, name, name, name)
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


def warm_latency(call, n_calls: int = BATCH) -> float:
    """Median per-call latency (ms) over N_RUNS batches of n_calls calls."""
    call()  # one warmup batch element outside the timer
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            call()
        samples.append((time.perf_counter() - t0) / n_calls)
    return 1e3 * statistics.median(samples)


def main() -> None:
    import pandas as pd
    import pandas_ta_classic as ta

    import ta_mojo

    info = ta_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    import importlib.metadata

    print(f"- oracle: pandas-ta-classic {importlib.metadata.version('pandas-ta-classic')} "
          f"(pandas {pd.__version__}, pandas code path: {not ta.Imports['talib']})")
    print(f"- ta_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    print(f"- seed: {SEED}; warm: median of {N_RUNS} batches x {BATCH} calls; "
          f"cold: median of {N_RUNS} fresh-process first calls at n={COLD_SIZE:,}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'ta_mojo'")

    print("\n== correctness gate (element-wise vs pandas-ta-classic) ==")
    inputs = make_inputs(SEED + 1, 20_000)
    series = {k: pd.Series(v) for k, v in inputs.items()}
    for name in INDICATORS:
        if name == "macd":
            ours = ta_mojo.macd(inputs["close"], 12, 26, 9)
            ref_df = ta.macd(series["close"], fast=12, slow=26, signal=9)
            pairs = [
                (ours.macd, ref_df["MACD_12_26_9"]),
                (ours.signal, ref_df["MACDs_12_26_9"]),
                (ours.histogram, ref_df["MACDh_12_26_9"]),
            ]
        elif name == "ema":
            pairs = [(ta_mojo.ema(inputs["close"], 10), ta.ema(series["close"], 10))]
        elif name == "rsi":
            pairs = [(ta_mojo.rsi(inputs["close"], 14), ta.rsi(series["close"], 14))]
        else:
            pairs = [
                (
                    ta_mojo.atr(inputs["high"], inputs["low"], inputs["close"], 14),
                    ta.atr(series["high"], series["low"], series["close"], 14),
                )
            ]
        worst = 0.0
        for ours_arr, ref_arr in pairs:
            ref_np = np.asarray(ref_arr, dtype=np.float64)
            assert np.array_equal(np.isnan(ours_arr), np.isnan(ref_np)), f"{name}: NaN mask mismatch"
            valid = ~np.isnan(ref_np)
            if valid.any():
                worst = max(worst, float(np.max(np.abs(ours_arr[valid] - ref_np[valid]))))
        status = "OK" if worst <= ATOL else "FAIL"
        print(f"  {name:>5}: max|diff| = {worst:.3e}  [{status}]")
        if worst > ATOL:
            sys.exit(f"correctness gate failed for {name}")

    print("\n== cold first-call latency (ms), n=100,000, median of 5 fresh processes ==")
    print(f"{'indicator':>10} | {'pandas-ta-classic':>18} | {'ta_mojo':>10} | {'speedup':>8}")
    print(f"{'-' * 10}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 8}")
    cold_rows = []
    for name in INDICATORS:
        c_ref = cold_first_call(name, ours=False)
        c_ours = cold_first_call(name, ours=True)
        cold_rows.append((name, c_ref, c_ours, c_ref / c_ours))
        print(f"{name:>10} | {c_ref:>18.3f} | {c_ours:>10.4f} | {c_ref / c_ours:>7.1f}x")

    print("\n== warm steady-state latency (ms/call), median of 5 batches ==")
    header = (
        f"{'indicator':>10} | {'bars':>10} | {'pandas-ta-classic':>18} | "
        f"{'ta_mojo':>10} | {'speedup':>8}"
    )
    print(header)
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 18}-+-{'-' * 10}-+-{'-' * 8}")
    warm_rows = []
    for name in INDICATORS:
        for n in SIZES:
            inputs = make_inputs(SEED + n, n)
            series = {k: pd.Series(v) for k, v in inputs.items()}
            t_ref = warm_latency(lambda: call_oracle(ta, pd, name, series))
            t_ours = warm_latency(lambda: call_ours(ta_mojo, name, inputs))
            warm_rows.append((name, n, t_ref, t_ours, t_ref / t_ours))
            print(f"{name:>10} | {n:>10,} | {t_ref:>18.3f} | {t_ours:>10.4f} "
                  f"| {t_ref / t_ours:>7.1f}x")

    print("\n== README paste block ==")
    print("Cold first call (n=100,000 bars, median of 5 fresh processes):")
    print()
    print("| indicator | pandas-ta-classic (ms) | ta-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|")
    for name, c_ref, c_ours, speedup in cold_rows:
        print(f"| {name} | {c_ref:.3f} | {c_ours:.4f} | {speedup:.1f}x |")
    print()
    print("Warm steady state (ms per call, median of 5 batches of 20 calls):")
    print()
    print("| indicator | bars | pandas-ta-classic (ms) | ta-mojo (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, n, t_ref, t_ours, speedup in warm_rows:
        print(f"| {name} | {n:,} | {t_ref:.3f} | {t_ours:.4f} | {speedup:.1f}x |")


if __name__ == "__main__":
    main()
