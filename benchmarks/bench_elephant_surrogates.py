#!/usr/bin/env python3
"""Reproducible benchmark: elephant (pip oracle) vs elephant_mojo.

Three cells mirror the accelerated paths:

- ``plain dither``: SPADE's surrogate generation — elephant's
  ``spade._generate_binned_surrogates`` (method ``dither_spikes``, the
  spade default; dither + bin into a BinnedSpikeTrain per surrogate) vs
  ``elephant_mojo.dither``. 10 trains x 500 spikes x 200 surrogates.
- ``refractory dither``: elephant's ``dither_spikes(refractory_period=..)``
  (a pure-Python loop over surrogates x spikes) + the same binning vs
  ``elephant_mojo.dither(method='dither_spikes_with_refractory_period')``.
  5 trains x 500 spikes x 50 surrogates.
- ``pvalue spectrum``: elephant's ``spade._get_pvalue_spec`` vs
  ``elephant_mojo.pvalue_spectrum`` on a (2000, 20) maximal-occurrence
  matrix (occurrences up to 300 -> ~6000 spectrum entries).

Every cell is timed two ways:

- **cold**: first call in a fresh interpreter (import + dlopen + first
  touch), measured inside a spawned subprocess;
- **warm**: median of 5 steady-state calls in this process.

Correctness is gated before timing: the dither cells assert survivor-rate
agreement within 1% and shape equality against the oracle (the full
statistical gate is the differential suite), and the spectrum cell
asserts bit-exact equality. Timings are single-threaded wall clock.

Run from the repository root:

    PYTHONPATH=python/elephant_mojo pixi run python benchmarks/bench_elephant_surrogates.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

N_RUNS = 5
SEED = 20260919

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Workload constants.
PLAIN_TRAINS, PLAIN_SPIKES, PLAIN_SURR = 10, 500, 200
REFR_TRAINS, REFR_SPIKES, REFR_SURR = 5, 500, 50
SPEC_ROWS, SPEC_SIZES, SPEC_MAX_OCC = 2000, 20, 300
T_START, T_STOP, BIN_SIZE, DITHER, REFR = 0.0, 1000.0, 5.0, 15.0, 5.0


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


def make_trains(n_trains: int, n_spikes: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [np.sort(rng.uniform(T_START, T_STOP, n_spikes)) for _ in range(n_trains)]


def make_max_occs() -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    return rng.integers(0, SPEC_MAX_OCC, size=(SPEC_ROWS, SPEC_SIZES)).astype(np.float64)


# ---------------------------------------------------------------------------
# Oracle (elephant) paths — measured exactly the way spade drives them.
# ---------------------------------------------------------------------------


def oracle_plain_dither(trains) -> np.ndarray:
    import quantities as pq

    import elephant.spade as espade

    sts = [
        __import__("neo").SpikeTrain(t * pq.ms, t_start=T_START * pq.ms, t_stop=T_STOP * pq.ms)
        for t in trains
    ]
    out = [
        bst.to_bool_array()
        for _, bst in espade._generate_binned_surrogates(
            sts,
            bin_size=BIN_SIZE * pq.ms,
            dither=DITHER * pq.ms,
            surr_method="dither_spikes",
            n_surrogates=PLAIN_SURR,
        )
    ]
    return np.stack(out)


def oracle_refractory_dither(trains) -> np.ndarray:
    import quantities as pq
    import neo
    from elephant.spike_train_surrogates import dither_spikes

    n_bins = int((T_STOP - T_START) / BIN_SIZE)
    out = np.zeros((REFR_SURR, len(trains), n_bins), dtype=bool)
    for i, t in enumerate(trains):
        st = neo.SpikeTrain(t * pq.ms, t_start=T_START * pq.ms, t_stop=T_STOP * pq.ms)
        for s in range(REFR_SURR):
            sur = np.asarray(
                dither_spikes(st, dither=DITHER * pq.ms, refractory_period=REFR * pq.ms)[
                    0
                ].magnitude
            )
            b = ((sur - T_START) / BIN_SIZE).astype(np.int64)
            out[s, i, b[(b >= 0) & (b < n_bins)]] = True
    return out


def oracle_pvalue_spec(max_occs: np.ndarray):
    import elephant.spade as espade

    return espade._get_pvalue_spec(max_occs.copy(), 2, 2 + SPEC_SIZES - 1, 2, SPEC_ROWS, 1, "#")


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def time_median(fn, n_runs: int = N_RUNS) -> float:
    fn()  # warmup (also the steady-state verifier)
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def time_cold(snippet: str, env_extra: dict | None = None) -> float:
    """First-call latency in a fresh interpreter (import + dlopen + call)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "python", "elephant_mojo") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    if env_extra:
        env.update(env_extra)
    code = (
        "import time, numpy as np\n"
        "t0 = time.perf_counter()\n"
        + snippet
        + "\nprint(f'{time.perf_counter() - t0:.6f}')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        cwd=REPO_ROOT,
    )
    return float(out.stdout.strip().splitlines()[-1])


def _mojo_snippet(call: str) -> str:
    return (
        "import elephant_mojo\n"
        f"{call}\n"
    )


def main() -> None:
    import elephant_mojo
    from elephant_mojo import dither, pvalue_spectrum

    info = elephant_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- elephant_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- seed: {SEED}; warm timings: median of {N_RUNS}; cold: first call in a fresh process")
    if not info["native_available"]:
        sys.exit(
            "native kernel unavailable; refusing to benchmark the fallback as 'elephant_mojo'"
        )

    import elephant  # noqa: F401  (oracle presence check)

    plain_trains = make_trains(PLAIN_TRAINS, PLAIN_SPIKES)
    refr_trains = plain_trains[:REFR_TRAINS]
    max_occs = make_max_occs()
    n_bins = int((T_STOP - T_START) / BIN_SIZE)

    def our_plain():
        return dither(
            plain_trains, BIN_SIZE, DITHER, n_surrogates=PLAIN_SURR,
            t_start=T_START, t_stop=T_STOP, seed=SEED,
        )

    def our_refr():
        return dither(
            refr_trains, BIN_SIZE, DITHER, n_surrogates=REFR_SURR,
            t_start=T_START, t_stop=T_STOP,
            method="dither_spikes_with_refractory_period",
            refractory_period=REFR, seed=SEED,
        )

    def our_spec():
        return pvalue_spectrum(
            max_occs, 2, 2 + SPEC_SIZES - 1, 2, n_surr=SPEC_ROWS, spectrum="#"
        )

    print("\n== correctness gate ==")
    np.random.seed(SEED)
    ora_plain = oracle_plain_dither(plain_trains)
    our_p = our_plain()
    assert ora_plain.shape == our_p.shape == (PLAIN_SURR, PLAIN_TRAINS, n_bins)
    rate_o = ora_plain.sum() / ora_plain.size
    rate_m = our_p.sum() / our_p.size
    assert abs(rate_o - rate_m) / rate_o < 0.01, (rate_o, rate_m)
    print(f"  plain dither: shape {our_p.shape}, survivor-rate diff {abs(rate_o - rate_m):.2e} [OK]")

    np.random.seed(SEED)
    ora_refr = oracle_refractory_dither(refr_trains)
    our_r = our_refr()
    assert ora_refr.shape == our_r.shape == (REFR_SURR, REFR_TRAINS, n_bins)
    rate_o = ora_refr.sum() / ora_refr.size
    rate_m = our_r.sum() / our_r.size
    assert abs(rate_o - rate_m) / rate_o < 0.01, (rate_o, rate_m)
    print(f"  refractory dither: shape {our_r.shape}, survivor-rate diff {abs(rate_o - rate_m):.2e} [OK]")

    spec_ref = oracle_pvalue_spec(max_occs)
    spec_ours = our_spec()
    assert [[int(c) for c in e[:-1]] + [float(e[-1])] for e in spec_ref] == spec_ours
    print(f"  pvalue spectrum: {len(spec_ours)} entries, bit-exact [OK]")

    rows: list[tuple[str, float, float, float, float]] = []  # label, oracle warm, ours cold, ours warm

    print("\n== warm steady-state (median of 5, seconds) ==")
    print(f"{'cell':>18} | {'elephant oracle':>15} | {'elephant_mojo':>13} | {'speedup':>8}")
    print(f"{'-' * 18}-+-{'-' * 15}-+-{'-' * 13}-+-{'-' * 8}")
    for label, oracle_fn, our_fn in [
        ("plain dither", lambda: oracle_plain_dither(plain_trains), our_plain),
        ("refractory dither", lambda: oracle_refractory_dither(refr_trains), our_refr),
        ("pvalue spectrum", lambda: oracle_pvalue_spec(max_occs), our_spec),
    ]:
        t_oracle = time_median(oracle_fn)
        t_ours = time_median(our_fn)
        rows.append((label, t_oracle, 0.0, t_ours))
        print(f"{label:>18} | {t_oracle:>15.4f} | {t_ours:>13.6f} | {t_oracle / t_ours:>7.1f}x")

    print("\n== cold first-call (fresh interpreter, seconds) ==")
    plain_call = (
        f"trains = [np.sort(np.random.default_rng({SEED}).uniform(0.0, 1000.0, {PLAIN_SPIKES})) for _ in range({PLAIN_TRAINS})]\n"
        f"elephant_mojo.dither(trains, {BIN_SIZE}, {DITHER}, n_surrogates={PLAIN_SURR}, t_stop=1000.0, seed={SEED})"
    )
    refr_call = (
        f"trains = [np.sort(np.random.default_rng({SEED}).uniform(0.0, 1000.0, {REFR_SPIKES})) for _ in range({REFR_TRAINS})]\n"
        f"elephant_mojo.dither(trains, {BIN_SIZE}, {DITHER}, n_surrogates={REFR_SURR}, t_stop=1000.0, method='dither_spikes_with_refractory_period', refractory_period={REFR}, seed={SEED})"
    )
    spec_call = (
        f"mo = np.random.default_rng({SEED + 1}).integers(0, {SPEC_MAX_OCC}, size=({SPEC_ROWS}, {SPEC_SIZES})).astype(float)\n"
        f"elephant_mojo.pvalue_spectrum(mo, 2, {2 + SPEC_SIZES - 1}, 2, n_surr={SPEC_ROWS})"
    )
    colds = []
    for label, snippet in [
        ("plain dither", plain_call),
        ("refractory dither", refr_call),
        ("pvalue spectrum", spec_call),
    ]:
        t_cold = time_cold(_mojo_snippet(snippet))
        colds.append(t_cold)
        print(f"  {label:>18}: {t_cold:.4f} s")
    rows = [(label, t_o, t_c, t_w) for (label, t_o, _, t_w), t_c in zip(rows, colds)]

    # Fallback context row (smallest meaningful cells): what non-native platforms get.
    os.environ["ELEPHANT_MOJO_DISABLE_NATIVE"] = "1"
    try:
        import elephant_mojo._native as nat

        nat._LIB, nat._LIB_SOURCE = None, None
        fb_plain = time_median(our_plain)
        fb_spec = time_median(our_spec)
        fb_refr = time_median(our_refr)
    finally:
        del os.environ["ELEPHANT_MOJO_DISABLE_NATIVE"]
        nat._LIB, nat._LIB_SOURCE = None, None
    print(f"\n== NumPy fallback context (warm, median of 5, seconds) ==")
    print(f"  plain dither:      {fb_plain:.4f}")
    print(f"  refractory dither: {fb_refr:.4f}")
    print(f"  pvalue spectrum:   {fb_spec:.6f}")

    print("\n== README paste block ==")
    print("| workload | elephant 1.2.1 warm (s) | elephant-mojo cold (s) | elephant-mojo warm (s) | warm speedup |")
    print("|---|---:|---:|---:|---:|")
    for label, t_o, t_c, t_w in rows:
        print(f"| {label} | {t_o:.3f} | {t_c:.4f} | {t_w:.6f} | {t_o / t_w:.0f}x |")
    print(f"| plain dither — NumPy fallback | {rows[0][1]:.3f} | — | {fb_plain:.4f} | {rows[0][1] / fb_plain:.1f}x |")
    print(f"| refractory dither — NumPy fallback | {rows[1][1]:.3f} | — | {fb_refr:.4f} | {rows[1][1] / fb_refr:.1f}x |")
    print(f"| pvalue spectrum — NumPy fallback | {rows[2][1]:.3f} | — | {fb_spec:.6f} | {rows[2][1] / fb_spec:.1f}x |")


if __name__ == "__main__":
    main()
