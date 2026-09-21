#!/usr/bin/env python3
"""Reproducible benchmark: msgpack (C backend and pure-Python fallback) vs
msgpack_mojo (native Mojo kernel and its own vendored fallback), on
packb/unpackb across nested structures.

Payloads are generated locally from fixed seeds (no network, no datasets):
a small ~0.5 KB API-style message, a ~64 KB mixed document, a ~2 MB mixed
document, plus three shape-isolating payloads (int array, string array, one
1 MB binary blob).

- warm steady-state: per-call latency, median of 5 runs (each run averages
  enough iterations for ~150 ms), per backend and payload.
- cold first-call: the very first packb()/unpackb() in a fresh Python
  process (includes the native library load + ABI handshake for
  msgpack_mojo), median of 5 processes.

Correctness is asserted (pack output byte-exact vs the oracle's C backend,
unpack values equal) before any timing happens, so the numbers below always
come from a verified-correct build. Speedups are reported against BOTH
oracle backends separately: the C extension (`msgpack._cmsgpack`, the
default) and the pure-Python `msgpack.fallback`.

Run from the repository root:

    PYTHONPATH="python/msgpack_mojo:.oracle-msgpack" PYTHONNOUSERSITE=1 \
        pixi run python benchmarks/bench_msgpack.py
"""

from __future__ import annotations

import math
import os
import platform
import random
import statistics
import string
import subprocess
import sys
import time

SEED = 20260920
N_RUNS = 5

# ---------------------------------------------------------------------------
# Payload generators (deterministic)
# ---------------------------------------------------------------------------


def _rand_text(rng: random.Random, n: int) -> str:
    pool = string.ascii_letters + string.digits + " .,;:!?-_"
    return "".join(rng.choice(pool) for _ in range(n))


def small_api() -> dict:
    rng = random.Random(SEED)
    return {
        "id": 987654,
        "method": "messages.list",
        "ok": True,
        "latency_ms": 12.75,
        "tags": ["alpha", "beta", "release-2026.09"],
        "params": {"limit": 50, "offset": 0, "filter": None},
        "trace": [rng.randrange(2**31) for _ in range(8)],
    }


def mixed_doc(target_bytes: int, seed: int) -> list:
    """A packed-size-targeted list of mixed records (maps of scalars)."""
    rng = random.Random(seed)
    records: list[dict] = []
    import msgpack  # local import: sizing only

    size = 16
    t = 0
    while size < target_bytes:
        t += 1
        rec = {
            f"name_{t}": _rand_text(rng, 16),
            f"int_{t}": rng.randrange(-(2**62), 2**62),
            f"float_{t}": rng.uniform(-1e4, 1e4),
            f"flag_{t}": rng.random() < 0.5,
            f"none_{t}": None,
            f"arr_{t}": [rng.randrange(0, 1000) for _ in range(rng.randrange(0, 9))],
            f"str_{t}": _rand_text(rng, rng.randrange(0, 48)),
            f"bin_{t}": bytes(rng.randrange(256) for _ in range(rng.randrange(0, 24))),
        }
        records.append(rec)
        size += len(msgpack.packb(rec))
    return records


def int_array() -> list:
    rng = random.Random(SEED + 3)
    return [rng.randrange(-(2**31), 2**31) for _ in range(100_000)]


def str_array() -> list:
    rng = random.Random(SEED + 4)
    return [_rand_text(rng, rng.randrange(4, 24)) for _ in range(20_000)]


def bin_blob() -> dict:
    rng = random.Random(SEED + 5)
    return {"kind": "blob", "data": bytes(rng.randrange(256) for _ in range(1 << 20))}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_warm(fn, arg, min_run_time: float = 0.15) -> float:
    """Median per-call seconds over N_RUNS runs of ~min_run_time each."""
    t0 = time.perf_counter()
    fn(arg)
    dt = time.perf_counter() - t0
    n = max(1, int(min_run_time / max(dt, 1e-6)))
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(arg)
        samples.append((time.perf_counter() - t0) / n)
    return statistics.median(samples)


def time_cold(kind: str, op: str, payload_name: str) -> float:
    """Median first-call latency across N_RUNS fresh Python processes."""
    samples = []
    for _ in range(N_RUNS):
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cold-one", kind, op, payload_name],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            check=True,
        )
        samples.append(float(r.stdout.strip()))
    return statistics.median(samples)


def _cold_one(kind: str, op: str, payload_name: str) -> None:
    """Child-process entry: import untimed, first call timed."""
    import msgpack
    from msgpack import fallback as fb

    import msgpack_mojo

    payloads = _payloads()
    obj, blob = payloads[payload_name]
    if kind == "oracle-c":
        fn = msgpack.packb if op == "pack" else msgpack.unpackb
    elif kind == "oracle-fallback":
        fn = fb.Packer().pack if op == "pack" else fb.unpackb
    elif kind == "mojo-fallback":
        os.environ["MSGPACK_MOJO_DISABLE_NATIVE"] = "1"
        fn = msgpack_mojo.packb if op == "pack" else msgpack_mojo.unpackb
    else:
        fn = msgpack_mojo.packb if op == "pack" else msgpack_mojo.unpackb
    arg = obj if op == "pack" else blob
    t0 = time.perf_counter()
    fn(arg)
    print(time.perf_counter() - t0)


def values_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b
    return a == b


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


def _payloads() -> dict:
    import msgpack

    objs = {
        "small_api (~0.5 KB)": small_api(),
        "medium (~64 KB)": mixed_doc(64 * 1024, SEED + 1),
        "large (~2 MB)": mixed_doc(2 * 1024 * 1024, SEED + 2),
        "int array (100k)": int_array(),
        "str array (20k)": str_array(),
        "bin blob (1 MB)": bin_blob(),
    }
    return {name: (obj, msgpack.packb(obj)) for name, obj in objs.items()}


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--cold-one":
        _cold_one(*sys.argv[2:])
        return

    import msgpack
    from msgpack import fallback as fb

    import msgpack_mojo

    info = msgpack_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- oracle: msgpack {'.'.join(map(str, msgpack.version))} (C backend + pure-Python fallback)")
    print(
        f"- msgpack_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- seeds: payloads from SEED={SEED}; runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'msgpack_mojo'")
    assert "_cmsgpack" in msgpack.Packer.__module__, "oracle must use its C backend"

    payloads = _payloads()

    print("\n== correctness gate (pack byte-exact vs oracle C backend; unpack values equal) ==")
    for name, (obj, blob) in payloads.items():
        ours_pack = msgpack_mojo.packb(obj)
        if ours_pack != blob:
            sys.exit(f"correctness gate failed (pack bytes differ) for {name}")
        if not values_equal(msgpack_mojo.unpackb(blob), msgpack.unpackb(blob)):
            sys.exit(f"correctness gate failed (unpack values differ) for {name}")
        fb_blob = fb.Packer().pack(obj)
        if fb_blob != blob:
            sys.exit(f"oracle fallback disagrees with its C backend for {name}")
        if not values_equal(fb.unpackb(blob), msgpack.unpackb(blob)):
            sys.exit(f"oracle fallback unpack disagrees with its C backend for {name}")
        print(f"  {name:>22}: {len(blob):>10,} packed bytes  [OK]")

    print("\n== cold first-call (fresh process, median of 5, import excluded) ==")
    kinds = ["oracle-c", "oracle-fallback", "mojo-native", "mojo-fallback"]
    for name in payloads:
        row = [f"{name:>22}"]
        for kind in kinds:
            ms = 1e3 * (time_cold(kind, "pack", name) + time_cold(kind, "unpack", name))
            row.append(f"{ms:>10.3f} ms")
        print("  " + " | ".join(row))
    print("  (columns: oracle C backend | oracle pure-Python fallback | msgpack_mojo native | msgpack_mojo fallback; pack+unpack first calls combined)")

    backends = [
        ("msgpack C (default)", None),
        ("msgpack fallback", None),
        ("msgpack_mojo native", None),
        ("msgpack_mojo fallback", None),
    ]
    readme_rows = []
    for op in ("pack", "unpack"):
        print(f"\n== warm steady-state {op}b (per call, median of 5 runs) ==")
        header = (
            f"{'payload':>22} | {'backend':>22} | {'ms/call':>10} | {'MB/s':>9}"
            f" | {'vs C':>8} | {'vs fallback':>11}"
        )
        print(header)
        print(f"{'-' * 22}-+-{'-' * 22}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 11}")
        for name, (obj, blob) in payloads.items():
            arg = obj if op == "pack" else blob
            base_c = base_fb = None
            for label in [b[0] for b in backends]:
                if label == "msgpack C (default)":
                    fn = msgpack.packb if op == "pack" else msgpack.unpackb
                    sec = time_warm(fn, arg)
                    base_c = sec
                elif label == "msgpack fallback":
                    if op == "pack":
                        packer = fb.Packer()
                        fn = packer.pack
                    else:
                        fn = fb.unpackb
                    sec = time_warm(fn, arg)
                    base_fb = sec
                elif label == "msgpack_mojo native":
                    fn = msgpack_mojo.packb if op == "pack" else msgpack_mojo.unpackb
                    sec = time_warm(fn, arg)
                else:
                    os.environ["MSGPACK_MOJO_DISABLE_NATIVE"] = "1"
                    try:
                        fn = msgpack_mojo.packb if op == "pack" else msgpack_mojo.unpackb
                        sec = time_warm(fn, arg)
                    finally:
                        del os.environ["MSGPACK_MOJO_DISABLE_NATIVE"]
                ms = sec * 1e3
                mbps = len(blob) / sec / 1e6
                vs_c = base_c / sec if base_c else 1.0
                vs_fb = base_fb / sec if base_fb else 1.0
                print(
                    f"{name:>22} | {label:>22} | {ms:>10.4f} | {mbps:>9.1f}"
                    f" | {vs_c:>7.2f}x | {vs_fb:>10.2f}x"
                )
                readme_rows.append((op, name, label, ms, mbps, vs_c, vs_fb))

    print("\n== README paste block ==")
    print("| op | payload | backend | ms/call | MB/s | vs oracle C | vs oracle fallback |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for op, name, label, ms, mbps, vs_c, vs_fb in readme_rows:
        print(f"| {op}b | {name} | {label} | {ms:.4f} | {mbps:.1f} | {vs_c:.2f}x | {vs_fb:.2f}x |")


if __name__ == "__main__":
    main()
