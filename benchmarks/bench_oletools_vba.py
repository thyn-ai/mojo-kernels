#!/usr/bin/env python3
"""Reproducible benchmark: oletools (oracle) vs oletools-mojo.

Workloads are generated locally from fixed seeds (no network, no datasets)
and compressed with the fresh MS-OVBA compressor in
tests/test_oletools_vba_fixtures.py:

  * realistic VBA macro source at two sizes (~210 KB, ~2.1 MB);
  * text-like 4 MiB (mixed literals and copy tokens);
  * RLE 8 MiB (copy-token-dense decode, overlapping offset-2 copies);
  * incompressible-random 1 MiB (RawChunk path);
  * mixed 2 MiB (compressible and incompressible 64 KiB stripes).

Note: matches can never span the 4096-byte chunk boundary (the decoder's
window is chunk-local), so large-scale repetition still lands in RawChunks
when individual chunks are incompressible — exactly what a real producer
emits. The workloads above are shaped to exercise each decode path.

Both sides are timed END TO END on the same CompressedContainer bytes:
`oletools.olevba.decompress_stream` vs `oletools_mojo.decompress_stream`.

Protocol per (workload, implementation): cold = the FIRST decompress call
for that workload (the first workload's cold also carries the one-time
dlopen/ABI handshake on the native side); the cold call doubles as the
correctness gate (its result must be byte-equal to the other side's), then
warm = median of 5 more runs. Numbers therefore always come from a
verified-correct build. Run from the repository root:

    PYTHONPATH=".oracle-oletools:python/oletools_mojo" pixi run python benchmarks/bench_oletools_vba.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from test_oletools_vba_fixtures import (  # noqa: E402
    OLETOOLS_VERSION,
    rng_bytes,
    vba_compress,
    vba_like_source,
)

N_RUNS = 5  # warm runs; median reported
SEED = 20260919


def workloads():
    rng = random.Random(SEED)
    out = []

    vba_small = vba_like_source(random.Random(SEED + 1), 3000)
    out.append(("VBA source ~210 KB", vba_small))

    vba_large = vba_like_source(random.Random(SEED + 2), 30000)
    out.append(("VBA source ~2.1 MB", vba_large))

    # Letter words from a small vocabulary: 3-byte keys repeat inside every
    # 4096-byte chunk, so the LZ path (literals + copy tokens) dominates.
    words = [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(2, 9))) for _ in range(300)]
    text = b" ".join(rng.choice(words) for _ in range(700_000))[: 4 * 1024 * 1024]
    out.append(("text-like 4 MiB", text))

    # Maximal copy density: offset-2 overlapping copies inside every chunk.
    rle = b"ab" * (4 * 1024 * 1024)
    out.append(("RLE 8 MiB", rle))

    rand = rng_bytes(rng, 1024 * 1024)
    out.append(("random 1 MiB (RawChunks)", rand))

    stripes = []
    for i in range(32):
        if i % 2 == 0:
            stripes.append(rng_bytes(rng, 65536))
        else:
            stripes.append(vba_like_source(random.Random(SEED + 100 + i), 1200))
    mixed = b"".join(stripes)[: 2 * 1024 * 1024]
    out.append(("mixed 2 MiB", mixed))
    return out


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


def measure(decode_oracle, decode_ours, n_warm: int = N_RUNS):
    """Cold calls double as the correctness gate; then warm medians."""
    t0 = time.perf_counter()
    result_oracle = decode_oracle()
    cold_o = time.perf_counter() - t0
    t0 = time.perf_counter()
    result_ours = decode_ours()
    cold_m = time.perf_counter() - t0
    assert result_ours == result_oracle, "correctness gate failed (byte mismatch)"
    warm_o_samples, warm_m_samples = [], []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        decode_oracle()
        warm_o_samples.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        decode_ours()
        warm_m_samples.append(time.perf_counter() - t0)
    return cold_o, statistics.median(warm_o_samples), cold_m, statistics.median(warm_m_samples)


def main() -> None:
    import oletools_mojo
    from oletools.olevba import decompress_stream as oracle_decompress

    info = oletools_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- oletools (oracle): {OLETOOLS_VERSION}")
    print(
        f"- oletools_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- protocol: cold first call (doubles as correctness gate), warm = median of {N_RUNS}; seed={SEED}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'oletools_mojo'")

    rows = []
    print("\n== MS-OVBA CompressedContainer decode (same bytes both sides) ==")
    for name, payload in workloads():
        container = vba_compress(payload)

        def decode_oracle(container=container):
            return oracle_decompress(container)

        def decode_ours(container=container):
            return oletools_mojo.decompress_stream(container)

        cold_o, warm_o, cold_m, warm_m = measure(decode_oracle, decode_ours)
        mb = len(payload) / 1e6
        ratio = len(payload) / len(container)
        rows.append((name, mb, ratio, cold_o, warm_o, cold_m, warm_m))
        print(
            f"  {name}: oletools cold {cold_o * 1e3:9.2f} ms warm {warm_o * 1e3:9.2f} ms "
            f"({mb / warm_o:7.1f} MB/s) | oletools_mojo cold {cold_m * 1e3:7.3f} ms "
            f"warm {warm_m * 1e3:7.3f} ms ({mb / warm_m:7.1f} MB/s) "
            f"-> warm {warm_o / warm_m:6.1f}x"
        )

    print("\n== README paste block ==")
    print("| stream | decompressed | oletools cold (ms) | oletools warm (ms) | oletools_mojo cold (ms) | oletools_mojo warm (ms) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, mb, ratio, cold_o, warm_o, cold_m, warm_m in rows:
        print(
            f"| {name} | {mb:.2f} MB | {cold_o * 1e3:.2f} | {warm_o * 1e3:.2f} "
            f"| {cold_m * 1e3:.3f} | {warm_m * 1e3:.3f} | {warm_o / warm_m:.1f}x |"
        )


if __name__ == "__main__":
    main()
