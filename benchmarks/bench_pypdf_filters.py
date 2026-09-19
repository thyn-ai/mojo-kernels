#!/usr/bin/env python3
"""Reproducible benchmark: pypdf (oracle) vs pypdf-filters-mojo.

Workloads are generated locally from fixed seeds (no network, no datasets)
by the fresh encoders in tests/test_pypdf_filters_fixtures.py:

  * PNG-predicted flate streams: RGB/RGBA image-like rows and a 16-bit
    grayscale scan, per-row filter bytes mixed the way real encoders emit
    them (predictor 15), plus one TIFF Predictor 2 stream;
  * LZW streams: text-like, incompressible-random, and RLE-heavy payloads.

Both sides are timed END TO END on the same bytes: the oracle decodes a
flate-wrapped stream via pypdf's decode_stream_data (inflate + predictor
reconstruction inside pypdf); pdf_mojo inflates with zlib and decodes with
the kernel — the zlib stage is common to both. LZW is timed on the raw LZW
bytes (no flate stage on either side).

Protocol per (workload, implementation): cold = the FIRST decode call for
that workload (the first workload's cold also carries the one-time
dlopen/ABI handshake on the native side); the cold call doubles as the
correctness gate (its result must be byte-equal to the other side's), then
warm = median of 5 more runs. Numbers therefore always come from a
verified-correct build. Run from the repository root:

    PYTHONPATH=python/pypdf_filters_mojo pixi run python benchmarks/bench_pypdf_filters.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from test_pypdf_filters_fixtures import (  # noqa: E402
    PYPDF_VERSION,
    lzw_encode,
    oracle_lzw,
    png_predict_encode,
    png_row_len,
    tiff_predict_encode,
)

N_RUNS = 5  # warm runs; median reported
SEED = 20260919


def gradient_noise_rows(rng: random.Random, rows: int, row_len: int, bpp: int) -> bytes:
    """Image-like content: smooth horizontal gradient plus small noise."""
    out = bytearray(rows * row_len)
    for r in range(rows):
        base = (r * 7) % 256
        for i in range(row_len):
            channel = i % bpp
            value = (base + (i // bpp) // 4 + channel * 13 + rng.randrange(8)) & 0xFF
            out[r * row_len + i] = value
    return bytes(out)


def realistic_filters(rng: random.Random, rows: int) -> list[int]:
    """Per-row filter mix the way real PNG-ish encoders emit: Up-heavy with
    Sub/Paeth/Average and the occasional None."""
    return [rng.choices((0, 1, 2, 3, 4), weights=(1, 3, 6, 2, 4))[0] for _ in range(rows)]


def png_workloads():
    rng = random.Random(SEED)
    out = []
    for name, columns, colors, bpc, rows in (
        ("RGB 1024x768x3 bpc=8", 1024, 3, 8, 768),
        ("RGBA 2048x512x4 bpc=8", 2048, 4, 8, 512),
        ("gray16 1600x1200 bpc=16", 1600, 1, 16, 1200),
    ):
        row_len = png_row_len(columns, colors, bpc)
        bpp = max(1, (colors * bpc) // 8)
        original = gradient_noise_rows(rng, rows, row_len, bpp)
        filters = realistic_filters(rng, rows)
        predicted = png_predict_encode(original, columns, colors, bpc, filters)
        out.append((name, predicted, columns, colors, bpc, 15, len(original)))
    # TIFF Predictor 2 stream (no filter bytes).
    columns, colors, bpc, rows = 2048, 3, 8, 1024
    row_len = png_row_len(columns, colors, bpc)
    original = gradient_noise_rows(rng, rows, row_len, colors)
    predicted = tiff_predict_encode(original, columns, colors, bpc)
    out.append(("TIFF2 2048x1024x3 bpc=8", predicted, columns, colors, bpc, 2, len(original)))
    return out


def lzw_workloads():
    rng = random.Random(SEED + 1)
    words = [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(2, 9))) for _ in range(4000)]
    text = b" ".join(rng.choice(words) for _ in range(700_000))[: 4 * 1024 * 1024]
    rand = bytes(rng.randrange(256) for _ in range(1024 * 1024))
    block = bytes(rng.randrange(256) for _ in range(4096))
    rle = block * (8 * 1024 * 1024 // len(block))
    return [
        ("text-like 4 MiB", text),
        ("random 1 MiB", rand),
        ("RLE 8 MiB", rle),
    ]


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
    import pdf_mojo
    from pypdf.filters import decode_stream_data
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    info = pdf_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- pypdf (oracle): {PYPDF_VERSION}")
    print(
        f"- pdf_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- protocol: cold first call (doubles as correctness gate), warm = median of {N_RUNS}; seed={SEED}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'pdf_mojo'")

    def oracle_flate_stream(fl, columns, colors, bpc, predictor):
        stream = DecodedStreamObject()
        stream.set_data(fl)
        stream[NameObject("/Filter")] = NameObject("/FlateDecode")
        stream[NameObject("/DecodeParms")] = DictionaryObject(
            {
                NameObject("/Predictor"): NumberObject(predictor),
                NameObject("/Columns"): NumberObject(columns),
                NameObject("/Colors"): NumberObject(colors),
                NameObject("/BitsPerComponent"): NumberObject(bpc),
            }
        )
        return decode_stream_data(stream)

    png_rows = []
    print("\n== PNG/TIFF predictor streams (end-to-end: pypdf flate decode vs zlib+kernel) ==")
    for name, predicted, columns, colors, bpc, predictor, raw_len in png_workloads():
        flated = zlib.compress(predicted, 6)
        decode_oracle = lambda: oracle_flate_stream(flated, columns, colors, bpc, predictor)
        decode_ours = lambda: pdf_mojo.decode_png_prediction(
            zlib.decompress(flated), columns, colors, bpc, predictor
        )
        cold_o, warm_o, cold_m, warm_m = measure(decode_oracle, decode_ours)
        mb = raw_len / 1e6
        png_rows.append((name, mb, cold_o, warm_o, cold_m, warm_m))
        print(
            f"  {name}: pypdf cold {cold_o * 1e3:8.2f} ms warm {warm_o * 1e3:8.2f} ms "
            f"({mb / warm_o:7.1f} MB/s) | pdf_mojo cold {cold_m * 1e3:7.3f} ms "
            f"warm {warm_m * 1e3:7.3f} ms ({mb / warm_m:7.1f} MB/s) "
            f"-> warm {warm_o / warm_m:6.1f}x"
        )

    lzw_rows = []
    print("\n== LZW streams (raw LZW bytes both sides) ==")
    for name, payload in lzw_workloads():
        encoded = lzw_encode(payload, early_change=1)
        decode_oracle = lambda: oracle_lzw(encoded, 1)
        decode_ours = lambda: pdf_mojo.decode_lzw(encoded, 1)
        cold_o, warm_o, cold_m, warm_m = measure(decode_oracle, decode_ours)
        mb = len(payload) / 1e6
        lzw_rows.append((name, mb, cold_o, warm_o, cold_m, warm_m))
        print(
            f"  {name}: pypdf cold {cold_o * 1e3:8.2f} ms warm {warm_o * 1e3:8.2f} ms "
            f"({mb / warm_o:7.1f} MB/s) | pdf_mojo cold {cold_m * 1e3:7.3f} ms "
            f"warm {warm_m * 1e3:7.3f} ms ({mb / warm_m:7.1f} MB/s) "
            f"-> warm {warm_o / warm_m:6.1f}x"
        )

    print("\n== README paste block ==")
    print("PNG/TIFF predictor reconstruction (end-to-end decode):")
    print("| stream | decoded | pypdf cold (ms) | pypdf warm (ms) | pdf_mojo cold (ms) | pdf_mojo warm (ms) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, mb, cold_o, warm_o, cold_m, warm_m in png_rows:
        print(
            f"| {name} | {mb:.2f} MB | {cold_o * 1e3:.2f} | {warm_o * 1e3:.2f} "
            f"| {cold_m * 1e3:.3f} | {warm_m * 1e3:.3f} | {warm_o / warm_m:.1f}x |"
        )
    print("\nLZW decode:")
    print("| stream | decoded | pypdf cold (ms) | pypdf warm (ms) | pdf_mojo cold (ms) | pdf_mojo warm (ms) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, mb, cold_o, warm_o, cold_m, warm_m in lzw_rows:
        print(
            f"| {name} | {mb:.2f} MB | {cold_o * 1e3:.2f} | {warm_o * 1e3:.2f} "
            f"| {cold_m * 1e3:.3f} | {warm_m * 1e3:.3f} | {warm_o / warm_m:.1f}x |"
        )


if __name__ == "__main__":
    main()
