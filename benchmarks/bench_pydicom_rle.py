#!/usr/bin/env python3
"""Reproducible benchmark: pydicom native RLE codec (oracle) vs pydicom-mojo.

Workloads are realistic medical volumes generated locally from fixed seeds
(no network, no datasets) by the builders in tests/test_pydicom_rle_fixtures.py:

  * a CT-style series: 60 frames of 512x512 16-bit monochrome (smooth
    anatomical structure plus fine acquisition noise);
  * a segmentation-style series: 40 frames of 512x512 8-bit binary blobs
    (the long-run content RLE compresses best);
  * a single 1024x1024 8-bit RGB frame (interleaved, 3 segments).

Both sides are timed on identical inputs at the function level — the exact
seam a pydicom decoding/encoding plugin runs at: the oracle side calls
pydicom 3.0.2's pure-Python `pydicom.pixels.decoders.rle._rle_decode_frame`
/ `pydicom.pixels.encoders.native._encode_frame`; the pydicom-mojo side
calls `pydicom_mojo.decode_frame` / `pydicom_mojo.encode_frame`. Outputs
must be byte-equal on every call.

Protocol per (workload, direction): cold = the FIRST call for that workload
(the first workload's cold also carries the one-time dlopen/ABI handshake
on the native side); the cold call doubles as the correctness gate (its
result must be byte-equal to the other side's), then warm = median of 5
more runs. Numbers therefore always come from a verified-correct build.
Run from the repository root:

    PYTHONPATH=python/pydicom_mojo .oracle-pydicom/bin/python benchmarks/bench_pydicom_rle.py
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

from test_pydicom_rle_fixtures import (  # noqa: E402
    PYDICOM_VERSION,
    FakeEncodeRunner,
    ct_like_frame,
    rgb_like_frame,
    seg_like_frame,
)

from pydicom.pixels.decoders.rle import _rle_decode_frame  # noqa: E402
from pydicom.pixels.encoders.native import _encode_frame  # noqa: E402

N_RUNS = 5  # warm runs; median reported
SEED = 20260919


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


class Volume:
    """One workload: geometry + the raw frames + their RLE encodings."""

    def __init__(self, name, rows, columns, samples, bits, frames):
        self.name = name
        self.rows, self.columns, self.samples, self.bits = rows, columns, samples, bits
        self.raw_frames = frames
        runner = FakeEncodeRunner(columns, samples, bits)
        self.rle_frames = [bytes(_encode_frame(f, runner)) for f in frames]

    @property
    def decoded_bytes(self) -> int:
        return sum(len(f) for f in self.raw_frames)

    @property
    def encoded_bytes(self) -> int:
        return sum(len(f) for f in self.rle_frames)

    def decode_oracle(self) -> bytes:
        return b"".join(
            bytes(_rle_decode_frame(f, self.rows, self.columns, self.samples, self.bits))
            for f in self.rle_frames
        )

    def encode_oracle(self) -> bytes:
        runner = FakeEncodeRunner(self.columns, self.samples, self.bits)
        return b"".join(bytes(_encode_frame(f, runner)) for f in self.raw_frames)


def build_volumes():
    rng = random.Random(SEED)
    ct_frames = [ct_like_frame(512, 512, seed=SEED + i) for i in range(8)]
    # 60-frame series: cycle the 8 distinct slices (same per-frame cost).
    ct = Volume(
        "CT series 512x512 16-bit, 60 frames",
        512, 512, 1, 16,
        [ct_frames[(i * 3 + 1) % 8] for i in range(60)],
    )
    seg_frames = [seg_like_frame(512, 512, seed=SEED + 100 + i) for i in range(8)]
    seg = Volume(
        "SEG series 512x512 8-bit, 40 frames",
        512, 512, 1, 8,
        [seg_frames[(i * 5 + 2) % 8] for i in range(40)],
    )
    rgb = Volume(
        "RGB frame 1024x1024 8-bit",
        1024, 1024, 3, 8,
        [rgb_like_frame(1024, 1024, seed=SEED + 200)],
    )
    _ = rng
    return [ct, seg, rgb]


def main() -> None:
    import pydicom_mojo

    info = pydicom_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- pydicom (oracle): {PYDICOM_VERSION}")
    print(
        f"- pydicom_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- protocol: cold first call (doubles as correctness gate), warm = median of {N_RUNS}; seed={SEED}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'pydicom_mojo'")

    volumes = build_volumes()
    decode_rows, encode_rows = [], []

    print("\n== RLE decode (whole volume per timed call) ==")
    for vol in volumes:

        def decode_ours(vol=vol) -> bytes:
            return b"".join(
                pydicom_mojo.decode_frame(f, vol.rows, vol.columns, vol.samples, vol.bits)
                for f in vol.rle_frames
            )

        cold_o, warm_o, cold_m, warm_m = measure(vol.decode_oracle, decode_ours)
        mb = vol.decoded_bytes / 1e6
        decode_rows.append((vol.name, vol.encoded_bytes, vol.decoded_bytes, cold_o, warm_o, cold_m, warm_m))
        print(
            f"  {vol.name}: pydicom cold {cold_o * 1e3:9.2f} ms warm {warm_o * 1e3:9.2f} ms "
            f"({mb / warm_o:7.1f} MB/s) | pydicom_mojo cold {cold_m * 1e3:8.3f} ms "
            f"warm {warm_m * 1e3:8.3f} ms ({mb / warm_m:7.1f} MB/s) "
            f"-> warm {warm_o / warm_m:6.1f}x"
        )

    print("\n== RLE encode (whole volume per timed call) ==")
    for vol in volumes:

        def encode_ours(vol=vol) -> bytes:
            return b"".join(
                pydicom_mojo.encode_frame(f, vol.columns, vol.samples, vol.bits)
                for f in vol.raw_frames
            )

        cold_o, warm_o, cold_m, warm_m = measure(vol.encode_oracle, encode_ours)
        mb = vol.decoded_bytes / 1e6
        encode_rows.append((vol.name, vol.encoded_bytes, vol.decoded_bytes, cold_o, warm_o, cold_m, warm_m))
        print(
            f"  {vol.name}: pydicom cold {cold_o * 1e3:9.2f} ms warm {warm_o * 1e3:9.2f} ms "
            f"({mb / warm_o:7.1f} MB/s) | pydicom_mojo cold {cold_m * 1e3:8.3f} ms "
            f"warm {warm_m * 1e3:8.3f} ms ({mb / warm_m:7.1f} MB/s) "
            f"-> warm {warm_o / warm_m:6.1f}x"
        )

    print("\n== README paste block ==")
    print("RLE decode (whole volume per call):")
    print("| volume | RLE size | decoded | pydicom cold (ms) | pydicom warm (ms) | pydicom_mojo cold (ms) | pydicom_mojo warm (ms) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, enc, dec, cold_o, warm_o, cold_m, warm_m in decode_rows:
        print(
            f"| {name} | {enc / 1e6:.2f} MB | {dec / 1e6:.2f} MB | {cold_o * 1e3:.2f} | {warm_o * 1e3:.2f} "
            f"| {cold_m * 1e3:.3f} | {warm_m * 1e3:.3f} | {warm_o / warm_m:.1f}x |"
        )
    print("\nRLE encode (whole volume per call):")
    print("| volume | RLE size | decoded | pydicom cold (ms) | pydicom warm (ms) | pydicom_mojo cold (ms) | pydicom_mojo warm (ms) | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, enc, dec, cold_o, warm_o, cold_m, warm_m in encode_rows:
        print(
            f"| {name} | {enc / 1e6:.2f} MB | {dec / 1e6:.2f} MB | {cold_o * 1e3:.2f} | {warm_o * 1e3:.2f} "
            f"| {cold_m * 1e3:.3f} | {warm_m * 1e3:.3f} | {warm_o / warm_m:.1f}x |"
        )


if __name__ == "__main__":
    main()
