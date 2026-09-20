#!/usr/bin/env python3
"""pydicom-mojo quickstart: DICOM RLE Lossless decode and encode.

Self-contained (stdlib + pydicom_mojo only — pydicom is NOT needed at
runtime). Builds a small 16-bit monochrome frame the way a CT scanner's
native pixel data looks (little-endian interleaved bytes), RLE-encodes it
into a standard DICOM RLE frame (64-byte header + one PackBits segment per
byte plane), then decodes it back. Run from anywhere against an installed
wheel:

    python examples/quickstart.py

Set PYDICOM_MOJO_DISABLE_NATIVE=1 to exercise the pure-Python fallback;
output is identical on both backends.
"""

from __future__ import annotations

import hashlib

import pydicom_mojo

ROWS, COLUMNS, SAMPLES, BITS = 8, 8, 1, 16


def synthetic_ct_slice() -> bytes:
    """A tiny 16-bit LE 'CT slice': a bright disc on a dark background."""
    out = bytearray(ROWS * COLUMNS * 2)
    for y in range(ROWS):
        for x in range(COLUMNS):
            value = 1050 if (y - 3.5) ** 2 + (x - 3.5) ** 2 < 6 else 80
            out[2 * (y * COLUMNS + x)] = value & 0xFF
            out[2 * (y * COLUMNS + x) + 1] = (value >> 8) & 0xFF
    return bytes(out)


def main() -> None:
    info = pydicom_mojo.backend_info()
    backend = "native Mojo kernel" if info["native_available"] else "pure-Python fallback"
    print(f"pydicom-mojo {pydicom_mojo.__version__} — backend: {backend}")

    native_pixels = synthetic_ct_slice()

    # Encode: native LE pixel bytes -> DICOM RLE frame (header + segments).
    rle_frame = pydicom_mojo.encode_frame(
        native_pixels, columns=COLUMNS, samples_per_pixel=SAMPLES, bits_allocated=BITS
    )
    assert rle_frame[:4] == (2).to_bytes(4, "little")  # 2 segments (MSB+LSB planes)
    print(f"encode: {len(native_pixels)} native bytes -> {len(rle_frame)} RLE bytes "
          f"(64-byte header + 2 byte-plane segments)")

    # Decode: RLE frame -> LE, planar configuration 1 pixel bytes (== the
    # native bytes for single-sample data).
    decoded = pydicom_mojo.decode_frame(
        rle_frame, rows=ROWS, columns=COLUMNS, samples_per_pixel=SAMPLES, bits_allocated=BITS
    )
    assert decoded == native_pixels
    print(f"decode: {len(rle_frame)} RLE bytes -> {len(decoded)} pixel bytes, "
          f"sha256 {hashlib.sha256(decoded).hexdigest()[:16]}… — lossless round trip OK")

    # To run inside pydicom's own pipelines, register the plugin:
    #
    #   from pydicom.pixels.decoders import RLELosslessDecoder
    #   from pydicom.pixels.encoders import RLELosslessEncoder
    #   RLELosslessDecoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "decode_frame"))
    #   RLELosslessEncoder.add_plugin("pydicom_mojo", ("pydicom_mojo.plugin", "encode_frame"))
    #   ds.decompress(decoding_plugin="pydicom_mojo")
    #   ds.compress(RLELossless, encoding_plugin="pydicom_mojo")


if __name__ == "__main__":
    main()
