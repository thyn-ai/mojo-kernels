#!/usr/bin/env python3
"""pypdf-filters-mojo quickstart: PNG-predictor and LZW decoding.

Self-contained (stdlib + pdf_mojo only — pypdf is NOT needed at runtime).
Builds a small RGB image, predictor-encodes its rows the way a PDF producer
would, then reconstructs them; also decodes an LZW stream. Run from
anywhere against an installed wheel:

    python examples/quickstart.py

Set PDF_MOJO_DISABLE_NATIVE=1 to exercise the pure-Python fallback; output
is identical on both backends.
"""

from __future__ import annotations

import hashlib
import zlib

import pdf_mojo

COLUMNS, COLORS, BPC, ROWS = 8, 3, 8, 4
PREDICTOR = 15  # PNG optimum: per-row filter bytes


def png_up_predict(rows: list[bytes]) -> bytes:
    """Toy producer: prefix every row with the PNG Up filter byte (2) and
    store the byte-wise difference to the previous row (mod 256)."""
    out = bytearray()
    prev = bytes(len(rows[0]))
    for row in rows:
        out.append(2)
        out += bytes((v - p) & 0xFF for v, p in zip(row, prev))
        prev = row
    return bytes(out)


def main() -> None:
    info = pdf_mojo.backend_info()
    backend = "native Mojo kernel" if info["native_available"] else "pure-Python fallback"
    print(f"pypdf-filters-mojo {pdf_mojo.__version__} — backend: {backend}")

    # --- PNG predictor round-trip ---
    row_len = COLUMNS * COLORS  # bpc=8
    image = bytes(
        (x * 9 + y * 5 + c * 3) & 0xFF
        for y in range(ROWS)
        for x in range(COLUMNS)
        for c in range(COLORS)
    )
    rows = [image[r * row_len : (r + 1) * row_len] for r in range(ROWS)]
    predicted = png_up_predict(rows)
    # In a real PDF these bytes would be flate-compressed; zlib is common
    # to both pypdf and pdf_mojo, so the kernel sees the inflated bytes.
    flate = zlib.compress(predicted)
    reconstructed = pdf_mojo.decode_png_prediction(
        zlib.decompress(flate), COLUMNS, COLORS, BPC, PREDICTOR
    )
    assert reconstructed == image
    print(f"PNG predictor: {len(image)} bytes reconstructed, "
          f"sha256 {hashlib.sha256(reconstructed).hexdigest()[:16]}… — OK")

    # --- LZW decode ---
    # LZW (EarlyChange=1) encoding of b"TOBEORNOTTOBEORTOBEORNOT".
    lzw_stream = bytes.fromhex("801509e422293ca44e2795205048342e0b0784c040")
    decoded = pdf_mojo.decode_lzw(lzw_stream, early_change=1)
    assert decoded == b"TOBEORNOTTOBEORTOBEORNOT"
    print(f"LZW: {len(lzw_stream)} bytes -> {decoded!r} — OK")


if __name__ == "__main__":
    main()
