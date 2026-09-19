"""Shared encoders and pypdf-oracle helpers for the pypdf-filters suite.

The encoders (PNG/TIFF predictor + LZW) are written fresh from the format
specifications (ISO 32000-1 §7.4.4, RFC 2083 row filters, TIFF-style LZW
with EarlyChange). The oracle helpers call the published PyPI `pypdf`
package strictly through its public generic-object API — pypdf is the
black-box reference, never read or adapted.

Everything here is deterministic (seeded content only, no network).

This module is imported by tests/test_pypdf_filters_differential.py and
benchmarks/bench_pypdf_filters.py; it contains no tests itself.
"""

from __future__ import annotations

import zlib

import pypdf
from pypdf.filters import decode_stream_data
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

PYPDF_VERSION = pypdf.__version__  # pinned in CI; recorded in test output


def png_row_len(columns: int, colors: int, bpc: int) -> int:
    return (colors * columns * bpc + 7) // 8


def png_bpp(colors: int, bpc: int) -> int:
    # floor(colors * bpc / 8): matches the observable oracle behaviour for
    # sub-byte bpc (0 → self-addition), NOT the RFC 2083 max(1, ...).
    return (colors * bpc) // 8


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def tiff_predict_encode(original: bytes, columns: int, colors: int, bpc: int) -> bytes:
    """Forward TIFF Predictor 2 (per-row horizontal differencing)."""
    row_len = png_row_len(columns, colors, bpc)
    bpp = png_bpp(colors, bpc)
    assert len(original) % row_len == 0
    out = bytearray()
    for r in range(len(original) // row_len):
        row = original[r * row_len : (r + 1) * row_len]
        for i, v in enumerate(row):
            left = row[i - bpp] if i >= bpp else 0
            out.append((v - left) & 0xFF)
    return bytes(out)


def png_predict_encode(
    original: bytes, columns: int, colors: int, bpc: int, filters: list[int]
) -> bytes:
    """Forward-apply PNG row filters; returns filter-byte-prefixed rows.

    `filters` holds one filter byte (0-4) per row of `original`.
    """
    row_len = png_row_len(columns, colors, bpc)
    bpp = png_bpp(colors, bpc)
    assert len(original) % row_len == 0
    assert len(filters) == len(original) // row_len
    out = bytearray()
    prev = bytes(row_len)
    for r, f in enumerate(filters):
        row = original[r * row_len : (r + 1) * row_len]
        out.append(f)
        if f == 0:
            out += row
        elif f == 1:
            for i, v in enumerate(row):
                left = row[i - bpp] if i >= bpp else 0
                out.append((v - left) & 0xFF)
        elif f == 2:
            for i, v in enumerate(row):
                out.append((v - prev[i]) & 0xFF)
        elif f == 3:
            for i, v in enumerate(row):
                left = row[i - bpp] if i >= bpp else 0
                out.append((v - (left + prev[i]) // 2) & 0xFF)
        elif f == 4:
            for i, v in enumerate(row):
                a = row[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                out.append((v - _paeth(a, prev[i], c)) & 0xFF)
        else:
            raise AssertionError(f"bad filter byte {f}")
        prev = row
    return bytes(out)


def lzw_encode(
    data: bytes,
    early_change: int = 1,
    clear_interval: int | None = None,
    emit_eod: bool = True,
) -> bytes:
    """Fresh LZW encoder: MSB-first codes, 9-12 bits.

    Width growth follows the pypdf 6.x decoder's observable rule — the code
    width grows at table index (1 << width) - 1 for BOTH EarlyChange values
    — so every stream produced here decodes correctly through the oracle
    (and through pdf_mojo) with either EarlyChange setting. The
    `early_change` argument is accepted for call-site clarity.

    Starts with a ClearTable code; `clear_interval` emits an additional
    ClearTable after every N emitted data codes (mid-stream reset test);
    `emit_eod=False` produces a truncated stream (no EOD marker).
    """
    out = bytearray()
    acc = 0
    nbits = 0

    def emit(code: int, width: int) -> None:
        nonlocal acc, nbits
        acc = (acc << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1

    table: dict[bytes, int] = {bytes([c]): c for c in range(256)}
    next_code = 258
    width = 9
    emitted_since_clear = 0

    emit(256, width)
    w = b""
    for byte in data:
        wb = bytes([byte])
        wc = w + wb
        if wc in table:
            w = wc
            continue
        emit(table[w], width)
        if next_code < 4096:
            table[wc] = next_code
            next_code += 1
            # The decoder adds its entry one code later than the encoder,
            # so it grows the width when ITS index hits (1 << width) - 1
            # (the oracle's observable rule); the encoder must emit at the
            # new width one entry later, when its own index hits 1 << width.
            if width < 12 and next_code == (1 << width):
                width += 1
        w = wb
        emitted_since_clear += 1
        if clear_interval and emitted_since_clear >= clear_interval:
            # Mid-stream ClearTable: reset both sides. `w` stays as the
            # current (single-byte) string — it will be emitted as the
            # literal the decoder expects first after a clear, and the
            # add/emit invariant is preserved.
            emit(256, width)
            table = {bytes([c]): c for c in range(256)}
            next_code = 258
            width = 9
            emitted_since_clear = 0
    if w:
        emit(table[w], width)
    if emit_eod:
        emit(257, width)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def oracle_flate_png(
    predicted: bytes, columns: int, colors: int, bpc: int, predictor: int
) -> bytes:
    """Decode with the pypdf oracle: flate stream + PNG/TIFF decode parms."""
    stream = DecodedStreamObject()
    stream.set_data(zlib.compress(predicted))
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


def oracle_lzw(data: bytes, early_change: int) -> bytes:
    """Decode with the pypdf oracle: LZWDecode stream + EarlyChange parm."""
    stream = DecodedStreamObject()
    stream.set_data(data)
    stream[NameObject("/Filter")] = NameObject("/LZWDecode")
    stream[NameObject("/DecodeParms")] = DictionaryObject(
        {NameObject("/EarlyChange"): NumberObject(early_change)}
    )
    return decode_stream_data(stream)
