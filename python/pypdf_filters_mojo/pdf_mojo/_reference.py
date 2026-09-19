"""Vendored pure-Python reference decoders (fallback backend).

Written fresh from the published file-format specifications (ISO 32000-1
section 7.4.4, TIFF Predictor 2, the PNG row filters of RFC 2083, and
TIFF-style LZW with EarlyChange). These are the exact same semantics the
Mojo kernel implements — including the byte-level behaviours the
differential suite pins against the PyPI `pypdf` oracle:

  * bytes-per-pixel is floor(colors * bits_per_component / 8) for both the
    TIFF and PNG paths (0 for sub-byte bpc with few colors → self-addition);
  * predictor math is per-byte mod 256 on a buffer pre-seeded with the raw
    row bytes;
  * PNG rows honour the per-row filter byte for any predictor 10-15, a
    ragged final row is zero-padded before filtering, filter bytes > 4 are
    an error;
  * TIFF Predictor 2 processes a short final row without padding;
  * LZW stops at EOD or when fewer than `width` bits remain, and any code
    >= the next free table entry decodes as prev_string + prev_string[0];
  * the LZW code width grows at table index (1 << width) - 1 — the pypdf
    6.x decoder's observable behaviour for BOTH EarlyChange 0 and 1 (the
    parameter is accepted for API compatibility; the oracle decodes both
    identically, and so do we).

This module is also the correctness oracle for platforms without a native
kernel build (e.g. Windows): identical output on every input, just slower.
"""

from __future__ import annotations

TABLE_SIZE = 4096
FIRST_FREE = 258


class PdfFilterError(Exception):  # noqa: N818
    """A stream could not be decoded (corrupt or unsupported content)."""


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_prediction_reference(
    data: bytes, columns: int, colors: int, bpc: int, predictor: int
) -> bytes:
    """Reconstruct a PNG/TIFF-predicted (already inflated) stream."""
    if predictor == 1:
        return bytes(data)

    row_len = (colors * columns * bpc + 7) // 8
    bpp = (colors * bpc) // 8

    if predictor == 2:
        # TIFF Predictor 2: per-row horizontal differencing; a short final
        # row is processed as-is (never padded).
        out = bytearray(data)
        pos = 0
        n = len(data)
        while pos < n:
            rlen = min(row_len, n - pos)
            for i in range(bpp, rlen):
                out[pos + i] = (out[pos + i] + out[pos + i - bpp]) & 0xFF
            pos += rlen
        return bytes(out)

    # PNG predictors 10-15: every row is prefixed with a filter byte.
    stride = row_len + 1
    rows = (len(data) + stride - 1) // stride
    out = bytearray(rows * row_len)
    for r in range(rows):
        base = r * stride
        f = data[base]
        if f > 4:
            raise PdfFilterError(f"unsupported PNG filter {f}")
        avail = min(row_len, len(data) - base - 1)
        ob = r * row_len
        # Pre-seed with the raw row bytes, zero-padded to a full row.
        out[ob : ob + avail] = data[base + 1 : base + 1 + avail]
        pb = ob - row_len
        if f == 0:
            continue
        if f == 1:  # Sub
            for i in range(bpp, row_len):
                out[ob + i] = (out[ob + i] + out[ob + i - bpp]) & 0xFF
        elif f == 2:  # Up
            if r > 0:
                for i in range(row_len):
                    out[ob + i] = (out[ob + i] + out[pb + i]) & 0xFF
        elif f == 3:  # Average
            for i in range(row_len):
                left = out[ob + i - bpp] if i >= bpp else 0
                up = out[pb + i] if r > 0 else 0
                out[ob + i] = (out[ob + i] + (left + up) // 2) & 0xFF
        else:  # Paeth
            for i in range(row_len):
                a = out[ob + i - bpp] if i >= bpp else 0
                b = out[pb + i] if r > 0 else 0
                c = out[pb + i - bpp] if (r > 0 and i >= bpp) else 0
                out[ob + i] = (out[ob + i] + _paeth(a, b, c)) & 0xFF
    return bytes(out)


def decode_lzw_reference(data: bytes, early_change: int) -> bytes:
    """Decode an LZW stream (MSB-first codes, 9-12 bits, EarlyChange 0/1)."""
    # Table entries 258..4095: (prefix code, suffix byte); literals are
    # their own single-byte strings.
    prefix = [0] * TABLE_SIZE
    suffix = [0] * TABLE_SIZE
    out = bytearray()
    walk = bytearray(TABLE_SIZE)  # scratch for emitting one string

    acc = 0  # MSB-first bit accumulator
    nbits = 0
    pos = 0
    width = 9
    next_code = FIRST_FREE
    prev = -1

    while True:
        while nbits < width and pos < len(data):
            acc = (acc << 8) | data[pos]
            pos += 1
            nbits += 8
        if nbits < width:
            return bytes(out)  # truncated tail: return what we have
        code = (acc >> (nbits - width)) & ((1 << width) - 1)
        nbits -= width

        if code == 256:  # ClearTable
            width = 9
            next_code = FIRST_FREE
            prev = -1
            continue
        if code == 257:  # EOD
            return bytes(out)
        if prev < 0:
            # First code after a clear (or stream start) must be a literal.
            if code >= 256:
                raise PdfFilterError(
                    "malformed LZW stream: non-literal code where a literal is required"
                )
            out.append(code)
            prev = code
            continue

        # Lenient KwKwK: any code >= next_code decodes as
        # string(prev) + first_byte(string(prev)).
        is_kwkwk = code >= next_code
        emit = prev if is_kwkwk else code

        length = 0
        c = emit
        while c >= 256:
            walk[length] = suffix[c]
            length += 1
            c = prefix[c]
        walk[length] = c
        length += 1
        fc = walk[length - 1]
        for i in range(length - 1, -1, -1):
            out.append(walk[i])
        if is_kwkwk:
            out.append(fc)

        if next_code < TABLE_SIZE:
            prefix[next_code] = prev
            suffix[next_code] = fc
            added = next_code
            next_code += 1
            # Width growth at (1 << width) - 1: the oracle's behaviour for
            # both EarlyChange values.
            if width < 12 and next_code == (1 << width) - 1:
                width += 1
            prev = added if is_kwkwk else code
        elif not is_kwkwk:
            prev = code
