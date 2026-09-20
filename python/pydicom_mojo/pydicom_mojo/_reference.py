"""Vendored pure-Python reference codec for DICOM RLE segments (fallback backend).

Written fresh from the published DICOM Standard, Part 5, Annex G ("RLE
Lossless Compression"): each byte plane of a frame is coded independently
as a sequence of PackBits packets. These are the exact same semantics the
Mojo kernel implements — including the observable behaviours the
differential suite pins against the PyPI `pydicom` oracle (pydicom 3.0.2,
pydicom.pixels.decoders.rle / pydicom.pixels.encoders.native):

  * decode packet header N: 0..127 copies the next N+1 bytes literally,
    128 is a no-op, 129..255 replicates the next byte 257-N times;
  * decode is lenient at the end of a segment: truncated literal packets
    are cut short, a trailing replicate header appends nothing, and
    decoding stops cleanly at the end of the input (no malformed-stream
    error exists at the segment level);
  * encode is row by row: runs of two or more equal bytes flush pending
    literals and code as replicate packets (full 128-runs as header 129, a
    remainder > 1 as header 257-remainder, a remainder of exactly 1 as a
    one-byte literal), and an odd-length segment gets one trailing 0x00
    padding byte.

This module is also the correctness oracle for platforms without a native
kernel build (e.g. Windows): identical output on every input, just slower.
"""

from __future__ import annotations

MAX_RUN = 128
_NOOP = 128


class RleCodecError(Exception):  # noqa: N818
    """An RLE segment could not be coded (native kernel failure only).

    Segment-level PackBits decoding has no malformed-stream error: every
    byte sequence decodes to something. This exception exists so the native
    loader can signal an unexpected kernel status; the public API then
    falls back to this pure-Python reference.
    """


def decode_segment_reference(src: bytes) -> bytes:
    """Decode one RLE segment (lenient PackBits) and return the bytes."""
    out = bytearray()
    pos = 0
    end = len(src)
    while pos < end:
        header = src[pos] + 1  # header byte N, as N + 1 (1..256)
        pos += 1
        if header > 129:
            # Replicate: the next byte repeated (258 - header) times. A
            # missing value byte appends nothing (empty slice * count).
            out += src[pos : pos + 1] * (258 - header)
            pos += 1
        elif header < 129:
            # Literal: copy the next `header` bytes, silently truncated at
            # the end of the input; the position advances past the full
            # claimed length, which ends the loop.
            out += src[pos : pos + header]
            pos += header
        # header == 129 (N == 128): no-op, only the header byte is consumed.
    return bytes(out)


def _encode_row_reference(row: bytes) -> bytes:
    """Code one row as PackBits packets (see the module docstring)."""
    out = bytearray()
    literal = bytearray()

    def flush_literal() -> None:
        offset = 0
        while offset < len(literal):
            chunk = literal[offset : offset + MAX_RUN]
            out.append(len(chunk) - 1)
            out.extend(chunk)
            offset += len(chunk)

    pos = 0
    end = len(row)
    while pos < end:
        value = row[pos]
        run = 1
        while pos + run < end and row[pos + run] == value:
            run += 1
        if run == 1:
            literal.append(value)
            pos += 1
            continue
        if literal:
            flush_literal()
            literal.clear()
        full, part = divmod(run, MAX_RUN)
        for _ in range(full):
            out.extend((257 - MAX_RUN, value))  # header 129: replicate 128 times
        if part > 1:
            out.extend((257 - part, value))
        elif part == 1:
            # A leftover single byte codes as a one-byte literal packet.
            out.extend((0, value))
        pos += run

    if literal:
        flush_literal()
    return bytes(out)


def encode_segment_reference(src: bytes, columns: int) -> bytes:
    """Encode one byte plane into an RLE segment, row by row.

    `columns` is the row length in bytes; a short final row is coded as-is.
    Mirrors the oracle exactly: `columns` of 0 raises ValueError (the
    oracle's range() call rejects a zero step), a negative `columns` codes
    no rows at all, and an odd-length result is padded with one trailing
    0x00 byte.
    """
    if columns == 0:
        raise ValueError("range() arg 3 must not be zero")
    out = bytearray()
    if columns > 0:
        for idx in range(0, len(src), columns):
            out += _encode_row_reference(src[idx : idx + columns])
    out += b"\x00" * (len(out) % 2)
    return bytes(out)
