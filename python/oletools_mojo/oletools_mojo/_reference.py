"""Vendored pure-Python MS-OVBA decompression reference (the fallback backend).

Clean-room implementation of the CompressedContainer codec of [MS-OVBA]
section 2.4.1 (CompressedChunk/RawChunk framing, TokenSequences, LiteralTokens
and CopyTokens with CopyToken Help bit geometry), written from the published
specification. Its observable behaviour — output bytes AND exception types —
is byte-for-byte identical to `oletools.olevba.decompress_stream` (the
oracle), including the oracle's quirks:

  * empty input -> IndexError (the signature byte is read unconditionally);
  * a container ending in a single trailing byte -> struct.error (strict
    16-bit unpack of the chunk header or of a CopyToken);
  * RawChunk size field must encode exactly 4098; a short final RawChunk is
    copied leniently;
  * CopyToken bit_count = max(4, ceil(log2(difference))) with exact integer
    arithmetic — agrees with the oracle's float64 math.log for every
    difference < 2**29 (verified exhaustively in the differential suite);
  * CopyToken at difference == 0 -> ValueError (the oracle's float log
    raises a math-domain error);
  * overlapping copies replicate byte-by-byte (RLE semantics);
  * a CopyToken offset beyond the decompressed length is resolved with
    Python negative-index wrap semantics, re-evaluated against the growing
    output at every byte; a still-out-of-range index -> IndexError.

This module has no third-party dependencies.
"""

from __future__ import annotations

import struct

__all__ = ["decompress_stream_reference", "copytoken_bit_count"]


def copytoken_bit_count(difference: int) -> int:
    """max(4, ceil(log2(difference))) for difference >= 1, integer-exact.

    MS-OVBA 2.4.1.3.19.1 CopyToken Help. Agrees with the oracle's float64
    computation for every difference < 2**29; beyond that (a single chunk
    decompressing to more than 512 MiB) the oracle's float log2 has
    power-of-two rounding anomalies this package intentionally does not
    reproduce.
    """
    if difference < 1:
        raise ValueError("math domain error")
    return max(4, (difference - 1).bit_length())


def _read_u16le(container: bytes | bytearray, pos: int) -> int:
    """Strict little-endian uint16 at `pos`; struct.error when truncated."""
    if pos + 2 > len(container):
        # Mirrors struct.unpack("<H", one_byte) in the oracle.
        raise struct.error("unpack requires a buffer of 2 bytes")
    return container[pos] | (container[pos + 1] << 8)


def decompress_stream_reference(compressed_container: bytes | bytearray) -> bytes:
    """Decompress one MS-OVBA CompressedContainer. See module docstring."""
    # The oracle reads the signature byte at offset 0 unconditionally.
    if len(compressed_container) == 0:
        raise IndexError("bytearray index out of range")
    sig_byte = compressed_container[0]
    if sig_byte != 0x01:
        raise ValueError("invalid signature byte {0:02X}".format(sig_byte))

    container = compressed_container  # local alias; read-only below
    n = len(container)
    out = bytearray()
    current = 1

    while current < n:
        chunk_start = current
        header = _read_u16le(container, chunk_start)
        chunk_size = (header & 0x0FFF) + 3
        chunk_signature = (header >> 12) & 0x07
        if chunk_signature != 0b011:
            raise ValueError("Invalid CompressedChunkSignature in VBA compressed stream")
        chunk_flag = (header >> 15) & 0x01
        # MS-OVBA 2.4.1.3.12: the flag==1 bound can never fire (the 12-bit
        # size field maxes out at 4098); the RawChunk equality check can.
        if chunk_flag == 0 and chunk_size != 4098:
            raise ValueError(
                "CompressedChunkSize=%d != 4098 but CompressedChunkFlag == 0" % chunk_size
            )

        compressed_end = min(n, chunk_start + chunk_size)
        current = chunk_start + 2

        if chunk_flag == 0:
            # RawChunk: copy up to 4096 bytes as-is (lenient short tail),
            # then advance a full 4096 regardless.
            out += container[current : current + 4096]
            current += 4096
            continue

        # CompressedChunk: TokenSequences until compressed_end.
        decompressed_chunk_start = len(out)
        while current < compressed_end:
            flag_byte = container[current]
            current += 1
            for bit_index in range(8):
                if current >= compressed_end:
                    break
                if (flag_byte >> bit_index) & 1 == 0:
                    # LiteralToken.
                    out.append(container[current])
                    current += 1
                else:
                    # CopyToken (MS-OVBA 2.4.1.3.19.2 Unpack CopyToken).
                    token = _read_u16le(container, current)
                    current += 2
                    bit_count = copytoken_bit_count(len(out) - decompressed_chunk_start)
                    length_mask = 0xFFFF >> bit_count
                    length = (token & length_mask) + 3
                    offset = ((token & (~length_mask & 0xFFFF)) >> (16 - bit_count)) + 1
                    copy_source = len(out) - offset
                    # Byte-by-byte: overlapping copies replicate, and a
                    # negative source wraps around the end exactly like the
                    # oracle's per-index bytearray access (the wrap is
                    # re-evaluated against the growing output each byte).
                    for index in range(copy_source, copy_source + length):
                        resolved = index
                        if resolved < 0:
                            resolved += len(out)
                        if resolved < 0 or resolved >= len(out):
                            raise IndexError("bytearray index out of range")
                        out.append(out[resolved])

    return bytes(out)
