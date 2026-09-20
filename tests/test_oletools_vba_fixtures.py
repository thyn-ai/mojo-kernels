"""Shared fixtures for the oletools-vba differential suite.

Everything is generated locally from explicit seeds — no network, no sample
files — so the suite is bit-reproducible on any machine.

The oracle is the published PyPI package oletools==0.60.2
(`oletools.olevba.decompress_stream`), installed into .oracle-oletools/ (see
scripts/test_all_oletools_vba.sh) and used strictly as a black box.

`vba_compress` is a fresh MS-OVBA 2.4.1.3.6-shaped compressor (oletools ships
no public compressor): chunks of up to 4096 bytes, greedy single-candidate
matching, RawChunk fallback when a chunk does not compress. Every stream it
emits is validated through the oracle in the suite (Mode B round-trips), so
the encoder itself is pinned before any kernel comparison relies on it.
"""

from __future__ import annotations

import importlib.metadata
import struct

from oletools.olevba import copytoken_help as oracle_copytoken_help
from oletools.olevba import decompress_stream as oracle_decompress

# The oracle bindings above are re-exported for the differential suite and
# the benchmark; this module holds no oracle-dependent logic of its own.
__all__ = [
    "OLETOOLS_VERSION",
    "oracle_copytoken_help",
    "oracle_decompress",
    "chunk_header",
    "copy_token",
    "literal_chunk",
    "raw_chunk",
    "container",
    "vba_compress",
    "vba_like_source",
    "rng_bytes",
]

OLETOOLS_VERSION = importlib.metadata.version("oletools")

# ---------------------------------------------------------------------------
# Stream builders (MS-OVBA 2.4.1)
# ---------------------------------------------------------------------------


def chunk_header(flag: int, size_field: int, signature: int = 0b011) -> bytes:
    """16-bit little-endian CompressedChunkHeader."""
    return struct.pack("<H", ((flag & 1) << 15) | ((signature & 7) << 12) | (size_field & 0xFFF))


def copy_token(offset: int, length: int, difference: int) -> bytes:
    """Encode a CopyToken valid at `difference` decompressed bytes into a chunk."""
    assert difference >= 1 and offset >= 1 and length >= 3
    bit_count = max(4, (difference - 1).bit_length())
    assert offset - 1 < (1 << bit_count)
    assert length - 3 <= (0xFFFF >> bit_count)
    return struct.pack("<H", ((offset - 1) << (16 - bit_count)) | (length - 3))


def literal_chunk(data: bytes) -> bytes:
    """One compressed chunk holding `data` as pure LiteralTokens (<= 3640B)."""
    assert 0 < len(data) <= 3640
    payload = bytearray()
    for base in range(0, len(data), 8):
        payload.append(0x00)
        payload += data[base : base + 8]
    return chunk_header(1, len(payload) - 1) + bytes(payload)


def raw_chunk(data: bytes) -> bytes:
    """One RawChunk (flag=0, size field 0xFFF); `data` may be short (tail)."""
    assert 0 < len(data) <= 4096
    return chunk_header(0, 0xFFF) + data


def container(*chunks: bytes) -> bytes:
    return b"\x01" + b"".join(chunks)


# ---------------------------------------------------------------------------
# Fresh compressor (Mode B fixtures)
# ---------------------------------------------------------------------------


def _bit_count(difference: int) -> int:
    """MS-OVBA 2.4.1.3.19.1 CopyToken Help: max(4, ceil(log2(difference)))."""
    return max(4, (difference - 1).bit_length())


def vba_compress(data: bytes, force_raw: bool = False) -> bytes:
    """Compress `data` into an MS-OVBA CompressedContainer.

    Greedy hash matcher (single candidate per position); emits a RawChunk
    whenever a chunk does not compress (or `force_raw` is set). Short final
    RawChunks are emitted as-is — the oracle decoder copies them leniently.
    """
    out = bytearray(b"\x01")
    pos = 0
    while pos < len(data):
        chunk = data[pos : pos + 4096]
        payload = bytearray()
        p = 0
        table: dict[bytes, int] = {}
        while p < len(chunk):
            flag_pos = len(payload)
            payload.append(0)
            flags = 0
            for bit in range(8):
                if p >= len(chunk):
                    break
                match = None
                if p > 0:
                    bc = _bit_count(p)
                    max_len = min((0xFFFF >> bc) + 3, len(chunk) - p)
                    if max_len >= 3:
                        key = bytes(chunk[p : p + 3])
                        cand = table.get(key, -1)
                        if cand >= 0 and p - cand <= (1 << bc):
                            length = 3
                            while length < max_len and chunk[cand + length] == chunk[p + length]:
                                length += 1
                            match = (p - cand, length)
                        table[key] = p
                if match is None:
                    payload.append(chunk[p])
                    p += 1
                else:
                    payload += copy_token(match[0], match[1], p)
                    flags |= 1 << bit
                    p += match[1]
            payload[flag_pos] = flags
        if force_raw or len(payload) >= len(chunk):
            out += raw_chunk(chunk)
        else:
            out += chunk_header(1, len(payload) - 1) + bytes(payload)
        pos += len(chunk)
    return bytes(out)


# ---------------------------------------------------------------------------
# Deterministic payload generators
# ---------------------------------------------------------------------------


def vba_like_source(rng, n_lines: int) -> bytes:
    """Realistic VBA macro text: keywords, indentation, repeated idioms."""
    lines = [
        "Attribute VB_Name = \"Module1\"",
        "Option Explicit",
        "Private Declare PtrSafe Function GetTickCount Lib \"kernel32\" () As Long",
    ]
    for i in range(n_lines):
        template = rng.choice(
            (
                "Sub Proc%d()\n    Dim x%d As Long\n    x%d = %d\nEnd Sub",
                "Function F%d(ByVal a%d As String) As String\n    F%d = a%d & \"suffix\"\nEnd Function",
                "    If sh%d.Name = \"Sheet1\" Then sh%d.Visible = xlSheetHidden",
                "    For Each c%d In Range(\"A1:D20\")\n        c%d.Value = c%d.Value * 2\n    Next",
                "Const MSG_%d = \"error while processing request %d\"",
                "    Call WriteLog(\"checkpoint %d\", Now, Err.Number)",
            )
        )
        lines.append(template % ((i,) * template.count("%d")))
    return ("\r\n".join(lines) + "\r\n").encode("latin-1")


def rng_bytes(rng, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))
