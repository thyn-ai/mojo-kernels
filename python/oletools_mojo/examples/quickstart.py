#!/usr/bin/env python3
"""oletools-mojo quickstart: MS-OVBA VBA decompression.

Self-contained (stdlib + oletools_mojo only — oletools is NOT needed at
runtime). Builds a CompressedContainer the way an Office producer would
(greedy literal/copy-token chunks per MS-OVBA 2.4.1.3.6) around a small VBA
macro, then decompresses it. Run from anywhere against an installed wheel:

    python examples/quickstart.py

Set OLETOOLS_MOJO_DISABLE_NATIVE=1 to exercise the pure-Python fallback;
output is identical on both backends.
"""

from __future__ import annotations

import hashlib
import struct

import oletools_mojo

VBA_MODULE = b"""\
Attribute VB_Name = "Module1"
Sub AutoOpen()
    Dim payload As String
    payload = "Hello from oletools-mojo"
    MsgBox payload
End Sub
"""


def _bit_count(difference: int) -> int:
    """MS-OVBA 2.4.1.3.19.1 CopyToken Help: max(4, ceil(log2(difference)))."""
    return max(4, (difference - 1).bit_length())


def compress_vba(data: bytes) -> bytes:
    """Minimal spec-shaped MS-OVBA compressor (test/demo-grade).

    Chunks of up to 4096 bytes; greedy single-candidate matching; falls back
    to a RawChunk when a chunk does not compress (the oracle decoder accepts
    a short final RawChunk leniently).
    """
    out = bytearray(b"\x01")  # CompressedContainer signature
    pos = 0
    while pos < len(data):
        chunk = data[pos : pos + 4096]
        payload = bytearray()
        p = 0
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
                        cand = chunk.rfind(key, 0, p)
                        if cand >= 0:
                            length = 3
                            while length < max_len and chunk[cand + length] == chunk[p + length]:
                                length += 1
                            match = (p - cand, length)
                if match is None:
                    payload.append(chunk[p])
                    p += 1
                else:
                    offset, length = match
                    bc = _bit_count(p)
                    token = ((offset - 1) << (16 - bc)) | (length - 3)
                    payload += struct.pack("<H", token)
                    flags |= 1 << bit
                    p += length
            payload[flag_pos] = flags
        if len(payload) >= len(chunk):
            # No savings (or would not fit): RawChunk, size field 0xFFF.
            out += struct.pack("<H", 0x3FFF) + chunk
        else:
            out += struct.pack("<H", 0xB000 | (len(payload) - 1)) + payload
        pos += len(chunk)
    return bytes(out)


def main() -> None:
    info = oletools_mojo.backend_info()
    backend = "native Mojo kernel" if info["native_available"] else "pure-Python fallback"
    print(f"oletools-mojo {oletools_mojo.__version__} — backend: {backend}")

    container = compress_vba(VBA_MODULE)
    recovered = oletools_mojo.decompress_stream(container)
    assert recovered == VBA_MODULE, "round-trip mismatch"
    print(
        f"round-trip: {len(VBA_MODULE)} bytes of VBA <- {len(container)} bytes compressed, "
        f"sha256 {hashlib.sha256(recovered).hexdigest()[:16]}… — OK"
    )

    # bytearray input (the type oletools' VBA_Parser passes) works the same.
    assert oletools_mojo.decompress_stream(bytearray(container)) == VBA_MODULE
    print("bytearray input — OK")

    # A hand-built stream exercising overlapping copy tokens (RLE):
    # one compressed chunk: literal 'A', then a copy token (offset=1, len=5).
    # difference=1 -> bit_count=4; token = (0 << 12) | (5-3) = 0x0002.
    stream = b"\x01" + struct.pack("<H", 0xB003) + b"\x02A\x02\x00"
    assert oletools_mojo.decompress_stream(stream) == b"AAAAAA"
    print("overlapping copy token (RLE) — OK")


if __name__ == "__main__":
    main()
