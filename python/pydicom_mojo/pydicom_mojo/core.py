"""Frame-level DICOM RLE codec, byte-compatible with pydicom's native codec.

The hot per-segment PackBits loops run on the native Mojo kernel when its
shared library is available (macOS arm64 / Linux x86_64 wheels) and
transparently fall back to the vendored pure-Python reference otherwise.
Frame assembly — the 64-byte RLE header, segment offsets, planar
configuration 1 byte interleaving, and segment-order correction — is shared
by both backends in this module, so results are identical either way; the
differential test suite asserts byte-for-byte agreement with the published
`pydicom` package on both paths.

Decode mirrors `pydicom.pixels.decoders.rle._rle_decode_frame` and encode
mirrors `pydicom.pixels.encoders.native._encode_frame` (pydicom 3.0.2),
including their validation errors, warning behaviour, and edge semantics.
"""

from __future__ import annotations

import math
import os
import warnings
from struct import pack, unpack

from pydicom_mojo import _native, _reference
from pydicom_mojo._native import RleCodecError  # re-exported

__all__ = [
    "decode_frame",
    "encode_frame",
    "decode_segment",
    "encode_segment",
    "get_backend",
    "RleCodecError",
]

# Cached native-availability probe (env disable always wins; failure to load
# is sticky for the process, like the other mojo-kernels wrappers).
_use_native: bool | None = None


def _backend() -> str:
    global _use_native
    if os.environ.get("PYDICOM_MOJO_DISABLE_NATIVE") == "1":
        return "fallback"
    if _use_native is None:
        _use_native = _native.native_available()
    return "native" if _use_native else "fallback"


def get_backend() -> str:
    """The backend the next codec call will use: "native" or "fallback"."""
    return _backend()


def _as_bytes(data, name: str) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"{name} must be a bytes-like object, got {type(data).__name__}")


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Segment level
# ---------------------------------------------------------------------------


def decode_segment(src) -> bytes:
    """Decode one RLE segment (lenient PackBits).

    Every byte sequence decodes to something: truncated literal packets are
    cut short, a trailing replicate header appends nothing, and a 0x80
    header is a no-op. There is no malformed-stream error at this level.
    """
    src = _as_bytes(src, "src")
    if _backend() == "native":
        try:
            return _native.decode_segment_bounded_native(src)
        except (RleCodecError, _native.NativeUnavailable):
            pass  # fall through to the reference, silently correct
    return _reference.decode_segment_reference(src)


def encode_segment(src, columns: int) -> bytes:
    """Encode one byte plane into an RLE segment, row by row.

    `columns` is the row length in bytes. Mirrors the oracle's edge
    semantics: `columns` of 0 raises ValueError, a negative `columns`
    codes no rows, and an odd-length result is padded with one 0x00 byte.
    """
    src = _as_bytes(src, "src")
    columns = _as_int(columns, "columns")
    if columns == 0:
        raise ValueError("range() arg 3 must not be zero")
    if columns < 0:
        return b""  # the oracle's range(0, len, columns) is empty
    if _backend() == "native":
        try:
            return _native.encode_segment_native(src, columns)
        except (RleCodecError, _native.NativeUnavailable):
            pass
    return _reference.encode_segment_reference(src, columns)


def _decode_segment_exact(src: bytes, expected: int) -> bytes:
    """Decode one segment expecting exactly `expected` bytes out.

    Conformant segments decode in one bounded pass; a non-conformant
    over-long segment (more decoded bytes than expected) is retried with
    the proven hard bound so the oracle's warn-and-truncate behaviour is
    reproduced exactly.
    """
    if _backend() == "native":
        try:
            return _native.decode_segment_native(src, expected)
        except _native._OutputTooSmall:
            return _native.decode_segment_native(src, 64 * len(src))
        except (RleCodecError, _native.NativeUnavailable):
            pass
    return _reference.decode_segment_reference(src)


# ---------------------------------------------------------------------------
# Frame level: header, planar layout, segment-order correction
# ---------------------------------------------------------------------------


def _parse_header(header: bytes) -> list[int]:
    """Return the segment offsets from the 64-byte RLE header."""
    if len(header) != 64:
        raise ValueError("The RLE header can only be 64 bytes long")
    nr_segments = unpack("<L", header[:4])[0]
    if nr_segments > 15:
        raise ValueError(
            f"The RLE header specifies an invalid number of segments ({nr_segments})"
        )
    return list(unpack(f"<{nr_segments}L", header[4 : 4 * (nr_segments + 1)]))


def decode_frame(
    src,
    rows: int,
    columns: int,
    samples_per_pixel: int,
    bits_allocated: int,
    segment_order: str = ">",
) -> bytes:
    """Decode one frame of DICOM RLE Lossless pixel data.

    `src` is one encoded frame including its 64-byte RLE header. Returns
    the decoded frame as little-endian, planar configuration 1 bytes (all
    of sample 0, then sample 1, ...; least-significant byte first within
    each pixel) — byte-identical to pydicom's native RLE decoder.

    `segment_order` is the RLE segment byte order of `src`: ">" for big
    endian (the DICOM default), "<" for little endian (non-conformant).
    As in the oracle, any other value iterates segments like ">" but skips
    the byte-order correction.

    Raises NotImplementedError when `bits_allocated` is not a multiple of
    8, ValueError for a malformed header or a segment-count/decoded-length
    mismatch, and warns (then truncates) on non-conformant over-long
    segments — all matching the oracle.
    """
    src = _as_bytes(src, "src")
    rows = _as_int(rows, "rows")
    columns = _as_int(columns, "columns")
    samples_per_pixel = _as_int(samples_per_pixel, "samples_per_pixel")
    bits_allocated = _as_int(bits_allocated, "bits_allocated")

    if bits_allocated % 8:
        raise NotImplementedError(
            "Unable to decode RLE encoded pixel data with "
            f"{bits_allocated} bits allocated"
        )

    # Parse the RLE Header
    offsets = _parse_header(src[:64])
    nr_segments = len(offsets)

    # Check that the actual number of segments is as expected
    bytes_per_sample = bits_allocated // 8
    if nr_segments != samples_per_pixel * bytes_per_sample:
        raise ValueError(
            "The number of RLE segments in the pixel data doesn't match the "
            f"expected amount ({nr_segments} vs. {samples_per_pixel * bytes_per_sample} "
            "segments)"
        )

    # Ensure the last segment gets decoded
    offsets.append(len(src))

    # Preallocate with null bytes
    decoded = bytearray(rows * columns * samples_per_pixel * bytes_per_sample)

    # Interleave each segment in a manner consistent with a planar
    # configuration of 1 (and little endian byte ordering); see the oracle.
    stride = bytes_per_sample * rows * columns
    for sample_number in range(samples_per_pixel):
        le_gen = range(bytes_per_sample)
        byte_offsets = le_gen if segment_order == "<" else reversed(le_gen)
        for byte_offset in byte_offsets:
            ii = sample_number * bytes_per_sample + byte_offset
            segment = _decode_segment_exact(
                src[offsets[ii] : offsets[ii + 1]], rows * columns
            )

            # Check that the number of decoded bytes is correct
            actual_length = len(segment)
            if actual_length < rows * columns:
                raise ValueError(
                    "The amount of decoded RLE segment data doesn't match the "
                    f"expected amount ({actual_length} vs. {rows * columns} bytes)"
                )
            elif actual_length != rows * columns:
                warnings.warn(
                    "The decoded RLE segment contains non-conformant padding "
                    f"- {actual_length} vs. {rows * columns} bytes expected"
                )

            if segment_order == ">":
                byte_offset = bytes_per_sample - byte_offset - 1

            start = byte_offset + (sample_number * stride)
            decoded[start : start + stride : bytes_per_sample] = segment[
                : rows * columns
            ]

    return bytes(decoded)


def encode_frame(
    src,
    columns: int,
    samples_per_pixel: int,
    bits_allocated: int,
    byteorder: str = "<",
) -> bytes:
    """Encode one frame of pixel data as DICOM RLE Lossless.

    `src` is one frame of little-endian, interleaved (planar configuration
    0) pixel data — e.g. RGB RGB ... for 8-bit RGB, or LSB MSB per pixel
    for 16-bit monochrome. Returns the encoded frame including its 64-byte
    RLE header, byte-identical to pydicom's native RLE encoder: byte planes
    are segmented most-significant byte first (big endian segment order,
    per the DICOM Standard), each plane is coded row by row, and odd-length
    segments are padded with one 0x00 byte.

    Raises ValueError for `byteorder == ">"` (unsupported, as in the
    oracle) or when the frame would need more than 15 RLE segments.
    """
    src = _as_bytes(src, "src")
    columns = _as_int(columns, "columns")
    samples_per_pixel = _as_int(samples_per_pixel, "samples_per_pixel")
    bits_allocated = _as_int(bits_allocated, "bits_allocated")

    if byteorder == ">":
        raise ValueError("Unsupported option \"byteorder = '>'\"")
    # Like the oracle, any other value is treated as little endian.

    bytes_allocated = math.ceil(bits_allocated / 8)

    nr_segments = bytes_allocated * samples_per_pixel
    if nr_segments > 15:
        raise ValueError(
            "Unable to encode as the DICOM standard only allows "
            "a maximum of 15 segments in RLE encoded data"
        )

    rle_data = bytearray()
    seg_lengths = []

    for sample_nr in range(samples_per_pixel):
        for byte_offset in reversed(range(bytes_allocated)):
            idx = byte_offset + bytes_allocated * sample_nr
            segment = encode_segment(src[idx::nr_segments], columns)
            rle_data.extend(segment)
            seg_lengths.append(len(segment))

    # Add the number of segments to the header
    rle_header = bytearray(pack("<L", len(seg_lengths)))

    # Add the segment offsets, starting at 64 for the first segment
    # We don't need an offset to any data at the end of the last segment
    offsets = [64]
    for ii, length in enumerate(seg_lengths[:-1]):
        offsets.append(offsets[ii] + length)
    rle_header.extend(pack(f"<{len(offsets)}L", *offsets))

    # Add trailing padding to make up the rest of the header (if required)
    rle_header.extend(b"\x00" * (64 - len(rle_header)))

    return bytes(rle_header + rle_data)
