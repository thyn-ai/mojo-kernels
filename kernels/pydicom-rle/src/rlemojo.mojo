"""Clean-room DICOM RLE Lossless segment codec (PackBits variant).

Written fresh from the published DICOM Standard, Part 5, Annex G ("RLE
Lossless Compression"): a frame holds up to 15 independently coded byte
segments, each a sequence of PackBits packets. No third-party code is used
or adapted. Behaviour is validated byte-for-byte against the PyPI `pydicom`
package, used strictly as a black-box oracle, including these observable
semantics of its pure-Python codec (pydicom 3.0.2,
pydicom.pixels.decoders.rle / pydicom.pixels.encoders.native):

  * Decode packet header byte N (0-255): N in 0..127 copies the next N+1
    bytes literally; N == 128 is a no-op; N in 129..255 replicates the next
    byte 257-N times.
  * Decode is lenient at the end of a segment: a literal packet whose
    payload overruns the input is silently truncated, a replicate packet
    missing its value byte appends nothing, and decoding stops cleanly at
    the end of the input. There is no error return for malformed segments;
    the only failure modes are invalid parameters and output-buffer
    overflow.
  * Encode works row by row (rows of `columns` bytes; a short final row is
    coded as-is). Runs of two or more equal bytes flush any pending literal
    bytes, then code as replicate packets: full 128-byte runs as header
    129, a remainder > 1 as header 257-remainder, and a remainder of
    exactly 1 as a one-byte literal packet (header 0). Runs of 128 are
    therefore always coded as one replicate packet, and 2-byte runs always
    as replicate (never literal), matching the oracle's documented note.
  * Literal runs are emitted in chunks of at most 128 bytes (header
    length-1), mid-row and at row end alike.
  * An odd-length encoded segment is padded with one trailing 0x00 byte.

Exported C ABI (v1):

    int32_t  rlemojo_abi_version(void)
    int64_t  rlemojo_decode_segment(const uint8_t* data, int64_t data_len,
                                    uint8_t* out_buf, int64_t out_cap)
    int64_t  rlemojo_encode_segment(const uint8_t* data, int64_t data_len,
                                    int64_t columns, uint8_t* out_buf,
                                    int64_t out_cap)

Both segment functions write into a caller-provided buffer and return the
number of bytes written. The decoded size of a segment is at most
64 * data_len (every 2-byte replicate packet expands to at most 128 bytes);
the encoded size is at most 2 * data_len + 2 (an isolated single byte costs
one header byte, and the trailing even-length pad is one byte), so callers
can size buffers exactly.

Negative return codes:
    -1  invalid parameters (negative lengths, columns < 1, null pointers)
    -3  caller-provided output buffer too small
"""

from std.memory import Pointer, unsafe_memcpy
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of anything passed in; this library allocates nothing).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]

# Native SIMD width for uint8 on the build target (replicate-packet fills).
comptime WIDTH = simd_width_of[DType.uint8]()

# PackBits packet geometry (DICOM PS3.5 Annex G, same as TIFF PackBits).
comptime MAX_RUN = 128  # longest run one packet encodes
comptime NOOP = 128  # header byte that encodes nothing


@export
def rlemojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def rlemojo_decode_segment(
    data: U8Ptr, data_len: Int64, out_buf: U8Ptr, out_cap: Int64
) abi("C") -> Int64:
    """Decode one RLE segment (PackBits) into `out_buf`.

    Lenient end-of-input handling mirrors the oracle exactly: truncated
    literal packets are cut short, a trailing replicate header appends
    nothing, and no malformed-input error exists. Returns the byte count
    written, or a negative status code.
    """
    if data_len < 0 or out_cap < 0:
        return -1

    var pos = Int64(0)
    var out_len = Int64(0)
    while pos < data_len:
        var n = Int(data[unsafe_offset=Int(pos)])
        pos += 1
        if n > NOOP:
            # Replicate: the next byte repeated (257 - n) times (2..128).
            if pos >= data_len:
                # Missing value byte: nothing is appended; the loop ends.
                break
            var v = data[unsafe_offset=Int(pos)]
            pos += 1
            var count = 257 - n
            if out_len + Int64(count) > out_cap:
                return -3
            var base = Int(out_len)
            var splat = SIMD[DType.uint8, WIDTH](v)
            var i = 0
            while i + WIDTH <= count:
                out_buf.unsafe_store(base + i, splat)
                i += WIDTH
            while i < count:
                out_buf[unsafe_offset=base + i] = v
                i += 1
            out_len += Int64(count)
        elif n < NOOP:
            # Literal: copy the next (n + 1) bytes; silently truncated at
            # the end of the input (the position still advances past the
            # full claimed length, ending the loop).
            var count = n + 1
            var avail = data_len - pos
            var take = Int64(count)
            if take > avail:
                take = avail
            if out_len + take > out_cap:
                return -3
            if take > 0:
                unsafe_memcpy(
                    dest=out_buf.unsafe_offset(Int(out_len)),
                    src=data.unsafe_offset(Int(pos)),
                    count=Int(take),
                )
            out_len += take
            pos += Int64(count)
        # n == NOOP (128): no-op packet, only the header byte is consumed.
    return out_len


def _emit_byte(out_buf: U8Ptr, out_len: Int64, out_cap: Int64, v: UInt8) -> Int64:
    """Append one byte; returns the new length, or -3 if the buffer is full."""
    if out_len + 1 > out_cap:
        return -3
    out_buf[unsafe_offset=Int(out_len)] = v
    return out_len + 1


def _flush_literal(
    data: U8Ptr,
    lit_start: Int64,
    lit_len: Int64,
    out_buf: U8Ptr,
    out_len: Int64,
    out_cap: Int64,
) -> Int64:
    """Emit `lit_len` source bytes as literal packets (128-byte chunks).

    Returns the new output length, or -3 if the buffer is full.
    """
    var written = out_len
    var offset = Int64(0)
    while offset < lit_len:
        var chunk = lit_len - offset
        if chunk > Int64(MAX_RUN):
            chunk = Int64(MAX_RUN)
        written = _emit_byte(out_buf, written, out_cap, UInt8(Int(chunk) - 1))
        if written < 0:
            return -3
        if written + chunk > out_cap:
            return -3
        unsafe_memcpy(
            dest=out_buf.unsafe_offset(Int(written)),
            src=data.unsafe_offset(Int(lit_start + offset)),
            count=Int(chunk),
        )
        written += chunk
        offset += chunk
    return written


@export
def rlemojo_encode_segment(
    data: U8Ptr, data_len: Int64, columns: Int64, out_buf: U8Ptr, out_cap: Int64
) abi("C") -> Int64:
    """Encode one byte plane into an RLE segment (PackBits, row by row).

    Rows are `columns` bytes each; a short final row is coded as-is. Runs
    of two or more equal bytes flush pending literals and code as replicate
    packets (128-byte full runs as header 129, remainder > 1 as header
    257-remainder, remainder == 1 as a one-byte literal). An odd-length
    result is padded with one trailing 0x00. Returns the byte count
    written, or a negative status code.
    """
    if data_len < 0 or columns < 1 or out_cap < 0:
        return -1

    var out_len = Int64(0)
    var row_start = Int64(0)
    while row_start < data_len:
        var row_len = columns
        if data_len - row_start < row_len:
            row_len = data_len - row_start

        # One row: runs of >= 2 are replicate packets, isolated bytes
        # accumulate into a (contiguous) literal run flushed on demand.
        var lit_start = Int64(0)
        var lit_len = Int64(0)
        var i = Int64(0)
        while i < row_len:
            var v = data[unsafe_offset=Int(row_start + i)]
            var run = Int64(1)
            while i + run < row_len and data[
                unsafe_offset=Int(row_start + i + run)
            ] == v:
                run += 1
            if run == 1:
                if lit_len == 0:
                    lit_start = row_start + i
                lit_len += 1
                i += 1
                continue

            if lit_len > 0:
                out_len = _flush_literal(
                    data, lit_start, lit_len, out_buf, out_len, out_cap
                )
                if out_len < 0:
                    return -3
                lit_len = 0

            # Replicate: full 128-byte runs as header 129 (decodes to a
            # 128-copy packet), then the remainder.
            var full = run // Int64(MAX_RUN)
            var part = run % Int64(MAX_RUN)
            for _ in range(Int(full)):
                out_len = _emit_byte(out_buf, out_len, out_cap, UInt8(257 - MAX_RUN))
                if out_len < 0:
                    return -3
                out_len = _emit_byte(out_buf, out_len, out_cap, v)
                if out_len < 0:
                    return -3
            if part > 1:
                out_len = _emit_byte(out_buf, out_len, out_cap, UInt8(257 - Int(part)))
                if out_len < 0:
                    return -3
                out_len = _emit_byte(out_buf, out_len, out_cap, v)
                if out_len < 0:
                    return -3
            elif part == 1:
                # A leftover single byte codes as a one-byte literal packet.
                out_len = _emit_byte(out_buf, out_len, out_cap, UInt8(0))
                if out_len < 0:
                    return -3
                out_len = _emit_byte(out_buf, out_len, out_cap, v)
                if out_len < 0:
                    return -3
            i += run

        if lit_len > 0:
            out_len = _flush_literal(data, lit_start, lit_len, out_buf, out_len, out_cap)
            if out_len < 0:
                return -3
        row_start += row_len

    # Odd-length segments are padded with one trailing 0x00 byte.
    if out_len % 2 == 1:
        out_len = _emit_byte(out_buf, out_len, out_cap, UInt8(0))
        if out_len < 0:
            return -3
    return out_len
