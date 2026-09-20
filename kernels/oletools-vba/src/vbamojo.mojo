"""Clean-room MS-OVBA VBA decompression kernel (the "CompressedContainer"
codec of [MS-OVBA] section 2.4.1), the pure interpreter byte-work behind
oletools' `olevba.decompress_stream`.

Written fresh from the published specification (MS-OVBA 2.4.1.1 through
2.4.1.3.19.2: CompressedContainer, CompressedChunk, RawChunk, TokenSequence,
LiteralToken, CopyToken and CopyToken Help). No third-party code is used or
adapted. Behaviour is validated byte-for-byte against the PyPI `oletools`
package (olevba.decompress_stream), used strictly as a black-box oracle,
including these observable semantics that go beyond the spec text:

  * An empty input raises (the oracle reads the signature byte at offset 0
    unconditionally); a signature byte other than 0x01 is an error.
  * The 16-bit little-endian chunk header is read with a strict 2-byte
    unpack: a container ending in a single trailing byte is an error.
  * Chunk signature bits (12-14) must be 0b011; a RawChunk's size field must
    encode exactly 4098 (0xFFF + 3). A RawChunk shorter than 4096 bytes at
    the end of the container is copied leniently (as many bytes as remain).
  * CopyToken bit geometry comes from CopyToken Help with
    bit_count = max(4, ceil(log2(difference))) where difference is the
    number of bytes already decompressed in the current chunk. The oracle
    computes this with float64 math.log(difference, 2); this kernel uses
    exact integer arithmetic, which agrees with the float computation for
    every difference < 2**29 (verified exhaustively against the oracle).
    A CopyToken at difference == 0 is an error (the oracle's float log
    raises a math-domain ValueError).
  * Overlapping copies (offset < length) replicate byte-by-byte, so the
    just-written bytes are visible to the remainder of the copy (RLE).
  * Malformed CopyTokens whose offset exceeds the decompressed length are
    resolved with Python negative-index wrap semantics — the oracle is a
    Python program and a negative copy source wraps around the end of the
    output bytearray; only a still-out-of-range index is an error. The
    wrap is re-evaluated against the growing output length at every copied
    byte, exactly like the oracle's per-index bytearray access.

Exported C ABI (v1):

    int32_t  vbamojo_abi_version(void)
    int64_t  vbamojo_decompress(const uint8_t* data, int64_t data_len,
                                uint8_t** out_ptr)
    void     vbamojo_free(uint8_t* ptr)

`vbamojo_decompress` returns the decompressed length (>= 0) with `out_ptr`
receiving a library-allocated buffer (free with `vbamojo_free`), or a
negative status code (and `out_ptr` left untouched):

    -1  empty input (oracle: IndexError reading the signature byte)
    -2  invalid signature byte (!= 0x01)
    -3  invalid CompressedChunkSignature (!= 0b011)
    -4  RawChunk with CompressedChunkSize != 4098
    -5  CopyToken at difference == 0 (oracle: math-domain ValueError)
    -6  truncated 16-bit read at the container end (chunk header or
        CopyToken; oracle: struct.error)
    -7  copy source index out of range after negative-index wrap
        (oracle: IndexError)
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U8PtrPtr = Pointer[U8Ptr, MutUntrackedOrigin]

# Status codes (see module docstring).
comptime ERR_EMPTY = Int64(-1)
comptime ERR_SIGNATURE = Int64(-2)
comptime ERR_CHUNK_SIGNATURE = Int64(-3)
comptime ERR_RAW_SIZE = Int64(-4)
comptime ERR_COPYTOKEN_AT_ZERO = Int64(-5)
comptime ERR_TRUNCATED_U16 = Int64(-6)
comptime ERR_INDEX = Int64(-7)


@export
def vbamojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


def _ceil_log2_max4(d: Int64) -> Int64:
    """max(4, ceil(log2(d))) for d >= 1, with exact integer arithmetic.

    Agrees with the oracle's float64 `ceil(math.log(d, 2))` for every
    d < 2**29 (the float computation first diverges at d == 2**29, far
    beyond any reachable single-chunk decompressed size of 4096 bytes for
    spec-conforming streams).
    """
    var bits = Int64(0)
    var v = d - 1
    while v > 0:
        bits += 1
        v >>= 1
    if bits < 4:
        bits = 4
    return bits


def _ensure_cap(mut buf: U8Ptr, mut cap: Int64, need: Int64):
    """Grow `buf` so it holds at least `need` bytes (amortized doubling)."""
    if need <= cap:
        return
    var new_cap = cap * 2
    if new_cap < need:
        new_cap = need
    var nb = unsafe_alloc[UInt8](Int(new_cap))
    unsafe_memcpy(dest=nb, src=buf, count=Int(cap))
    buf.unsafe_free()
    buf = nb
    cap = new_cap


@export
def vbamojo_decompress(
    data: U8Ptr,
    data_len: Int64,
    out_ptr: U8PtrPtr,
) abi("C") -> Int64:
    """Decompress one MS-OVBA CompressedContainer.

    `data`/`data_len` describe the whole container (signature byte
    included). On success `out_ptr` receives a library-allocated buffer
    (free with `vbamojo_free`) and the decompressed length is returned; a
    negative value is an error (and `out_ptr` is left untouched).
    """
    if data_len < 0:
        return ERR_EMPTY
    if data_len == 0:
        # The oracle indexes the signature byte unconditionally.
        return ERR_EMPTY
    if data[unsafe_offset=0] != 0x01:
        return ERR_SIGNATURE

    # Output buffer. One raw chunk already justifies 4 KiB; start there.
    var cap = data_len * 2 + 64
    if cap < 4096:
        cap = 4096
    var buf = unsafe_alloc[UInt8](Int(cap))
    var out_len = Int64(0)

    var current = Int64(1)
    var status = Int64(0)
    var done = False

    while current < data_len and not done:
        var chunk_start = current
        # Strict 2-byte unpack of the chunk header (oracle: struct.error
        # on a single trailing byte).
        if chunk_start + 2 > data_len:
            status = ERR_TRUNCATED_U16
            break
        var header = Int64(data[unsafe_offset=Int(chunk_start)]) | (
            Int64(data[unsafe_offset=Int(chunk_start) + 1]) << 8
        )
        var chunk_size = (header & 0x0FFF) + 3
        var chunk_signature = (header >> 12) & 0x07
        if chunk_signature != 3:
            status = ERR_CHUNK_SIGNATURE
            break
        var chunk_flag = (header >> 15) & 0x01
        # The flag==1 size check of MS-OVBA 2.4.1.3.12 can never fire (the
        # 12-bit size field maxes out at 4098); the RawChunk check can.
        if chunk_flag == 0 and chunk_size != 4098:
            status = ERR_RAW_SIZE
            break

        var compressed_end = chunk_start + chunk_size
        if compressed_end > data_len:
            compressed_end = data_len
        current = chunk_start + 2

        if chunk_flag == 0:
            # RawChunk: copy up to 4096 bytes as-is (lenient short tail),
            # then advance a full 4096 regardless (ends the chunk loop).
            var n = data_len - current
            if n > 4096:
                n = 4096
            if n > 0:
                _ensure_cap(buf, cap, out_len + n)
                unsafe_memcpy(
                    dest=buf.unsafe_offset(Int(out_len)),
                    src=data.unsafe_offset(Int(current)),
                    count=Int(n),
                )
                out_len += n
            current += 4096
        else:
            # CompressedChunk: TokenSequences until compressed_end.
            var decompressed_chunk_start = out_len
            while current < compressed_end and not done:
                var flag_byte = Int(data[unsafe_offset=Int(current)])
                current += 1
                if flag_byte == 0 and current + 8 <= compressed_end:
                    # Fast path: 8 LiteralTokens in one copy.
                    _ensure_cap(buf, cap, out_len + 8)
                    unsafe_memcpy(
                        dest=buf.unsafe_offset(Int(out_len)),
                        src=data.unsafe_offset(Int(current)),
                        count=8,
                    )
                    out_len += 8
                    current += 8
                    continue
                for bit_index in range(8):
                    if current >= compressed_end:
                        break
                    var flag_bit = (flag_byte >> bit_index) & 1
                    if flag_bit == 0:
                        # LiteralToken.
                        _ensure_cap(buf, cap, out_len + 1)
                        buf[unsafe_offset=Int(out_len)] = data[
                            unsafe_offset=Int(current)
                        ]
                        out_len += 1
                        current += 1
                    else:
                        # CopyToken: strict 2-byte unpack against the
                        # container end (not compressed_end).
                        if current + 2 > data_len:
                            status = ERR_TRUNCATED_U16
                            done = True
                            break
                        var token = Int64(data[unsafe_offset=Int(current)]) | (
                            Int64(data[unsafe_offset=Int(current) + 1]) << 8
                        )
                        current += 2
                        var difference = out_len - decompressed_chunk_start
                        if difference == 0:
                            status = ERR_COPYTOKEN_AT_ZERO
                            done = True
                            break
                        var bit_count = _ceil_log2_max4(difference)
                        var length_mask = Int64(0xFFFF) >> bit_count
                        var length = (token & length_mask) + 3
                        var offset = (
                            (token & (~length_mask & 0xFFFF)) >> (16 - bit_count)
                        ) + 1
                        var copy_source = out_len - offset
                        if copy_source >= 0 and offset >= length:
                            # Non-overlapping copy fully inside the output:
                            # identical to the oracle's byte loop.
                            _ensure_cap(buf, cap, out_len + length)
                            unsafe_memcpy(
                                dest=buf.unsafe_offset(Int(out_len)),
                                src=buf.unsafe_offset(Int(copy_source)),
                                count=Int(length),
                            )
                            out_len += length
                        elif copy_source >= 0:
                            # Overlapping copy: byte-by-byte replication.
                            _ensure_cap(buf, cap, out_len + length)
                            for i in range(Int(length)):
                                buf[unsafe_offset=Int(out_len)] = buf[
                                    unsafe_offset=Int(copy_source) + i
                                ]
                                out_len += 1
                        else:
                            # Malformed offset: mirror the oracle's Python
                            # negative-index wrap, re-evaluated against the
                            # growing output length at every byte.
                            var j = copy_source
                            while j < copy_source + length:
                                var idx = j
                                if idx < 0:
                                    idx += out_len
                                if idx < 0 or idx >= out_len:
                                    status = ERR_INDEX
                                    done = True
                                    break
                                _ensure_cap(buf, cap, out_len + 1)
                                buf[unsafe_offset=Int(out_len)] = buf[
                                    unsafe_offset=Int(idx)
                                ]
                                out_len += 1
                                j += 1
                            if done:
                                break

    if status != 0:
        buf.unsafe_free()
        return status
    out_ptr[] = buf
    return out_len


@export
def vbamojo_free(ptr: U8Ptr) abi("C"):
    """Release a buffer returned by `vbamojo_decompress`."""
    ptr.unsafe_free()
