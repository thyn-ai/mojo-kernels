"""Clean-room MessagePack byte engine (msgpackmojo).

Written fresh from the public MessagePack specification (the format grammar
of prefix bytes, big-endian integer payloads, and the str/bin/array/map/ext
families). No third-party Mojo or C code is used or adapted; behavioral
edge cases (smallest-integer encoding, str8 gating on use_bin_type, ext
header choice by payload length) were probed black-box against the published
`msgpack` PyPI package, never read from its sources.

The kernel is a pure byte transformer with no knowledge of Python objects:

* pack:   instruction stream -> MessagePack bytes
* unpack: MessagePack bytes  -> record stream

The Python wrapper walks the object graph into an instruction stream (pack)
or assembles the record stream into objects (unpack). Both streams share one
opcode layout, all integers little-endian:

    0x00 nil | 0x01 false | 0x02 true
    0x03 s64:  8-byte little-endian signed integer
    0x04 u64:  8-byte little-endian unsigned integer
    0x05 f64:  8-byte little-endian IEEE-754 double
    0x06 str:  u32 byte length + UTF-8 bytes
    0x07 bin:  u32 byte length + bytes
    0x08 array: u32 element count (elements follow in order)
    0x09 map:  u32 pair count (2*count elements follow, key then value)
    0x0A ext:  i8 type code + u32 byte length + bytes

Exported C ABI (single-shot: one whole value per call):

    int32_t  msgpackmojo_abi_version(void)
    void*    msgpackmojo_pack(const uint8_t* stream, int64_t length, int64_t flags)
    void*    msgpackmojo_unpack(const uint8_t* data, int64_t length, int64_t flags)
    int32_t  msgpackmojo_result_status(void* handle)     -> 0 ok / 1 format / 2 truncated / 3 extra data
    int64_t  msgpackmojo_result_error_pos(void* handle)  -> input byte offset, -1 if ok
    int64_t  msgpackmojo_result_size(void* handle)       -> output byte length
    uint8_t* msgpackmojo_result_data(void* handle)       -> output bytes
    void     msgpackmojo_result_destroy(void* handle)

Pack flags: bit0 = use_bin_type (0 packs bytes into the legacy raw family and
gates str8 off, matching the reference), bit1 = use_single_float (f32 wire
form). Unpack flags: bit0 = raw (str-family payloads surface as bin records).

Pack output never exceeds the instruction-stream length (every opcode's wire
form is at most as long as its instruction encoding), so the output buffer is
allocated once at input size. Unpack runs two passes over the input: pass 1
validates the whole value and computes the exact record-stream size (a value
is complete when a 64-bit pending-element counter reaches zero, so nesting
needs no call stack and cannot overflow the kernel's stack), pass 2 writes
the records. Both passes are straight-line byte loops.
"""

from std.memory import Pointer, bitcast, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime Handle = Optional[Pointer[UInt8, MutUntrackedOrigin]]

# Instruction/record opcodes.
comptime OP_NIL: Int = 0x00
comptime OP_FALSE: Int = 0x01
comptime OP_TRUE: Int = 0x02
comptime OP_S64: Int = 0x03
comptime OP_U64: Int = 0x04
comptime OP_F64: Int = 0x05
comptime OP_STR: Int = 0x06
comptime OP_BIN: Int = 0x07
comptime OP_ARRAY: Int = 0x08
comptime OP_MAP: Int = 0x09
comptime OP_EXT: Int = 0x0A

# Result status codes (read by the wrapper to pick the exception type).
comptime ST_OK: Int32 = 0
comptime ST_FORMAT: Int32 = 1  # malformed input (unknown header / bad stream)
comptime ST_TRUNC: Int32 = 2  # ran out of input mid-value
comptime ST_EXTRA: Int32 = 3  # trailing bytes after one top-level value

# Pack/unpack flag bits.
comptime FLAG_USE_BIN_TYPE: Int = 1
comptime FLAG_SINGLE_FLOAT: Int = 2
comptime FLAG_RAW: Int = 1


struct Result(Copyable, Movable):
    """Owned engine outcome, read back through the result accessors."""

    var data: U8Ptr  # [size] output bytes (always an owned allocation)
    var size: Int64
    var status: Int32
    var err_pos: Int64  # input byte offset of the failure, -1 when ok

    def __init__(
        out self, data: U8Ptr, size: Int64, status: Int32, err_pos: Int64
    ):
        self.data = data
        self.size = size
        self.status = status
        self.err_pos = err_pos


# ---- little-endian helpers (instruction/record streams) --------------------


def read_u32le(p: U8Ptr, i: Int) -> Int:
    return (
        Int(p[i])
        | (Int(p[i + 1]) << 8)
        | (Int(p[i + 2]) << 16)
        | (Int(p[i + 3]) << 24)
    )


def read_u64le(p: U8Ptr, i: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(8):
        v |= UInt64(p[i + k]) << UInt64(8 * k)
    return v


def write_u64le(p: U8Ptr, i: Int, v: UInt64):
    for k in range(8):
        p[i + k] = UInt8((v >> UInt64(8 * k)) & 0xFF)

def write_u32le(p: U8Ptr, i: Int, v: UInt64):
    for k in range(4):
        p[i + k] = UInt8((v >> UInt64(8 * k)) & 0xFF)


# ---- big-endian helpers (MessagePack wire format) --------------------------


def read_u64be(p: U8Ptr, i: Int, nbytes: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(nbytes):
        v = (v << 8) | UInt64(p[i + k])
    return v


def write_u64be(p: U8Ptr, i: Int, v: UInt64, nbytes: Int):
    for k in range(nbytes):
        p[i + k] = UInt8((v >> UInt64(8 * (nbytes - 1 - k))) & 0xFF)


# ---- pack engine: instruction stream -> MessagePack bytes -------------------


def pack_int(dst: U8Ptr, op: Int, bits: UInt64, signed_negative: Bool) -> Int:
    """Write the smallest MessagePack integer encoding; return bytes written.

    `bits` carries the 64-bit pattern; `signed_negative` is set when the value
    arrived as a negative signed integer (the unsigned chain is used for every
    non-negative value, matching the reference packer's observable output).
    """
    if signed_negative:
        var sv = bitcast[DType.int64](bits)
        if sv >= -32:
            dst[op] = UInt8(bits & 0xFF)
            return 1
        if sv >= -128:
            dst[op] = 0xD0
            dst[op + 1] = UInt8(bits & 0xFF)
            return 2
        if sv >= -32768:
            dst[op] = 0xD1
            write_u64be(dst, op + 1, bits & 0xFFFF, 2)
            return 3
        if sv >= -2147483648:
            dst[op] = 0xD2
            write_u64be(dst, op + 1, bits & 0xFFFFFFFF, 4)
            return 5
        dst[op] = 0xD3
        write_u64be(dst, op + 1, bits, 8)
        return 9
    if bits < 128:
        dst[op] = UInt8(bits)
        return 1
    if bits < 256:
        dst[op] = 0xCC
        dst[op + 1] = UInt8(bits)
        return 2
    if bits < 65536:
        dst[op] = 0xCD
        write_u64be(dst, op + 1, bits, 2)
        return 3
    if bits < 4294967296:
        dst[op] = 0xCE
        write_u64be(dst, op + 1, bits, 4)
        return 5
    dst[op] = 0xCF
    write_u64be(dst, op + 1, bits, 8)
    return 9


def pack_str_header(dst: U8Ptr, op: Int, n: Int, allow_str8: Bool) -> Int:
    """str/raw family header; the legacy raw form (use_bin_type=False) never
    emits str8, matching the reference packer's observable output."""
    if n <= 31:
        dst[op] = UInt8(0xA0 | n)
        return 1
    if allow_str8 and n <= 255:
        dst[op] = 0xD9
        dst[op + 1] = UInt8(n)
        return 2
    if n <= 65535:
        dst[op] = 0xDA
        write_u64be(dst, op + 1, UInt64(n), 2)
        return 3
    dst[op] = 0xDB
    write_u64be(dst, op + 1, UInt64(n), 4)
    return 5


def pack_bin_header(dst: U8Ptr, op: Int, n: Int) -> Int:
    if n <= 255:
        dst[op] = 0xC4
        dst[op + 1] = UInt8(n)
        return 2
    if n <= 65535:
        dst[op] = 0xC5
        write_u64be(dst, op + 1, UInt64(n), 2)
        return 3
    dst[op] = 0xC6
    write_u64be(dst, op + 1, UInt64(n), 4)
    return 5


def pack_array_header(dst: U8Ptr, op: Int, n: Int) -> Int:
    if n <= 15:
        dst[op] = UInt8(0x90 | n)
        return 1
    if n <= 65535:
        dst[op] = 0xDC
        write_u64be(dst, op + 1, UInt64(n), 2)
        return 3
    dst[op] = 0xDD
    write_u64be(dst, op + 1, UInt64(n), 4)
    return 5


def pack_map_header(dst: U8Ptr, op: Int, n: Int) -> Int:
    if n <= 15:
        dst[op] = UInt8(0x80 | n)
        return 1
    if n <= 65535:
        dst[op] = 0xDE
        write_u64be(dst, op + 1, UInt64(n), 2)
        return 3
    dst[op] = 0xDF
    write_u64be(dst, op + 1, UInt64(n), 4)
    return 5


def pack_run(stream: U8Ptr, n: Int, flags: Int, dst: U8Ptr) raises -> Int:
    """Translate the instruction stream; returns the MessagePack byte count.

    The caller guarantees `out` has room for at least `n` bytes; every
    opcode's wire form is at most as long as its instruction encoding.
    """
    var use_bin = (flags & FLAG_USE_BIN_TYPE) != 0
    var single = (flags & FLAG_SINGLE_FLOAT) != 0
    var ip = 0
    var op = 0
    var pending: Int = 1  # values still expected to close the top level
    while pending > 0:
        if ip >= n:
            raise Error("truncated instruction stream")
        var tag = Int(stream[ip])
        ip += 1
        pending -= 1
        if tag == OP_NIL:
            dst[op] = 0xC0
            op += 1
        elif tag == OP_FALSE:
            dst[op] = 0xC2
            op += 1
        elif tag == OP_TRUE:
            dst[op] = 0xC3
            op += 1
        elif tag == OP_S64 or tag == OP_U64:
            if ip + 8 > n:
                raise Error("truncated integer instruction")
            var bits = read_u64le(stream, ip)
            ip += 8
            var negative = tag == OP_S64 and (bits >> 63) != 0
            op += pack_int(dst, op, bits, negative)
        elif tag == OP_F64:
            if ip + 8 > n:
                raise Error("truncated float instruction")
            var bits = read_u64le(stream, ip)
            ip += 8
            if single:
                var f = bitcast[DType.float64](bits)
                var f32 = Float32(f)
                dst[op] = 0xCA
                write_u64be(dst, op + 1, UInt64(bitcast[DType.uint32](f32)), 4)
                op += 5
            else:
                dst[op] = 0xCB
                write_u64be(dst, op + 1, bits, 8)
                op += 9
        elif tag == OP_STR or tag == OP_BIN:
            if ip + 4 > n:
                raise Error("truncated bytes instruction")
            var ln = read_u32le(stream, ip)
            ip += 4
            if ln < 0 or ip + ln > n:
                raise Error("truncated bytes payload")
            if tag == OP_STR:
                op += pack_str_header(dst, op, ln, use_bin)
            elif use_bin:
                op += pack_bin_header(dst, op, ln)
            else:
                op += pack_str_header(dst, op, ln, False)
            if ln > 0:
                unsafe_memcpy(dest=dst + op, src=stream + ip, count=ln)
            op += ln
            ip += ln
        elif tag == OP_ARRAY or tag == OP_MAP:
            if ip + 4 > n:
                raise Error("truncated container instruction")
            var count = read_u32le(stream, ip)
            ip += 4
            if tag == OP_ARRAY:
                op += pack_array_header(dst, op, count)
                pending += count
            else:
                op += pack_map_header(dst, op, count)
                pending += 2 * count
        elif tag == OP_EXT:
            if ip + 5 > n:
                raise Error("truncated ext instruction")
            var code = stream[ip]
            ip += 1
            var ln = read_u32le(stream, ip)
            ip += 4
            if ln < 0 or ip + ln > n:
                raise Error("truncated ext payload")
            if ln == 1:
                dst[op] = 0xD4
                dst[op + 1] = code
                op += 2
            elif ln == 2:
                dst[op] = 0xD5
                dst[op + 1] = code
                op += 2
            elif ln == 4:
                dst[op] = 0xD6
                dst[op + 1] = code
                op += 2
            elif ln == 8:
                dst[op] = 0xD7
                dst[op + 1] = code
                op += 2
            elif ln == 16:
                dst[op] = 0xD8
                dst[op + 1] = code
                op += 2
            elif ln <= 255:
                dst[op] = 0xC7
                dst[op + 1] = UInt8(ln)
                dst[op + 2] = code
                op += 3
            elif ln <= 65535:
                dst[op] = 0xC8
                write_u64be(dst, op + 1, UInt64(ln), 2)
                dst[op + 3] = code
                op += 4
            else:
                dst[op] = 0xC9
                write_u64be(dst, op + 1, UInt64(ln), 4)
                dst[op + 5] = code
                op += 6
            if ln > 0:
                unsafe_memcpy(dest=dst + op, src=stream + ip, count=ln)
            op += ln
            ip += ln
        else:
            raise Error("unknown instruction opcode")
    if ip != n:
        raise Error("trailing instruction bytes")
    return op


# ---- unpack engine: MessagePack bytes -> record stream ----------------------


struct UnpackOutcome(Copyable, Movable):
    var status: Int32
    var err_pos: Int64
    var record_size: Int64  # pass 1: exact record-stream byte count

    def __init__(out self, status: Int32, err_pos: Int64, record_size: Int64):
        self.status = status
        self.err_pos = err_pos
        self.record_size = record_size


def unpack_scan(data: U8Ptr, n: Int) -> UnpackOutcome:
    """Pass 1: validate one top-level value and size its record stream.

    A value is complete when the pending-element counter reaches zero, so
    nesting depth never touches the call stack. Multi-byte payload reads are
    bounds-checked before every access.
    """
    var ip = 0
    var pending: Int = 1
    var total: Int = 0
    while pending > 0:
        if ip >= n:
            return UnpackOutcome(ST_TRUNC, Int64(ip), 0)
        var b = Int(data[ip])
        var at = ip
        ip += 1
        pending -= 1
        if b <= 0x7F or b >= 0xE0:
            total += 9  # fixint -> s64 record
        elif b >= 0xA0 and b <= 0xBF:
            var ln = b & 0x1F
            if ip + ln > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += ln
            total += 5 + ln
        elif b >= 0x90 and b <= 0x9F:
            pending += b & 0x0F
            total += 5
        elif b >= 0x80 and b <= 0x8F:
            pending += 2 * (b & 0x0F)
            total += 5
        elif b == 0xC0 or b == 0xC2 or b == 0xC3:
            total += 1
        elif b == 0xC1:
            return UnpackOutcome(ST_FORMAT, Int64(at), 0)
        elif b >= 0xC4 and b <= 0xC6:  # bin 8/16/32
            var nlen = 1 << (b - 0xC4)
            if ip + nlen > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen
            if ip + ln > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += ln
            total += 5 + ln
        elif b >= 0xC7 and b <= 0xC9:  # ext 8/16/32
            var nlen = 1 << (b - 0xC7)
            if ip + nlen + 1 > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen + 1  # skip the length bytes and the type-code byte
            if ip + ln > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += ln
            total += 6 + ln
        elif b == 0xCA or b == 0xCB:  # f32 / f64
            var nlen = 4 if b == 0xCA else 8
            if ip + nlen > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += nlen
            total += 9
        elif b >= 0xCC and b <= 0xD3:  # uint 8/16/32/64, int 8/16/32/64
            var nlen = 1 << (b & 0x03)
            if ip + nlen > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += nlen
            total += 9
        elif b >= 0xD4 and b <= 0xD8:  # fixext 1/2/4/8/16
            var ln = 1 << (b - 0xD4)
            if ip + 1 + ln > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += 1 + ln
            total += 6 + ln
        elif b >= 0xD9 and b <= 0xDB:  # str 8/16/32
            var nlen = 1 << (b - 0xD9)
            if ip + nlen > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen
            if ip + ln > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            ip += ln
            total += 5 + ln
        elif b >= 0xDC and b <= 0xDF:  # array 16/32, map 16/32
            var nlen = 2 if b == 0xDC or b == 0xDE else 4
            if ip + nlen > n:
                return UnpackOutcome(ST_TRUNC, Int64(at), 0)
            var count = Int(read_u64be(data, ip, nlen))
            ip += nlen
            if b >= 0xDE:
                pending += 2 * count
            else:
                pending += count
            total += 5
        else:
            return UnpackOutcome(ST_FORMAT, Int64(at), 0)
    if ip != n:
        return UnpackOutcome(ST_EXTRA, Int64(ip), 0)
    return UnpackOutcome(ST_OK, -1, Int64(total))


def unpack_write(data: U8Ptr, n: Int, flags: Int, dst: U8Ptr):
    """Pass 2: re-parse the (already validated) value, writing records.

    Only called after `unpack_scan` returned ST_OK, so every read is in
    bounds and every prefix byte is known; the decode below mirrors the
    scan's dispatch exactly.
    """
    var raw = (flags & FLAG_RAW) != 0
    var ip = 0
    var op = 0
    var pending: Int = 1
    while pending > 0:
        var b = Int(data[ip])
        ip += 1
        pending -= 1
        if b <= 0x7F or b >= 0xE0:
            dst[op] = UInt8(OP_S64)
            var sv: Int64 = Int64(b) if b <= 0x7F else Int64(b) - 256
            write_u64le(dst, op + 1, bitcast[DType.uint64](sv))
            op += 9
        elif b >= 0xA0 and b <= 0xBF:
            var ln = b & 0x1F
            dst[op] = UInt8(OP_BIN if raw else OP_STR)
            write_u32le(dst, op + 1, UInt64(ln))
            unsafe_memcpy(dest=dst + op + 5, src=data + ip, count=ln)
            op += 5 + ln
            ip += ln
        elif b >= 0x90 and b <= 0x9F:
            var count = b & 0x0F
            dst[op] = UInt8(OP_ARRAY)
            write_u32le(dst, op + 1, UInt64(count))
            op += 5
            pending += count
        elif b >= 0x80 and b <= 0x8F:
            var count = b & 0x0F
            dst[op] = UInt8(OP_MAP)
            write_u32le(dst, op + 1, UInt64(count))
            op += 5
            pending += 2 * count
        elif b == 0xC0:
            dst[op] = UInt8(OP_NIL)
            op += 1
        elif b == 0xC2:
            dst[op] = UInt8(OP_FALSE)
            op += 1
        elif b == 0xC3:
            dst[op] = UInt8(OP_TRUE)
            op += 1
        elif b >= 0xC4 and b <= 0xC6:  # bin
            var nlen = 1 << (b - 0xC4)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen
            dst[op] = UInt8(OP_BIN)
            write_u32le(dst, op + 1, UInt64(ln))
            unsafe_memcpy(dest=dst + op + 5, src=data + ip, count=ln)
            op += 5 + ln
            ip += ln
        elif b >= 0xC7 and b <= 0xC9:  # ext 8/16/32
            var nlen = 1 << (b - 0xC7)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen
            var code = data[ip]
            ip += 1
            dst[op] = UInt8(OP_EXT)
            dst[op + 1] = code
            write_u32le(dst, op + 2, UInt64(ln))
            unsafe_memcpy(dest=dst + op + 6, src=data + ip, count=ln)
            op += 6 + ln
            ip += ln
        elif b == 0xCA:
            var bits32 = UInt32(read_u64be(data, ip, 4))
            ip += 4
            var f = Float64(bitcast[DType.float32](bits32))
            dst[op] = UInt8(OP_F64)
            write_u64le(dst, op + 1, bitcast[DType.uint64](f))
            op += 9
        elif b == 0xCB:
            var bits = read_u64be(data, ip, 8)
            ip += 8
            dst[op] = UInt8(OP_F64)
            write_u64le(dst, op + 1, bits)
            op += 9
        elif b >= 0xCC and b <= 0xCF:  # unsigned ints
            var nlen = 1 << (b & 0x03)
            var v = read_u64be(data, ip, nlen)
            ip += nlen
            dst[op] = UInt8(OP_U64)
            write_u64le(dst, op + 1, v)
            op += 9
        elif b >= 0xD0 and b <= 0xD3:  # signed ints
            var nlen = 1 << (b & 0x03)
            var uv = read_u64be(data, ip, nlen)
            ip += nlen
            var sv: Int64
            if nlen == 8:
                sv = bitcast[DType.int64](uv)
            else:
                var bits = 8 * nlen
                var shifted = (uv << UInt64(64 - bits))
                sv = bitcast[DType.int64](shifted) >> Int64(64 - bits)
            dst[op] = UInt8(OP_S64)
            write_u64le(dst, op + 1, bitcast[DType.uint64](sv))
            op += 9
        elif b >= 0xD4 and b <= 0xD8:  # fixext
            var ln = 1 << (b - 0xD4)
            var code = data[ip]
            ip += 1
            dst[op] = UInt8(OP_EXT)
            dst[op + 1] = code
            write_u32le(dst, op + 2, UInt64(ln))
            unsafe_memcpy(dest=dst + op + 6, src=data + ip, count=ln)
            op += 6 + ln
            ip += ln
        elif b >= 0xD9 and b <= 0xDB:  # str 8/16/32
            var nlen = 1 << (b - 0xD9)
            var ln = Int(read_u64be(data, ip, nlen))
            ip += nlen
            dst[op] = UInt8(OP_BIN if raw else OP_STR)
            write_u32le(dst, op + 1, UInt64(ln))
            unsafe_memcpy(dest=dst + op + 5, src=data + ip, count=ln)
            op += 5 + ln
            ip += ln
        elif b >= 0xDC and b <= 0xDF:  # array 16/32, map 16/32
            var nlen = 2 if b == 0xDC or b == 0xDE else 4
            var count = Int(read_u64be(data, ip, nlen))
            ip += nlen
            if b >= 0xDE:
                dst[op] = UInt8(OP_MAP)
                write_u32le(dst, op + 1, UInt64(count))
                pending += 2 * count
            else:
                dst[op] = UInt8(OP_ARRAY)
                write_u32le(dst, op + 1, UInt64(count))
                pending += count
            op += 5
        else:
            # unpack_scan validated the input; reaching here is a kernel bug.
            dst[op] = UInt8(OP_NIL)
            op += 1


# ---- C ABI --------------------------------------------------------------------


def make_error_result(status: Int32, err_pos: Int64) -> Handle:
    var res = unsafe_alloc[Result](1)
    # One-byte dummy allocation so `data` is always freeable; size 0
    # signals "no output".
    var dummy = unsafe_alloc[UInt8](1)
    res[] = Result(dummy, 0, status, err_pos)
    return res.unsafe_bitcast[UInt8]()


@export
def msgpackmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def msgpackmojo_pack(stream: U8Ptr, length: Int64, flags: Int64) abi("C") -> Handle:
    """Translate one instruction stream into MessagePack bytes."""
    var n = Int(length)
    if n <= 0:
        return make_error_result(ST_FORMAT, 0)
    # Every opcode's wire form is at most as long as its instruction
    # encoding, so `n` bytes of output always suffice.
    var buf = unsafe_alloc[UInt8](n)
    try:
        var size = pack_run(stream, n, Int(flags), buf)
        var res = unsafe_alloc[Result](1)
        res[] = Result(buf, Int64(size), ST_OK, -1)
        return res.unsafe_bitcast[UInt8]()
    except:
        buf.unsafe_free()
        return make_error_result(ST_FORMAT, 0)


@export
def msgpackmojo_unpack(data: U8Ptr, length: Int64, flags: Int64) abi("C") -> Handle:
    """Translate MessagePack bytes into one record stream."""
    var n = Int(length)
    if n < 0:
        n = 0
    var scan = unpack_scan(data, n)
    if scan.status != ST_OK:
        return make_error_result(scan.status, scan.err_pos)
    var buf = unsafe_alloc[UInt8](Int(scan.record_size))
    unpack_write(data, n, Int(flags), buf)
    var res = unsafe_alloc[Result](1)
    res[] = Result(buf, scan.record_size, ST_OK, -1)
    return res.unsafe_bitcast[UInt8]()


@export
def msgpackmojo_result_status(handle: Handle) abi("C") -> Int32:
    if not handle:
        return ST_FORMAT
    return handle.value().unsafe_bitcast[Result]()[].status


@export
def msgpackmojo_result_error_pos(handle: Handle) abi("C") -> Int64:
    if not handle:
        return -1
    return handle.value().unsafe_bitcast[Result]()[].err_pos


@export
def msgpackmojo_result_size(handle: Handle) abi("C") -> Int64:
    if not handle:
        return 0
    return handle.value().unsafe_bitcast[Result]()[].size


@export
def msgpackmojo_result_data(handle: Handle) abi("C") -> Handle:
    if not handle:
        return None
    return handle.value().unsafe_bitcast[Result]()[].data


@export
def msgpackmojo_result_destroy(handle: Handle) abi("C"):
    if not handle:
        return
    var res = handle.value().unsafe_bitcast[Result]()
    # `data` is always an owned allocation (a 1-byte dummy on errors).
    res[].data.unsafe_free()
    res.unsafe_free()
