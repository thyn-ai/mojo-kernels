"""Clean-room PDF stream filter kernels: PNG/TIFF predictor reconstruction
and LZW decoding, as used by PDF FlateDecode/LZWDecode streams.

Written fresh from the published file-format specifications (ISO 32000-1
section 7.4.4 decode parameters, TIFF Predictor 2, the PNG row filters of
RFC 2083, and TIFF-style LZW with EarlyChange). No third-party code is used
or adapted. Behaviour is validated byte-for-byte against the PyPI `pypdf`
package, used strictly as a black-box oracle, including these observable
semantics:

  * bytes-per-pixel is floor(colors * bits_per_component / 8) for BOTH the
    TIFF and PNG paths. For sub-byte bpc with few colors this floors to 0,
    which makes the "left neighbour" the byte itself (self-addition);
    reproduced exactly.
  * All predictor arithmetic is per-byte with wrapping uint8 math, on a
    buffer pre-seeded with the raw row bytes (so bpp == 0 doubles bytes).
  * PNG rows are lenient: the per-row filter byte (0-4) is honoured for any
    predictor 10-15; a ragged final row is zero-padded to a full row BEFORE
    filtering; a filter byte > 4 is an error.
  * TIFF Predictor 2 also processes a short final row, without padding:
    output length always equals input length.
  * LZW: MSB-first codes, 9-12 bits, lenient KwKwK (any code >= the next
    free entry decodes as prev_string + prev_string[0]), decoding stops at
    the EOD code or when fewer than `width` bits remain. The code width
    grows when the table index reaches (1 << width) - 1 — the pypdf 6.x
    decoder's observable behaviour for BOTH EarlyChange 0 and 1 (the
    parameter is accepted for API compatibility; the oracle decodes both
    identically, and so do we).

Exported C ABI (v1):

    int32_t  pdfmojo_abi_version(void)
    int64_t  pdfmojo_png_decode(data, data_len, columns, colors, bpc,
                                predictor, out_buf, out_cap)
    int64_t  pdfmojo_lzw_decode(data, data_len, early_change, out_ptr)
    void     pdfmojo_free(ptr)

`pdfmojo_png_decode` writes into a caller-provided buffer and returns the
number of bytes written. `pdfmojo_lzw_decode` returns a library-allocated
buffer through `out_ptr` (free with `pdfmojo_free`) and returns its length.

Negative return codes:
    -1  invalid parameters
    -2  malformed stream (LZW: non-literal where a literal is required)
    -3  caller-provided output buffer too small
    -5  unsupported PNG row filter byte (> 4)
"""

from std.memory import Pointer, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.sys import simd_width_of

comptime ABI_VERSION: Int32 = 1

# C-side pointer spellings (untracked origin: the caller owns the lifetime of
# anything passed in; the library owns what it allocates).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U8PtrPtr = Pointer[U8Ptr, MutUntrackedOrigin]

# Native SIMD width for uint8 on the build target (Up-filter row adds).
comptime WIDTH = simd_width_of[DType.uint8]()

# LZW table geometry (ISO 32000-1 §7.4.4.1): codes 0-255 are literals,
# 256 clears the table, 257 is end-of-data, entries are added from 258 up to
# 4095 inclusive (12-bit codes maximum).
comptime TABLE_SIZE = 4096
comptime FIRST_FREE = 258


@export
def pdfmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def pdfmojo_png_decode(
    data: U8Ptr,
    data_len: Int64,
    columns: Int64,
    colors: Int64,
    bpc: Int64,
    predictor: Int64,
    out_buf: U8Ptr,
    out_cap: Int64,
) abi("C") -> Int64:
    """Reconstruct one PNG/TIFF-predicted (already inflated) stream.

    `out_buf` must hold the reconstructed size: `data_len` bytes for
    predictors 1-2, or ceil(data_len / (row_len + 1)) * row_len bytes for
    predictors 10-15 where row_len = ceil(columns * colors * bpc / 8).
    Returns the byte count written, or a negative status code.
    """
    if data_len < 0 or out_cap < 0:
        return -1
    if columns < 1 or colors < 1:
        return -1
    if bpc != 1 and bpc != 2 and bpc != 4 and bpc != 8 and bpc != 16:
        return -1
    var is_png = predictor >= 10 and predictor <= 15
    if predictor != 1 and predictor != 2 and not is_png:
        return -1

    if predictor == 1:
        # No prediction: verbatim copy of any length.
        if out_cap < data_len:
            return -3
        if data_len > 0:
            unsafe_memcpy(dest=out_buf, src=data, count=Int(data_len))
        return data_len

    var row_len = (colors * columns * bpc + 7) // 8
    # floor(colors * bpc / 8): 0 for sub-byte bpc with few colors (the
    # oracle's observable self-addition behaviour), else bytes per pixel.
    var bpp = (colors * bpc) // 8

    if predictor == 2:
        # TIFF Predictor 2: per-row horizontal differencing over the whole
        # buffer; a short final row is processed as-is (never padded).
        if out_cap < data_len:
            return -3
        if data_len == 0:
            return 0
        unsafe_memcpy(dest=out_buf, src=data, count=Int(data_len))
        var pos = Int64(0)
        while pos < data_len:
            var rlen = row_len
            if data_len - pos < rlen:
                rlen = data_len - pos
            for i in range(Int(bpp), Int(rlen)):
                var j = Int(pos) + i
                out_buf[unsafe_offset=j] = out_buf[unsafe_offset=j] +
                    out_buf[unsafe_offset=j - Int(bpp)]
            pos += rlen
        return data_len

    # PNG predictors 10-15: every row is prefixed with a filter byte.
    var stride = row_len + 1
    var rows = data_len // stride
    if data_len % stride != 0:
        rows += 1
    var need = rows * row_len
    if out_cap < need:
        return -3
    for r in range(Int(rows)):
        var in_base = Int64(r) * stride
        var f = data[unsafe_offset=Int(in_base)]
        if f > 4:
            return -5
        # Raw bytes present for this row (short only for a ragged final
        # row); missing bytes are zero-padded before filtering.
        var avail = row_len
        if in_base + stride > data_len:
            avail = data_len - in_base - 1
        var ob = Int64(r) * row_len
        for i in range(Int(avail)):
            out_buf[unsafe_offset=Int(ob) + i] = data[
                unsafe_offset=Int(in_base) + 1 + i
            ]
        for i in range(Int(avail), Int(row_len)):
            out_buf[unsafe_offset=Int(ob) + i] = 0
        var pb = ob - row_len  # previous reconstructed row (invalid at r == 0)
        if f == 0:
            continue  # None: the pre-seeded raw row is already correct
        elif f == 1:
            # Sub: out[i] += out[i - bpp]
            for i in range(Int(bpp), Int(row_len)):
                var j = Int(ob) + i
                out_buf[unsafe_offset=j] = out_buf[unsafe_offset=j] +
                    out_buf[unsafe_offset=j - Int(bpp)]
        elif f == 2:
            # Up: out[i] += prev[i] — no loop-carried dependency, vectorized.
            if r > 0:
                var i = 0
                while i + WIDTH <= Int(row_len):
                    out_buf.unsafe_store(
                        Int(ob) + i,
                        out_buf.unsafe_load[width=WIDTH](Int(ob) + i) +
                        out_buf.unsafe_load[width=WIDTH](Int(pb) + i),
                    )
                    i += WIDTH
                while i < Int(row_len):
                    out_buf[unsafe_offset=Int(ob) + i] = out_buf[
                        unsafe_offset=Int(ob) + i
                    ] + out_buf[unsafe_offset=Int(pb) + i]
                    i += 1
        elif f == 3:
            # Average: out[i] += floor((left + up) / 2)
            for i in range(Int(row_len)):
                var left = 0
                if i >= Int(bpp):
                    left = Int(out_buf[unsafe_offset=Int(ob) + i - Int(bpp)])
                var up = 0
                if r > 0:
                    up = Int(out_buf[unsafe_offset=Int(pb) + i])
                var j = Int(ob) + i
                out_buf[unsafe_offset=j] = out_buf[unsafe_offset=j] + UInt8(
                    (left + up) // 2
                )
        else:
            # Paeth: out[i] += paeth(left, up, up_left)
            for i in range(Int(row_len)):
                var a = 0
                if i >= Int(bpp):
                    a = Int(out_buf[unsafe_offset=Int(ob) + i - Int(bpp)])
                var b = 0
                if r > 0:
                    b = Int(out_buf[unsafe_offset=Int(pb) + i])
                var c = 0
                if r > 0 and i >= Int(bpp):
                    c = Int(out_buf[unsafe_offset=Int(pb) + i - Int(bpp)])
                var p = a + b - c
                var pa = abs(p - a)
                var pbb = abs(p - b)
                var pc = abs(p - c)
                var pred: Int
                if pa <= pbb and pa <= pc:
                    pred = a
                elif pbb <= pc:
                    pred = b
                else:
                    pred = c
                var j = Int(ob) + i
                out_buf[unsafe_offset=j] = out_buf[unsafe_offset=j] + UInt8(pred)
    return need


@export
def pdfmojo_lzw_decode(
    data: U8Ptr,
    data_len: Int64,
    early_change: Int64,
    out_ptr: U8PtrPtr,
) abi("C") -> Int64:
    """Decode one LZW stream (MSB-first codes, 9-12 bits, EarlyChange 0/1).

    Decoding stops at the EOD code or when fewer bits than the current code
    width remain; output decoded so far is returned either way. On success
    `out_ptr` receives a library-allocated buffer (free with
    `pdfmojo_free`) and the length is returned; a negative value is an error
    (and `out_ptr` is left untouched).
    """
    if data_len < 0 or (early_change != 0 and early_change != 1):
        return -1

    var prefix = unsafe_alloc[Int32](TABLE_SIZE)
    var suffix = unsafe_alloc[UInt8](TABLE_SIZE)
    # Scratch for emitting one string (max chain depth < TABLE_SIZE).
    var walk = unsafe_alloc[UInt8](TABLE_SIZE)

    var cap = 4 * data_len + 64
    if cap < 1024:
        cap = 1024
    var buf = unsafe_alloc[UInt8](Int(cap))
    var out_len = Int64(0)

    var acc = UInt64(0)  # MSB-first bit accumulator
    var nbits = 0
    var pos = Int64(0)
    var width = 9
    var next_code = FIRST_FREE
    var prev = -1
    var error = Int64(0)

    while True:
        while nbits < width and pos < data_len:
            acc = (acc << 8) | UInt64(data[unsafe_offset=Int(pos)])
            pos += 1
            nbits += 8
        if nbits < width:
            break  # truncated tail: return output decoded so far
        var code = Int((acc >> UInt64(nbits - width)) &
            UInt64((1 << width) - 1))
        nbits -= width

        if code == 256:  # ClearTable
            width = 9
            next_code = FIRST_FREE
            prev = -1
            continue
        if code == 257:  # EOD
            break
        if prev < 0:
            # First code after a clear (or stream start) must be a literal.
            if code >= 256:
                error = -2
                break
            if out_len + 1 > cap:
                cap = cap * 2
                var nb = unsafe_alloc[UInt8](Int(cap))
                unsafe_memcpy(dest=nb, src=buf, count=Int(out_len))
                buf.unsafe_free()
                buf = nb
            buf[unsafe_offset=Int(out_len)] = UInt8(code)
            out_len += 1
            prev = code
            continue

        # Lenient KwKwK: any code >= next_code decodes as
        # string(prev) + first_byte(string(prev)).
        var is_kwkwk = code >= next_code
        var emit = prev if is_kwkwk else code

        # Walk the prefix chain to collect string(emit) reversed.
        var length = 0
        var c = emit
        while c >= 256:
            walk[unsafe_offset=length] = suffix[unsafe_offset=c]
            length += 1
            c = Int(prefix[unsafe_offset=c])
        walk[unsafe_offset=length] = UInt8(c)
        length += 1
        var fc = walk[unsafe_offset=length - 1]  # root byte = first char

        var extra = length + (1 if is_kwkwk else 0)
        if out_len + Int64(extra) > cap:
            var new_cap = cap * 2
            if new_cap < out_len + Int64(extra) + 64:
                new_cap = out_len + Int64(extra) + 64
            var nb = unsafe_alloc[UInt8](Int(new_cap))
            unsafe_memcpy(dest=nb, src=buf, count=Int(out_len))
            buf.unsafe_free()
            buf = nb
            cap = new_cap
        for i in range(length):
            buf[unsafe_offset=Int(out_len)] = walk[
                unsafe_offset=length - 1 - i
            ]
            out_len += 1
        if is_kwkwk:
            buf[unsafe_offset=Int(out_len)] = fc
            out_len += 1

        # Add string(prev) + fc to the table (unless it is already full),
        # then grow the code width at (1 << width) - 1 — the oracle's
        # behaviour for both EarlyChange values.
        if next_code < TABLE_SIZE:
            prefix[unsafe_offset=next_code] = Int32(prev)
            suffix[unsafe_offset=next_code] = fc
            var added = next_code
            next_code += 1
            if width < 12 and next_code == (1 << width) - 1:
                width += 1
            prev = added if is_kwkwk else code
        else:
            if not is_kwkwk:
                prev = code

    suffix.unsafe_free()
    prefix.unsafe_free()
    walk.unsafe_free()
    if error != 0:
        buf.unsafe_free()
        return error
    out_ptr[] = buf
    return out_len


@export
def pdfmojo_free(ptr: U8Ptr) abi("C"):
    """Release a buffer returned by `pdfmojo_lzw_decode`."""
    ptr.unsafe_free()
