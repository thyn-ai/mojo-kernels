"""Clean-room ROOT TBranch basket walker for std::vector and std::string data.

Written fresh from the documented ROOT I/O serialization format (ROOT's
io/doc/TFile and ttree.md specifications): split-level-0 ``std::vector<T>``
and ``std::string`` branch baskets, as produced both by uproot.recreate
(raw items + an entry-offset table) and by CERN ROOT (per-entry collection
headers). No third-party Mojo code is used or adapted.

Exported C ABI (batch-shaped: one whole uncompressed basket in, whole
offsets/content buffers out, so FFI cost amortizes to noise):

    int32_t uprootmojo_abi_version(void)
    int32_t uprootmojo_scan(data, borders, n_entries, mode, itemsize,
                            out_mode, out_total_items, out_total_bytes)
    int32_t uprootmojo_fill(data, borders, n_entries, mode, itemsize,
                            out_offsets, out_content, out_string_offsets)

``data`` is the uncompressed basket data region; ``borders`` holds
n_entries+1 byte offsets (entry starts plus the end of the data region).
The caller (Python wrapper) allocates every output buffer from the scan
totals, so this library never allocates.

Per-entry layouts (all integers big-endian, per the ROOT spec):

  * MODE_NUM_RAW (uproot.recreate jagged numeric): entry = raw items.
    Item count = entry byte length / itemsize.
  * MODE_NUM_HDR (CERN ROOT std::vector<T>): entry = 4-byte byte count
    (kByteCountMask 0x40000000 set, value counts the bytes after the
    field) + 2-byte version + 4-byte item count + raw items.
  * MODE_STR_ONE (std::string branch): entry = one TString: 1-byte length
    (<255), or 0xFF followed by a 4-byte length, then the bytes.
  * MODE_STR_VEC (CERN ROOT std::vector<std::string>): entry = collection
    header as MODE_NUM_HDR + count TStrings.
  * MODE_AUTO_NUM / MODE_AUTO_STR: scan detects RAW-vs-HDR (numeric) or
    ONE-vs-VEC (string) with strict validation and reports the detected
    mode through out_mode; fill is then called with the concrete mode.

Content bytes are copied verbatim (big-endian preserved, as stored); the
wrapper byteswaps numeric content to native endianness for delivery, so the
returned arrays match uproot's delivered buffers byte-for-byte.
"""

from std.memory import Pointer, unsafe_memcpy
from std.origin import MutUntrackedOrigin

comptime ABI_VERSION: Int32 = 1

# Entry-layout modes.
comptime MODE_NUM_RAW: Int32 = 0
comptime MODE_NUM_HDR: Int32 = 1
comptime MODE_STR_ONE: Int32 = 2
comptime MODE_STR_VEC: Int32 = 3
comptime MODE_AUTO_NUM: Int32 = 4
comptime MODE_AUTO_STR: Int32 = 5

# Status codes (stable part of the ABI; the wrapper maps them to exceptions).
comptime OK: Int32 = 0
comptime ERR_MODE: Int32 = 1  # unknown mode, or fill called with an AUTO mode
comptime ERR_ITEMSIZE: Int32 = 2  # numeric itemsize not in {4, 8}
comptime ERR_TRUNCATED: Int32 = 3  # entry/header overruns the data region
comptime ERR_BYTECOUNT: Int32 = 4  # collection byte-count mask/length invalid
comptime ERR_COUNT: Int32 = 5  # item count inconsistent with byte length
comptime ERR_STRING: Int32 = 6  # TString length prefix invalid/truncated
comptime ERR_TOTAL: Int32 = 7  # fill totals exceed scan totals (internal)
comptime ERR_BORDERS: Int32 = 8  # borders not monotonic within the data region

comptime K_BYTE_COUNT_MASK: UInt32 = 0x40000000
comptime COLLECTION_HEADER_BYTES: Int64 = 10  # 4 byte count + 2 version + 4 count

# C-side pointer spellings (untracked origin: the caller owns the lifetime
# of every buffer; this library allocates nothing).
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]


def _be32(data: U8Ptr, pos: Int64) -> UInt32:
    """Big-endian uint32 at byte position pos."""
    return (
        (UInt32(data[unsafe_offset=pos]) << 24)
        | (UInt32(data[unsafe_offset=pos + 1]) << 16)
        | (UInt32(data[unsafe_offset=pos + 2]) << 8)
        | UInt32(data[unsafe_offset=pos + 3])
    )


def _tstring_end(data: U8Ptr, start: Int64, end: Int64) -> Int64:
    """End position of the TString starting at `start`, or -1 on bad input.

    TString length prefix: one byte for lengths < 255, otherwise 0xFF and a
    4-byte big-endian length.
    """
    if start >= end:
        return -1
    var n = Int64(Int(data[unsafe_offset=start]))
    var header = Int64(1)
    if n == 255:
        if start + 5 > end:
            return -1
        n = Int64(Int(_be32(data, start + 1)))
        header = 5
    var stop = start + header + n
    if stop > end:
        return -1
    return stop


def _header_valid(data: U8Ptr, start: Int64, end: Int64) -> Bool:
    """True if the entry [start, end) carries a valid collection byte count.

    The byte count has kByteCountMask set and counts the bytes after its own
    4-byte field, i.e. value == (end - start) - 4.
    """
    if end - start < 4:
        return False
    var bc = _be32(data, start)
    return (bc & K_BYTE_COUNT_MASK) != 0 and Int64(Int(bc & ~K_BYTE_COUNT_MASK)) == (
        end - start
    ) - 4


def _borders_valid(borders: I64Ptr, n_entries: Int64, data_len: Int64) -> Bool:
    """Borders must be non-decreasing and stay inside the data region."""
    var prev = borders[unsafe_offset=0]
    if prev < 0 or prev > data_len:
        return False
    for i in range(1, Int(n_entries) + 1):
        var cur = borders[unsafe_offset=i]
        if cur < prev or cur > data_len:
            return False
        prev = cur
    return True


def _scan_num(
    data: U8Ptr,
    borders: I64Ptr,
    n_entries: Int64,
    mode: Int32,
    itemsize: Int64,
    out_total_items: I64Ptr,
    out_total_bytes: I64Ptr,
) -> Int32:
    """Scan numeric entries (MODE_NUM_RAW or MODE_NUM_HDR must be concrete)."""
    var total_items = Int64(0)
    var total_bytes = Int64(0)
    for i in range(Int(n_entries)):
        var start = borders[unsafe_offset=i]
        var end = borders[unsafe_offset=i + 1]
        var length = end - start
        if mode == MODE_NUM_HDR:
            if length < COLLECTION_HEADER_BYTES:
                return ERR_TRUNCATED
            if not _header_valid(data, start, end):
                return ERR_BYTECOUNT
            var n = Int64(Int(_be32(data, start + 6)))
            if n * itemsize != length - COLLECTION_HEADER_BYTES:
                return ERR_COUNT
            total_items += n
            total_bytes += n * itemsize
        else:  # MODE_NUM_RAW
            if length % itemsize != 0:
                return ERR_COUNT
            total_items += length // itemsize
            total_bytes += length
    out_total_items[unsafe_offset=0] = total_items
    out_total_bytes[unsafe_offset=0] = total_bytes
    return OK


def _scan_str_one(
    data: U8Ptr,
    borders: I64Ptr,
    n_entries: Int64,
    out_total_bytes: I64Ptr,
) -> Int32:
    """Scan std::string entries (one TString per entry, ending on the border)."""
    var total_bytes = Int64(0)
    for i in range(Int(n_entries)):
        var start = borders[unsafe_offset=i]
        var end = borders[unsafe_offset=i + 1]
        var stop = _tstring_end(data, start, end)
        if stop < 0:
            return ERR_STRING
        if stop != end:
            return ERR_BYTECOUNT  # trailing bytes: not a plain std::string entry
        var header = Int64(1)
        if data[unsafe_offset=start] == 255:
            header = 5
        total_bytes += end - start - header
    out_total_bytes[unsafe_offset=0] = total_bytes
    return OK


def _scan_str_vec(
    data: U8Ptr,
    borders: I64Ptr,
    n_entries: Int64,
    out_total_items: I64Ptr,
    out_total_bytes: I64Ptr,
) -> Int32:
    """Scan std::vector<std::string> entries (collection header + TStrings)."""
    var total_items = Int64(0)
    var total_bytes = Int64(0)
    for i in range(Int(n_entries)):
        var start = borders[unsafe_offset=i]
        var end = borders[unsafe_offset=i + 1]
        if end - start < COLLECTION_HEADER_BYTES:
            return ERR_TRUNCATED
        if not _header_valid(data, start, end):
            return ERR_BYTECOUNT
        var n = Int64(Int(_be32(data, start + 6)))
        var pos = start + COLLECTION_HEADER_BYTES
        for _ in range(Int(n)):
            var stop = _tstring_end(data, pos, end)
            if stop < 0:
                return ERR_STRING
            var header = Int64(1)
            if data[unsafe_offset=pos] == 255:
                header = 5
            total_bytes += stop - pos - header
            pos = stop
        if pos != end:
            return ERR_COUNT  # string walk must land exactly on the border
        total_items += n
    out_total_items[unsafe_offset=0] = total_items
    out_total_bytes[unsafe_offset=0] = total_bytes
    return OK


@export
def uprootmojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def uprootmojo_scan(
    data: U8Ptr,
    data_len: Int64,
    borders: I64Ptr,
    n_entries: Int64,
    mode: Int32,
    itemsize: Int32,
    out_mode: I64Ptr,
    out_total_items: I64Ptr,
    out_total_bytes: I64Ptr,
) abi("C") -> Int32:
    """Validate one basket and report the detected mode and output totals.

    For AUTO modes the detected concrete mode is written to out_mode; for
    concrete modes out_mode echoes the input. out_total_items is the item
    count (numeric items, or strings for STR_VEC; n_entries for STR_ONE);
    out_total_bytes is the exact content byte count.
    """
    if n_entries < 0 or data_len < 0:
        return ERR_BORDERS
    if not _borders_valid(borders, n_entries, data_len):
        return ERR_BORDERS
    out_total_items[unsafe_offset=0] = 0
    out_total_bytes[unsafe_offset=0] = 0

    if mode == MODE_AUTO_NUM:
        # Layout A (raw items) entries always divide evenly by itemsize;
        # Layout B (collection header) entries never do, because the 10-byte
        # header is 2 mod 4 and 2 mod 8. Empty baskets default to RAW.
        if itemsize != 4 and itemsize != 8:
            return ERR_ITEMSIZE
        var detected = MODE_NUM_RAW
        for i in range(Int(n_entries)):
            var length = borders[unsafe_offset=i + 1] - borders[unsafe_offset=i]
            if length % Int64(itemsize) != 0:
                detected = MODE_NUM_HDR
                break
        out_mode[unsafe_offset=0] = Int64(detected)
        return _scan_num(
            data, borders, n_entries, detected, Int64(itemsize),
            out_total_items, out_total_bytes,
        )

    if mode == MODE_AUTO_STR:
        # A valid collection byte count is decisive: real byte counts have
        # kByteCountMask set and equal entry_length - 4 exactly. std::string
        # entries (TString) can only alias this by coincidence, and then the
        # strict string-count walk below still has to land on the border.
        var vec_ok = _scan_str_vec(
            data, borders, n_entries, out_total_items, out_total_bytes
        )
        if vec_ok == OK:
            out_mode[unsafe_offset=0] = Int64(MODE_STR_VEC)
            return OK
        out_total_items[unsafe_offset=0] = 0
        out_total_bytes[unsafe_offset=0] = 0
        var one_ok = _scan_str_one(data, borders, n_entries, out_total_bytes)
        if one_ok == OK:
            out_total_items[unsafe_offset=0] = n_entries
            out_mode[unsafe_offset=0] = Int64(MODE_STR_ONE)
            return OK
        return vec_ok  # report the more specific (vector) failure

    if mode == MODE_NUM_RAW or mode == MODE_NUM_HDR:
        if itemsize != 4 and itemsize != 8:
            return ERR_ITEMSIZE
        out_mode[unsafe_offset=0] = Int64(mode)
        return _scan_num(
            data, borders, n_entries, mode, Int64(itemsize),
            out_total_items, out_total_bytes,
        )
    if mode == MODE_STR_ONE:
        out_mode[unsafe_offset=0] = Int64(mode)
        var rc = _scan_str_one(data, borders, n_entries, out_total_bytes)
        out_total_items[unsafe_offset=0] = n_entries
        return rc
    if mode == MODE_STR_VEC:
        out_mode[unsafe_offset=0] = Int64(mode)
        return _scan_str_vec(
            data, borders, n_entries, out_total_items, out_total_bytes
        )
    return ERR_MODE


@export
def uprootmojo_fill(
    data: U8Ptr,
    data_len: Int64,
    borders: I64Ptr,
    n_entries: Int64,
    mode: Int32,
    itemsize: Int32,
    out_offsets: I64Ptr,
    out_content: U8Ptr,
    out_string_offsets: I64Ptr,
    total_items: Int64,
    total_bytes: Int64,
) abi("C") -> Int32:
    """Fill offsets/content buffers for one basket (scan must have passed).

    out_offsets receives n_entries+1 cumulative item counts (items per entry:
    numeric items, or strings for STR_VEC, or bytes for STR_ONE). out_content
    receives the concatenated raw content bytes (big-endian preserved).
    out_string_offsets is only used for MODE_STR_VEC: total_items+1 cumulative
    string byte counts. The caller passes the scan totals; they are re-checked
    here so the two calls can never disagree silently.
    """
    if mode != MODE_NUM_RAW and mode != MODE_NUM_HDR and mode != MODE_STR_ONE and mode != MODE_STR_VEC:
        return ERR_MODE
    if (mode == MODE_NUM_RAW or mode == MODE_NUM_HDR) and itemsize != 4 and itemsize != 8:
        return ERR_ITEMSIZE
    if n_entries < 0 or data_len < 0:
        return ERR_BORDERS
    if not _borders_valid(borders, n_entries, data_len):
        return ERR_BORDERS

    var isz = Int64(itemsize)
    var items_done = Int64(0)
    var bytes_done = Int64(0)
    out_offsets[unsafe_offset=0] = 0
    if mode == MODE_STR_VEC:
        out_string_offsets[unsafe_offset=0] = 0

    for i in range(Int(n_entries)):
        var start = borders[unsafe_offset=i]
        var end = borders[unsafe_offset=i + 1]
        var length = end - start

        if mode == MODE_NUM_RAW:
            if length % isz != 0:
                return ERR_COUNT
            var n = length // isz
            if items_done + n > total_items or bytes_done + length > total_bytes:
                return ERR_TOTAL
            if length > 0:
                unsafe_memcpy(
                    dest=out_content.unsafe_offset(Int(bytes_done)),
                    src=data.unsafe_offset(Int(start)),
                    count=Int(length),
                )
            items_done += n
            bytes_done += length
            out_offsets[unsafe_offset=i + 1] = items_done

        elif mode == MODE_NUM_HDR:
            if length < COLLECTION_HEADER_BYTES:
                return ERR_TRUNCATED
            if not _header_valid(data, start, end):
                return ERR_BYTECOUNT
            var n = Int64(Int(_be32(data, start + 6)))
            var nbytes = n * isz
            if nbytes != length - COLLECTION_HEADER_BYTES:
                return ERR_COUNT
            if items_done + n > total_items or bytes_done + nbytes > total_bytes:
                return ERR_TOTAL
            if nbytes > 0:
                unsafe_memcpy(
                    dest=out_content.unsafe_offset(Int(bytes_done)),
                    src=data.unsafe_offset(Int(start + COLLECTION_HEADER_BYTES)),
                    count=Int(nbytes),
                )
            items_done += n
            bytes_done += nbytes
            out_offsets[unsafe_offset=i + 1] = items_done

        elif mode == MODE_STR_ONE:
            var stop = _tstring_end(data, start, end)
            if stop < 0:
                return ERR_STRING
            if stop != end:
                return ERR_BYTECOUNT
            var header = Int64(1)
            if end > start and data[unsafe_offset=start] == 255:
                header = 5
            var nbytes = length - header
            if bytes_done + nbytes > total_bytes:
                return ERR_TOTAL
            if nbytes > 0:
                unsafe_memcpy(
                    dest=out_content.unsafe_offset(Int(bytes_done)),
                    src=data.unsafe_offset(Int(start + header)),
                    count=Int(nbytes),
                )
            bytes_done += nbytes
            items_done += 1
            out_offsets[unsafe_offset=i + 1] = bytes_done

        else:  # MODE_STR_VEC
            if length < COLLECTION_HEADER_BYTES:
                return ERR_TRUNCATED
            if not _header_valid(data, start, end):
                return ERR_BYTECOUNT
            var n = Int64(Int(_be32(data, start + 6)))
            if items_done + n > total_items:
                return ERR_TOTAL
            var pos = start + COLLECTION_HEADER_BYTES
            for _ in range(Int(n)):
                var stop = _tstring_end(data, pos, end)
                if stop < 0:
                    return ERR_STRING
                var header = Int64(1)
                if data[unsafe_offset=pos] == 255:
                    header = 5
                var nbytes = stop - pos - header
                if bytes_done + nbytes > total_bytes:
                    return ERR_TOTAL
                if nbytes > 0:
                    unsafe_memcpy(
                        dest=out_content.unsafe_offset(Int(bytes_done)),
                        src=data.unsafe_offset(Int(pos + header)),
                        count=Int(nbytes),
                    )
                bytes_done += nbytes
                items_done += 1
                out_string_offsets[unsafe_offset=Int(items_done)] = bytes_done
                pos = stop
            if pos != end:
                return ERR_COUNT
            out_offsets[unsafe_offset=i + 1] = items_done

    if items_done != total_items or bytes_done != total_bytes:
        return ERR_TOTAL
    return OK
