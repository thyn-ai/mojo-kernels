"""Vendored pure-Python basket walker (fallback backend).

Used when the native Mojo kernel is unavailable (unsupported platform,
missing shared library, ABI mismatch, or ``UPROOT_MOJO_DISABLE_NATIVE=1``).
It implements exactly the same per-entry layouts, validations, and error
codes as the kernel (see ``kernels/uproot-branches/src/uprootmojo.mojo``),
so both backends produce identical offsets/content buffers or fail with
identical codes.
"""

from __future__ import annotations

import numpy as np

from uproot_mojo._rootfile import BasketDataError

# Entry-layout modes (must match the kernel).
MODE_NUM_RAW = 0
MODE_NUM_HDR = 1
MODE_STR_ONE = 2
MODE_STR_VEC = 3
MODE_AUTO_NUM = 4
MODE_AUTO_STR = 5

# Status codes (must match the kernel).
OK = 0
ERR_MODE = 1
ERR_ITEMSIZE = 2
ERR_TRUNCATED = 3
ERR_BYTECOUNT = 4
ERR_COUNT = 5
ERR_STRING = 6
ERR_TOTAL = 7
ERR_BORDERS = 8

_ERROR_NAMES = {
    ERR_MODE: "ERR_MODE",
    ERR_ITEMSIZE: "ERR_ITEMSIZE",
    ERR_TRUNCATED: "ERR_TRUNCATED",
    ERR_BYTECOUNT: "ERR_BYTECOUNT",
    ERR_COUNT: "ERR_COUNT",
    ERR_STRING: "ERR_STRING",
    ERR_TOTAL: "ERR_TOTAL",
    ERR_BORDERS: "ERR_BORDERS",
}

_COLLECTION_HEADER_BYTES = 10
_K_BYTE_COUNT_MASK = 0x40000000


def error_name(code: int) -> str:
    return _ERROR_NAMES.get(code, f"ERR_{code}")


def raise_for_code(code: int, where: str) -> None:
    if code == OK:
        return
    raise BasketDataError(
        code, f"{where}: basket walk failed with {error_name(code)} ({code})"
    )


def _be32(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "big")


def _tstring_end(data: bytes, start: int, end: int) -> int:
    """End of the TString at start, or -1 (same rules as the kernel)."""
    if start >= end:
        return -1
    n = data[start]
    header = 1
    if n == 255:
        if start + 5 > end:
            return -1
        n = _be32(data, start + 1)
        header = 5
    stop = start + header + n
    if stop > end:
        return -1
    return stop


def _header_valid(data: bytes, start: int, end: int) -> bool:
    if end - start < 4:
        return False
    bc = _be32(data, start)
    return (bc & _K_BYTE_COUNT_MASK) != 0 and (bc & ~_K_BYTE_COUNT_MASK) == (
        end - start
    ) - 4


def _check_borders(borders: np.ndarray, data_len: int) -> None:
    if len(borders) and (borders[0] < 0 or borders[0] > data_len):
        raise BasketDataError(
            ERR_BORDERS, "ERR_BORDERS: borders start outside the data region"
        )
    if np.any(borders[1:] < borders[:-1]) or (len(borders) and borders[-1] > data_len):
        raise BasketDataError(
            ERR_BORDERS, "ERR_BORDERS: borders not monotonic within the data region"
        )


def _scan_num(data, borders, mode, itemsize):
    total_items = 0
    total_bytes = 0
    for i in range(len(borders) - 1):
        start, end = int(borders[i]), int(borders[i + 1])
        length = end - start
        if mode == MODE_NUM_HDR:
            if length < _COLLECTION_HEADER_BYTES:
                return ERR_TRUNCATED, 0, 0
            if not _header_valid(data, start, end):
                return ERR_BYTECOUNT, 0, 0
            n = _be32(data, start + 6)
            if n * itemsize != length - _COLLECTION_HEADER_BYTES:
                return ERR_COUNT, 0, 0
            total_items += n
            total_bytes += n * itemsize
        else:
            if length % itemsize:
                return ERR_COUNT, 0, 0
            total_items += length // itemsize
            total_bytes += length
    return OK, total_items, total_bytes


def _scan_str_one(data, borders):
    total_bytes = 0
    for i in range(len(borders) - 1):
        start, end = int(borders[i]), int(borders[i + 1])
        stop = _tstring_end(data, start, end)
        if stop < 0:
            return ERR_STRING, 0
        if stop != end:
            return ERR_BYTECOUNT, 0
        header = 5 if data[start] == 255 else 1
        total_bytes += end - start - header
    return OK, total_bytes


def _scan_str_vec(data, borders):
    total_items = 0
    total_bytes = 0
    for i in range(len(borders) - 1):
        start, end = int(borders[i]), int(borders[i + 1])
        if end - start < _COLLECTION_HEADER_BYTES:
            return ERR_TRUNCATED, 0, 0
        if not _header_valid(data, start, end):
            return ERR_BYTECOUNT, 0, 0
        n = _be32(data, start + 6)
        pos = start + _COLLECTION_HEADER_BYTES
        for _ in range(n):
            stop = _tstring_end(data, pos, end)
            if stop < 0:
                return ERR_STRING, 0, 0
            header = 5 if data[pos] == 255 else 1
            total_bytes += stop - pos - header
            pos = stop
        if pos != end:
            return ERR_COUNT, 0, 0
        total_items += n
    return OK, total_items, total_bytes


def scan_basket(data, borders, mode, itemsize):
    """Validate a basket; return (code, detected_mode, total_items, total_bytes)."""
    _check_borders(borders, len(data))
    if mode == MODE_AUTO_NUM:
        if itemsize not in (4, 8):
            return ERR_ITEMSIZE, mode, 0, 0
        detected = MODE_NUM_RAW
        for i in range(len(borders) - 1):
            if (borders[i + 1] - borders[i]) % itemsize:
                detected = MODE_NUM_HDR
                break
        code, total_items, total_bytes = _scan_num(data, borders, detected, itemsize)
        return code, detected, total_items, total_bytes
    if mode == MODE_AUTO_STR:
        code, total_items, total_bytes = _scan_str_vec(data, borders)
        if code == OK:
            return OK, MODE_STR_VEC, total_items, total_bytes
        code_one, total_bytes = _scan_str_one(data, borders)
        if code_one == OK:
            return OK, MODE_STR_ONE, len(borders) - 1, total_bytes
        return code, mode, 0, 0
    if mode in (MODE_NUM_RAW, MODE_NUM_HDR):
        if itemsize not in (4, 8):
            return ERR_ITEMSIZE, mode, 0, 0
        code, total_items, total_bytes = _scan_num(data, borders, mode, itemsize)
        return code, mode, total_items, total_bytes
    if mode == MODE_STR_ONE:
        code, total_bytes = _scan_str_one(data, borders)
        return code, mode, len(borders) - 1, total_bytes
    if mode == MODE_STR_VEC:
        code, total_items, total_bytes = _scan_str_vec(data, borders)
        return code, mode, total_items, total_bytes
    return ERR_MODE, mode, 0, 0


def fill_basket(data, borders, mode, itemsize, total_items, total_bytes):
    """Walk a basket into (offsets, content, string_offsets); raises on error."""
    offsets = np.zeros(len(borders), dtype=np.int64)
    content = bytearray()
    string_offsets = [0] if mode == MODE_STR_VEC else None

    def guard(items_done, bytes_done):
        if items_done > total_items or bytes_done > total_bytes:
            raise BasketDataError(ERR_TOTAL, "fill totals exceed scan totals")

    items_done = 0
    for i in range(len(borders) - 1):
        start, end = int(borders[i]), int(borders[i + 1])
        length = end - start
        if mode == MODE_NUM_RAW:
            if length % itemsize:
                raise BasketDataError(ERR_COUNT, "entry byte length not divisible by itemsize")
            n = length // itemsize
            guard(items_done + n, len(content) + length)
            content += data[start:end]
            items_done += n
            offsets[i + 1] = items_done
        elif mode == MODE_NUM_HDR:
            if length < _COLLECTION_HEADER_BYTES:
                raise BasketDataError(ERR_TRUNCATED, "entry shorter than the collection header")
            if not _header_valid(data, start, end):
                raise BasketDataError(ERR_BYTECOUNT, "invalid collection byte count")
            n = _be32(data, start + 6)
            if n * itemsize != length - _COLLECTION_HEADER_BYTES:
                raise BasketDataError(ERR_COUNT, "item count inconsistent with byte length")
            guard(items_done + n, len(content) + n * itemsize)
            content += data[start + _COLLECTION_HEADER_BYTES : end]
            items_done += n
            offsets[i + 1] = items_done
        elif mode == MODE_STR_ONE:
            stop = _tstring_end(data, start, end)
            if stop < 0:
                raise BasketDataError(ERR_STRING, "invalid TString length prefix")
            if stop != end:
                raise BasketDataError(ERR_BYTECOUNT, "trailing bytes after std::string")
            header = 5 if data[start] == 255 else 1
            guard(items_done + 1, len(content) + length - header)
            content += data[start + header : end]
            items_done += 1
            offsets[i + 1] = len(content)
        elif mode == MODE_STR_VEC:
            if length < _COLLECTION_HEADER_BYTES:
                raise BasketDataError(ERR_TRUNCATED, "entry shorter than the collection header")
            if not _header_valid(data, start, end):
                raise BasketDataError(ERR_BYTECOUNT, "invalid collection byte count")
            n = _be32(data, start + 6)
            guard(items_done + n, len(content))
            pos = start + _COLLECTION_HEADER_BYTES
            for _ in range(n):
                stop = _tstring_end(data, pos, end)
                if stop < 0:
                    raise BasketDataError(ERR_STRING, "invalid TString length prefix")
                header = 5 if data[pos] == 255 else 1
                guard(items_done + 1, len(content) + (stop - pos - header))
                content += data[pos + header : stop]
                items_done += 1
                string_offsets.append(len(content))
                pos = stop
            if pos != end:
                raise BasketDataError(ERR_COUNT, "string walk did not land on the entry border")
            offsets[i + 1] = items_done
        else:
            raise BasketDataError(ERR_MODE, f"unknown fill mode {mode}")

    if items_done != total_items or len(content) != total_bytes:
        raise BasketDataError(ERR_TOTAL, "fill totals disagree with scan totals")
    return (
        offsets,
        bytes(content),
        np.asarray(string_offsets, dtype=np.int64) if string_offsets is not None else None,
    )


def walk_basket(data, borders, mode, itemsize, where):
    """Scan + fill one basket. Same signature/semantics as the native backend."""
    code, detected, total_items, total_bytes = scan_basket(data, borders, mode, itemsize)
    raise_for_code(code, where)
    offsets, content, string_offsets = fill_basket(
        data, borders, detected, itemsize, total_items, total_bytes
    )
    return detected, offsets, content, string_offsets
