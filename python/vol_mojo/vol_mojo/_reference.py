"""Vendored pure-Python reference implementation of the two kernels.

Used when the native Mojo kernel is unavailable (forced with
``VOL_MOJO_DISABLE_NATIVE=1``). It implements exactly the semantics
documented in ``kernels/volatility3-hive/src/volatility3hivemojo.mojo`` —
same traversal order, same skip/fatal split, same constraint checks — so
the two backends cannot disagree about outputs. The differential suite
runs against the volatility3 oracle once per backend.
"""

from __future__ import annotations

import re

_HIVE_BINS_BASE = 0x1000
_ADDR_MASK = 0x7FFFFFFF
_VOLATILE_BIT = 0x80000000

_SIG_NK = 0x6B6E
_SIG_LF = 0x666C
_SIG_LH = 0x686C
_SIG_RI = 0x6972

_NK_SUBKEY_LISTS = 0x1C
_NK_NAME_LENGTH = 0x48
_NK_NAME = 0x4C

_OPS_CAP = 1 << 30

_POOL_TAG_OFFSET = 4
_PAGE_TYPE_PAGED = 1
_PAGE_TYPE_NONPAGED = 2
_PAGE_TYPE_FREE = 4

_LAYOUT_X64 = 0
_LAYOUT_X86 = 1


class _FatalHive(Exception):
    """Internal: the hive image is malformed where the oracle raises."""


class _GuardTripped(Exception):
    """Internal: the iteration guard tripped (pathological image)."""


def _u16le(data: bytes, off: int) -> int:
    return data[off] | (data[off + 1] << 8)


def _u32le(data: bytes, off: int) -> int:
    return (
        data[off]
        | (data[off + 1] << 8)
        | (data[off + 2] << 16)
        | (data[off + 3] << 24)
    )


def _translate(
    off: int, need: int, storage_len: int, volatile_len: int, data_len: int
) -> int:
    """Cell-space offset -> file offset; -1 when not readable."""
    limit = volatile_len if off & _VOLATILE_BIT else storage_len
    if (off & _ADDR_MASK) > limit:
        return -1
    foff = _HIVE_BINS_BASE + (off & _ADDR_MASK)
    if foff + need > data_len:
        return -1
    return foff


def walk_hive_reference(
    data: bytes, root_cell: int, storage_len: int, volatile_len: int
) -> tuple[list[int], list[bytes]]:
    """Pre-order walk of the key-node tree; see core.walk_hive for the contract."""
    if len(data) < _HIVE_BINS_BASE:
        raise _FatalHive("image smaller than the 4 KiB base block")
    maxaddr = _VOLATILE_BIT | volatile_len
    offsets: list[int] = []
    names: list[bytes] = []
    stack = [root_cell]
    ops = 0
    data_len = len(data)

    def push_entries(cell: int, dstack: list[int]) -> None:
        nonlocal ops
        fsig = _translate(cell + 4, 4, storage_len, volatile_len, data_len)
        if fsig < 0:
            raise _FatalHive(f"index cell at {cell:#x} is not decodable")
        sig = _u16le(data, fsig)
        if sig == _SIG_RI:
            stride = 4
        elif sig in (_SIG_LH, _SIG_LF):
            stride = 8
        else:
            return
        count = _u16le(data, fsig + 2)
        base = fsig + 4
        run: list[int] = []
        for j in range(count):
            ops += 1
            if ops > _OPS_CAP:
                raise _GuardTripped
            eoff = base + j * stride
            if eoff + 4 > data_len:
                raise _FatalHive(f"index list at {cell:#x} is truncated")
            run.append(_u32le(data, eoff))
        dstack.extend(reversed(run))

    def descend(list_cell: int, children: list[int]) -> None:
        nonlocal ops
        ftop = _translate(list_cell + 4, 2, storage_len, volatile_len, data_len)
        if ftop < 0:
            return  # the oracle's signature read fails soft here: skip
        if _u16le(data, ftop) not in (_SIG_RI, _SIG_LH, _SIG_LF):
            return
        dstack: list[int] = []
        push_entries(list_cell, dstack)
        while dstack:
            e = dstack.pop()
            ops += 1
            if ops > _OPS_CAP:
                raise _GuardTripped
            if (e & _ADDR_MASK) > maxaddr:
                continue
            fe = _translate(e + 4, 2, storage_len, volatile_len, data_len)
            if fe < 0:
                continue
            esig = _u16le(data, fe)
            if esig == _SIG_NK:
                children.append(e)
            elif esig in (_SIG_RI, _SIG_LH, _SIG_LF):
                push_entries(e, dstack)
            # anything else (vk/sk/db/unknown) is skipped by the oracle

    while stack:
        cell = stack.pop()
        ops += 1
        if ops > _OPS_CAP:
            raise _GuardTripped
        fsig = _translate(cell + 4, 2, storage_len, volatile_len, data_len)
        if fsig < 0 or _u16le(data, fsig) != _SIG_NK:
            raise _FatalHive(f"node to visit at {cell:#x} is not a key node")
        fsl = _translate(cell + 4 + _NK_SUBKEY_LISTS, 8, storage_len, volatile_len, data_len)
        fnl = _translate(cell + 4 + _NK_NAME_LENGTH, 2, storage_len, volatile_len, data_len)
        if fsl < 0 or fnl < 0:
            raise _FatalHive(f"key node at {cell:#x} is truncated")
        name_len = _u16le(data, fnl)
        fname = _translate(
            cell + 4 + _NK_NAME, name_len, storage_len, volatile_len, data_len
        )
        if fname < 0:
            raise _FatalHive(f"key node name at {cell:#x} is truncated")
        offsets.append(cell + 4)
        names.append(data[fname : fname + name_len])
        children: list[int] = []
        for idx in range(2):
            descend(_u32le(data, fsl + idx * 4), children)
        stack.extend(reversed(children))
    return offsets, names


def scan_pool_headers_reference(
    data: bytes,
    constraints_sorted: list[tuple[bytes, int, int, int, int, int]],
    alignment: int,
    layout: int,
    vista_semantics: bool,
) -> list[tuple[int, int]]:
    """Batch pool-header validation; see core.scan_pool_headers for the contract."""
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    data_len = len(data)
    if data_len < 2:
        raise ValueError("data must be at least 2 bytes to back a scan layer")
    # The oracle constructs headers on a flat layer whose address mask is
    # (1 << ceil(log2(size - 1))) - 1; object offsets are normalized through
    # it (a negative raw header offset wraps, as do member offsets).
    pow2 = 1
    while pow2 < data_len - 1:
        pow2 <<= 1
    addr_mask = pow2 - 1
    hits: list[tuple[int, int]] = []
    # Leftmost, non-overlapping, longest-match-at-position tag scan. The
    # alternation is ordered by descending tag length (rows arrive
    # pre-sorted), so the first alternative that can match at a position
    # is the longest — exactly the oracle's regex-trie semantics — and
    # re.finditer resumes after each match, exactly like the oracle.
    pattern = re.compile(
        b"|".join(re.escape(row[0]) for row in constraints_sorted)
    )
    row_index = {row[0]: i for i, row in enumerate(constraints_sorted)}
    for match in pattern.finditer(data):
        pos = match.start()
        matched = row_index[match.group()]
        row = constraints_sorted[matched]
        tag, size_min, size_max, page_type, index_min, index_max = row
        h = (pos - _POOL_TAG_OFFSET) & addr_mask
        need_block = size_min != 0 or size_max != 0
        need_type = page_type != 0
        need_index = index_min != 0 or index_max != 0
        # u16 member reads: BlockSize/PoolType word at relative offset 2
        # (member offsets are masked too), PoolIndex word at offset 0.
        roff2 = (h + 2) & addr_mask
        passes = True
        if (need_block or need_type) and roff2 + 2 > data_len:
            passes = False
        if passes and need_index and h + 2 > data_len:
            passes = False
        if passes and need_block:
            if layout == _LAYOUT_X64:
                block_size = data[roff2]
            else:
                block_size = _u16le(data, roff2) & 0x1FF
            total = alignment * block_size
            if size_min != 0 and total < size_min:
                passes = False
            if passes and size_max != 0 and total > size_max:
                passes = False
        if passes and need_type:
            if layout == _LAYOUT_X64:
                pool_type = data[roff2 + 1]
            else:
                pool_type = (_u16le(data, roff2) >> 9) & 0x7F
            is_free = pool_type == 0
            if vista_semantics:
                is_paged = pool_type % 2 == 1
                is_nonpaged = pool_type % 2 == 0 and pool_type > 0
            else:
                is_paged = pool_type % 2 == 0 and pool_type > 0
                is_nonpaged = pool_type % 2 == 1
            type_ok = (
                (page_type & _PAGE_TYPE_FREE and is_free)
                or (page_type & _PAGE_TYPE_NONPAGED and is_nonpaged)
                or (page_type & _PAGE_TYPE_PAGED and is_paged)
            )
            if not type_ok:
                passes = False
        if passes and need_index:
            if layout == _LAYOUT_X64:
                pool_index = data[h + 1]
            else:
                pool_index = (_u16le(data, h) >> 9) & 0x7F
            if index_min != 0 and pool_index < index_min:
                passes = False
            if passes and index_max != 0 and pool_index > index_max:
                passes = False
        if passes:
            hits.append((h, matched))
    return hits
