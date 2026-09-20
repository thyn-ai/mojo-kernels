"""Public API: batch hive-tree walking and pool-header constraint validation.

``walk_hive`` traverses the key-node tree of a flat Windows registry hive
image (.dat file: regf base block + hbins) in exactly the pre-order a
``volatility3.framework.layers.registry.RegistryHive.visit_nodes`` visitor
observes, emitting one ``(offset, type, value)`` tuple per visited key node.
``scan_pool_headers`` validates pool-header tag hits in a byte buffer against
a constraint set with exactly the per-hit semantics of
``volatility3.plugins.windows.poolscanner.PoolHeaderScanner.__call__``.

Both entry points run on the native Mojo kernel when available and on the
vendored pure-Python reference otherwise; input validation and result
shaping are shared in this module, so the backends cannot disagree about
inputs. Outputs are deterministic and identical across backends (the
differential suite checks both against the volatility3 oracle).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple, Union

from vol_mojo import _reference
from vol_mojo._native import (
    LAYOUT_X64,
    LAYOUT_X86,
    NativeKernelError,
    NativeUnavailable,
    _load,
    scan_pool_headers_native,
    walk_hive_native,
)

_BASE_BLOCK_SIZE = 0x1000
_ROOT_CELL_FALLBACK = 0x20
_NODE_TYPE = "_CM_KEY_NODE"

_LAYOUTS = {"x64": LAYOUT_X64, "x86": LAYOUT_X86}


class HiveError(ValueError):
    """Malformed hive image or walk input (structured, fail-fast)."""


class PoolScanError(ValueError):
    """Malformed pool-scan input (structured, fail-fast)."""


@dataclasses.dataclass(frozen=True)
class PoolConstraint:
    """One pool-tag constraint, mirroring poolscanner.PoolConstraint's tests.

    Attributes
    ----------
    tag : bytes
        The 4-byte (or longer/shorter) pool tag to scan for. Tags must be
        unique within a scan.
    size : (int, int) or None
        Inclusive (min, max) bounds for ``alignment * BlockSize``; either
        side may be 0/None for "no bound" (mirroring the oracle's
        truthiness tests).
    page_type : int or None
        Bitmask of PAGE_TYPE_PAGED (1) | PAGE_TYPE_NONPAGED (2) |
        PAGE_TYPE_FREE (4); None/0 disables the pool-type test.
    index : (int, int) or None
        Inclusive (min, max) bounds for PoolIndex; 0/None disables a side.
    """

    tag: bytes
    size: Optional[Tuple[Optional[int], Optional[int]]] = None
    page_type: Optional[int] = None
    index: Optional[Tuple[Optional[int], Optional[int]]] = None


PAGE_TYPE_PAGED = 1
PAGE_TYPE_NONPAGED = 2
PAGE_TYPE_FREE = 4


def _as_bytes(name: str, value: object) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)  # one defensive copy; also pins the buffer
    raise HiveError(f"{name} must be a bytes-like object, got {type(value).__name__}")


def _as_bytes_pool(name: str, value: object) -> bytes:
    try:
        return _as_bytes(name, value)
    except HiveError as exc:
        raise PoolScanError(str(exc)) from None


def _validate_cell_index(name: str, value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HiveError(f"{name} must be an integer, got {value!r}")
    if value < 0 or value > maximum:
        raise HiveError(f"{name} must be within [0, {maximum:#x}], got {value:#x}")
    return value


def walk_hive(
    hive: object,
    *,
    root_cell_offset: Optional[int] = None,
    storage_length: Optional[int] = None,
    volatile_storage_length: int = 0,
) -> list[tuple[int, str, str]]:
    """Walk a flat registry hive image in ``visit_nodes`` pre-order.

    Parameters
    ----------
    hive : bytes-like
        The complete flat hive image: 4 KiB ``regf`` base block followed by
        the hive bins (hbins). This is the on-disk layout of Windows
        ``SOFTWARE``/``SYSTEM``/``.dat`` hive files. Hives carved out of a
        memory image through an in-memory HMAP are out of scope (see the
        package README).
    root_cell_offset : int or None
        Cell index of the root key node. Default: read it from the base
        block exactly like the oracle's ``root_cell_offset`` property
        (``RootCell`` when the base block signature is ``regf``, else
        ``0x20``).
    storage_length : int or None
        Length of the non-volatile hive storage in bytes (the invalid-cell
        bound of the oracle's address translation). Default:
        ``len(hive) - 0x1000`` — the full bins area of a flat file.
    volatile_storage_length : int
        Length of the volatile storage. Flat .dat files carry no volatile
        storage; keep the default 0 (volatile cells are then skipped, as
        the oracle does when its volatile storage is empty).

    Returns
    -------
    list of (offset, type, value)
        One tuple per visited key node, in ``visit_nodes`` pre-order:
        ``offset`` is the node's hive-space struct offset (cell index + 4),
        ``type`` is ``"_CM_KEY_NODE"`` (every visited node is a key node by
        construction), ``value`` is the node's name decoded as latin-1.

    Raises
    ------
    HiveError
        On invalid input, or when the image is malformed exactly where the
        oracle's traversal would raise (bad root node, truncated visited
        node or index list). Cells that merely fail to decode inside a
        subkey list are skipped, mirroring the oracle.
    """
    data = _as_bytes("hive", hive)
    if len(data) < _BASE_BLOCK_SIZE:
        raise HiveError(
            f"hive image must be at least {_BASE_BLOCK_SIZE} bytes (base block), "
            f"got {len(data)}"
        )
    if root_cell_offset is None:
        # Mirrors RegistryHive.root_cell_offset: "regf" -> RootCell else 0x20.
        if data[0:4] == b"regf":
            root_cell = int.from_bytes(data[0x24:0x28], "little")
        else:
            root_cell = _ROOT_CELL_FALLBACK
    else:
        root_cell = _validate_cell_index("root_cell_offset", root_cell_offset, 0xFFFFFFFF)
    if storage_length is None:
        storage_length = len(data) - _BASE_BLOCK_SIZE
    storage_len = _validate_cell_index("storage_length", storage_length, 0xFFFFFFFF)
    volatile_len = _validate_cell_index(
        "volatile_storage_length", volatile_storage_length, 0xFFFFFFFF
    )

    try:
        _load()  # fail fast here if the kernel cannot be used at all
        offsets, names = walk_hive_native(data, root_cell, storage_len, volatile_len)
    except NativeUnavailable:
        try:
            offsets, names = _reference.walk_hive_reference(
                data, root_cell, storage_len, volatile_len
            )
        except _reference._FatalHive as exc:
            raise HiveError(f"malformed hive image: {exc}") from None
        except _reference._GuardTripped:
            raise HiveError(
                "malformed hive image: iteration guard tripped "
                "(pathological or cyclic structure)"
            ) from None
    except NativeKernelError as exc:
        # Kernel status 2/5 are the native spellings of the shared
        # malformed-hive paths (the oracle raises on the same inputs).
        raise HiveError(f"malformed hive image: {exc}") from None
    return [(off, _NODE_TYPE, name.decode("latin-1")) for off, name in zip(offsets, names)]


def _validate_bound(owner: str, name: str, value: object) -> int:
    """A size/index bound: None or 0 disables; must be a non-negative int."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise PoolScanError(f"{owner}{name} must be an integer or None, got {value!r}")
    if value < 0:
        raise PoolScanError(f"{owner}{name} must be >= 0, got {value}")
    return value


def _normalize_constraints(
    constraints: object,
) -> list[tuple[PoolConstraint, tuple[bytes, int, int, int, int, int]]]:
    """Validate constraints and build (original, kernel-row) pairs.

    Accepts vol_mojo.PoolConstraint instances or any duck-typed object with
    ``tag``/``size``/``page_type``/``index`` attributes (e.g. the oracle's
    own PoolConstraint). Kernel rows are (tag, size_min, size_max,
    page_type, index_min, index_max) sorted with descending tag length so
    the first match at a position is the longest — the exact semantics of
    the oracle's regex-trie scanner.
    """
    if not isinstance(constraints, Sequence) or isinstance(constraints, (bytes, bytearray)):
        raise PoolScanError("constraints must be a sequence of PoolConstraint-like objects")
    if len(constraints) == 0:
        raise PoolScanError("constraints must not be empty")
    pairs: list[tuple[PoolConstraint, tuple[bytes, int, int, int, int, int]]] = []
    seen: set[bytes] = set()
    for i, raw in enumerate(constraints):
        owner = f"constraints[{i}]."
        try:
            tag = raw.tag  # type: ignore[union-attr]
            size = raw.size  # type: ignore[union-attr]
            page_type = raw.page_type  # type: ignore[union-attr]
            index = raw.index  # type: ignore[union-attr]
        except AttributeError:
            raise PoolScanError(
                f"constraints[{i}] must expose tag/size/page_type/index attributes"
            ) from None
        if not isinstance(tag, (bytes, bytearray)) or len(tag) == 0:
            raise PoolScanError(f"{owner}tag must be non-empty bytes, got {tag!r}")
        tag_b = bytes(tag)
        if tag_b in seen:
            raise PoolScanError(
                f"constraint tag is used for more than one constraint: {tag_b!r}"
            )
        seen.add(tag_b)
        if size is None:
            size_min = size_max = 0
        else:
            if not isinstance(size, tuple) or len(size) != 2:
                raise PoolScanError(f"{owner}size must be a (min, max) tuple or None")
            size_min = _validate_bound(owner, "size[0]", size[0])
            size_max = _validate_bound(owner, "size[1]", size[1])
        if page_type is None:
            ptype = 0
        else:
            if isinstance(page_type, bool) or not isinstance(page_type, int):
                raise PoolScanError(
                    f"{owner}page_type must be an int bitmask or None, got {page_type!r}"
                )
            ptype = int(page_type)
            if ptype < 0 or ptype > 0x7FFFFFFF:
                raise PoolScanError(f"{owner}page_type out of range: {ptype}")
        if index is None:
            index_min = index_max = 0
        else:
            if not isinstance(index, tuple) or len(index) != 2:
                raise PoolScanError(f"{owner}index must be a (min, max) tuple or None")
            index_min = _validate_bound(owner, "index[0]", index[0])
            index_max = _validate_bound(owner, "index[1]", index[1])
        original = raw if isinstance(raw, PoolConstraint) else PoolConstraint(
            tag_b,
            None if size is None else (size[0], size[1]),
            page_type,
            None if index is None else (index[0], index[1]),
        )
        pairs.append(
            (original, (tag_b, size_min, size_max, ptype, index_min, index_max))
        )
    # Sort by descending tag length (stable): longest match at a position.
    pairs.sort(key=lambda pair: -len(pair[1][0]))
    return pairs


def scan_pool_headers(
    data: object,
    constraints: object,
    *,
    alignment: int = 0x10,
    layout: str = "x64",
    vista_semantics: bool = True,
) -> list[tuple[int, bytes]]:
    """Scan a buffer for pool tags and validate each header per constraint.

    Equivalent to feeding ``data`` to
    ``poolscanner.PoolHeaderScanner.__call__`` and collecting the offsets of
    the headers that pass their constraint's size/pool-type/index tests.

    Parameters
    ----------
    data : bytes-like
        The buffer to scan (e.g. a physical-memory chunk or page file).
    constraints : sequence
        ``PoolConstraint`` rows (or duck-typed equivalents exposing
        ``tag``/``size``/``page_type``/``index``). Tags must be unique.
    alignment : int
        Pool block alignment multiplier for the size test (0x10 on x64,
        0x8 on x86 — exactly the oracle's values).
    layout : {'x64', 'x86'}
        ``_POOL_HEADER`` byte layout: ``'x64'`` covers both bundled x64
        tables (poolheader-x64 and poolheader-x64-win7 are byte-identical);
        ``'x86'`` is the 32-bit bitfield layout.
    vista_semantics : bool
        Pool-type mapping: True for Vista-and-later (paged == odd
        PoolType), False for pre-Vista (paged == even non-zero PoolType) —
        the oracle's POOL_HEADER_VISTA vs POOL_HEADER classes.

    Returns
    -------
    list of (offset, tag)
        ``(header_offset, tag)`` for every hit whose header passed, in
        scan order (ascending offset, leftmost non-overlapping tags).
        ``header_offset`` is exactly what the oracle yields as
        ``header.vol.offset``: the raw tag-offset-minus-4 normalized
        through the scan layer's address mask
        ``(1 << ceil(log2(len(data) - 1))) - 1`` — the two differ only
        for tags in the first 4 bytes of the buffer (negative raw
        offsets wrap through the mask).

    Raises
    ------
    PoolScanError
        On invalid input (bad constraints, alignment, layout, or a buffer
        smaller than 2 bytes — the oracle cannot build a scan layer for
        those either).
    """
    buf = _as_bytes_pool("data", data)
    if len(buf) < 2:
        raise PoolScanError(
            f"data must be at least 2 bytes to back a scan layer, got {len(buf)}"
        )
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise PoolScanError(f"alignment must be a positive integer, got {alignment!r}")
    if layout not in _LAYOUTS:
        raise PoolScanError(f"layout must be one of {sorted(_LAYOUTS)}, got {layout!r}")
    if not isinstance(vista_semantics, bool):
        raise PoolScanError(f"vista_semantics must be a bool, got {vista_semantics!r}")
    pairs = _normalize_constraints(constraints)
    rows = [row for _, row in pairs]
    layout_id = _LAYOUTS[layout]

    try:
        _load()  # fail fast here if the kernel cannot be used at all
        hits = scan_pool_headers_native(buf, rows, alignment, layout_id, vista_semantics)
    except NativeUnavailable:
        hits = _reference.scan_pool_headers_reference(
            buf, rows, alignment, layout_id, vista_semantics
        )
    except NativeKernelError as exc:
        raise PoolScanError(f"native pool scan failed: {exc}") from None
    return [(off, rows[idx][0]) for off, idx in hits]
