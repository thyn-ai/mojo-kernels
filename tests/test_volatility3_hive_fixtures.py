"""Fixtures for the volatility3-hive differential suite (not a test module).

Contains:

* ``HiveBuilder`` — a generator of small but valid flat registry hive
  images (regf base block + hbins + nk/lh/lf/ri/vk cells), so the suite
  never needs a copyrighted Windows hive. Everything is deterministic.
* ``add_management`` — appends the in-memory management superstructure
  (_CMHIVE/_HHIVE/_HMAP_*) the oracle's ``RegistryHive`` layer needs to
  walk a hive that lives in a flat file, plus the fake ``nt_symbols`` ISF
  those structs are typed with (``fake_nt_isf_json``). The on-disk hive
  bytes themselves are real-format; only the superstructure is synthetic,
  and the HMAP reproduces the identity page mapping of a flat .dat file.
* ``oracle_walk_tuples`` / ``oracle_pool_hits`` — harnesses driving the
  real volatility3 (the oracle) over the generated artifacts. volatility3
  is imported lazily so this module loads without the oracle installed.
"""

from __future__ import annotations

import functools
import json
import struct
from typing import Optional

PAGE = 0x1000
HBIN_HEADER = 0x20


def _pad8(n: int) -> int:
    return (n + 7) & ~7


class HiveBuilder:
    """Builds a flat .dat hive image; cells are 8-aligned, hbins page-sized."""

    def __init__(self) -> None:
        self._hbins: list[bytearray] = []  # finalized cell areas
        self._cur = bytearray()
        self._bins_base = 0  # bins offset of the current hbin

    def _cur_used(self) -> int:
        return HBIN_HEADER + len(self._cur)

    def _close_hbin(self) -> None:
        self._hbins.append(self._cur)
        self._bins_base += (self._cur_used() + PAGE - 1) & ~(PAGE - 1)
        self._cur = bytearray()

    def add_cell(self, data: bytes) -> int:
        """Append an allocated cell; returns its hcell index (bins offset)."""
        total = _pad8(4 + len(data))
        if self._cur_used() + total > PAGE:
            self._close_hbin()
        cell_off = self._bins_base + self._cur_used()
        self._cur += struct.pack("<i", -total) + data
        self._cur += b"\x00" * (total - 4 - len(data))
        return cell_off

    @staticmethod
    def nk(
        name: bytes,
        *,
        flags: int = 0x20,
        parent: int = 0xFFFFFFFF,
        subkey_count: int = 0,
        subkey_list: int = 0xFFFFFFFF,
        subkey_count_volatile: int = 0,
        subkey_list_volatile: int = 0xFFFFFFFF,
        value_count: int = 0,
        value_list: int = 0xFFFFFFFF,
    ) -> bytes:
        """_CM_KEY_NODE cell payload (without the 4-byte cell size)."""
        d = bytearray(76)
        struct.pack_into("<H", d, 0, 0x6B6E)  # "nk"
        struct.pack_into("<H", d, 2, flags)
        struct.pack_into("<I", d, 16, parent & 0xFFFFFFFF)
        struct.pack_into("<I", d, 20, subkey_count)
        struct.pack_into("<I", d, 24, subkey_count_volatile)
        struct.pack_into("<I", d, 28, subkey_list & 0xFFFFFFFF)
        struct.pack_into("<I", d, 32, subkey_list_volatile & 0xFFFFFFFF)
        struct.pack_into("<I", d, 36, value_count)  # ValueList.Count
        struct.pack_into("<I", d, 40, value_list & 0xFFFFFFFF)  # ValueList.List
        struct.pack_into("<H", d, 72, len(name))  # NameLength
        return bytes(d) + name

    @staticmethod
    def index(sig: bytes, entries: list) -> bytes:
        """lh/lf: entries = [(cell, hash)]; ri: entries = [cell]."""
        d = bytearray(sig + b"\x00\x00")
        struct.pack_into("<H", d, 2, len(entries))
        if sig in (b"lh", b"lf"):
            for cell, h in entries:
                d += struct.pack("<II", cell & 0xFFFFFFFF, h & 0xFFFFFFFF)
        elif sig == b"ri":
            for cell in entries:
                d += struct.pack("<I", cell & 0xFFFFFFFF)
        else:
            raise ValueError(sig)
        return bytes(d)

    @staticmethod
    def vk(name: bytes, vtype: int = 4, data: int = 0) -> bytes:
        """_CM_KEY_VALUE cell payload (inline data)."""
        d = bytearray(20)
        struct.pack_into("<H", d, 0, 0x6B76)  # "vk"
        struct.pack_into("<H", d, 2, len(name))
        struct.pack_into("<I", d, 4, 0x80000004)  # DataLength: 4, inline
        struct.pack_into("<I", d, 8, data)
        struct.pack_into("<I", d, 12, vtype)
        struct.pack_into("<H", d, 16, 1)  # flags: ascii name
        return bytes(d) + name

    def build_dat(self, root_cell: int) -> tuple[bytes, int]:
        """Assemble the .dat image; returns (image, hbins_length)."""
        if self._cur or not self._hbins:
            self._close_hbin()
        hbins_len = self._bins_base + (self._cur_used() + PAGE - 1) & ~(PAGE - 1)
        # _close_hbin appended the last area; recompute exactly.
        hbins_len = sum(
            (HBIN_HEADER + len(area) + PAGE - 1) & ~(PAGE - 1) for area in self._hbins
        )
        base = bytearray(PAGE)
        base[0:4] = b"regf"
        struct.pack_into("<I", base, 4, 1)  # primary sequence
        struct.pack_into("<I", base, 8, 1)  # secondary sequence
        struct.pack_into("<I", base, 20, 1)  # major version
        struct.pack_into("<I", base, 24, 5)  # minor version
        struct.pack_into("<I", base, 32, 1)  # format
        struct.pack_into("<I", base, 36, root_cell)  # RootCell
        struct.pack_into("<I", base, 40, hbins_len)  # Length
        out = bytearray(base)
        for area in self._hbins:
            size = (HBIN_HEADER + len(area) + PAGE - 1) & ~(PAGE - 1)
            hbin = bytearray(size)
            hbin[0:4] = b"hbin"
            struct.pack_into("<I", hbin, 4, len(out) - PAGE)  # FileOffset
            struct.pack_into("<I", hbin, 8, size)  # Size
            hbin[HBIN_HEADER : HBIN_HEADER + len(area)] = area
            free = HBIN_HEADER + len(area)
            if size - free >= 8:
                struct.pack_into("<i", hbin, free, size - free)  # free cell
            out += hbin
        return bytes(out), hbins_len


# ---------------------------------------------------------------------------
# Fake nt_symbols ISF (format 4.0.0) for the management superstructure.
# ---------------------------------------------------------------------------


def _base(name, kind, size, signed=False):
    return (name, {"endian": "little", "kind": kind, "signed": signed, "size": size})


def _field(offset, type_):
    return {"offset": offset, "type": type_}


def _b(name):
    return {"kind": "base", "name": name}


def _ptr(name):
    return {"kind": "pointer", "subtype": {"kind": "struct", "name": name}}


def _arr(count, subtype):
    return {"kind": "array", "count": count, "subtype": subtype}


def _struct(name, size, fields):
    return (name, {"kind": "struct", "size": size, "fields": fields})


def fake_nt_isf() -> dict:
    """Minimal ISF: _CMHIVE/_HHIVE/_HMAP_* plus the on-disk cell structs."""
    u32 = _b("unsigned long")
    nk_fields = {
        "Signature": _field(0, _b("unsigned short")),
        "Flags": _field(2, _b("unsigned short")),
        "LastWriteTime": _field(4, _b("long long")),
        "Spare": _field(12, u32),
        "Parent": _field(16, u32),
        "SubKeyCounts": _field(20, _arr(2, u32)),
        "SubKeyLists": _field(28, _arr(2, u32)),
        "ValueList": _field(36, {"kind": "struct", "name": "_CHILD_LIST"}),
        "Security": _field(44, u32),
        "Class": _field(48, u32),
        "LargestSubKeyNameLength": _field(52, u32),
        "LargestSubKeyClassLength": _field(56, u32),
        "LargestValueNameLength": _field(60, u32),
        "LargestValueDataLength": _field(64, u32),
        "WorkVar": _field(68, u32),
        "NameLength": _field(72, _b("unsigned short")),
        "ClassLength": _field(74, _b("unsigned short")),
        "Name": _field(76, _arr(1, _b("unsigned char"))),
    }
    user_types = dict(
        [
            _struct(
                "_UNICODE_STRING",
                16,
                {
                    "Length": _field(0, _b("unsigned short")),
                    "MaximumLength": _field(2, _b("unsigned short")),
                    "Buffer": _field(8, {"kind": "pointer", "subtype": _b("void")}),
                },
            ),
            _struct(
                "_CMHIVE",
                56,
                {
                    "Hive": _field(0, _ptr("_HHIVE")),
                    "FileFullPath": _field(8, {"kind": "struct", "name": "_UNICODE_STRING"}),
                    "FileUserName": _field(24, {"kind": "struct", "name": "_UNICODE_STRING"}),
                    "HiveRootPath": _field(40, {"kind": "struct", "name": "_UNICODE_STRING"}),
                },
            ),
            _struct(
                "_HHIVE",
                48,
                {
                    "Signature": _field(0, u32),
                    "BaseBlock": _field(8, _ptr("_HBASE_BLOCK")),
                    "Storage": _field(16, _arr(2, {"kind": "struct", "name": "_DUAL"})),
                },
            ),
            _struct(
                "_DUAL",
                16,
                {
                    "Length": _field(0, u32),
                    "Map": _field(8, _ptr("_HMAP_DIRECTORY")),
                },
            ),
            _struct(
                "_HMAP_DIRECTORY",
                8192,
                {"Directory": _field(0, _arr(1024, _ptr("_HMAP_TABLE")))},
            ),
            _struct(
                "_HMAP_TABLE",
                8192,
                {"Table": _field(0, _arr(512, {"kind": "struct", "name": "_HMAP_ENTRY"}))},
            ),
            _struct(
                "_HMAP_ENTRY",
                16,
                {
                    "BlockOffset": _field(0, _b("unsigned long long")),
                    "PermanentBinAddress": _field(8, _b("unsigned long long")),
                },
            ),
            _struct(
                "_HBASE_BLOCK",
                4096,
                {
                    "Signature": _field(0, _arr(4, _b("char"))),
                    "PrimarySequenceNumber": _field(4, u32),
                    "SecondarySequenceNumber": _field(8, u32),
                    "LastWritten": _field(12, _b("long long")),
                    "MajorVersion": _field(20, u32),
                    "MinorVersion": _field(24, u32),
                    "Type": _field(28, u32),
                    "Format": _field(32, u32),
                    "RootCell": _field(36, u32),
                    "Length": _field(40, u32),
                },
            ),
            _struct("_CHILD_LIST", 8, {"Count": _field(0, u32), "List": _field(4, u32)}),
            _struct("_CM_KEY_NODE", 77, nk_fields),
            _struct(
                "_CM_KEY_VALUE",
                21,
                {
                    "Signature": _field(0, _b("unsigned short")),
                    "NameLength": _field(2, _b("unsigned short")),
                    "DataLength": _field(4, u32),
                    "Data": _field(8, u32),
                    "Type": _field(12, u32),
                    "Flags": _field(16, _b("unsigned short")),
                    "Spare": _field(18, _b("unsigned short")),
                    "Name": _field(20, _arr(1, _b("unsigned char"))),
                },
            ),
            _struct(
                "_CM_KEY_INDEX",
                8,
                {
                    "Signature": _field(0, _b("unsigned short")),
                    "Count": _field(2, _b("unsigned short")),
                    "List": _field(4, _arr(1, u32)),
                },
            ),
            _struct(
                "_CM_KEY_SECURITY",
                21,
                {
                    "Signature": _field(0, _b("unsigned short")),
                    "Reserved": _field(2, _b("unsigned short")),
                    "Flink": _field(4, u32),
                    "Blink": _field(8, u32),
                    "ReferenceCount": _field(12, u32),
                    "DescriptorLength": _field(16, u32),
                    "Descriptor": _field(20, _arr(1, _b("unsigned char"))),
                },
            ),
            _struct(
                "_CM_BIG_DATA",
                8,
                {
                    "Signature": _field(0, _b("unsigned short")),
                    "Count": _field(2, _b("unsigned short")),
                    "List": _field(4, u32),
                },
            ),
            (
                "__unnamed_u",
                {
                    "kind": "union",
                    "size": 77,
                    "fields": {
                        "KeyNode": _field(0, {"kind": "struct", "name": "_CM_KEY_NODE"}),
                        "KeyValue": _field(0, {"kind": "struct", "name": "_CM_KEY_VALUE"}),
                        "KeySecurity": _field(
                            0, {"kind": "struct", "name": "_CM_KEY_SECURITY"}
                        ),
                        "KeyIndex": _field(0, {"kind": "struct", "name": "_CM_KEY_INDEX"}),
                        "ValueData": _field(0, {"kind": "struct", "name": "_CM_BIG_DATA"}),
                        "KeyList": _field(0, _arr(1, u32)),
                    },
                },
            ),
            _struct(
                "_CELL_DATA", 77, {"u": _field(0, {"kind": "union", "name": "__unnamed_u"})}
            ),
        ]
    )
    return {
        "metadata": {
            "producer": {
                "version": "0.0.1",
                "name": "vol-mojo differential harness by hand",
                "datetime": "2026-09-19T00:00:00",
            },
            "format": "4.0.0",
        },
        "symbols": {},
        "enums": {},
        "base_types": dict(
            [
                _base("void", "void", 0),
                _base("char", "char", 1, signed=True),
                _base("unsigned char", "char", 1),
                _base("unsigned short", "int", 2),
                _base("short", "int", 2, signed=True),
                _base("unsigned long", "int", 4),
                _base("long", "int", 4, signed=True),
                _base("unsigned long long", "int", 8),
                _base("long long", "int", 8, signed=True),
                _base("pointer", "int", 8),
            ]
        ),
        "user_types": user_types,
    }


def fake_nt_isf_json() -> str:
    return json.dumps(fake_nt_isf())


def add_management(
    dat: bytes, hbins_len: int, mgmt_off: Optional[int] = None
) -> tuple[bytes, int]:
    """Append the synthetic _CMHIVE/_HHIVE/_HMAP superstructure.

    The HMAP reproduces the flat .dat page mapping (page k -> file offset
    0x1000 + k*0x1000), which is exactly what the in-memory map of a
    file-backed hive reduces to. Returns (image, cmhive_offset).
    """
    if mgmt_off is None:
        mgmt_off = (len(dat) + PAGE - 1) & ~(PAGE - 1)
    npages = (hbins_len + PAGE - 1) // PAGE
    n_tables = (npages + 511) // 512
    CMHIVE = mgmt_off
    HHIVE = mgmt_off + 0x40
    HMAPDIR = mgmt_off + 0x100
    HMAPTAB = mgmt_off + 0x100 + 0x2000
    img = bytearray(dat)
    img += b"\x00" * (HMAPTAB + n_tables * 0x2000 - len(img))
    # _CMHIVE (the _UNICODE_STRING name fields stay zeroed -> "[NONAME]")
    struct.pack_into("<Q", img, CMHIVE, HHIVE)
    # _HHIVE
    struct.pack_into("<I", img, HHIVE, 0xBEE0BEE0)
    struct.pack_into("<Q", img, HHIVE + 8, 0)  # BaseBlock -> file offset 0
    struct.pack_into("<I", img, HHIVE + 16, hbins_len)  # Storage[0].Length
    struct.pack_into("<Q", img, HHIVE + 24, HMAPDIR)  # Storage[0].Map
    struct.pack_into("<I", img, HHIVE + 32, 0)  # Storage[1].Length
    struct.pack_into("<Q", img, HHIVE + 40, HMAPDIR)  # Storage[1].Map
    # _HMAP_DIRECTORY -> one _HMAP_TABLE per 512 pages
    for t in range(n_tables):
        struct.pack_into("<Q", img, HMAPDIR + t * 8, HMAPTAB + t * 0x2000)
    for page in range(npages):
        tab = HMAPTAB + (page // 512) * 0x2000 + (page % 512) * 16
        struct.pack_into("<Q", img, tab, PAGE + page * PAGE)  # BlockOffset
        struct.pack_into("<Q", img, tab + 8, 0)  # PermanentBinAddress
    return bytes(img), CMHIVE


# ---------------------------------------------------------------------------
# Oracle harnesses (volatility3 imported lazily).
# ---------------------------------------------------------------------------


def _make_hive_layer(img_path: str, cmhive_off: int, isf_path: str):
    """Build a vol3 Context with the synthetic hive stacked as a RegistryHive."""
    from volatility3.framework import contexts
    from volatility3.framework.layers import physical, registry
    from volatility3.framework.symbols import intermed
    from volatility3.framework.symbols.windows.extensions import registry as regext

    ctx = contexts.Context()
    table_name = ctx.symbol_space.free_table_name("fakent")
    table = intermed.IntermediateSymbolTable(
        context=ctx,
        config_path="fakent",
        name=table_name,
        isf_url="file://" + isf_path,
        class_types={
            "_CMHIVE": regext.CMHIVE,
            "_CM_KEY_NODE": regext.CM_KEY_NODE,
            "_CM_KEY_VALUE": regext.CM_KEY_VALUE,
            "_HMAP_ENTRY": regext.HMAP_ENTRY,
        },
    )
    ctx.symbol_space.append(table)
    ctx.config["base.location"] = "file://" + img_path
    ctx.add_layer(physical.FileLayer(ctx, "base", "base"))
    ctx.config["hivecfg.base_layer"] = "base"
    ctx.config["hivecfg.hive_offset"] = cmhive_off
    ctx.config["hivecfg.nt_symbols"] = table_name
    ctx.config["hivecfg.kernel_module_name"] = "no_such_module"
    hive = registry.RegistryHive(ctx, "hivecfg", "hive")
    ctx.add_layer(hive)
    return hive


@functools.lru_cache(maxsize=None)
def _oracle_walk_cached(img_path: str, cmhive_off: int, isf_path: str, node_off: int):
    hive = _make_hive_layer(img_path, cmhive_off, isf_path)
    tuples = []

    def visitor(node):
        tname = node.vol.type_name.split("!")[-1]
        tuples.append((node.vol.offset, tname, node.get_name()))

    node = hive.get_node(node_off) if node_off is not None else None
    hive.visit_nodes(visitor, node=node)
    return tuple(tuples)


def oracle_walk_tuples(
    img_path: str, cmhive_off: int, isf_path: str, node_off: Optional[int] = None
) -> list:
    """The oracle's visit_nodes tuples for the image (cached per image)."""
    return list(_oracle_walk_cached(img_path, cmhive_off, isf_path, node_off))


def oracle_pool_scanner_factory(
    layer_path: str,
    constraints,
    pool_table: str,
    vista_semantics: bool,
    alignment: int,
):
    """Build the oracle's PoolHeaderScanner once; return a run(data) callable.

    The returned callable performs exactly the per-hit work a plugin pays
    for: ``[(constraint, header) for ... in scanner(data, 0)]`` collected as
    (header_offset, tag) pairs. Scanner/context construction (regex
    compilation, symbol tables) happens once, up front.
    """
    from volatility3.framework import contexts
    from volatility3.framework.layers import physical
    from volatility3.framework.symbols import intermed
    from volatility3.framework.symbols.windows.extensions import pool as poolext  # noqa: F401  (import first: breaks the pool<->poolscanner cycle)
    from volatility3.plugins.windows import poolscanner

    ctx = contexts.Context()
    ctx.config["phys.location"] = "file://" + layer_path
    ctx.add_layer(physical.FileLayer(ctx, "phys", "phys"))
    class_type = poolext.POOL_HEADER_VISTA if vista_semantics else poolext.POOL_HEADER
    table_name = intermed.IntermediateSymbolTable.create(
        context=ctx,
        config_path="poolhdr",
        sub_path="windows",
        filename=pool_table,
        class_types={"_POOL_HEADER": class_type},
    )
    module = ctx.module(table_name, "phys", offset=0)
    scanner = poolscanner.PoolHeaderScanner(
        module, {c.tag: c for c in constraints}, alignment
    )

    def run(data: bytes) -> list:
        return [
            (header.vol.offset, constraint.tag) for constraint, header in scanner(data, 0)
        ]

    return run


def oracle_pool_hits(
    data: bytes,
    layer_path: str,
    constraints,
    pool_table: str,
    vista_semantics: bool,
    alignment: int,
) -> list:
    """The oracle's PoolHeaderScanner.__call__ over ``data``.

    ``pool_table`` is one of the bundled tables (poolheader-x64,
    poolheader-x64-win7, poolheader-x86); ``vista_semantics`` selects the
    extension class exactly like PoolScanner.get_pool_header_table does.
    Returns [(header_offset, tag)] in scan order.
    """
    return oracle_pool_scanner_factory(
        layer_path, constraints, pool_table, vista_semantics, alignment
    )(data)


# ---------------------------------------------------------------------------
# Pool buffer helpers.
# ---------------------------------------------------------------------------


def put_header_x64(buf: bytearray, off: int, pool_index: int, block_size: int,
                   pool_type: int, tag: bytes) -> None:
    """x64 _POOL_HEADER: byte1=PoolIndex, byte2=BlockSize, byte3=PoolType."""
    struct.pack_into("<BBBB", buf, off, 0, pool_index, block_size, pool_type)
    buf[off + 4 : off + 8] = tag


def put_header_x86(buf: bytearray, off: int, pool_index: int, block_size: int,
                   pool_type: int, tag: bytes) -> None:
    """x86 _POOL_HEADER: 9-bit sizes + 7-bit index/type packed in two u16."""
    w0 = (pool_index & 0x7F) << 9
    w1 = (block_size & 0x1FF) | ((pool_type & 0x7F) << 9)
    struct.pack_into("<HH", buf, off, w0, w1)
    buf[off + 4 : off + 8] = tag
