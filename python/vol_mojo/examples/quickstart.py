#!/usr/bin/env python3
"""vol-mojo quickstart: walk a synthetic hive + scan a synthetic pool buffer.

Builds a small valid in-memory hive image (regf + hbin + nk/lh cells) and a
synthetic pool buffer, then runs both public entry points and prints
deterministic checksums. Output must be byte-identical on the native and
fallback backends — the wheel smoke tests assert exactly that.

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import hashlib
import struct

import vol_mojo


def _pad8(n: int) -> int:
    return (n + 7) & ~7


def build_demo_hive() -> bytes:
    """A tiny valid flat hive: ROOT -> {Microsoft, Software{Windows, NT}}."""

    cells = bytearray()
    size = 0x20  # hbin header

    def add_cell(payload: bytes) -> int:
        nonlocal size
        total = _pad8(4 + len(payload))
        off = size
        cells.extend(struct.pack("<i", -total) + payload)
        cells.extend(b"\x00" * (total - 4 - len(payload)))
        size += total
        return off

    def nk(name: bytes, sub_count: int = 0, sub_list: int = 0xFFFFFFFF) -> bytes:
        d = bytearray(76)
        struct.pack_into("<H", d, 0, 0x6B6E)  # "nk"
        struct.pack_into("<H", d, 2, 0x20)  # KEY_COMP_NAME
        struct.pack_into("<I", d, 16, 0xFFFFFFFF)  # Parent
        struct.pack_into("<I", d, 20, sub_count)
        struct.pack_into("<I", d, 28, sub_list)
        struct.pack_into("<I", d, 32, 0xFFFFFFFF)
        struct.pack_into("<H", d, 72, len(name))
        return bytes(d) + name

    def lh(entries) -> bytes:
        d = bytearray(b"lh\x00\x00")
        struct.pack_into("<H", d, 2, len(entries))
        for cell, h in entries:
            d += struct.pack("<II", cell, h)
        return bytes(d)

    leaf_ms = add_cell(nk(b"Microsoft"))
    leaf_win = add_cell(nk(b"Windows"))
    leaf_nt = add_cell(nk(b"NT"))
    idx_sw = add_cell(lh([(leaf_win, 0x11111111), (leaf_nt, 0x22222222)]))
    mid_sw = add_cell(nk(b"Software", 2, idx_sw))
    idx_root = add_cell(lh([(leaf_ms, 0xAAAAAAAA), (mid_sw, 0xBBBBBBBB)]))
    root = add_cell(nk(b"ROOT", 2, idx_root))

    hbins_len = (size + 0xFFF) & ~0xFFF
    base = bytearray(0x1000)
    base[0:4] = b"regf"
    struct.pack_into("<I", base, 20, 1)  # major version
    struct.pack_into("<I", base, 24, 5)  # minor version
    struct.pack_into("<I", base, 32, 1)  # format
    struct.pack_into("<I", base, 36, root)  # RootCell
    struct.pack_into("<I", base, 40, hbins_len)  # Length
    hbin = bytearray(hbins_len)
    hbin[0:4] = b"hbin"
    struct.pack_into("<I", hbin, 8, hbins_len)
    hbin[0x20 : 0x20 + len(cells)] = cells
    free = 0x20 + len(cells)
    if hbins_len - free >= 8:
        struct.pack_into("<i", hbin, free, hbins_len - free)
    return bytes(base) + bytes(hbin)


def build_demo_pool() -> bytes:
    buf = bytearray(0x400)

    def header(off: int, block: int, ptype: int, tag: bytes) -> None:
        struct.pack_into("<BBBB", buf, off, 0, 0, block, ptype)
        buf[off + 4 : off + 8] = tag

    header(0x40, 0x41, 2, b"Proc")  # 0x41*0x10 = 1040 >= 600, nonpaged -> pass
    header(0x80, 0x10, 2, b"Proc")  # 0x10*0x10 = 256 < 600 -> reject
    header(0xC0, 0x41, 0, b"Proc")  # free pool -> pass (FREE in mask)
    header(0x100, 0x20, 1, b"CM10")  # paged, 0x20*0x10 = 512 < 800 -> reject
    header(0x180, 0x40, 1, b"CM10")  # paged, 0x40*0x10 = 1024 >= 800 -> pass
    return bytes(buf)


def main() -> None:
    info = vol_mojo.backend_info()
    print(
        "backend:",
        "native" if info["native_available"] else "fallback",
        f"({info.get('native_source') or info.get('error')})",
    )

    tuples = vol_mojo.walk_hive(build_demo_hive())
    print(f"hive walk: {len(tuples)} key nodes")
    for t in tuples:
        print(f"  {t}")
    walk_checksum = hashlib.sha256(repr(tuples).encode()).hexdigest()[:16]
    print(f"walk checksum: {walk_checksum}")

    constraints = [
        vol_mojo.PoolConstraint(
            b"Proc",
            size=(600, None),
            page_type=vol_mojo.PAGE_TYPE_NONPAGED | vol_mojo.PAGE_TYPE_FREE,
        ),
        vol_mojo.PoolConstraint(
            b"CM10",
            size=(800, None),
            page_type=vol_mojo.PAGE_TYPE_PAGED | vol_mojo.PAGE_TYPE_FREE,
        ),
    ]
    hits = vol_mojo.scan_pool_headers(
        build_demo_pool(), constraints, alignment=0x10, layout="x64"
    )
    print(f"pool scan: {len(hits)} passing headers")
    for off, tag in hits:
        print(f"  {off:#x} {tag!r}")
    scan_checksum = hashlib.sha256(repr(hits).encode()).hexdigest()[:16]
    print(f"scan checksum: {scan_checksum}")


if __name__ == "__main__":
    main()
