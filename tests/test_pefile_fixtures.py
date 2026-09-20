"""Synthetic PE fixtures for the pefile-mojo differential suite.

Every fixture is generated locally by ``build_pe`` (pure ``struct`` packing,
deterministic, no network, no toolchain, no system binaries) and — where noted
— re-serialized through the oracle's own ``pefile.PE.write()`` support. No PE
from this machine (or any other) is used: the repo is public and fixtures must
be provenance-clean.

``build_pe`` produces a minimal but well-formed PE32/PE32+ image: DOS header,
PE signature, COFF header, optional header with 16 data directories, a ``.text``
section (filler) and an ``.idata`` section holding the import directory, the
ILT/IAT thunk tables, hint/name strings and DLL names. An unreferenced overlay
can be appended to exercise the checksum's non-dword-aligned tail path.
"""

from __future__ import annotations

import struct

import pytest

pefile = pytest.importorskip("pefile", reason="oracle package pefile is required")

# ---------------------------------------------------------------------------
# PE builder
# ---------------------------------------------------------------------------

IMPORT_DIR_INDEX = 1
ORDINAL = object()  # marker for "import by ordinal" entries


def _dos_header(e_lfanew: int) -> bytes:
    # IMAGE_DOS_HEADER: only e_magic and e_lfanew are load-bearing here.
    hdr = bytearray(64)
    hdr[0:2] = b"MZ"
    struct.pack_into("<I", hdr, 0x3C, e_lfanew)
    return bytes(hdr)


def _coff_header(machine: int, n_sections: int, size_opt: int, characteristics: int) -> bytes:
    return struct.pack("<HHIIIHH", machine, n_sections, 0x6566F1A0, 0, 0, size_opt,
                       characteristics)


def _optional_header(bits: int, *, image_base: int, size_of_image: int,
                     size_of_headers: int, checksum: int,
                     import_rva: int, import_size: int, entry_point: int) -> bytes:
    if bits == 32:
        fixed = struct.pack(
            "<HBBIIIIIIIIIHHHHHHIIIIHHIIIIII",
            0x10B, 14, 0,          # Magic, linker version
            0x200, 0x200, 0,       # SizeOfCode / InitializedData / UninitializedData
            entry_point, 0x1000, 0x2000,  # AddressOfEntryPoint, BaseOfCode, BaseOfData
            image_base,
            0x1000, 0x200,         # SectionAlignment, FileAlignment
            6, 0, 0, 0, 6, 0,      # OS/Image/Subsystem versions
            0,                     # Win32VersionValue
            size_of_image, size_of_headers, checksum,
            3, 0,                  # Subsystem (CUI), DllCharacteristics
            0x100000, 0x1000, 0x100000, 0x1000,
            0, 16,                 # LoaderFlags, NumberOfRvaAndSizes
        )
    else:
        fixed = struct.pack(
            "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
            0x20B, 14, 0,          # Magic, linker version
            0x200, 0x200, 0,       # SizeOfCode / InitializedData / UninitializedData
            entry_point, 0x1000,   # AddressOfEntryPoint, BaseOfCode (no BaseOfData)
            image_base,
            0x1000, 0x200,         # SectionAlignment, FileAlignment
            6, 0, 0, 0, 6, 0,      # OS/Image/Subsystem versions
            0,                     # Win32VersionValue
            size_of_image, size_of_headers, checksum,
            3, 0,                  # Subsystem (CUI), DllCharacteristics
            0x100000, 0x1000, 0x100000, 0x1000,
            0, 16,                 # LoaderFlags, NumberOfRvaAndSizes
        )
    dirs = bytearray(16 * 8)
    struct.pack_into("<II", dirs, IMPORT_DIR_INDEX * 8, import_rva, import_size)
    return fixed + bytes(dirs)


def _section_header(name: bytes, vsize: int, vaddr: int, raw_size: int,
                    raw_ptr: int, characteristics: int) -> bytes:
    return struct.pack("<8sIIIIIIHHI", name, vsize, vaddr, raw_size, raw_ptr,
                       0, 0, 0, 0, characteristics)


def build_import_blob(bits: int, dlls: list, *, base_rva: int = 0x2000,
                      bound_iat: bool = False) -> tuple[bytes, int]:
    """Serialize the import directory payload for a section mapped at base_rva.

    ``dlls`` is a list of ``(dll_name, entries)`` where each entry is
    ``(name_bytes, hint)`` or ``(ORDINAL, ordinal)``. Returns the section
    payload and the import-directory size in bytes.
    """
    thunk_fmt, thunk_size = ("<I", 4) if bits == 32 else ("<Q", 8)
    ordinal_flag = 0x80000000 if bits == 32 else 0x8000000000000000

    desc_area = (len(dlls) + 1) * 20
    cursor = desc_area
    tables: list[dict] = []  # ILT/IAT pieces, patched after names are placed
    hintname_rva: dict[tuple[int, int], int] = {}
    dll_rvas: list[int] = []

    # Pass 1: thunk tables (name slots zero-filled, patched in pass 3).
    for d, (dll_name, entries) in enumerate(dlls):
        for kind in ("ilt", "iat"):
            rva = base_rva + cursor
            raw = bytearray()
            for e, entry in enumerate(entries):
                if entry[0] is ORDINAL:
                    val = ordinal_flag | entry[1]
                else:
                    val = 0  # patched later
                if kind == "iat" and bound_iat:
                    val = 0x7C000000 + 0x10 * e  # bound addresses: IAT != ILT
                raw += struct.pack(thunk_fmt, val)
            raw += struct.pack(thunk_fmt, 0)  # terminator
            tables.append({"rva": rva, "raw": raw, "kind": kind, "desc": d})
            cursor += len(raw)

    # Pass 2: hint/name strings and DLL names.
    blob = bytearray(cursor)
    for d, (dll_name, entries) in enumerate(dlls):
        for e, entry in enumerate(entries):
            if entry[0] is ORDINAL:
                continue
            name, hint = entry
            entry_rva = base_rva + cursor
            raw = struct.pack("<H", hint) + name + b"\0"
            if len(raw) % 2:
                raw += b"\0"
            blob += raw
            cursor += len(raw)
            hintname_rva[(d, e)] = entry_rva
    for d, (dll_name, entries) in enumerate(dlls):
        dll_rvas.append(base_rva + cursor)
        raw = dll_name + b"\0"
        if len(raw) % 2:
            raw += b"\0"
        blob += raw
        cursor += len(raw)

    # Pass 3: write tables with patched name RVAs, then the descriptors.
    for t in tables:
        raw = t["raw"]
        if not (t["kind"] == "iat" and bound_iat):
            raw = bytearray(raw)
            for e, entry in enumerate(dlls[t["desc"]][1]):
                if entry[0] is ORDINAL:
                    continue
                struct.pack_into(thunk_fmt, raw, e * thunk_size,
                                 hintname_rva[(t["desc"], e)])
        off = t["rva"] - base_rva
        blob[off:off + len(raw)] = raw

    descs = bytearray()
    for d, (dll_name, entries) in enumerate(dlls):
        ilt = next(t["rva"] for t in tables if t["desc"] == d and t["kind"] == "ilt")
        iat = next(t["rva"] for t in tables if t["desc"] == d and t["kind"] == "iat")
        descs += struct.pack("<IIIII", ilt, 0, 0, dll_rvas[d], iat)
    descs += b"\0" * 20  # null descriptor
    blob[0:len(descs)] = descs
    return bytes(blob), desc_area


def build_pe(*, bits: int = 32, dlls: list, overlay: bytes = b"",
             checksum_field: int = 0xDEADBEEF, bound_iat: bool = False,
             text_filler: int = 0x90) -> bytes:
    """Assemble a complete, well-formed synthetic PE image."""
    assert bits in (32, 64)
    idata, dir_size = build_import_blob(bits, dlls, bound_iat=bound_iat)
    idata_raw_size = (len(idata) + 0x1FF) & ~0x1FF
    text_raw_size = 0x200
    headers_size = 0x200
    text_ptr = headers_size
    idata_ptr = text_ptr + text_raw_size

    machine = 0x14C if bits == 32 else 0x8664
    characteristics = 0x0102 if bits == 32 else 0x0022
    size_opt = 0xE0 if bits == 32 else 0xF0
    image_base = 0x400000 if bits == 32 else 0x140000000
    size_of_image = 0x3000

    e_lfanew = 0x40
    headers = bytearray()
    headers += _dos_header(e_lfanew)
    assert len(headers) == e_lfanew
    headers += b"PE\0\0"
    headers += _coff_header(machine, 2, size_opt, characteristics)
    headers += _optional_header(
        bits,
        image_base=image_base,
        size_of_image=size_of_image,
        size_of_headers=headers_size,
        checksum=checksum_field,
        import_rva=0x2000,
        import_size=dir_size,
        entry_point=0x1000,
    )
    headers += _section_header(b".text\0\0\0", 1, 0x1000, text_raw_size, text_ptr,
                               0x60000020)
    headers += _section_header(b".idata\0\0", len(idata), 0x2000, idata_raw_size,
                               idata_ptr, 0xC0000040)
    headers += b"\0" * (headers_size - len(headers))

    text = bytes([text_filler]) * text_raw_size
    idata_padded = idata + b"\0" * (idata_raw_size - len(idata))
    return bytes(headers) + text + idata_padded + overlay


# ---------------------------------------------------------------------------
# Named fixture family (deterministic; covers PE32/PE32+, ordinal/named,
# bound IAT, empty ILT, invalid names, dead descriptors, overlay lengths)
# ---------------------------------------------------------------------------

def fixture_dlls_plain() -> list:
    return [
        (b"KERNEL32.dll", [(b"ComputeHash", 0), (b"RenderFrame", 3), (b"MapBuffer", 9)]),
        (b"USER32.dll", [(b"DrawTextA", 12), (b"HitTest", 4)]),
    ]


def fixture_dlls_ordinal() -> list:
    return [
        (b"VENDORDRV.dll", [(ORDINAL, 12), (ORDINAL, 3), (b"ResetDevice", 1)]),
        (b"AUDIOHAL.dll", [(ORDINAL, 255)]),
    ]


def fixture_dlls_charset() -> list:
    # Exercise every extra character pefile accepts in import names.
    return [
        (b"LIBCMT.dll", [
            (b"Func.With.Dot", 0),
            (b"_imp__Foo@4", 1),
            (b"?Cpp@@YAHH@Z", 2),
            (b"array(idx)", 3),
            (b"tmpl<int>", 4),
            (b"cost$Fn", 5),
        ]),
    ]


def fixture_dlls_invalid_name() -> list:
    return [
        (b"SHLWAPI.dll", [(b"GoodNameCheck", 0), (b"Bad Name", 1), (b"AlsoGood", 2)]),
    ]


def fixture_dlls_dead_desc() -> list:
    return [
        (b"", []),  # no thunks at all -> pefile skips this descriptor
        (b"MSVCRT.dll", [(b"strlen", 0), (b"memcpy", 1)]),
    ]


def make_fixtures() -> dict[str, bytes]:
    fx: dict[str, bytes] = {}
    fx["pe32_plain"] = build_pe(bits=32, dlls=fixture_dlls_plain())
    fx["pe32_overlay1"] = build_pe(bits=32, dlls=fixture_dlls_plain(), overlay=b"X")
    fx["pe32_overlay2"] = build_pe(bits=32, dlls=fixture_dlls_plain(), overlay=b"YZ")
    fx["pe32_overlay3"] = build_pe(bits=32, dlls=fixture_dlls_plain(), overlay=b"\xAA" * 3)
    fx["pe64_ordinal"] = build_pe(bits=64, dlls=fixture_dlls_ordinal())
    fx["pe32_charset"] = build_pe(bits=32, dlls=fixture_dlls_charset())
    fx["pe64_charset"] = build_pe(bits=64, dlls=fixture_dlls_charset(), overlay=b"Q")
    fx["pe32_bound"] = build_pe(bits=32, dlls=fixture_dlls_plain(), bound_iat=True)
    fx["pe64_bound"] = build_pe(bits=64, dlls=fixture_dlls_ordinal(), bound_iat=True)
    fx["pe32_empty_ilt"] = build_pe(bits=32, dlls=fixture_dlls_plain())
    fx["pe32_invalid_name"] = build_pe(bits=32, dlls=fixture_dlls_invalid_name())
    fx["pe64_invalid_name"] = build_pe(bits=64, dlls=fixture_dlls_invalid_name())
    fx["pe32_dead_desc"] = build_pe(bits=32, dlls=fixture_dlls_dead_desc())
    fx["pe32_no_overlay_text_alt"] = build_pe(bits=32, dlls=fixture_dlls_plain(),
                                              text_filler=0xCC)
    return fx


def make_big_pe(n_dlls: int = 40, n_funcs: int = 25, *, bits: int = 64,
                text_bytes: int = 0) -> bytes:
    """Larger synthetic image used by the benchmark (deterministic names)."""
    dlls = [
        (
            f"VENDOR{i:03d}.dll".encode(),
            [(f"EntryPoint{i:03d}_{j:03d}".encode(), j % 0x400) for j in range(n_funcs)]
            + [(ORDINAL, (i * 7 + 1) % 0x800)],
        )
        for i in range(n_dlls)
    ]
    data = build_pe(bits=bits, dlls=dlls)
    if text_bytes:
        # Grow the .text section by rebuilding with a bigger filler tail.
        data = data + bytes((i * 131 + 7) & 0xFF for i in range(text_bytes))
    return data


# pefile treats OriginalFirstThunk==0 as "use the IAT"; rebuild one fixture
# with OFT zeroed out (pefile's `table = ilt or iat` path).
def make_empty_ilt_variant() -> bytes:
    data = bytearray(build_pe(bits=32, dlls=fixture_dlls_plain()))
    pe = pefile.PE(data=bytes(data))
    desc_file_off = pe.get_offset_from_rva(
        pe.OPTIONAL_HEADER.DATA_DIRECTORY[IMPORT_DIR_INDEX].VirtualAddress
    )
    for i in range(len(fixture_dlls_plain())):
        struct.pack_into("<I", data, desc_file_off + i * 20, 0)  # OriginalFirstThunk = 0
    return bytes(data)


FIXTURES = make_fixtures()
FIXTURES["pe32_empty_ilt"] = make_empty_ilt_variant()

# Every fixture, re-serialized through the oracle's own write() support. This
# is the "build tiny synthetic PEs with pefile's own write support" half of the
# fixture family: provenance = local generator + pefile 2024.8.26 write().
WRITE_ROUNDTRIP = {name: bytes(pefile.PE(data=blob).write()) for name, blob in FIXTURES.items()}


# ---------------------------------------------------------------------------
# Fixture sanity tests (the differential suite itself lives in
# tests/test_pefile_differential.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_parses_clean_with_oracle(name):
    pe = pefile.PE(data=FIXTURES[name])
    assert pe.OPTIONAL_HEADER.DATA_DIRECTORY[IMPORT_DIR_INDEX].VirtualAddress == 0x2000
    assert pe.DIRECTORY_ENTRY_IMPORT, f"{name}: oracle found no imports"


def test_empty_ilt_variant_really_has_zero_oft():
    pe = pefile.PE(data=FIXTURES["pe32_empty_ilt"])
    entry = pe.DIRECTORY_ENTRY_IMPORT[0]
    assert entry.struct.OriginalFirstThunk == 0
    assert entry.imports, "oracle must fall back to the IAT"


def test_bound_fixture_really_is_bound():
    pe = pefile.PE(data=FIXTURES["pe32_bound"])
    bounds = [sym.bound for d in pe.DIRECTORY_ENTRY_IMPORT for sym in d.imports]
    assert any(b is not None for b in bounds)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_oracle_write_roundtrips_fixture(name):
    # Provenance gate for the checksum differential: pefile's generate_checksum
    # hashes pe.write(), so a drop-in over the on-disk bytes is only identical
    # when write() reproduces the input. Assert it for every fixture.
    assert WRITE_ROUNDTRIP[name] == FIXTURES[name]


def test_no_ordlookup_shadowing():
    # Our ordinal-import DLLs must NOT be in pefile's ordlookup database, or
    # the oracle would resolve names where pefile_mojo (documented scope)
    # returns name=None.
    from ordlookup import ords

    for name, blob in FIXTURES.items():
        pe = pefile.PE(data=blob)
        for desc in pe.DIRECTORY_ENTRY_IMPORT:
            assert desc.dll.lower() not in ords, (name, desc.dll)


def test_overlay_lengths_cover_all_remainders():
    remainders = {len(FIXTURES[n]) % 4 for n in
                  ("pe32_plain", "pe32_overlay1", "pe32_overlay2", "pe32_overlay3")}
    assert remainders == {0, 1, 2, 3}
