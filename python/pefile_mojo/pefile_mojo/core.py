"""Public API: PE checksum and import-table parsing, native kernel + fallback.

The functions here are hot-loop replacements for the corresponding
``pefile.PE`` methods, not a full PE parser. They take the raw file bytes
plus the header facts pefile itself would compute, and return the same
values pefile would (asserted bit-exact by the differential suite).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from pefile_mojo import _native, _reference

OPTIONAL_HEADER_MAGIC_PE = 0x10B
OPTIONAL_HEADER_MAGIC_PE_PLUS = 0x20B
_IMAGE_DIRECTORY_ENTRY_IMPORT = 1
_MAX_SECTIONS = 0x800  # pefile's cap on parsed section headers
_IMAGE_NT_SIGNATURE = 0x00004550
_IMAGE_DOS_SIGNATURE = 0x5A4D
_BOUND_NONE = 0xFFFFFFFFFFFFFFFF


class PEFormatError(ValueError):  # noqa: N818
    """The input bytes are not a parseable PE image (mirrors pefile's error)."""


@dataclass(frozen=True)
class ImportSymbol:
    """One imported symbol (pefile.PE.DIRECTORY_ENTRY_IMPORT[].imports[]).

    ``name`` is None for ordinal imports (see README for the ordlookup
    scope note); ``ordinal`` is None for named imports; ``hint`` is None
    when the hint word is not fully present in the file; ``bound`` is the
    bound IAT address when the file is bound, else None.
    """

    name: bytes | None
    ordinal: int | None
    hint: int | None
    address: int
    bound: int | None


@dataclass(frozen=True)
class ImportDescriptor:
    """One imported DLL and its symbols (pefile's ImportDescData)."""

    dll: bytes
    imports: tuple[ImportSymbol, ...] = field(default_factory=tuple)


def checksum_field_offset(data: bytes | bytearray | memoryview) -> int:
    """File offset of the OptionalHeader.CheckSum field.

    Equivalent to ``pefile.PE(data).OPTIONAL_HEADER.get_file_offset() + 0x40``
    (the offset is the same for PE32 and PE32+). Raises PEFormatError when
    the headers are not parseable.
    """
    view = memoryview(data)
    opt_off = _optional_header_offset(view)
    return opt_off + 0x40


def generate_checksum(data: bytes | bytearray | memoryview, checksum_offset: int) -> int:
    """Bit-exact replacement for ``pefile.PE(data).generate_checksum()``.

    ``data`` is the serialized image (what ``pe.write()`` returns — for an
    unmodified file read from disk, the file bytes themselves; see the
    README drop-in note) and ``checksum_offset`` the file offset of the
    CheckSum field (see :func:`checksum_field_offset`).

    Uses the native kernel when available, else the vendored pure-Python
    implementation; both are bit-exact against pefile.
    """
    _validate_checksum_inputs(data, checksum_offset)
    try:
        return _native.native_checksum(data, checksum_offset)
    except _native.NativeUnavailable:
        return _reference.checksum(data, checksum_offset)


def parse_imports(data: bytes | bytearray | memoryview) -> list[ImportDescriptor]:
    """Import table of a PE image, identical to pefile's DIRECTORY_ENTRY_IMPORT.

    Equivalent to ``pefile.PE(data).parse_data_directories(directories=[1])``
    followed by reading ``.DIRECTORY_ENTRY_IMPORT``: same descriptors in the
    same order, same per-symbol (name, ordinal, hint, address, bound) values,
    same skip rules for invalid entries. Raises PEFormatError when the PE
    headers themselves are not parseable; a missing/empty import directory
    returns [].
    """
    header = _parse_headers(data)
    if header["import_dir_rva"] == 0:
        return []
    try:
        flat = _native.native_parse_imports(header)
        return _flat_to_descriptors(flat)
    except _native.NativeUnavailable:
        rows = _reference.parse_imports(header)
        return [
            ImportDescriptor(dll=dll, imports=tuple(ImportSymbol(**s) for s in syms))
            for dll, syms in rows
        ]


def backend_info() -> dict:
    """Diagnostics for the active backend. Never raises."""
    return _native.backend_info()


def native_available() -> bool:
    """True if the native kernel can be used right now. Never raises."""
    return _native.native_available()


# ---------------------------------------------------------------------------
# Header parsing (shared by the native call and the fallback)
# ---------------------------------------------------------------------------


def _validate_checksum_inputs(data, checksum_offset) -> None:
    view = memoryview(data)
    if view.ndim != 1:
        raise ValueError("data must be a flat bytes-like object")
    if not isinstance(checksum_offset, int) or checksum_offset < 0:
        raise ValueError("checksum_offset must be a non-negative int")


def _optional_header_offset(view: memoryview) -> int:
    if len(view) < 0x40:
        raise PEFormatError("not a PE file: truncated DOS header")
    if struct.unpack_from("<H", view, 0)[0] != _IMAGE_DOS_SIGNATURE:
        raise PEFormatError("not a PE file: missing MZ signature")
    e_lfanew = struct.unpack_from("<I", view, 0x3C)[0]
    if e_lfanew + 24 > len(view):
        raise PEFormatError("not a PE file: truncated NT headers")
    if struct.unpack_from("<I", view, e_lfanew)[0] != _IMAGE_NT_SIGNATURE:
        raise PEFormatError("not a PE file: missing PE\\0\\0 signature")
    opt_off = e_lfanew + 24
    # pefile tolerates a short optional header by zero-padding it (Tiny PE
    # behavior): it raises "No Optional Header found" only when fewer than
    # MINIMUM_VALID_OPTIONAL_HEADER_RAW_SIZE bytes remain (69, or 73 for the
    # PE32+ padded retry).
    avail = len(view) - opt_off
    if avail < 69:
        raise PEFormatError("not a PE file: no optional header")
    if avail < 73 and struct.unpack_from("<H", view, opt_off)[0] == OPTIONAL_HEADER_MAGIC_PE_PLUS:
        raise PEFormatError("not a PE file: no optional header (PE32+)")
    return opt_off


def _parse_headers(data) -> dict:
    """Extract exactly the header facts the import walk needs.

    Mirrors pefile's own parsing: data-directory entry #1 (IMPORT), the
    section table (sorted by VirtualAddress, stopping at pefile's cap or a
    null header), and pefile's `self.header` length computation.
    """
    view = memoryview(data)
    if view.ndim != 1:
        raise ValueError("data must be a flat bytes-like object")
    opt_off = _optional_header_offset(view)
    coff_off = opt_off - 20
    n_sections_declared, size_opt = struct.unpack_from("<HxxxxxxxxxxxxH", view, coff_off + 2)
    avail = len(view) - opt_off

    def zread(off: int, size: int) -> int:
        # pefile zero-pads a short optional header before unpacking.
        chunk = bytes(view[opt_off + off : opt_off + off + size])
        return int.from_bytes(chunk.ljust(size, b"\0"), "little")

    magic = zread(0, 2)
    if magic == OPTIONAL_HEADER_MAGIC_PE_PLUS:
        image_base = zread(24, 8)
        num_rva_off, dirs_off = 108, 112
    else:
        # PE32 layout; pefile also falls back to 32-bit for unknown magics.
        image_base = zread(28, 4)
        num_rva_off, dirs_off = 92, 96
    section_alignment = zread(32, 4)
    file_alignment = zread(36, 4)
    number_of_rva_and_sizes = zread(num_rva_off, 4)

    import_dir_rva = 0
    if number_of_rva_and_sizes > _IMAGE_DIRECTORY_ENTRY_IMPORT:
        # pefile stops unpacking directory entries at the first short read;
        # entry #1 then never exists and the import parse is skipped.
        if avail >= dirs_off + (_IMAGE_DIRECTORY_ENTRY_IMPORT + 1) * 8:
            import_dir_rva = zread(dirs_off + _IMAGE_DIRECTORY_ENTRY_IMPORT * 8, 4)

    sections_off = opt_off + size_opt
    section_rows: list[tuple[int, int, int, int]] = []
    for i in range(min(n_sections_declared, _MAX_SECTIONS)):
        off = sections_off + 40 * i
        remaining = len(view) - off
        if remaining <= 0:
            break
        if remaining < 40:
            # pefile's SectionStructure.__unpack__ raises on a short read.
            raise PEFormatError("not a PE file: truncated section table")
        raw = bytes(view[off : off + 40])
        if raw == b"\0" * 40:
            break  # pefile stops at a null section header
        _name, vsize, vaddr, rawsize, rawptr = struct.unpack_from("<8sIIII", raw, 0)
        section_rows.append((vaddr, vsize, rawsize, rawptr))
    section_rows.sort(key=lambda r: r[0])  # pefile sorts by VirtualAddress

    # pefile's self.header: up to the first section's raw data (or the end of
    # the section table when there are no sections / odd layouts).
    if n_sections_declared > 0 and section_rows:
        sections_end = sections_off + 40 * n_sections_declared
    else:
        sections_end = sections_off
    rawptrs = [r[3] & ~0x1FF for r in section_rows if r[3] > 0]
    lowest = min(rawptrs) if rawptrs else None
    if not lowest or lowest < sections_end:
        header_len = sections_end
    else:
        header_len = lowest
    header_len = min(header_len, len(view))

    return {
        "data": view,
        "pe_type": magic,
        "image_base": image_base,
        "import_dir_rva": import_dir_rva,
        "section_alignment": section_alignment,
        "file_alignment": file_alignment,
        "header_len": header_len,
        "sections": section_rows,
    }


def _flat_to_descriptors(flat: dict) -> list[ImportDescriptor]:
    pool = bytes(flat["pool"])
    out: list[ImportDescriptor] = []
    for i in range(flat["n_desc"]):
        dll_off = flat["desc_dll_off"][i]
        dll_len = flat["desc_dll_len"][i]
        dll = pool[dll_off : dll_off + dll_len]
        start = flat["desc_sym_start"][i]
        count = flat["desc_sym_count"][i]
        syms = []
        for j in range(start, start + count):
            name_off = flat["sym_name_off"][j]
            name_len = flat["sym_name_len"][j]
            name = pool[name_off : name_off + name_len] if name_len >= 0 else None
            ordinal = flat["sym_ordinal"][j]
            hint = flat["sym_hint"][j]
            bound = flat["sym_bound"][j]
            syms.append(
                ImportSymbol(
                    name=name,
                    ordinal=ordinal if ordinal >= 0 else None,
                    hint=hint if hint >= 0 else None,
                    address=flat["sym_address"][j],
                    bound=bound if bound != _BOUND_NONE else None,
                )
            )
        out.append(ImportDescriptor(dll=dll, imports=tuple(syms)))
    return out
