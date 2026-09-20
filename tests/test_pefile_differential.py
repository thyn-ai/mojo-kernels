"""Differential suite: pefile_mojo vs the PyPI pefile oracle.

Runs twice (see scripts/test_all_pefile.sh): once against the native Mojo
kernel, once with PEFILE_MOJO_DISABLE_NATIVE=1 forcing the pure-Python
fallback. Both backends must agree with the oracle exactly:

* generate_checksum: bit-exact (tolerance 0) on every fixture, on the
  oracle-write() round-trips, and on seeded byte/length mutations.
* parse_imports: identical (dll, name, ordinal, hint, address, bound) tables
  on every fixture; identical *or* explicitly-erroring behavior on mutations
  (corrupt-input divergence is documented in the package README).
"""

from __future__ import annotations

import random

import pytest

pefile = pytest.importorskip("pefile", reason="oracle package pefile is required")

import pefile_mojo
from test_pefile_fixtures import FIXTURES, WRITE_ROUNDTRIP, make_big_pe

BLOBS: dict[str, bytes] = {}
BLOBS.update(FIXTURES)
BLOBS.update({f"{name}#write": blob for name, blob in WRITE_ROUNDTRIP.items()})


def ours_imports(blob: bytes):
    return [
        (d.dll, [(s.name, s.ordinal, s.hint, s.address, s.bound) for s in d.imports])
        for d in pefile_mojo.parse_imports(blob)
    ]


def oracle_imports(blob: bytes):
    pe = pefile.PE(data=blob)
    # pefile only sets DIRECTORY_ENTRY_IMPORT when the parse produced entries.
    entries = getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
    return [
        (d.dll, [(s.name, s.ordinal, s.hint, s.address, s.bound) for s in d.imports])
        for d in entries
    ]


def ours_checksum(blob: bytes) -> int:
    return pefile_mojo.generate_checksum(blob, pefile_mojo.checksum_field_offset(blob))


def oracle_checksum(pe) -> int:
    return pe.generate_checksum()


# ---------------------------------------------------------------------------
# Exact parity on well-formed fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BLOBS))
def test_checksum_bit_exact(name):
    blob = BLOBS[name]
    pe = pefile.PE(data=blob)
    assert ours_checksum(blob) == oracle_checksum(pe)


@pytest.mark.parametrize("name", sorted(BLOBS))
def test_imports_identical(name):
    assert ours_imports(BLOBS[name]) == oracle_imports(BLOBS[name])


def test_big_synthetic_image_parity():
    blob = make_big_pe(n_dlls=40, n_funcs=25, bits=64, text_bytes=100_003)
    pe = pefile.PE(data=blob)
    assert ours_checksum(blob) == oracle_checksum(pe)
    assert ours_imports(blob) == oracle_imports(blob)


def test_import_dir_rva_zero_returns_empty():
    # Zero the import directory RVA: pefile skips the parser entirely.
    blob = bytearray(FIXTURES["pe32_plain"])
    pe = pefile.PE(data=FIXTURES["pe32_plain"])
    dd_off = pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].get_file_offset()
    blob[dd_off : dd_off + 8] = b"\0" * 8
    blob = bytes(blob)
    pe2 = pefile.PE(data=blob)
    assert not hasattr(pe2, "DIRECTORY_ENTRY_IMPORT")
    assert pefile_mojo.parse_imports(blob) == []


# ---------------------------------------------------------------------------
# Seeded mutations: checksum stays bit-exact wherever the oracle parses;
# imports agree exactly wherever the oracle parses.
# ---------------------------------------------------------------------------


def _mutations(seed: int, count: int) -> list[tuple[str, bytes]]:
    rng = random.Random(seed)
    names = sorted(FIXTURES)
    out = []
    for i in range(count):
        name = rng.choice(names)
        blob = bytearray(FIXTURES[name])
        for _ in range(rng.randint(1, 8)):
            pos = rng.randrange(len(blob))
            blob[pos] = rng.randrange(256)
        if i % 4 == 3:
            blob = blob[: rng.randrange(1, len(blob))]  # truncation
        out.append((f"{name}#mut{i}", bytes(blob)))
    return out


MUTATIONS = _mutations(seed=20260919, count=48)


@pytest.mark.parametrize("label", [m[0] for m in MUTATIONS])
def test_mutated_blobs_match_oracle(label):
    blob = dict(MUTATIONS)[label]
    try:
        pe = pefile.PE(data=blob)
    except Exception:
        # The oracle rejects the bytes. pefile's full PE() parse also walks
        # structures outside pefile_mojo's documented scope (resources,
        # relocations, ...), so pefile_mojo may legitimately still succeed;
        # it must never raise anything but its declared header-level errors.
        try:
            pefile_mojo.parse_imports(blob)
            pefile_mojo.checksum_field_offset(blob)
        except (pefile_mojo.PEFormatError, ValueError):
            pass
        return
    if pe.OPTIONAL_HEADER is None:
        pytest.skip("oracle parsed without an optional header")
    # Checksum drop-in contract: identical over the bytes pefile would hash
    # (pe.write() output). When write() round-trips — every well-formed real
    # PE — that is the input itself; assert the stronger form then too.
    rewritten = bytes(pe.write())
    assert pefile_mojo.generate_checksum(
        rewritten, pefile_mojo.checksum_field_offset(rewritten)
    ) == oracle_checksum(pe)
    if rewritten == blob:
        assert ours_checksum(blob) == oracle_checksum(pe)
    assert ours_imports(blob) == oracle_imports(blob)


# ---------------------------------------------------------------------------
# Header-level truncation parity: both must raise (deterministic cases)
# ---------------------------------------------------------------------------


def _truncations() -> dict[str, bytes]:
    blob = FIXTURES["pe64_ordinal"]
    pe = pefile.PE(data=blob)
    opt_off = pe.OPTIONAL_HEADER.get_file_offset()
    sec_off = pe.sections[0].get_file_offset()
    return {
        "empty": b"",
        "dos-short": blob[:17],
        "nt-sig-short": blob[:0x41],
        "coff-short": blob[:0x4A],
        # 60 < 69 (MINIMUM_VALID_OPTIONAL_HEADER_RAW_SIZE): both must raise.
        "optional-below-min": blob[: opt_off + 60],
        # 70 < 73 with a PE32+ magic: the padded retry fails, both raise.
        "optional64-below-min": blob[: opt_off + 70],
        # A present-but-short section header raises in pefile's direct unpack.
        "section-short": blob[: sec_off + 25],
        "bad-mz": b"NZ" + blob[2:],
        "bad-pe-sig": blob[:0x40] + b"PX" + blob[0x42:],
    }


def _padded_optional_headers() -> dict[str, bytes]:
    blob = FIXTURES["pe64_ordinal"]
    opt_off = pefile.PE(data=blob).OPTIONAL_HEADER.get_file_offset()
    return {
        # 128 >= 73: pefile zero-pads the optional header and parses on.
        "optional64-padded": blob[: opt_off + 128],
        # Above 73 but below the 112-byte PE32+ fixed part: directory entry
        # #1 is truncated away, so pefile sees no import directory at all.
        "optional64-no-dirs": blob[: opt_off + 100],
    }


@pytest.mark.parametrize("label", sorted(_truncations()))
def test_truncated_headers_raise_like_oracle(label):
    blob = _truncations()[label]
    with pytest.raises(pefile.PEFormatError):
        pefile.PE(data=blob)
    with pytest.raises(pefile_mojo.PEFormatError):
        pefile_mojo.parse_imports(blob)
    if label != "section-short":
        # checksum_field_offset validates only the structures it reads
        # (DOS/NT/optional headers); the section table is outside its scope.
        with pytest.raises(pefile_mojo.PEFormatError):
            pefile_mojo.checksum_field_offset(blob)


@pytest.mark.parametrize("label", sorted(_padded_optional_headers()))
def test_padded_optional_headers_match_oracle(label):
    blob = _padded_optional_headers()[label]
    pe = pefile.PE(data=blob)  # must not raise (Tiny PE zero-padding behavior)
    assert ours_imports(blob) == oracle_imports(blob)
    # pefile's generate_checksum hashes pe.write(), which differs from the
    # input bytes for a truncated (zero-padded) header. The drop-in contract
    # is over the same serialized bytes, so compare against write() output.
    rewritten = bytes(pe.write())
    assert pefile_mojo.generate_checksum(
        rewritten, pefile_mojo.checksum_field_offset(rewritten)
    ) == oracle_checksum(pe)


# ---------------------------------------------------------------------------
# Header-level error parity
# ---------------------------------------------------------------------------


def test_empty_input_raises_like_oracle():
    with pytest.raises(pefile.PEFormatError):
        pefile.PE(data=b"")
    with pytest.raises(pefile_mojo.PEFormatError):
        pefile_mojo.parse_imports(b"")
    with pytest.raises(pefile_mojo.PEFormatError):
        pefile_mojo.checksum_field_offset(b"")


def test_garbage_input_raises_like_oracle():
    blob = b"\x7fELF not a PE at all" * 8
    with pytest.raises(pefile.PEFormatError):
        pefile.PE(data=blob)
    with pytest.raises(pefile_mojo.PEFormatError):
        pefile_mojo.parse_imports(blob)


def test_generate_checksum_validates_offset():
    with pytest.raises(ValueError):
        pefile_mojo.generate_checksum(b"MZ", -1)


def test_checksum_on_plain_bytes_is_pure_function():
    # generate_checksum over explicit (bytes, offset) needs no PE structure;
    # cross-check native vs fallback on raw bytes including a non-aligned tail.
    from pefile_mojo import _reference

    blob = bytes(range(256)) * 5 + b"\xaa" * 3
    want = _reference.checksum(blob, 64)
    if pefile_mojo.native_available():
        assert pefile_mojo.generate_checksum(blob, 64) == want
