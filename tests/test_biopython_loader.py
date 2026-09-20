"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import io
import os

import pytest

import bio_mojo
from bio_mojo import _native
from bio_mojo.errors import NativeUnavailable

_GB = (
    "LOCUS       T            4 bp    DNA             UNA       01-JAN-2000\n"
    "DEFINITION  x.\nACCESSION   A1\nVERSION     A1.1\n"
    "ORIGIN\n        1 acgt\n//\n"
)


def _parse_both():
    assert [r.id for r in bio_mojo.parse_fasta(io.StringIO(">a\nAC\n"))] == ["a"]
    recs = list(bio_mojo.parse_genbank(io.StringIO(_GB)))
    assert recs[0].id == "A1.1"
    assert str(recs[0].seq) == "ACGT"


def test_backend_info_shape():
    info = bio_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("BIO_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (the test script builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("BIO_MOJO_DISABLE_NATIVE", "1")
    assert bio_mojo.native_available() is False
    assert bio_mojo.backend() == "fallback"
    _parse_both()


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash parsing: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("BIO_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("BIO_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        _parse_both()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("BIO_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert bio_mojo.backend() == "fallback"
        _parse_both()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def bioparse_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("BIO_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libbioparse.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
