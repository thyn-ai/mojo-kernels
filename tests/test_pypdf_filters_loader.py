"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import pdf_mojo
from pdf_mojo import _native
from pdf_mojo._native import NativeUnavailable

# One row, PNG Up filter, columns=3: reconstructs to [10, 20, 30] twice.
PNG_STREAM = bytes([2, 10, 20, 30, 2, 30, 30, 30])
PNG_EXPECTED = bytes([10, 20, 30, 40, 50, 60])


def _lzw_stream(codes, width=9) -> bytes:
    """Pack codes MSB-first at a fixed width, zero-padded to a byte."""
    bits = "".join(f"{c:0{width}b}" for c in codes)
    bits += "0" * ((8 - len(bits) % 8) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


# clear(256) 'A'(65) EOD(257), 9-bit MSB-first, zero-padded to a byte.
LZW_STREAM = _lzw_stream([256, 65, 257])
LZW_EXPECTED = b"A"


def _reset_backend_caches(monkeypatch=None):
    _native._LIB, _native._LIB_SOURCE = None, None
    pdf_mojo._use_native = None


def test_backend_info_shape():
    info = pdf_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("PDF_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (bash kernels/pypdf-filters/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_get_backend_matches_env():
    expected = "fallback" if os.environ.get("PDF_MOJO_DISABLE_NATIVE") == "1" else "native"
    assert pdf_mojo.get_backend() == expected


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("PDF_MOJO_DISABLE_NATIVE", "1")
    _reset_backend_caches()
    try:
        assert pdf_mojo.get_backend() == "fallback"
        assert pdf_mojo.native_available() is False
        assert pdf_mojo.decode_png_prediction(PNG_STREAM, 3, 1, 8, 12) == PNG_EXPECTED
        assert pdf_mojo.decode_lzw(LZW_STREAM) == LZW_EXPECTED
    finally:
        _reset_backend_caches()


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash decoding: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("PDF_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("PDF_MOJO_DISABLE_NATIVE", raising=False)
    _reset_backend_caches()
    try:
        assert pdf_mojo.decode_png_prediction(PNG_STREAM, 3, 1, 8, 12) == PNG_EXPECTED
        assert pdf_mojo.decode_lzw(LZW_STREAM) == LZW_EXPECTED
    finally:
        _reset_backend_caches()


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("PDF_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert pdf_mojo.get_backend() == "fallback"
        assert pdf_mojo.decode_png_prediction(PNG_STREAM, 3, 1, 8, 12) == PNG_EXPECTED
        assert pdf_mojo.decode_lzw(LZW_STREAM) == LZW_EXPECTED
    finally:
        _reset_backend_caches()


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def pdfmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("PDF_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libpdfmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _reset_backend_caches()
