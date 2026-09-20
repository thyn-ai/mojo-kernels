"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import oletools_mojo
from oletools_mojo import _native
from oletools_mojo._native import NativeUnavailable

# One RawChunk holding b"VBA!" (short tail, copied leniently), sig byte first.
STREAM = b"\x01" + (0x3FFF).to_bytes(2, "little") + b"VBA!"
EXPECTED = b"VBA!"


def _reset_backend_caches(monkeypatch=None):
    _native._LIB, _native._LIB_SOURCE = None, None
    oletools_mojo._use_native = None


def test_backend_info_shape():
    info = oletools_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("OLETOOLS_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (bash kernels/oletools-vba/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_get_backend_matches_env():
    expected = "fallback" if os.environ.get("OLETOOLS_MOJO_DISABLE_NATIVE") == "1" else "native"
    assert oletools_mojo.get_backend() == expected


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("OLETOOLS_MOJO_DISABLE_NATIVE", "1")
    _reset_backend_caches()
    try:
        assert oletools_mojo.get_backend() == "fallback"
        assert oletools_mojo.native_available() is False
        assert oletools_mojo.decompress_stream(STREAM) == EXPECTED
    finally:
        _reset_backend_caches()


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash decoding: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("OLETOOLS_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("OLETOOLS_MOJO_DISABLE_NATIVE", raising=False)
    _reset_backend_caches()
    try:
        assert oletools_mojo.decompress_stream(STREAM) == EXPECTED
    finally:
        _reset_backend_caches()


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("OLETOOLS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert oletools_mojo.get_backend() == "fallback"
        assert oletools_mojo.decompress_stream(STREAM) == EXPECTED
    finally:
        _reset_backend_caches()


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def vbamojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("OLETOOLS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libvbamojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _reset_backend_caches()


def test_integration_install_uninstall():
    # The olevba monkeypatch helper swaps and restores the module attribute.
    from oletools import olevba

    from oletools_mojo import olevba_integration

    original = olevba.decompress_stream
    olevba_integration.install()
    try:
        assert olevba.decompress_stream is oletools_mojo.decompress_stream
        assert olevba_integration.is_installed()
        assert olevba.decompress_stream(STREAM) == EXPECTED
    finally:
        olevba_integration.uninstall()
    assert olevba.decompress_stream is original
    assert not olevba_integration.is_installed()
