"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import langdetect_mojo
from langdetect_mojo import _native
from langdetect_mojo._native import NativeUnavailable

TEXT = "This is a sample English sentence for language detection."


def test_backend_info_shape():
    info = langdetect_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("LANGDETECT_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (scripts/test_all_langdetect.sh builds it first).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("LANGDETECT_MOJO_DISABLE_NATIVE", "1")
    assert langdetect_mojo.backend() == "fallback"
    assert langdetect_mojo.native_available() is False
    # ... and detection still works (pure-Python path)
    assert langdetect_mojo.detect(TEXT) == "en"


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash detection: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("LANGDETECT_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("LANGDETECT_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module caches
    langdetect_mojo.core._STATE = None
    try:
        assert langdetect_mojo.detect(TEXT) == "en"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
        langdetect_mojo.core._STATE = None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("LANGDETECT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    langdetect_mojo.core._STATE = None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert langdetect_mojo.detect(TEXT) == "en"
        assert langdetect_mojo.backend() == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
        langdetect_mojo.core._STATE = None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def langdetectmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("LANGDETECT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/liblangdetectmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_native_model_rejects_garbage_blob():
    # The kernel validates the blob header; a garbage blob must not crash.
    if os.environ.get("LANGDETECT_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native kernel disabled")
    with pytest.raises(NativeUnavailable, match="rejected"):
        _native.NativeModel(b"not a profile blob at all", 55)
