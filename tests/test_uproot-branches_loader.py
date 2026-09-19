"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import uproot_mojo
from uproot_mojo import _fallback, _native
from uproot_mojo._native import NativeUnavailable


def _one_basket():
    # Minimal std::string basket: two entries, "ab" and "".
    data = b"\x02ab\x00"
    import numpy as np

    return data, np.array([0, 3, 4], dtype=np.int64)


def test_backend_info_shape():
    info = uproot_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("UPROOT_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (scripts build it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("UPROOT_MOJO_DISABLE_NATIVE", "1")
    assert uproot_mojo.native_available() is False
    info = uproot_mojo.backend_info()
    assert info["native_available"] is False
    assert info["disabled_by_env"] is True


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it
    # and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("UPROOT_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("UPROOT_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        data, borders = _one_basket()
        detected, offsets, content, _ = _native.walk_basket(
            data, borders, _fallback.MODE_AUTO_STR, 0, "loader test"
        )
        assert detected == _fallback.MODE_STR_ONE
        assert content == b"ab"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("UPROOT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        walk, backend = uproot_mojo.core._backend_walk()
        assert backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def uprootmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("UPROOT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libuprootmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
