"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import capa_mojo
from capa_mojo import _native
from capa_mojo._native import NativeUnavailable

RULE = """rule:
  meta:
    name: loader probe
    authors: [capa_mojo test]
    scopes:
      static: function
      dynamic: unsupported
  features:
    - and:
      - number: 0x10
      - or:
        - api: CreateFileA
        - count(number(0x20)): 2 or more
"""

FMAP = {("number", 0x10): {1}, ("api", "CreateFileA"): {2}}


def test_backend_info_shape():
    info = capa_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("CAPA_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (scripts build it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("CAPA_MOJO_DISABLE_NATIVE", "1")
    rs = capa_mojo.load_rules([RULE])
    assert rs.backend == "fallback"
    assert "loader probe" in rs.match(FMAP)
    assert capa_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it and
    # continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("CAPA_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("CAPA_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        rs = capa_mojo.load_rules([RULE])
        assert "loader probe" in rs.match(FMAP)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("CAPA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert capa_mojo.load_rules([RULE]).backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def capamojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("CAPA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libcapamojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
