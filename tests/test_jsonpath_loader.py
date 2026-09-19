"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import jsonpath_mojo
from jsonpath_mojo import _native
from jsonpath_mojo._native import NativeUnavailable

DOC = {"a": {"b": [1, 2, 3]}, "c": "x"}
EXPR = "$.a.b[?(@ > 1)]"


def test_backend_info_shape():
    info = jsonpath_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("JSONPATH_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("JSONPATH_MOJO_DISABLE_NATIVE", "1")
    assert jsonpath_mojo.native_available() is False
    # the public API stays correct on the forced fallback
    assert jsonpath_mojo.find(EXPR, DOC) == [(2, "a.b.[1]"), (3, "a.b.[2]")]


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the query: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("JSONPATH_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("JSONPATH_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert jsonpath_mojo.find(EXPR, DOC) == [(2, "a.b.[1]"), (3, "a.b.[2]")]
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("JSONPATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert jsonpath_mojo.find(EXPR, DOC) == [(2, "a.b.[1]"), (3, "a.b.[2]")]
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def jsonpathmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("JSONPATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libjsonpathmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_find_rejects_non_str_expression():
    with pytest.raises(TypeError):
        jsonpath_mojo.find(42, DOC)
