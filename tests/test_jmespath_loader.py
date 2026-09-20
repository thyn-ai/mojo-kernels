"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import jmespath_mojo
from jmespath_mojo import _native
from jmespath_mojo._native import NativeUnavailable

DOC = {"a": [{"b": 1}, {"b": 2}]}


def test_backend_info_shape():
    info = jmespath_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("JMESPATH_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("JMESPATH_MOJO_DISABLE_NATIVE", "1")
    before = dict(_native._STATS)
    assert jmespath_mojo.search("a[*].b", DOC) == [1, 2]
    assert jmespath_mojo.native_available() is False
    after = dict(_native._STATS)
    assert after["native_ok"] == before["native_ok"]


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the search: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("JMESPATH_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("JMESPATH_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert jmespath_mojo.search("a[*].b", DOC) == [1, 2]
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("JMESPATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert jmespath_mojo.search("a[*].b", DOC) == [1, 2]
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def jmespathmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("JMESPATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libjmespathmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_fallback_engagement_counter(monkeypatch):
    monkeypatch.delenv("JMESPATH_MOJO_DISABLE_NATIVE", raising=False)
    if not jmespath_mojo.native_available():
        pytest.skip("native kernel not built")
    before = dict(_native._STATS)
    # in-scope: native
    assert jmespath_mojo.search("a[*].b", DOC) == [1, 2]
    # multiselect: outside the kernel subset -> fallback counter moves,
    # result still correct.
    assert jmespath_mojo.search("a[0].{x: b}", DOC) == {"x": 1}
    # error path: fallback raises the reference error class
    with pytest.raises(Exception, match="abs"):
        jmespath_mojo.search("abs('x')", DOC)
    after = dict(_native._STATS)
    assert after["native_ok"] - before["native_ok"] == 1
    assert after["fallback_engaged"] - before["fallback_engaged"] == 2
