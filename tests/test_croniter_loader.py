"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

import croniter_mojo
from croniter_mojo import _native
from croniter_mojo._native import NativeUnavailable

START = datetime(2026, 9, 19, 14, 30, 45)


def test_backend_info_shape():
    info = croniter_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("CRONITER_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("CRONITER_MOJO_DISABLE_NATIVE", "1")
    assert croniter_mojo.native_available() is False
    # ... and the public API still produces the oracle-identical result.
    assert croniter_mojo.get_next("0 9 * * mon", START) == datetime(2026, 9, 21, 9, 0)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it
    # and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("CRONITER_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("CRONITER_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert croniter_mojo.get_next("0 9 * * mon", START) == datetime(2026, 9, 21, 9, 0)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("CRONITER_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert croniter_mojo.get_next("0 9 * * mon", START) == datetime(2026, 9, 21, 9, 0)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def cronitermojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("CRONITER_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libcronitermojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_parse_cache_reuses_schedule():
    import croniter_mojo.core as core

    core._cache.clear()
    croniter_mojo.get_next("0 9 * * *", START)
    size_after_first = len(core._cache)
    croniter_mojo.get_next("0 9 * * *", START)
    assert len(core._cache) == size_after_first  # second call is a cache hit
