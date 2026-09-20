"""Loader/backend behaviour for vader_mojo: ABI handshake, fallback forcing,
diagnostics."""

from __future__ import annotations

import os

import pytest

import vader_mojo
from vader_mojo import _native
from vader_mojo._native import NativeUnavailable

TEXT = "VADER is smart, handsome, and funny!"


def test_backend_info_shape():
    info = vader_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("VADER_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (scripts/test_all_vader.sh builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("VADER_MOJO_DISABLE_NATIVE", "1")
    analyzer = vader_mojo.SentimentIntensityAnalyzer()
    assert analyzer.backend == "fallback"
    assert vader_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash construction: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("VADER_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("VADER_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        analyzer = vader_mojo.SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(TEXT)
        assert set(scores) == {"neg", "neu", "pos", "compound"}
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("VADER_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert vader_mojo.SentimentIntensityAnalyzer().backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def vadermojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("VADER_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libvadermojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_analyzer_close_and_reuse(monkeypatch):
    monkeypatch.delenv("VADER_MOJO_DISABLE_NATIVE", raising=False)
    analyzer = vader_mojo.SentimentIntensityAnalyzer()
    if analyzer.backend != "native":
        pytest.skip("native kernel unavailable in this run")
    analyzer.close()
    with pytest.raises(NativeUnavailable):
        analyzer._native.polarity(TEXT)
    # closing twice is a no-op, and __del__ must not raise afterwards
    analyzer.close()
