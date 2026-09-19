"""ta-mojo loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import ta_mojo
from ta_mojo import _native
from ta_mojo._native import NativeUnavailable

CLOSE = np.linspace(100.0, 110.0, 64) + np.sin(np.linspace(0.0, 6.0, 64))


def test_backend_info_shape():
    info = ta_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("TA_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/ta/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("TA_MOJO_DISABLE_NATIVE", "1")
    out = ta_mojo.ema(CLOSE, 10)
    assert out.shape == CLOSE.shape
    assert ta_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("TA_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("TA_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        out = ta_mojo.rsi(CLOSE, 14)
        assert out.shape == CLOSE.shape
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("TA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        out = ta_mojo.atr(CLOSE, CLOSE, CLOSE, 14)
        assert out.shape == CLOSE.shape
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def tamojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("TA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libtamojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_macd_native_path_returns_three_arrays():
    if os.environ.get("TA_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback run")
    result = ta_mojo.macd(CLOSE, 12, 26, 9)
    assert isinstance(result, ta_mojo.MacdResult)
    for arr in (result.macd, result.signal, result.histogram):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
        assert arr.shape == CLOSE.shape
