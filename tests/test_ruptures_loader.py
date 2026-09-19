"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import ruptures_mojo
from ruptures_mojo import _native
from ruptures_mojo._native import NativeUnavailable

SIGNAL = [0.1, 1.2, -0.4, 2.0, 0.3, -1.1, 0.9, 0.0, 1.7, -0.6, 0.5, 0.8]


def test_backend_info_shape():
    info = ruptures_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("RUPTURES_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (build.sh runs first).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("RUPTURES_MOJO_DISABLE_NATIVE", "1")
    bkps = ruptures_mojo.detect(SIGNAL, "dynp", n_bkps=1)
    assert ruptures_mojo.last_backend() == "fallback"
    assert bkps[-1] == len(SIGNAL)
    assert ruptures_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash detection: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("RUPTURES_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("RUPTURES_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        bkps = ruptures_mojo.detect(SIGNAL, "binseg", n_bkps=1)
        assert bkps[-1] == len(SIGNAL)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("RUPTURES_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        ruptures_mojo.detect(SIGNAL, "pelt", pen=1.0)
        assert ruptures_mojo.last_backend() == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def rupturesmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("RUPTURES_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/librupturesmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_native_and_fallback_agree_on_tie_heavy_signal():
    # Integer-valued signal: many exact cost ties; both backends must return
    # the identical breakpoint list (the differential suite proves this
    # against ruptures itself; here we pin backend-vs-backend agreement).
    import numpy as np

    sig = np.resize(np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0]), 96)
    native_bkps = ruptures_mojo.detect(sig, "dynp", model="l2", min_size=2, jump=2, n_bkps=4)
    if ruptures_mojo.last_backend() != "native":
        pytest.skip("native kernel unavailable in this run")
    os.environ["RUPTURES_MOJO_DISABLE_NATIVE"] = "1"
    try:
        fallback_bkps = ruptures_mojo.detect(
            sig, "dynp", model="l2", min_size=2, jump=2, n_bkps=4
        )
    finally:
        del os.environ["RUPTURES_MOJO_DISABLE_NATIVE"]
    assert native_bkps == fallback_bkps


def test_native_too_large_falls_back(monkeypatch):
    # When the native kernel reports the problem exceeds its capacity (e.g.
    # a Dynp admissible grid above its matrix cap), the wrapper must
    # transparently serve the vendored fallback with identical results.
    monkeypatch.delenv("RUPTURES_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(
        _native, "native_detect", lambda *a: _native.STATUS_TOO_LARGE
    )
    bkps = ruptures_mojo.detect(SIGNAL, "dynp", n_bkps=1)
    assert ruptures_mojo.last_backend() == "fallback"
    assert bkps[-1] == len(SIGNAL)
