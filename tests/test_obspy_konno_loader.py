"""Loader/backend behaviour for obspy-mojo: ABI handshake, fallback forcing,
native-vs-fallback dispatch, and diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import obspy_mojo
from obspy_mojo import _native
from obspy_mojo._native import NativeUnavailable

FREQS = np.logspace(-1, 2, 64)
SPECTRA = np.random.default_rng(7).random(64) * 100.0


def _small_smooth():
    return obspy_mojo.konno_ohmachi_smoothing(SPECTRA, FREQS, normalize=True)


def test_backend_info_shape():
    info = obspy_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("OBSPY_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (scripts/test_all_obspy_konno.sh builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("OBSPY_MOJO_DISABLE_NATIVE", "1")
    out = _small_smooth()
    assert out.shape == (64,)
    assert obspy_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash smoothing: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("OBSPY_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("OBSPY_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        out = _small_smooth()
        assert out.shape == (64,)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("OBSPY_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        out = _small_smooth()
        assert out.shape == (64,)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def konnomojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("OBSPY_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libkonnomojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


# --------------------------------------------------------------------------
# dispatch: each public call must be served by the expected backend
# --------------------------------------------------------------------------


def _dispatch_spy(monkeypatch):
    """Count successful calls into each backend implementation (native attempts
    that raise NativeUnavailable count separately as failed attempts)."""
    calls = {
        "native_loop": 0,
        "native_matrix": 0,
        "native_loop_fail": 0,
        "native_matrix_fail": 0,
        "ref_loop": 0,
        "ref_matrix": 0,
    }
    from obspy_mojo import _reference

    real_nl, real_nm = _native.smooth_loop, _native.window_matrix
    real_rl, real_rm = _reference.smooth_loop, _reference.smoothing_matrix

    def spy_native_loop(*args):
        try:
            result = real_nl(*args)
        except NativeUnavailable:
            calls["native_loop_fail"] += 1
            raise
        calls["native_loop"] += 1
        return result

    def spy_native_matrix(*args):
        try:
            result = real_nm(*args)
        except NativeUnavailable:
            calls["native_matrix_fail"] += 1
            raise
        calls["native_matrix"] += 1
        return result

    def spy_ref_loop(*args, **kwargs):
        calls["ref_loop"] += 1
        return real_rl(*args, **kwargs)

    def spy_ref_matrix(*args, **kwargs):
        calls["ref_matrix"] += 1
        return real_rm(*args, **kwargs)

    monkeypatch.setattr(_native, "smooth_loop", spy_native_loop)
    monkeypatch.setattr(_native, "window_matrix", spy_native_matrix)
    monkeypatch.setattr(_reference, "smooth_loop", spy_ref_loop)
    monkeypatch.setattr(_reference, "smoothing_matrix", spy_ref_matrix)
    return calls


def _expected_native() -> bool:
    return os.environ.get("OBSPY_MOJO_DISABLE_NATIVE") != "1"


def test_dispatch_loop_path(monkeypatch):
    calls = _dispatch_spy(monkeypatch)
    out = obspy_mojo.konno_ohmachi_smoothing(SPECTRA, FREQS, normalize=True)
    assert out.shape == (64,)
    if _expected_native():
        assert (calls["native_loop"], calls["ref_loop"]) == (1, 0)
        assert calls["native_loop_fail"] == 0
    else:
        assert (calls["native_loop"], calls["native_loop_fail"], calls["ref_loop"]) == (0, 1, 1)
    assert calls["native_matrix"] == 0 and calls["ref_matrix"] == 0


def test_dispatch_matrix_path(monkeypatch):
    calls = _dispatch_spy(monkeypatch)
    spectra = np.random.default_rng(8).random((3, 64))
    out = obspy_mojo.konno_ohmachi_smoothing(spectra, FREQS, normalize=True)
    assert out.shape == (3, 64)
    if _expected_native():
        # native builds the window matrix; NumPy owns normalize/power/matmul
        assert (calls["native_matrix"], calls["ref_matrix"]) == (1, 0)
        assert calls["native_matrix_fail"] == 0
    else:
        assert (calls["native_matrix"], calls["native_matrix_fail"], calls["ref_matrix"]) == (0, 1, 1)
    assert calls["native_loop"] == 0 and calls["ref_loop"] == 0


def test_dispatch_count2_matrix_and_loop(monkeypatch):
    calls = _dispatch_spy(monkeypatch)
    obspy_mojo.konno_ohmachi_smoothing(SPECTRA, FREQS, count=2, normalize=True)
    obspy_mojo.konno_ohmachi_smoothing(
        SPECTRA, FREQS, count=2, max_memory_usage=0, normalize=True
    )
    if _expected_native():
        assert (calls["native_matrix"], calls["native_loop"]) == (1, 1)
        assert calls["ref_loop"] == 0 and calls["ref_matrix"] == 0
    else:
        assert (calls["ref_matrix"], calls["ref_loop"]) == (1, 1)
        assert calls["native_matrix"] == 0 and calls["native_loop"] == 0
