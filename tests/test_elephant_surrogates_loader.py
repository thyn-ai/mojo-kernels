"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import elephant_mojo
from elephant_mojo import _native
from elephant_mojo._native import NativeUnavailable

MAX_OCCS = np.array([[2.0, 0.0], [3.0, 1.0], [5.0, 0.0]])
TRAINS = [np.array([100.0, 250.0, 700.0])]


def _small_spectrum():
    return elephant_mojo.pvalue_spectrum(MAX_OCCS, 2, 3, 1)


def _small_dither():
    return elephant_mojo.dither(
        TRAINS, 5.0, 15.0, n_surrogates=2, t_stop=1000.0, seed=0
    )


def test_backend_info_shape():
    info = elephant_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("ELEPHANT_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("ELEPHANT_MOJO_DISABLE_NATIVE", "1")
    entries = _small_spectrum()
    assert entries == [
        [2, 1, 1.0],
        [2, 2, 1.0],
        [2, 3, 2 / 3],
        [2, 4, 1 / 3],
        [2, 5, 1 / 3],
        [3, 1, 1 / 3],
    ]
    assert elephant_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash evaluation: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("ELEPHANT_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("ELEPHANT_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert _small_spectrum()
        assert _small_dither().shape == (2, 1, 200)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("ELEPHANT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert _small_spectrum()
        assert _small_dither().shape == (2, 1, 200)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def elephantsurrogatesmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("ELEPHANT_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native,
        "_candidate_paths",
        lambda: [("fake", "/fake/libelephantsurrogatesmojo.dylib")],
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
