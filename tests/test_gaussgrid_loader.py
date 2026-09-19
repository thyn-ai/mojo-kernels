"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import gaussgrid_fixtures as fx

import cclib_mojo
from cclib_mojo import _native
from cclib_mojo._native import NativeUnavailable

GBASIS, COORDS = fx.h2o_sto3g()
COEFF = np.ones(7)
GRID = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)


def _small_density():
    origin, step, shape = GRID
    return cclib_mojo.density_on_grid(GBASIS, COORDS, COEFF, origin, step, shape)


def test_backend_info_shape():
    info = cclib_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("CCLIB_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (pixi run test builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("CCLIB_MOJO_DISABLE_NATIVE", "1")
    grid = _small_density()
    assert grid.shape == (5, 5, 5)
    assert cclib_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash evaluation: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("CCLIB_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("CCLIB_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        grid = _small_density()
        assert grid.shape == (5, 5, 5)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("CCLIB_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        grid = _small_density()
        assert grid.shape == (5, 5, 5)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def gaussgridmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("CCLIB_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libgaussgridmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
