"""Loader/backend behaviour and input validation for metpy_mojo."""

from __future__ import annotations

import os

import numpy as np
import pytest

import metpy_mojo
from metpy_mojo import _native
from metpy_mojo._native import NativeUnavailable

P = np.linspace(1000.0, 200.0, 41)
Z = -np.log(P / 1000.0) * 7.5
T = np.maximum(300.0 - 6.5 * Z, 222.0)
TD = np.maximum(294.0 - 5.0 * Z, 200.0)


def test_backend_info_shape():
    info = metpy_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("METPY_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("METPY_MOJO_DISABLE_NATIVE", "1")
    assert metpy_mojo.native_available() is False
    cape, cin = metpy_mojo.cape_cin(P, T, TD)
    assert cape > 0.0 and cin <= 0.0  # fallback computes correctly


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("METPY_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("METPY_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        cape, cin = metpy_mojo.cape_cin(P, T, TD)
        assert cape > 0.0
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("METPY_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        cape, _ = metpy_mojo.cape_cin(P, T, TD)
        assert cape > 0.0
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def metpycapemojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("METPY_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libmetpycapemojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="same shape"):
        metpy_mojo.cape_cin(P, T, TD[:-1])
    with pytest.raises(ValueError, match="strictly decreasing"):
        metpy_mojo.cape_cin(P[::-1], T[::-1], TD[::-1])
    with pytest.raises(ValueError, match="finite"):
        metpy_mojo.cape_cin(P, T, TD.copy().astype(float) ** np.nan)
    with pytest.raises(ValueError, match="at least 2 levels"):
        metpy_mojo.cape_cin([1000.0], [300.0], [295.0])
    with pytest.raises(ValueError, match="which"):
        metpy_mojo.cape_cin(P, T, TD, which="mixed_layer")
    with pytest.raises(ValueError, match="standard-atmosphere scope"):
        # es(320 K) ~ 106 hPa exceeds the ambient pressure at both levels.
        metpy_mojo.cape_cin([100.0, 50.0], [320.0, 320.0], [290.0, 290.0])


def test_nan_lfc_when_never_buoyant():
    # Arctic-stable column: no LFC, so (0, 0) and NaN diagnostics.
    p = np.linspace(1000.0, 100.0, 91)
    z = -np.log(p / 1000.0) * 7.5
    T_s = np.maximum(268.0 - 4.0 * z, 210.0)
    Td_s = np.maximum(258.0 - 5.0 * z, 180.0)
    cape, cin = metpy_mojo.cape_cin(p, T_s, Td_s)
    assert cape == 0.0 and cin == 0.0
    diag = metpy_mojo.parcel_diagnostics(p, T_s, Td_s)
    assert np.isnan(diag.lfc_pressure) and np.isnan(diag.el_pressure)
    assert diag.lcl_pressure > 0.0


def test_el_nan_integrates_to_top():
    # The doc-6 sounding has no EL crossing: EL is NaN, CAPE still computed.
    p = np.array([959., 779.2, 751.3, 724.3, 700., 269.])
    T6 = np.array([22.2, 14.6, 12., 9.4, 7., -49.]) + 273.15
    Td6 = np.array([19., -11.2, -10.8, -10.4, -10., -53.]) + 273.15
    diag = metpy_mojo.parcel_diagnostics(p, T6, Td6)
    assert diag.cape > 0.0
    assert np.isnan(diag.el_pressure)
