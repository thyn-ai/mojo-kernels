"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import pykalman_mojo
from pykalman_mojo import _native
from pykalman_mojo._native import NativeUnavailable

PARAMS = dict(
    transition_matrices=[[1.0, 0.1], [0.0, 1.0]],
    observation_matrices=[[1.0, 0.0]],
)
OBS = np.arange(12, dtype=np.float64).reshape(12, 1)


def test_backend_info_shape():
    info = pykalman_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("PYKALMAN_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("PYKALMAN_MOJO_DISABLE_NATIVE", "1")
    kf = pykalman_mojo.KalmanFilter(**PARAMS)
    fm, fc = kf.filter(OBS)
    assert kf.backend == "fallback"
    assert fm.shape == (12, 2) and fc.shape == (12, 2, 2)
    assert pykalman_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash construction: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("PYKALMAN_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("PYKALMAN_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        kf = pykalman_mojo.KalmanFilter(**PARAMS)
        fm, _ = kf.filter(OBS)
        assert fm.shape == (12, 2)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("PYKALMAN_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert pykalman_mojo.KalmanFilter(**PARAMS).backend == "fallback"
        fm, _ = pykalman_mojo.KalmanFilter(**PARAMS).filter(OBS)
        assert fm.shape == (12, 2)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def pykalmanmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("PYKALMAN_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libpykalmanmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_singular_system_served_by_fallback(monkeypatch):
    # A zero observation covariance makes the innovation covariance S
    # singular; the kernel reports status 3 and the wrapper serves that call
    # from the reference implementation (whose pseudo-inverse copes).
    if os.environ.get("PYKALMAN_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native-only behaviour")
    pykalman = pytest.importorskip("pykalman")
    params = dict(
        transition_matrices=[[0.5]],
        observation_matrices=[[1.0]],
        observation_covariance=[[0.0]],
        transition_covariance=[[1.0]],
        initial_state_covariance=[[0.0]],
    )
    obs = np.zeros((5, 1))
    kf = pykalman_mojo.KalmanFilter(**params)
    fm, fc = kf.filter(obs)
    ref_fm, ref_fc = pykalman.KalmanFilter(**params).filter(obs)
    # P0 = 0, R = 0 => S = 0 at t=0: pinv(0) = 0 => gain 0, filtered = prior.
    assert fm[0, 0] == pytest.approx(0.0)
    assert np.isfinite(fm).all() and np.isfinite(fc).all()
    np.testing.assert_allclose(fm, ref_fm, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fc, ref_fc, rtol=0, atol=1e-10)
