"""spatialmath-mojo loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import spatialmath_mojo
from spatialmath_mojo import _native
from spatialmath_mojo._native import NativeUnavailable

POSE = np.eye(4)[np.newaxis, :, :].repeat(4, axis=0)
POINTS = np.linspace(-1.0, 1.0, 15).reshape(5, 3)


def test_backend_info_shape():
    info = spatialmath_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("SPATIALMATH_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/spatialmath/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("SPATIALMATH_MOJO_DISABLE_NATIVE", "1")
    out = spatialmath_mojo.inverse(POSE)
    assert out.shape == POSE.shape
    assert spatialmath_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("SPATIALMATH_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("SPATIALMATH_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        out = spatialmath_mojo.compose(POSE, POSE)
        assert out.shape == POSE.shape
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("SPATIALMATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        out = spatialmath_mojo.transform(POSE, POINTS)
        assert out.shape == (4, 5, 3)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def spatialmathmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("SPATIALMATH_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libspatialmathmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_native_paths_return_fresh_float64_arrays():
    if os.environ.get("SPATIALMATH_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback run")
    composed = spatialmath_mojo.compose(POSE, POSE)
    inverted = spatialmath_mojo.inverse(POSE)
    transformed = spatialmath_mojo.transform(POSE, POINTS)
    for arr in (composed, inverted, transformed):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
    np.testing.assert_allclose(composed, POSE, rtol=0, atol=0.0)
    np.testing.assert_allclose(inverted, POSE, rtol=0, atol=0.0)
    np.testing.assert_allclose(
        transformed, np.broadcast_to(POINTS, (4, 5, 3)), rtol=0, atol=0.0
    )
