"""motmetrics-mojo loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pytest

import motmetrics_mojo as M
from motmetrics_mojo import _native
from motmetrics_mojo._native import NativeUnavailable


def _quick_acc():
    acc = M.MOTAccumulator(auto_id=True)
    acc.update([1, 2], [10, 20], [[0.1, 0.9], [0.8, 0.2]])
    acc.update([1, 2], [10, 20], [[0.2, 0.9], [0.9, 0.1]])
    return acc


def test_backend_info_shape():
    info = M.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("MOTMETRICS_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/motmetrics/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("MOTMETRICS_MOJO_DISABLE_NATIVE", "1")
    acc = _quick_acc()
    assert acc.native_backend is False
    assert M.native_available() is False
    s = M.compute(acc, metrics=["mota", "motp", "idf1"])
    assert s["mota"] == 1.0 and s["idf1"] == 1.0
    assert s["motp"] == pytest.approx(0.15)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("MOTMETRICS_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("MOTMETRICS_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        acc = _quick_acc()
        assert M.compute(acc, metrics=["mota"])["mota"] == 1.0
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("MOTMETRICS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        acc = _quick_acc()
        assert acc.native_backend is False
        assert M.compute(acc, metrics=["idf1"])["idf1"] == 1.0
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def motmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("MOTMETRICS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libmotmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_native_path_counters(monkeypatch):
    if os.environ.get("MOTMETRICS_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("fallback run")
    acc = _quick_acc()
    assert acc.native_backend is True
    counts = acc.counts()
    assert counts["num_frames"] == 2
    assert counts["num_objects"] == 4
    assert counts["num_predictions"] == 4
    assert counts["num_matches"] == 4
    assert counts["dist_sum"] == pytest.approx(0.6, abs=1e-15)
    for key, value in counts.items():
        if key == "dist_sum":
            assert isinstance(value, float)
        else:
            assert isinstance(value, int) and not isinstance(value, bool)


def test_array_like_inputs_and_empty():
    acc = M.MOTAccumulator(auto_id=True)
    # lists, tuples, numpy arrays; scalar distance for the 1x1 case
    acc.update((1,), [10], 0.3)
    acc.update(np.array([1, 2], dtype=np.int32), np.array([10]), np.array([[0.2], [0.4]]))
    acc.update([], [], np.zeros((0, 0)))
    s = M.compute(acc, metrics=["num_frames", "num_matches", "num_misses"])
    assert s.to_dict() == {"num_frames": 3, "num_matches": 2, "num_misses": 1}
