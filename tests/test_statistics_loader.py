"""Loader/backend behaviour for statistics_mojo: ABI handshake, fallback
forcing, diagnostics, and the load-time CPython-layout self-test."""

from __future__ import annotations

import os

import pytest

import statistics_mojo
from statistics_mojo import _native
from statistics_mojo._native import NativeUnavailable

DATA = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]


def test_backend_info_shape():
    info = statistics_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("STATISTICS_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (the build step runs first).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("STATISTICS_MOJO_DISABLE_NATIVE", "1")
    assert statistics_mojo.native_available() is False
    # The public API keeps working (stdlib delegation), bit-identically.
    import statistics

    assert statistics_mojo.variance(DATA) == statistics.variance(DATA)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it and
    # continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("STATISTICS_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("STATISTICS_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None
    try:
        assert statistics_mojo.variance(DATA) == pytest.approx(1.3720238095238095)
    finally:
        _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("STATISTICS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        import statistics

        assert statistics_mojo.stdev(DATA) == statistics.stdev(DATA)
    finally:
        _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def statsmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("STATISTICS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libstatsmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib, pydll: None)
    _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None


def test_self_test_failure_rejected(monkeypatch):
    # A kernel that fails the load-time self-test (e.g. CPython layout drift)
    # must be treated as unavailable, never trusted.
    monkeypatch.delenv("STATISTICS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(
        _native,
        "_self_test",
        lambda lib, pydll: (_ for _ in ()).throw(
            NativeUnavailable("native self-test failed: forced")
        ),
    )
    _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None
    try:
        with pytest.raises(NativeUnavailable, match="self-test"):
            _native._load()
        import statistics

        assert statistics_mojo.mean(DATA) == statistics.mean(DATA)
    finally:
        _native._LIB, _native._PYLIB, _native._LIB_SOURCE = None, None, None
