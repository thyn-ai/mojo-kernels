"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import pymavlink_mojo
from pymavlink_mojo import _native
from pymavlink_mojo.errors import NativeUnavailable


def _parse_both_modes():
    # one v2 HEARTBEAT frame under each framing mode
    frame = bytes.fromhex("fd090000002a070000003930000002030405038606")
    raw = pymavlink_mojo.parse_buffer(frame)
    assert raw.messages[0].get_type() == "HEARTBEAT"
    assert raw.consumed == len(frame)
    tlog = pymavlink_mojo.parse_buffer(b"\x00\x06=\xfbp\xde\xd0\x00" + frame,
                                       timestamps=True)
    assert tlog.messages[0].get_type() == "HEARTBEAT"
    assert tlog.messages[0]._timestamp == 1757000000.0
    assert tlog.consumed == 8 + len(frame)


def test_backend_info_shape():
    info = pymavlink_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("PYMAVLINK_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
        assert pymavlink_mojo.backend() == "fallback"
    else:
        # The suite's native run requires a built kernel (the test script builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]
        assert pymavlink_mojo.backend() == "native"


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("PYMAVLINK_MOJO_DISABLE_NATIVE", "1")
    assert pymavlink_mojo.native_available() is False
    assert pymavlink_mojo.backend() == "fallback"
    _parse_both_modes()


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash parsing: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("PYMAVLINK_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("PYMAVLINK_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        _parse_both_modes()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("PYMAVLINK_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert pymavlink_mojo.backend() == "fallback"
        _parse_both_modes()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def pymavmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("PYMAVLINK_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [("fake", "/fake")])
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda p: FakeLib())
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
        assert pymavlink_mojo.backend() == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_input_types():
    frame = bytes.fromhex("fd090000002a070000003930000002030405038606")
    for data in (frame, bytearray(frame), memoryview(frame)):
        res = pymavlink_mojo.parse_buffer(data)
        assert res.messages[0].get_type() == "HEARTBEAT"
    with pytest.raises(TypeError):
        pymavlink_mojo.parse_buffer("not bytes")
    with pytest.raises(TypeError):
        pymavlink_mojo.parse_buffer(12345)
