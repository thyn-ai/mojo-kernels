"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os
import struct

import pytest

import pydicom_mojo
from pydicom_mojo import _native, core
from pydicom_mojo._native import NativeUnavailable


def _rle_header(nr_segments: int, offsets: list[int]) -> bytes:
    values = [nr_segments] + list(offsets)
    values += [0] * (16 - len(values))
    return struct.pack("<16L", *values)


# 4x4 16-bit frame, MSB segment (all zeros) then LSB segment (0..15), each
# a single 16-byte literal packet.
_SEG_MSB = bytes([15]) + bytes(16)
_SEG_LSB = bytes([15]) + bytes(range(16))
FRAME = _rle_header(2, [64, 64 + len(_SEG_MSB)]) + _SEG_MSB + _SEG_LSB
FRAME_EXPECTED = bytes(b for pair in zip(range(16), bytes(16)) for b in pair)

PLANE = bytes(range(64))
PLANE_ENCODED = pydicom_mojo._reference.encode_segment_reference(PLANE, 8)


def _reset_backend_caches(monkeypatch=None):
    _native._LIB, _native._LIB_SOURCE = None, None
    core._use_native = None


def test_frame_fixture_decodes():
    assert pydicom_mojo.decode_frame(FRAME, 4, 4, 1, 16) == FRAME_EXPECTED
    assert pydicom_mojo.encode_segment(PLANE, 8) == PLANE_ENCODED


def test_backend_info_shape():
    info = pydicom_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("PYDICOM_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (bash kernels/pydicom-rle/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_get_backend_matches_env():
    expected = "fallback" if os.environ.get("PYDICOM_MOJO_DISABLE_NATIVE") == "1" else "native"
    assert pydicom_mojo.get_backend() == expected


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("PYDICOM_MOJO_DISABLE_NATIVE", "1")
    _reset_backend_caches()
    try:
        assert pydicom_mojo.get_backend() == "fallback"
        assert pydicom_mojo.native_available() is False
        assert pydicom_mojo.decode_frame(FRAME, 4, 4, 1, 16) == FRAME_EXPECTED
        assert pydicom_mojo.encode_segment(PLANE, 8) == PLANE_ENCODED
    finally:
        _reset_backend_caches()


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash decoding: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("PYDICOM_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("PYDICOM_MOJO_DISABLE_NATIVE", raising=False)
    _reset_backend_caches()
    try:
        assert pydicom_mojo.decode_frame(FRAME, 4, 4, 1, 16) == FRAME_EXPECTED
        assert pydicom_mojo.encode_segment(PLANE, 8) == PLANE_ENCODED
    finally:
        _reset_backend_caches()


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("PYDICOM_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert pydicom_mojo.get_backend() == "fallback"
        assert pydicom_mojo.decode_frame(FRAME, 4, 4, 1, 16) == FRAME_EXPECTED
        assert pydicom_mojo.encode_segment(PLANE, 8) == PLANE_ENCODED
    finally:
        _reset_backend_caches()


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def rlemojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("PYDICOM_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/librlemojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _reset_backend_caches()
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _reset_backend_caches()
