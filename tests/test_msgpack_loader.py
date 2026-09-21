"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import msgpack_mojo
from msgpack_mojo import _native
from msgpack_mojo._native import NativeUnavailable

DOC = {"a": [1, 2.5, "x"], "b": b"\x00\xff"}


def test_backend_info_shape():
    info = msgpack_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("MSGPACK_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/msgpack/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("MSGPACK_MOJO_DISABLE_NATIVE", "1")
    assert msgpack_mojo.native_available() is False
    assert msgpack_mojo.unpackb(msgpack_mojo.packb(DOC)) == DOC


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash pack/unpack: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("MSGPACK_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("MSGPACK_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert msgpack_mojo.unpackb(msgpack_mojo.packb(DOC)) == DOC
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_falls_back_transparently(monkeypatch):
    monkeypatch.delenv("MSGPACK_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently to the
        # vendored pure-Python engine.
        assert msgpack_mojo.unpackb(msgpack_mojo.packb(DOC)) == DOC
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def msgpackmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("MSGPACK_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libmsgpackmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_both_engines_produce_identical_bytes():
    """The native pack engine and the vendored pure-Python engine must agree
    byte-for-byte on the same instruction stream (backend interchangeability,
    independent of the oracle)."""
    from msgpack_mojo import _engine_py
    from msgpack_mojo._encode import encode_stream

    doc = {"k": [1, -33, 2**40, 1.5, "héllo", b"\x00\xff", None, True], "z": {}}
    stream = encode_stream(doc)
    for flags in (0, 1, 2, 3):
        try:
            native = _native.pack_bytes(stream, flags)
        except NativeUnavailable:
            continue  # forced-fallback run: nothing to compare against
        assert _engine_py.pack_engine(stream, flags) == native


def test_both_engines_unpack_identically():
    from msgpack_mojo import _engine_py

    blob = msgpack_mojo.packb(DOC)
    status_n, pos_n, rec_n = (None, None, None)
    try:
        status_n, pos_n, rec_n = _native.unpack_bytes(blob, 0)
    except NativeUnavailable:
        pass
    status_p, pos_p, rec_p = _engine_py.unpack_engine(blob, 0)
    assert status_p == 0 and pos_p == -1
    if status_n is not None:
        assert (status_n, pos_n, rec_n) == (status_p, pos_p, rec_p)
