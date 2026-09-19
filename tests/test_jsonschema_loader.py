"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import jsonschema_mojo
from jsonschema_mojo import _native
from jsonschema_mojo._native import NativeUnavailable

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
INSTANCE = {"a": 1}


def test_backend_info_shape():
    info = jsonschema_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("JSONSCHEMA_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (bash kernels/jsonschema/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("JSONSCHEMA_MOJO_DISABLE_NATIVE", "1")
    validator = jsonschema_mojo.Validator(SCHEMA)
    assert validator.backend == "fallback"
    assert jsonschema_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash construction: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("JSONSCHEMA_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("JSONSCHEMA_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        validator = jsonschema_mojo.Validator(SCHEMA)
        assert validator.is_valid(INSTANCE) is True
        assert validator.is_valid({"a": "x"}) is False
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("JSONSCHEMA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert jsonschema_mojo.Validator(SCHEMA).backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def jsonschemamojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("JSONSCHEMA_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native,
        "_candidate_paths",
        lambda: [("fake", "/fake/libjsonschemamojo.dylib")],
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_native_schema_rejects_garbage_text():
    # The kernel must cleanly reject non-JSON schema text (no crash); the
    # public API then falls back.
    if not _native.native_available():
        pytest.skip("native kernel not built")
    with pytest.raises(NativeUnavailable):
        _native.NativeSchema(b"{not json")


def test_handle_close_and_redel():
    if not _native.native_available():
        pytest.skip("native kernel not built")
    schema = _native.NativeSchema(b'{"type":"integer"}')
    schema.close()
    schema.close()  # double close is a no-op
    with pytest.raises(NativeUnavailable):
        schema.validate_records(b"1")
