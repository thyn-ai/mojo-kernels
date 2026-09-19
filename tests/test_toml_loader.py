"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import toml_mojo
from toml_mojo import _native
from toml_mojo._native import NativeUnavailable

DOC = "[a]\nb = 1\n"


def test_backend_info_shape():
    info = toml_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("TOML_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (kernels/toml/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("TOML_MOJO_DISABLE_NATIVE", "1")
    assert toml_mojo.native_available() is False
    assert toml_mojo.loads(DOC) == {"a": {"b": 1}}


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash parsing: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("TOML_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("TOML_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert toml_mojo.loads(DOC) == {"a": {"b": 1}}
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("TOML_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently to tomllib.
        assert toml_mojo.loads(DOC) == {"a": {"b": 1}}
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def tomlmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("TOML_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libtomlmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_loads_type_error_like_tomllib():
    import tomllib

    with pytest.raises(TypeError):
        tomllib.loads(b"a = 1")
    with pytest.raises(TypeError):
        toml_mojo.loads(b"a = 1")


def test_load_binary_file_like_tomllib(tmp_path):
    import tomllib

    p = tmp_path / "doc.toml"
    p.write_bytes("[a]\n b = [1, 2.5]\n".encode("utf-8"))
    with open(p, "rb") as f:
        assert toml_mojo.load(f) == tomllib.loads("[a]\n b = [1, 2.5]\n")
