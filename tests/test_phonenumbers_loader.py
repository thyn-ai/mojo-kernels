"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import phonenumbers_mojo
from phonenumbers_mojo import _native
from phonenumbers_mojo._native import NativeUnavailable

TEXT = "+1 212-555-1234"


def test_backend_info_shape():
    info = phonenumbers_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (
        os.environ.get("PHONENUMBERS_MOJO_DISABLE_NATIVE") == "1"
    )
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (scripts/test_all_phonenumbers.sh builds it first).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("PHONENUMBERS_MOJO_DISABLE_NATIVE", "1")
    assert phonenumbers_mojo.backend() == "fallback"
    assert phonenumbers_mojo.native_available() is False
    # ... and parsing still works (pure-Python path)
    n = phonenumbers_mojo.parse(TEXT)
    assert n.country_code == 1
    assert n.national_number == 2125551234


def _reset_caches(monkeypatch):
    monkeypatch.setattr(_native, "_LIB", None)
    monkeypatch.setattr(_native, "_LIB_SOURCE", None)
    monkeypatch.setattr(phonenumbers_mojo.core, "_STORE", None)
    monkeypatch.setattr(phonenumbers_mojo.core, "_BACKEND", None)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash parsing: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("PHONENUMBERS_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("PHONENUMBERS_MOJO_DISABLE_NATIVE", raising=False)
    _reset_caches(monkeypatch)
    try:
        assert phonenumbers_mojo.parse(TEXT).national_number == 2125551234
    finally:
        _reset_caches(monkeypatch)


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("PHONENUMBERS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _reset_caches(monkeypatch)
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert phonenumbers_mojo.parse(TEXT).national_number == 2125551234
        assert phonenumbers_mojo.backend() == "fallback"
    finally:
        _reset_caches(monkeypatch)


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def phonenumbersmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("PHONENUMBERS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libphonenumbersmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    monkeypatch.setattr(_native, "_LIB", None)
    monkeypatch.setattr(_native, "_LIB_SOURCE", None)
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        monkeypatch.setattr(_native, "_LIB", None)
        monkeypatch.setattr(_native, "_LIB_SOURCE", None)


def test_native_store_rejects_garbage_blob():
    # The kernel validates the blob header; a garbage blob must not crash.
    if os.environ.get("PHONENUMBERS_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native kernel disabled")
    with pytest.raises(NativeUnavailable, match="rejected"):
        _native.NativeStore(b"not a metadata blob at all")


def test_native_store_rejects_truncated_blob():
    if os.environ.get("PHONENUMBERS_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native kernel disabled")
    from phonenumbers_mojo._data import load_blob

    blob = load_blob()
    with pytest.raises(NativeUnavailable, match="rejected"):
        _native.NativeStore(blob[: len(blob) // 2])


def test_phone_number_repr_matches_oracle_shape():
    n = phonenumbers_mojo.parse(TEXT)
    assert repr(n) == (
        "PhoneNumber(country_code=1, national_number=2125551234, extension=None, "
        "italian_leading_zero=None, number_of_leading_zeros=None, "
        "country_code_source=0, preferred_domestic_carrier_code=None)"
    )
