"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import ipaddress_mojo
from ipaddress_mojo import _native
from ipaddress_mojo._native import NativeUnavailable


def test_backend_info_shape():
    info = ipaddress_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("IPADDRESS_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (the test script builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("IPADDRESS_MOJO_DISABLE_NATIVE", "1")
    assert ipaddress_mojo.native_available() is False
    # The public API still answers correctly via the fallback.
    assert int(ipaddress_mojo.ip_address("1.2.3.4")) == 0x01020304
    assert list(ipaddress_mojo.contains_many(
        [ipaddress_mojo.ip_network("10.0.0.0/8")],
        [ipaddress_mojo.ip_address("10.1.2.3")],
    )) == [True]


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash anything: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("IPADDRESS_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("IPADDRESS_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert int(ipaddress_mojo.parse_many(["1.2.3.4"])[0]) == 0x01020304
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("IPADDRESS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert int(ipaddress_mojo.parse_many(["1.2.3.4"])[0]) == 0x01020304
        assert [str(x) for x in ipaddress_mojo.collapse_batch(
            [ipaddress_mojo.ip_network("10.0.0.0/24"), ipaddress_mojo.ip_network("10.0.1.0/24")]
        )] == ["10.0.0.0/23"]
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def ipaddressmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("IPADDRESS_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libipaddressmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
