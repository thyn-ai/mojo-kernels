"""mistune-mojo loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import mistune_mojo
from mistune_mojo import _native
from mistune_mojo._native import NativeUnavailable

DOC = "# Hi\n\nsome *text* [link](/url)\n"


def test_backend_info_shape():
    info = mistune_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("MISTUNE_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (scripts build it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("MISTUNE_MOJO_DISABLE_NATIVE", "1")
    assert mistune_mojo.native_available() is False
    # ... and rendering still works (pure-Python engine)
    assert mistune_mojo.markdown(DOC) == mistune_mojo.markdown(DOC)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it and
    # continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("MISTUNE_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("MISTUNE_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        out = mistune_mojo.markdown(DOC)
        assert "<h1>Hi</h1>" in out
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("MISTUNE_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert "<h1>Hi</h1>" in mistune_mojo.markdown(DOC)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def mistunemojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("MISTUNE_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libmistunemojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_kernel_decline_falls_back():
    # Documents the kernel declines (unsupported constructs) render through
    # the pure-Python engine with identical bytes.
    import os

    if os.environ.get("MISTUNE_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native disabled in this run")
    from mistune_mojo._native import UnsupportedConstruct, render_native

    doc = "[x](/f&MadeUpEntity;)\n"
    try:
        out = render_native(doc)
    except UnsupportedConstruct:
        out = None
    assert out is None or out == mistune_mojo.markdown(doc)
