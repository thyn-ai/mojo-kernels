"""Loader/backend behaviour for jinja2_mojo: ABI handshake, fallback forcing,
diagnostics, and the public API surface."""

from __future__ import annotations

import os

import pytest

import jinja2_mojo
from jinja2_mojo import _native
from jinja2_mojo._native import NativeUnavailable

SRC = "Hello {{ name }}!"


def test_backend_info_shape():
    info = jinja2_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("JINJA2_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("JINJA2_MOJO_DISABLE_NATIVE", "1")
    assert jinja2_mojo.native_available() is False
    # ... and the public API still produces the oracle-identical result.
    assert jinja2_mojo.compile_template(SRC).render({"name": "W"}) == "Hello W!"


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it
    # and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("JINJA2_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("JINJA2_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        assert jinja2_mojo.compile_template(SRC).render({"name": "W"}) == "Hello W!"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("JINJA2_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert jinja2_mojo.compile_template(SRC).render({"name": "W"}) == "Hello W!"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def jinja2mojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("JINJA2_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libjinja2mojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_compiled_template_surface():
    t = jinja2_mojo.compile_template(SRC)
    assert t.render({"name": "A"}) == "Hello A!"
    assert t.render(name="B") == "Hello B!"
    assert t({"name": "C"}) == "Hello C!"
    assert t(name="D") == "Hello D!"
    assert "".join(t.generate({"name": "E"})) == "Hello E!"
    assert t.source == SRC
    assert t.backend in ("native", "stock")
    import jinja2

    assert isinstance(t.template, jinja2.Template)
    assert isinstance(t.environment, jinja2.Environment)


def test_batch_empty_and_single():
    assert jinja2_mojo.compile_batch([]) == []
    one = jinja2_mojo.compile_batch([SRC])
    assert len(one) == 1
    assert one[0].render({"name": "Q"}) == "Hello Q!"


def test_scan_records_declined_gracefully():
    # Templates outside the native subset must return None from scan (never
    # raise, never guess), letting the wrapper run the stock lexer.
    from jinja2_mojo import _tokens

    if not jinja2_mojo.native_available():
        pytest.skip("native kernel unavailable")
    assert _tokens.build_tokens("{% raw %}{{ x }}{% endraw %}") is None
    assert _tokens.build_tokens("a {# unterminated") is None
    assert _tokens.build_tokens("{{ 'unterminated }}") is None
    assert _tokens.build_tokens("{{ (1 }}") is None
    plain = _tokens.build_tokens("plain text, no tags")
    assert plain is not None and len(plain) == 1 and plain[0].type == "data"
