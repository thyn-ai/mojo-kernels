"""Loader/backend behaviour for sacrebleu-mojo: ABI handshake, fallback
forcing, diagnostics."""

from __future__ import annotations

import os

import pytest

import sacrebleu_mojo
from sacrebleu_mojo import _native
from sacrebleu_mojo._native import NativeUnavailable

HYPS = ["the quick brown fox", "hello world"]
REFS = [["the fast brown fox", "hello there world"]]


def test_backend_info_shape():
    info = sacrebleu_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("SACREBLEU_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel.
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("SACREBLEU_MOJO_DISABLE_NATIVE", "1")
    bleu = sacrebleu_mojo.corpus_bleu(HYPS, REFS)
    assert bleu.backend == "fallback"
    chrf = sacrebleu_mojo.corpus_chrf(HYPS, REFS)
    assert chrf.backend == "fallback"
    assert sacrebleu_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash: the resolver skips it and
    # continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("SACREBLEU_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("SACREBLEU_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        bleu = sacrebleu_mojo.corpus_bleu(HYPS, REFS)
        assert bleu.score > 0
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("SACREBLEU_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert sacrebleu_mojo.corpus_bleu(HYPS, REFS).backend == "fallback"
        assert sacrebleu_mojo.corpus_chrf(HYPS, REFS).backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def sacremojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("SACREBLEU_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libsacremojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None
