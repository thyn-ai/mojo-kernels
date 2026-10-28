"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import bm25_mojo
import numpy as np
import pytest
from bm25_mojo import _native
from bm25_mojo._native import NativeUnavailable

CORPUS = [["a", "b"], ["b", "c"], ["a", "b", "c"]]


def test_backend_info_shape():
    info = bm25_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("BM25_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel (pixi run test builds it).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("BM25_MOJO_DISABLE_NATIVE", "1")
    bm25 = bm25_mojo.BM25Okapi(CORPUS)
    assert bm25.backend == "fallback"
    assert bm25_mojo.native_available() is False


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash construction: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("BM25_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("BM25_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        bm25 = bm25_mojo.BM25Okapi(CORPUS)
        scores = bm25.get_scores(["a"])
        assert scores.shape == (3,)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("BM25_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        assert bm25_mojo.BM25Okapi(CORPUS).backend == "fallback"
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def bm25mojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("BM25_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libbm25mojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_v3_loads_and_uses_python_build_path(monkeypatch):
    """An ABI v3 library lacks the v4 corpus builders; it still loads for
    scoring but the constructor must fall back to the Python build path."""
    class FakeLib:
        def bm25mojo_abi_version(self):
            return 3

    monkeypatch.delenv("BM25_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake-v3", "/fake/libbm25mojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    # Bind only the ABI v3 scoring symbols; no v4 build symbols.
    monkeypatch.setattr(_native, "_bind_abi", lambda lib, abi: None)
    _native._LIB, _native._LIB_SOURCE, _native._LIB_ABI = None, None, None
    try:
        bm25 = bm25_mojo.BM25Okapi(CORPUS)
        assert bm25.backend == "fallback"
        scores = bm25.get_scores(["a"])
        assert scores.shape == (3,)
        assert scores.dtype == "float64"
    finally:
        _native._LIB, _native._LIB_SOURCE, _native._LIB_ABI = None, None, None


def test_non_str_tokens_fall_back_to_python_path():
    """The native fast paths require str tokens; non-str corpora must still
    construct and score correctly (the index streams are built in Python and
    scoring may still run natively)."""
    from rank_bm25 import BM25Okapi as RefBM25Okapi

    corpus = [["a", 1], [1, "c"], ["a", "c", "c"]]
    bm25 = bm25_mojo.BM25Okapi(corpus)
    ref = RefBM25Okapi(corpus)
    assert bm25.corpus_size == ref.corpus_size
    assert bm25.avgdl == ref.avgdl
    assert bm25.doc_freqs == ref.doc_freqs
    assert bm25.idf == ref.idf
    scores = bm25.get_scores(["a"])
    assert scores.shape == (3,)
    assert scores.dtype == "float64"
    np.testing.assert_allclose(scores, ref.get_scores(["a"]), rtol=0, atol=1e-8)


def test_native_build_result_struct_fields_are_sane():
    """The Mojo BuildResult struct must map to the ctypes _BuildResultC layout.

    The explicit ``_pad`` field keeps the C-side and Python-side layouts
    bit-identical; if the padding drifted, ``n_docs``/``avgdl`` would read
    garbage and these assertions would fail.
    """
    if os.environ.get("BM25_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native struct layout test requires native backend")
    corpus = [["alpha", "beta"], ["beta", "gamma", "delta"], ["alpha"]]
    bm25 = bm25_mojo.BM25Okapi(corpus)
    if bm25.backend != "native":
        pytest.skip("native backend not available")
    assert bm25.corpus_size == len(corpus)
    assert bm25.avgdl > 0
    assert set(bm25.idf.keys()) == {"alpha", "beta", "gamma", "delta"}
