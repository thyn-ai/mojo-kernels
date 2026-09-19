"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics."""

from __future__ import annotations

import os

import networkx as nx
import pytest

import nx_mojo
from nx_mojo import _native
from nx_mojo._native import NativeUnavailable

G = nx.path_graph(5)


def test_backend_info_shape():
    info = nx_mojo.backend_info()
    assert info["abi_version_expected"] == _native.ABI_VERSION
    assert info["disabled_by_env"] == (os.environ.get("NX_MOJO_DISABLE_NATIVE") == "1")
    if info["disabled_by_env"]:
        assert info["native_available"] is False
    else:
        # The suite's native run requires a built kernel
        # (kernels/networkx-graph/build.sh).
        assert info["native_available"] is True
        assert info["abi_version_native"] == _native.ABI_VERSION
        assert info["native_source"]


def test_env_forces_fallback(monkeypatch):
    monkeypatch.setenv("NX_MOJO_DISABLE_NATIVE", "1")
    assert nx_mojo.native_available() is False
    # ... and the public API still computes correct results.
    bc = nx_mojo.betweenness_centrality(G)
    assert bc == nx.betweenness_centrality(G)


def test_broken_override_falls_back_to_candidates(monkeypatch, tmp_path):
    # A corrupt/unloadable override must not crash the call: the resolver
    # skips it and continues down the candidate list.
    bogus = tmp_path / "not-a-real-lib.dylib"
    bogus.write_text("definitely not a mach-o")
    monkeypatch.setenv("NX_MOJO_NATIVE_LIB", str(bogus))
    monkeypatch.delenv("NX_MOJO_DISABLE_NATIVE", raising=False)
    _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
    try:
        bc = nx_mojo.betweenness_centrality(G)
        assert bc == nx.betweenness_centrality(G)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_no_candidates_raises_native_unavailable(monkeypatch):
    monkeypatch.delenv("NX_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable):
            _native._load()
        # ... and the public API then falls back transparently.
        bc = nx_mojo.betweenness_centrality(G)
        assert bc == nx.betweenness_centrality(G)
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


def test_abi_mismatch_rejected(monkeypatch):
    class FakeLib:
        def nxgraphmojo_abi_version(self):
            return _native.ABI_VERSION + 1

    monkeypatch.delenv("NX_MOJO_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [("fake", "/fake/libnxgraphmojo.dylib")]
    )
    monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
    monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
    _native._LIB, _native._LIB_SOURCE = None, None
    try:
        with pytest.raises(NativeUnavailable, match="ABI"):
            _native._load()
    finally:
        _native._LIB, _native._LIB_SOURCE = None, None


# --------------------------------------------------------------------------
# official networkx backend interface (entry points resolve only when the
# package is installed; the wheel smoke test exercises the real dispatch)
# --------------------------------------------------------------------------


def test_backend_interface_unit():
    from nx_mojo.backend import BackendInterface, get_info

    G = nx.path_graph(5)
    # Identity conversions: the kernel consumes networkx graphs directly.
    assert BackendInterface.convert_from_nx(G, edge_attrs={}, name="x") is G
    bc = BackendInterface.betweenness_centrality(G)
    assert bc == nx.betweenness_centrality(G)
    assert BackendInterface.convert_to_nx(bc) is bc
    # can_run / should_run: in-scope calls run; out-of-scope give a reason.
    assert BackendInterface.can_run("betweenness_centrality", (G,), {}) is True
    assert BackendInterface.should_run("betweenness_centrality", (G,), {}) is True
    reason = BackendInterface.can_run("betweenness_centrality", (G,), {"k": 3})
    assert isinstance(reason, str) and reason
    reason = BackendInterface.can_run(
        "betweenness_centrality", (G,), {"endpoints": True}
    )
    assert isinstance(reason, str) and reason
    MG = nx.MultiGraph([(0, 1), (0, 1)])
    reason = BackendInterface.can_run("betweenness_centrality", (MG,), {})
    assert isinstance(reason, str) and reason
    reason = BackendInterface.can_run(
        "single_source_dijkstra", (G,), {"weight": lambda u, v, d: 1.0}
    )
    assert isinstance(reason, str) and reason
    # backend_info schema: "functions" must be a dict keyed by function name
    # (networkx's doc builder indexes it; a list breaks networkx at import).
    info = get_info()
    assert isinstance(info["functions"], dict)
    assert set(info["functions"]) == {
        "betweenness_centrality",
        "single_source_dijkstra",
    }
    assert info["short_summary"]


def test_backend_entry_point_dispatch_if_installed():
    # Entry points exist only in installed distributions (wheel smoke test
    # environment); in the repo dev suite they are absent.
    from importlib.metadata import entry_points

    eps = entry_points(group="networkx.backends")
    if "nx_mojo" not in {ep.name for ep in eps}:
        pytest.skip("nx-mojo not installed; entry points absent")
    G = nx.karate_club_graph()
    dispatched = nx.betweenness_centrality(G, backend="nx_mojo")
    assert dispatched == nx.betweenness_centrality(G)
