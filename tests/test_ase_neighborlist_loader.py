"""Loader tests: ABI handshake, resolution order, forced fallback, quickstart.

These tests do not need the ASE oracle. The suite is run twice (native and
ASE_MOJO_DISABLE_NATIVE=1) by scripts/test_all_ase_neighborlist.sh.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import ase_mojo
from ase_mojo import _native
from ase_mojo._native import NativeUnavailable


def _kernel_lib_path() -> str:
    here = os.path.dirname(__file__)
    name = _native._lib_basename()
    return os.path.abspath(
        os.path.join(here, "..", "kernels", "ase_neighborlist", "build", name)
    )


@pytest.fixture
def _reset_loader(monkeypatch):
    """Reset the loader's cached CDLL around each test."""
    monkeypatch.setattr(_native, "_LIB", None)
    monkeypatch.setattr(_native, "_LIB_SOURCE", None)
    monkeypatch.setattr(_native, "_LOAD_ERROR", None)
    yield


def test_abi_handshake(_reset_loader, monkeypatch):
    if not os.path.exists(_kernel_lib_path()):
        pytest.skip("native kernel not built on this machine")
    monkeypatch.delenv("ASE_MOJO_DISABLE_NATIVE", raising=False)
    lib = _native._load()
    assert int(lib.aseneighborlistmojo_abi_version()) == _native.ABI_VERSION


def test_env_override_resolution(_reset_loader, monkeypatch):
    if not os.path.exists(_kernel_lib_path()):
        pytest.skip("native kernel not built on this machine")
    monkeypatch.setenv("ASE_MOJO_NATIVE_LIB", _kernel_lib_path())
    monkeypatch.delenv("ASE_MOJO_DISABLE_NATIVE", raising=False)
    assert ase_mojo.native_available()
    info = ase_mojo.backend_info()
    assert info["native_available"]
    assert "env ASE_MOJO_NATIVE_LIB" in info["native_source"]
    assert info["abi_version_native"] == _native.ABI_VERSION


def test_env_override_garbage_falls_back(_reset_loader, monkeypatch, tmp_path):
    monkeypatch.delenv("ASE_MOJO_DISABLE_NATIVE", raising=False)
    garbage = tmp_path / "not_a_kernel.dylib"
    garbage.write_bytes(b"not a shared library")
    monkeypatch.setenv("ASE_MOJO_NATIVE_LIB", str(garbage))
    if os.path.exists(_kernel_lib_path()):
        # Falls through to the repo-dev build and loads fine.
        assert ase_mojo.native_available()
    else:
        assert not ase_mojo.native_available()


def test_disable_native_forces_fallback(_reset_loader, monkeypatch):
    monkeypatch.setenv("ASE_MOJO_DISABLE_NATIVE", "1")
    assert not ase_mojo.native_available()
    info = ase_mojo.backend_info()
    assert info["disabled_by_env"]
    assert not info["native_available"]
    assert "disabled" in info["error"]
    # The public API still works (fallback backend).
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])
    i, j, S = ase_mojo.primitive_neighbor_list("ijS", [True] * 3, cell, pos, 1.5)
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1), (1, 0)}


def test_backend_info_shape():
    info = ase_mojo.backend_info()
    for key in [
        "native_available",
        "native_source",
        "abi_version_expected",
        "abi_version_native",
        "disabled_by_env",
        "platform",
        "error",
    ]:
        assert key in info
    assert info["abi_version_expected"] == _native.ABI_VERSION


def test_fallback_and_native_agree_tiny_case(_reset_loader, monkeypatch):
    """Same input through both backends (in one process) gives identical sets."""
    rng = np.random.default_rng(123)
    cell = np.diag(rng.uniform(3.0, 5.0, 3))
    pos = rng.random((24, 3)) @ cell
    cutoff = 1.3

    monkeypatch.delenv("ASE_MOJO_DISABLE_NATIVE", raising=False)
    native_ok = ase_mojo.native_available()
    if native_ok:
        ni, nj, nS = ase_mojo.primitive_neighbor_list(
            "ijS", [True] * 3, cell, pos, cutoff
        )
    monkeypatch.setenv("ASE_MOJO_DISABLE_NATIVE", "1")
    fi, fj, fS = ase_mojo.primitive_neighbor_list("ijS", [True] * 3, cell, pos, cutoff)
    if native_ok:
        assert set(zip(ni.tolist(), nj.tolist(), map(tuple, nS.tolist()))) == set(
            zip(fi.tolist(), fj.tolist(), map(tuple, fS.tolist()))
        )


def test_quickstart_end_to_end():
    """The README quickstart, verified against an O(n^2) brute force."""
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.8, 3.0]])
    rng = np.random.default_rng(42)
    pos = rng.random((16, 3)) @ cell
    cutoff = 1.6
    i, j, S, d = ase_mojo.primitive_neighbor_list(
        "ijSd", [True, True, True], cell, pos, cutoff
    )
    # brute force
    want = set()
    for a in range(16):
        for b in range(16):
            if a == b:
                continue
            for s0 in (-1, 0, 1):
                for s1 in (-1, 0, 1):
                    for s2 in (-1, 0, 1):
                        s = (s0, s1, s2)
                        D = pos[b] - pos[a] + np.array(s) @ cell
                        if np.linalg.norm(D) < cutoff:
                            want.add((a, b, s))
    got = set(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))
    assert got == want
    assert d.dtype == np.float64
    assert np.all(d < cutoff)
