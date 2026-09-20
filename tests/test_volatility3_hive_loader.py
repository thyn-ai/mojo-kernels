"""Loader/backend behaviour: ABI handshake, fallback forcing, diagnostics,
and input validation. Does not require the volatility3 oracle."""

from __future__ import annotations

import os

import pytest

from test_volatility3_hive_fixtures import HiveBuilder

import vol_mojo
from vol_mojo import _native
from vol_mojo._native import NativeUnavailable


def _small_hive() -> bytes:
    b = HiveBuilder()
    leaf = b.add_cell(b.nk(b"Child"))
    idx = b.add_cell(b.index(b"lh", [(leaf, 0xAAAAAAAA)]))
    root = b.add_cell(b.nk(b"ROOT", subkey_count=1, subkey_list=idx))
    dat, _ = b.build_dat(root)
    return dat


def _small_pool() -> bytes:
    buf = bytearray(b"\x00" * 0x100)
    buf[0x42] = 0x41  # BlockSize
    buf[0x43] = 0x02  # PoolType (nonpaged, vista)
    buf[0x44:0x48] = b"Proc"
    return bytes(buf)


def _constraints():
    return [
        vol_mojo.PoolConstraint(
            b"Proc",
            size=(600, None),
            page_type=vol_mojo.PAGE_TYPE_NONPAGED | vol_mojo.PAGE_TYPE_FREE,
        )
    ]


def _small_walk():
    return vol_mojo.walk_hive(_small_hive())


def _small_scan():
    return vol_mojo.scan_pool_headers(_small_pool(), _constraints(), alignment=0x10)


class TestBackendDiagnostics:
    def test_backend_info_shape(self):
        info = vol_mojo.backend_info()
        assert info["abi_version_expected"] == _native.ABI_VERSION
        assert info["disabled_by_env"] == (
            os.environ.get("VOL_MOJO_DISABLE_NATIVE") == "1"
        )
        if info["disabled_by_env"]:
            assert info["native_available"] is False
        else:
            # The suite's native run requires a built kernel.
            assert info["native_available"] is True
            assert info["abi_version_native"] == _native.ABI_VERSION
            assert info["native_source"]

    def test_env_forces_fallback(self, monkeypatch):
        monkeypatch.setenv("VOL_MOJO_DISABLE_NATIVE", "1")
        assert [t[2] for t in _small_walk()] == ["ROOT", "Child"]
        assert _small_scan() == [(0x40, b"Proc")]
        assert vol_mojo.native_available() is False


class TestResolver:
    def test_broken_override_falls_back_to_candidates(self, monkeypatch, tmp_path):
        # A corrupt/unloadable override must not crash evaluation: the
        # resolver skips it and continues down the candidate list.
        bogus = tmp_path / "not-a-real-lib.dylib"
        bogus.write_text("definitely not a mach-o")
        monkeypatch.setenv("VOL_MOJO_NATIVE_LIB", str(bogus))
        monkeypatch.delenv("VOL_MOJO_DISABLE_NATIVE", raising=False)
        _native._LIB, _native._LIB_SOURCE = None, None  # reset module cache
        try:
            assert [t[2] for t in _small_walk()] == ["ROOT", "Child"]
            assert _small_scan() == [(0x40, b"Proc")]
        finally:
            _native._LIB, _native._LIB_SOURCE = None, None

    def test_no_candidates_raises_native_unavailable(self, monkeypatch):
        monkeypatch.delenv("VOL_MOJO_DISABLE_NATIVE", raising=False)
        monkeypatch.setattr(_native, "_candidate_paths", lambda: [])
        _native._LIB, _native._LIB_SOURCE = None, None
        try:
            with pytest.raises(NativeUnavailable):
                _native._load()
            # ... and the public API then falls back transparently.
            assert [t[2] for t in _small_walk()] == ["ROOT", "Child"]
            assert _small_scan() == [(0x40, b"Proc")]
        finally:
            _native._LIB, _native._LIB_SOURCE = None, None

    def test_abi_mismatch_rejected(self, monkeypatch):
        class FakeLib:
            def volatility3hivemojo_abi_version(self):
                return _native.ABI_VERSION + 1

        monkeypatch.delenv("VOL_MOJO_DISABLE_NATIVE", raising=False)
        monkeypatch.setattr(_native.ctypes, "CDLL", lambda path: FakeLib())
        monkeypatch.setattr(
            _native,
            "_candidate_paths",
            lambda: [("fake", "/fake/libvolatility3hivemojo.dylib")],
        )
        monkeypatch.setattr(_native.os.path, "exists", lambda p: True)
        monkeypatch.setattr(_native, "_bind_abi", lambda lib: None)
        _native._LIB, _native._LIB_SOURCE = None, None
        try:
            with pytest.raises(NativeUnavailable, match="ABI"):
                _native._load()
        finally:
            _native._LIB, _native._LIB_SOURCE = None, None


class TestInputValidation:
    def test_walk_rejects_non_bytes(self):
        with pytest.raises(vol_mojo.HiveError):
            vol_mojo.walk_hive("not bytes")

    def test_walk_rejects_short_image(self):
        with pytest.raises(vol_mojo.HiveError):
            vol_mojo.walk_hive(b"\x00" * 100)

    def test_walk_rejects_bad_root(self):
        with pytest.raises(vol_mojo.HiveError):
            vol_mojo.walk_hive(_small_hive(), root_cell_offset=-1)
        with pytest.raises(vol_mojo.HiveError):
            vol_mojo.walk_hive(_small_hive(), root_cell_offset=1 << 40)
        with pytest.raises(vol_mojo.HiveError):
            vol_mojo.walk_hive(_small_hive(), root_cell_offset="x")

    def test_scan_rejects_non_bytes(self):
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(12345, _constraints())

    def test_scan_rejects_bad_alignment(self):
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), _constraints(), alignment=0)
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), _constraints(), alignment=-8)
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), _constraints(), alignment=True)

    def test_scan_rejects_bad_layout(self):
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), _constraints(), layout="x128")

    def test_scan_rejects_duplicate_tags(self):
        cs = _constraints() + _constraints()
        with pytest.raises(vol_mojo.PoolScanError, match="more than one constraint"):
            vol_mojo.scan_pool_headers(_small_pool(), cs)

    def test_scan_rejects_empty_constraints(self):
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), [])

    def test_scan_rejects_bad_constraint_shape(self):
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(_small_pool(), [object()])
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(
                _small_pool(), [vol_mojo.PoolConstraint(b"")]
            )
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(
                _small_pool(), [vol_mojo.PoolConstraint(b"Ab", size=(-1, None))]
            )
        with pytest.raises(vol_mojo.PoolScanError):
            vol_mojo.scan_pool_headers(
                _small_pool(), [vol_mojo.PoolConstraint(b"Ab", page_type="paged")]
            )


class TestDeterminism:
    def test_walk_is_deterministic(self):
        assert _small_walk() == _small_walk()

    def test_scan_is_deterministic(self):
        assert _small_scan() == _small_scan()
