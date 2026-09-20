"""Differential tests: cclib_mojo must match the PyQuante reference path.

Run twice by `scripts/test_all_cclib.sh`: once against the native Mojo
kernel and once with CCLIB_MOJO_DISABLE_NATIVE=1 (forced NumPy fallback).
Both backends must agree with the reference within 1e-10 relative
(in practice the agreement is at the 1e-13..1e-16 level).

Oracles:

- ``pyquante1_oracle`` — a Python-3 transcription of the PyQuante 1.6.5
  pure-Python amplitude path that cclib's ``pyamp`` hot loop executes
  (see that module's header for provenance). Always present.
- ``pyquante2`` — the pip-installable PyQuante rewrite (an independent
  implementation by the original PyQuante author), used directly and
  through cclib's own ``Volume.wavefunction`` / ``electrondensity``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import gaussgrid_fixtures as fx
import pyquante1_oracle as oracle

import cclib_mojo
from cclib_mojo import density_on_grid, wavefunction_on_grid

RTOL = 1e-10  # documented tolerance (task contract)
ATOL = 1e-12  # absolute floor for far-field points whose values are ~0


def _expected_backend() -> str:
    # scripts/test_all_cclib.sh runs the suite once per backend.
    return "fallback" if os.environ.get("CCLIB_MOJO_DISABLE_NATIVE") == "1" else "native"


def _active_backend() -> str:
    return "native" if cclib_mojo.native_available() else "fallback"


def assert_grid_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def pyquante2_wavefunction(gbasis, atomcoords, mocoeffs, origin, step, shape):
    """Independent oracle: pyquante2 cgbf objects evaluated per grid point."""
    from pyquante2 import cgbf

    bfs = []
    for i, atom_shells in enumerate(gbasis):
        center = tuple(float(v) * oracle.ANG2BOHR for v in atomcoords[i])
        for sym, prims in atom_shells:
            for power in oracle.SYM2POWERLIST[sym]:
                exps = [a for a, _ in prims]
                coefs = [c for _, c in prims]
                bfs.append(cgbf(center, powers=power, exps=exps, coefs=coefs))
    ax, ay, az = oracle.grid_axes(origin, step, shape)
    data = np.zeros(int(np.prod(shape)))
    for bs, bf in enumerate(bfs):
        if abs(mocoeffs[bs]) > 0.0:
            for p, (x, y, z) in enumerate(
                (xp, yp, zp) for xp in ax for yp in ay for zp in az
            ):
                data[p] += bf(x, y, z) * mocoeffs[bs]
    return data.reshape(shape)


# ---------------------------------------------------------------------------
# PyQuante1-oracle differential tests (every shell type, both backends)
# ---------------------------------------------------------------------------


def test_h2o_sto3g_wavefunction_pyquante1():
    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=7, n_mo=1, n_bf=7)[0]
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    actual = wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.wavefunction(gbasis, atomcoords, coeff, origin, step, shape)
    assert _active_backend() == _expected_backend()
    assert_grid_close(actual, expected)


def test_h2o_sto3g_wavefunction_pyquante2():
    pytest.importorskip("pyquante2", reason="pyquante2 oracle not installed")
    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=7, n_mo=1, n_bf=7)[0]
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    actual = wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = pyquante2_wavefunction(gbasis, atomcoords, coeff, origin, step, shape)
    assert_grid_close(actual, expected)


def test_h2o_sto3g_density_multi_mo():
    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=11, n_mo=3, n_bf=7)
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    actual = density_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.electrondensity_spin(
        gbasis, atomcoords, coeff, origin, step, shape
    )
    assert_grid_close(actual, expected)


def test_d_shell_on_oxygen():
    gbasis, atomcoords = fx.h2o_sto3g_d()  # 13 basis functions incl. 6 Cartesian d
    coeff = fx.seeded_coeffs(seed=13, n_mo=2, n_bf=13)
    origin, step, shape = (-1.8, -2.2, -1.5), (0.9, 1.1, 0.75), (4, 5, 6)
    actual = density_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.electrondensity_spin(
        gbasis, atomcoords, coeff, origin, step, shape
    )
    assert_grid_close(actual, expected)


def test_f_shell_on_carbon():
    gbasis, atomcoords = fx.carbon_sto3g_df()  # s + p + d + f on one atom
    coeff = fx.seeded_coeffs(seed=17, n_mo=2, n_bf=21)
    origin, step, shape = (-1.4, -1.2, -1.6), (0.6, 0.55, 0.65), (6, 5, 7)
    actual = density_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.electrondensity_spin(
        gbasis, atomcoords, coeff, origin, step, shape
    )
    assert_grid_close(actual, expected)


def test_mo_index_matches_wavefunction_and_density_of_square():
    gbasis, atomcoords = fx.h2o_sto3g_d()
    coeff = fx.seeded_coeffs(seed=19, n_mo=3, n_bf=13)
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    for k in range(3):
        via_index = density_on_grid(
            gbasis, atomcoords, coeff, origin, step, shape, mo_index=k
        )
        direct = wavefunction_on_grid(
            gbasis, atomcoords, coeff[k], origin, step, shape
        )
        assert_grid_close(via_index, direct)
        # Density of the single MO is its square (cclib electrondensity_spin).
        single = density_on_grid(gbasis, atomcoords, coeff[k], origin, step, shape)
        assert_grid_close(single, direct * direct)


def test_noncubic_anisotropic_grid():
    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=23, n_mo=2, n_bf=7)
    origin, step, shape = (-1.5, -2.3, -0.7), (0.9, 0.7, 1.1), (4, 6, 5)
    actual = density_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.electrondensity_spin(
        gbasis, atomcoords, coeff, origin, step, shape
    )
    assert_grid_close(actual, expected)


def test_grid_offset_from_origin_and_atoms():
    # Grid far from the molecular frame: tiny (but nonzero) far-field values.
    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=29, n_mo=1, n_bf=7)[0]
    origin, step, shape = (3.1, -4.4, 2.7), (0.8, 0.9, 0.7), (4, 4, 4)
    actual = wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.wavefunction(gbasis, atomcoords, coeff, origin, step, shape)
    assert_grid_close(actual, expected)


def test_grid_point_ordering():
    # [i, j, k] must be the point (ax[i], ay[j], az[k]) — catches transposes.
    gbasis, atomcoords = fx.h2o_sto3g_d()
    coeff = fx.seeded_coeffs(seed=31, n_mo=1, n_bf=13)[0]
    origin, step, shape = (-1.5, -2.3, -0.7), (0.9, 0.7, 1.1), (4, 6, 5)
    grid = wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    bfs = oracle.getbfs(gbasis, atomcoords)
    ax, ay, az = oracle.grid_axes(origin, step, shape)
    for i, j, k in [(0, 0, 0), (3, 5, 4), (1, 4, 2), (2, 0, 3)]:
        point_val = sum(
            coeff[bs] * bfs[bs].amp(ax[i], ay[j], az[k]) for bs in range(len(bfs))
        )
        assert grid[i, j, k] == pytest.approx(point_val, rel=RTOL, abs=ATOL)


def test_zero_coefficients_skipped_like_cclib():
    gbasis, atomcoords = fx.h2o_sto3g_d()
    coeff = fx.seeded_coeffs(seed=37, n_mo=1, n_bf=13)[0]
    coeff[2] = 0.0
    coeff[7] = 0.0
    coeff[12] = -0.0
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    actual = wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.wavefunction(gbasis, atomcoords, coeff, origin, step, shape)
    assert_grid_close(actual, expected)


def test_benzene_6_31g_star_spot_grid():
    # 102 basis functions incl. d shells across 12 atoms, modest grid.
    from cclib_mojo._basis import flatten_gbasis

    gbasis, atomcoords = fx.benzene_6_31g_star()
    n_bf = flatten_gbasis(gbasis, atomcoords).n_bf
    assert n_bf == 102
    coeff = fx.seeded_coeffs(seed=41, n_mo=2, n_bf=n_bf)
    origin, step, shape = (-3.0, -3.0, -2.0), (1.0, 1.0, 1.0), (7, 6, 5)
    actual = density_on_grid(gbasis, atomcoords, coeff, origin, step, shape)
    expected = oracle.electrondensity_spin(
        gbasis, atomcoords, coeff, origin, step, shape
    )
    assert_grid_close(actual, expected)


# ---------------------------------------------------------------------------
# cclib integration: the actual cclib Volume functions (pyquante2 backend)
# ---------------------------------------------------------------------------


def _ccdata(gbasis, atomcoords):
    from types import SimpleNamespace

    return SimpleNamespace(gbasis=gbasis, atomcoords=atomcoords[None, :, :])


def test_cclib_wavefunction_equivalence():
    cclib_volume = pytest.importorskip(
        "cclib.method.volume", reason="cclib not installed"
    )
    # cclib's wavefunction()/electrondensity*() evaluate through its pyquante2
    # backend and raise ImportError without it; pyquante2 is sdist-only and
    # does not build everywhere (MSVC rejects its crys.h).
    pytest.importorskip("pyquante2", reason="cclib's volume functions need pyquante2")
    from cclib_mojo import cclib_integration

    gbasis, atomcoords = fx.h2o_sto3g_d()
    coeff = fx.seeded_coeffs(seed=43, n_mo=1, n_bf=13)[0]
    ccdata = _ccdata(gbasis, atomcoords)
    vol = cclib_volume.Volume(origin=(-2.0, -2.0, -2.0), topcorner=(2.0, 2.0, 2.0),
                              spacing=(1.0, 1.0, 1.0))
    ref = cclib_volume.wavefunction(ccdata, vol, coeff)
    ours = cclib_integration.wavefunction(ccdata, vol, coeff)
    assert _active_backend() == _expected_backend()
    assert_grid_close(ours.data, ref.data)


def test_cclib_electrondensity_equivalence():
    cclib_volume = pytest.importorskip(
        "cclib.method.volume", reason="cclib not installed"
    )
    pytest.importorskip("pyquante2", reason="cclib's volume functions need pyquante2")
    from cclib_mojo import cclib_integration

    gbasis, atomcoords = fx.h2o_sto3g()
    coeff = fx.seeded_coeffs(seed=47, n_mo=3, n_bf=7)
    ccdata = _ccdata(gbasis, atomcoords)
    vol = cclib_volume.Volume(origin=(-2.0, -2.0, -2.0), topcorner=(2.0, 2.0, 2.0),
                              spacing=(1.0, 1.0, 1.0))
    ref = cclib_volume.electrondensity_spin(ccdata, vol, [coeff])
    ours = cclib_integration.electrondensity_spin(ccdata, vol, [coeff])
    assert_grid_close(ours.data, ref.data)
    # Restricted total density doubles the spin density (cclib convention).
    ref_total = cclib_volume.electrondensity(ccdata, vol, [coeff])
    ours_total = cclib_integration.electrondensity(ccdata, vol, [coeff])
    assert_grid_close(ours_total.data, ref_total.data)


# ---------------------------------------------------------------------------
# Validation / structured errors
# ---------------------------------------------------------------------------


def test_validation_errors():
    gbasis, atomcoords = fx.h2o_sto3g()
    origin, step, shape = (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), (5, 5, 5)
    coeff = np.ones(7)

    with pytest.raises(cclib_mojo.BasisError, match="unsupported shell"):
        wavefunction_on_grid(
            [[("G", [(1.0, 1.0)])]] + gbasis[1:], atomcoords, coeff, origin, step, shape
        )
    with pytest.raises(cclib_mojo.BasisError, match="empty primitive"):
        wavefunction_on_grid(
            [[("S", [])]] + gbasis[1:], atomcoords, coeff, origin, step, shape
        )
    with pytest.raises(cclib_mojo.BasisError, match="exponent"):
        wavefunction_on_grid(
            [[("S", [(-1.0, 1.0)])]] + gbasis[1:], atomcoords, coeff, origin, step, shape
        )
    with pytest.raises(cclib_mojo.BasisError, match="n_atoms"):
        wavefunction_on_grid(gbasis, atomcoords[:2], coeff, origin, step, shape)
    with pytest.raises(cclib_mojo.GridError, match="columns"):
        wavefunction_on_grid(gbasis, atomcoords, np.ones(6), origin, step, shape)
    with pytest.raises(cclib_mojo.GridError, match="positive integers"):
        wavefunction_on_grid(gbasis, atomcoords, coeff, origin, step, (5, 0, 5))
    with pytest.raises(cclib_mojo.GridError, match="positive"):
        wavefunction_on_grid(gbasis, atomcoords, coeff, origin, (1.0, -1.0, 1.0), shape)
    with pytest.raises(cclib_mojo.GridError, match="mo_index"):
        density_on_grid(
            gbasis, atomcoords, coeff[None, :], origin, step, shape, mo_index=1
        )
    with pytest.raises(cclib_mojo.GridError, match="1-D"):
        wavefunction_on_grid(
            gbasis, atomcoords, coeff[None, :], origin, step, shape
        )
    with pytest.raises(cclib_mojo.BasisError, match="finite"):
        bad = atomcoords.copy()
        bad[0, 0] = np.nan
        wavefunction_on_grid(gbasis, bad, coeff, origin, step, shape)
