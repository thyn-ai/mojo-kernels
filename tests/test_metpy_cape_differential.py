"""Differential tests: metpy_mojo must match the MetPy oracle.

The oracle is the published PyPI package (`pip install metpy`; the suite
was developed against metpy 1.7.1). Run twice by
`scripts/test_all_metpy_cape.sh`: once against the native Mojo kernel and
once with METPY_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback).

Parity contract (measured on this suite, stated honestly in the package
README): RK45 vs the oracle's LSODA is not bit-exact by construction, so
CAPE agrees within 2% relative or 1.0 J/kg absolute (whichever is
larger), and CIN within 1% relative or 0.5 J/kg. Measured agreement is
far tighter for meteorologically substantial CAPE (<= 0.9% for
CAPE >= 100 J/kg, <= 1.6% for 50-100 J/kg); the absolute tolerances
cover marginal soundings where a sub-hPa LFC placement difference flips
a few J/kg either way.
"""

from __future__ import annotations

import numpy as np
import pytest
import metpy.calc as mpcalc
from metpy.units import units

import metpy_mojo
from metpy_mojo import _native, _reference

# Documented parity tolerances (see module docstring).
CAPE_RTOL, CAPE_ATOL = 2e-2, 1.0
CIN_RTOL, CIN_ATOL = 1e-2, 0.5
# Native vs fallback implement the same algorithm; agreement is at the
# 1e-7 level (float libm differences only).
BACKEND_RTOL, BACKEND_ATOL = 1e-6, 1e-3
LCL_ATOL_HPA = 0.5


def _suite() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Deterministic standard-sounding suite (name -> (p, T, Td), hPa/K)."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.RandomState(20260919)

    p = np.linspace(1000.0, 100.0, 91)
    z = -np.log(p / 1000.0) * 7.5
    out["classic-91"] = (p, np.maximum(300.0 - 6.5 * z, 222.0),
                         np.maximum(294.0 - 5.0 * z, 200.0))
    out["doc-6"] = (np.array([959., 779.2, 751.3, 724.3, 700., 269.]),
                    np.array([22.2, 14.6, 12., 9.4, 7., -49.]) + 273.15,
                    np.array([19., -11.2, -10.8, -10.4, -10., -53.]) + 273.15)
    Tt = np.maximum(302.0 - 6.0 * z, 218.0)
    out["tropical"] = (p, Tt, np.minimum(np.maximum(298.0 - 4.5 * z, 210.0), Tt - 0.5))
    Ti = 300.0 - 6.5 * z
    inv = (p < 850.0) & (p > 750.0)
    Ti[inv] += 4.0 * np.sin(np.pi * (np.log(p[inv] / 850.0) / np.log(750.0 / 850.0)))
    Ti = np.maximum(Ti, 222.0)
    out["inversion"] = (p, Ti, np.minimum(np.maximum(292.0 - 5.5 * z, 200.0), Ti - 0.5))
    out["arctic-stable"] = (p, np.maximum(268.0 - 4.0 * z, 210.0),
                            np.maximum(258.0 - 5.0 * z, 180.0))
    Ts = np.maximum(299.0 - 6.5 * z, 222.0)
    Tds = np.maximum(293.0 - 5.0 * z, 200.0)
    Tds[0] = Ts[0]
    out["saturated-sfc"] = (p, Ts, Tds)
    T7 = np.maximum(298.0 - 6.8 * z, 220.0)
    Td7 = np.maximum(283.0 - 5.0 * z, 195.0)
    i850 = np.argmin(np.abs(p - 850.0))
    Td7[:i850 + 1] = np.minimum(T7[:i850 + 1] - 1.0, 295.0 - 2.0 * z[:i850 + 1])
    out["moist-aloft"] = (p, T7, Td7)
    for k in range(24):
        n = rng.randint(40, 120)
        pk = np.linspace(1000.0, rng.uniform(80.0, 150.0), n)
        zk = -np.log(pk / 1000.0) * rng.uniform(6.8, 8.2)
        Tk = rng.uniform(288.0, 306.0) - rng.uniform(5.5, 7.2) * zk
        for _ in range(3):
            Tk = Tk + rng.uniform(0.0, 1.5) * np.sin(
                rng.uniform(0.5, 2.5) * np.log(pk / 100.0) + rng.uniform(0, 2 * np.pi))
        Tk = np.maximum(Tk, rng.uniform(205.0, 225.0))
        dep = rng.uniform(4.0, 12.0) + rng.uniform(2.0, 8.0) * zk / zk.max()
        for _ in range(2):
            dep = dep + rng.uniform(0.0, 2.0) * np.sin(
                rng.uniform(0.5, 2.0) * np.log(pk / 100.0) + rng.uniform(0, 2 * np.pi))
        out[f"random-{k:02d}"] = (pk, Tk, Tk - np.maximum(dep, 1.0))
    return out


SUITE = _suite()
SOUNDING_IDS = sorted(SUITE)


def _oracle_cape_cin(p, T, Td, which):
    pq, Tq, Tdq = p * units.hPa, T * units.kelvin, Td * units.kelvin
    if which == "surface":
        cape_o, cin_o = mpcalc.surface_based_cape_cin(pq, Tq, Tdq)
    else:
        cape_o, cin_o = mpcalc.most_unstable_cape_cin(pq, Tq, Tdq)
    return float(cape_o.m), float(cin_o.m)


def _assert_cape_close(actual, expected):
    assert abs(actual - expected) <= CAPE_ATOL + CAPE_RTOL * abs(expected), (
        f"CAPE {actual:.3f} vs oracle {expected:.3f} "
        f"(diff {abs(actual - expected):.3f} J/kg)"
    )


def _assert_cin_close(actual, expected):
    assert abs(actual - expected) <= CIN_ATOL + CIN_RTOL * abs(expected), (
        f"CIN {actual:.3f} vs oracle {expected:.3f} "
        f"(diff {abs(actual - expected):.3f} J/kg)"
    )


def _expected_backend() -> str:
    import os
    return "fallback" if os.environ.get("METPY_MOJO_DISABLE_NATIVE") == "1" else "native"


@pytest.mark.parametrize("name", SOUNDING_IDS)
@pytest.mark.parametrize("which", ["surface", "most_unstable"])
def test_cape_cin_vs_oracle(name, which):
    p, T, Td = SUITE[name]
    cape, cin = metpy_mojo.cape_cin(p, T, Td, which=which)
    cape_o, cin_o = _oracle_cape_cin(p, T, Td, which)
    _assert_cape_close(cape, cape_o)
    _assert_cin_close(cin, cin_o)


@pytest.mark.parametrize("name", SOUNDING_IDS)
def test_backends_agree(name):
    """Native kernel and vendored fallback compute the same numbers."""
    if not metpy_mojo.native_available():
        pytest.skip("native kernel unavailable on this run")
    p, T, Td = SUITE[name]
    for which, code in [("surface", 0), ("most_unstable", 1)]:
        native = _native.cape_cin_batched(
            np.atleast_2d(p), np.atleast_2d(T), np.atleast_2d(Td), code,
            _reference.MU_DEPTH_HPA,
        )
        fallback = _reference.cape_cin_columns(
            np.atleast_2d(p), np.atleast_2d(T), np.atleast_2d(Td), code,
            _reference.MU_DEPTH_HPA,
        )
        for a, b in zip(native, fallback):
            np.testing.assert_allclose(
                a, b, rtol=BACKEND_RTOL, atol=BACKEND_ATOL,
                err_msg=f"{name}/{which}: native vs fallback",
            )


@pytest.mark.parametrize("name", SOUNDING_IDS)
def test_diagnostics_consistency(name):
    """parcel_diagnostics agrees with cape_cin and has sane level ordering."""
    p, T, Td = SUITE[name]
    for which in ["surface", "most_unstable"]:
        diag = metpy_mojo.parcel_diagnostics(p, T, Td, which=which)
        cape, cin = metpy_mojo.cape_cin(p, T, Td, which=which)
        assert diag.cape == cape and diag.cin == cin
        assert diag.lcl_pressure <= p[diag.parcel_level] + 1e-9
        if cape > 0.0:
            assert np.isfinite(diag.lfc_pressure)
            # LFC sits at or above the LCL (they coincide for a parcel
            # saturated at its start level).
            assert diag.lfc_pressure <= diag.lcl_pressure + 1e-6
            if np.isfinite(diag.el_pressure):
                assert diag.el_pressure < diag.lfc_pressure
        else:
            assert cin == 0.0 and not np.isfinite(diag.lfc_pressure)


def test_mu_parcel_level_matches_oracle():
    """The most-unstable start level matches metpy's most_unstable_parcel."""
    for name in ["classic-91", "inversion", "moist-aloft", "tropical"]:
        p, T, Td = SUITE[name]
        mu = mpcalc.most_unstable_parcel(
            p * units.hPa, T * units.kelvin, Td * units.kelvin
        )
        diag = metpy_mojo.parcel_diagnostics(p, T, Td, which="most_unstable")
        assert diag.parcel_level == int(mu[3]), f"{name}: MU level mismatch"


def test_lcl_matches_oracle():
    p, T, Td = SUITE["classic-91"]
    diag = metpy_mojo.parcel_diagnostics(p, T, Td)
    p_lcl_o, _ = mpcalc.lcl(
        p[0] * units.hPa, T[0] * units.kelvin, Td[0] * units.kelvin
    )
    assert abs(diag.lcl_pressure - float(p_lcl_o.m)) <= LCL_ATOL_HPA


def test_grid_matches_columns_and_oracle():
    """The 3-D grid API returns the per-column answers, oracle-checked."""
    p, T, Td = SUITE["classic-91"]
    _, T2, Td2 = SUITE["tropical"]
    n = len(p)
    T3 = np.stack([T, T2, T, T2, T, T2], axis=1).reshape(n, 3, 2)
    Td3 = np.stack([Td, Td2, Td2, Td, Td, Td2], axis=1).reshape(n, 3, 2)
    for which in ["surface", "most_unstable"]:
        cape2d, cin2d = metpy_mojo.cape_cin_grid(p, T3, Td3, which=which)
        assert cape2d.shape == (3, 2) and cin2d.shape == (3, 2)
        flat_T = [T, T2, T, T2, T, T2]
        flat_Td = [Td, Td2, Td2, Td, Td, Td2]
        for idx, (yy, xx) in enumerate([(i, j) for i in range(3) for j in range(2)]):
            cape_c, cin_c = metpy_mojo.cape_cin(p, flat_T[idx], flat_Td[idx], which=which)
            assert cape2d[yy, xx] == cape_c
            assert cin2d[yy, xx] == cin_c
            cape_o, cin_o = _oracle_cape_cin(p, flat_T[idx], flat_Td[idx], which)
            _assert_cape_close(float(cape2d[yy, xx]), cape_o)
            _assert_cin_close(float(cin2d[yy, xx]), cin_o)


def test_grid_with_3d_pressure():
    """Pressure may vary per column (e.g. sigma-level-like fields)."""
    p, T, Td = SUITE["classic-91"]
    n = len(p)
    T3 = np.broadcast_to(T[:, None, None], (n, 2, 2)).copy()
    Td3 = np.broadcast_to(Td[:, None, None], (n, 2, 2)).copy()
    p3 = np.broadcast_to(p[:, None, None], (n, 2, 2)).copy()
    cape_a, _ = metpy_mojo.cape_cin_grid(p, T3, Td3)
    cape_b, _ = metpy_mojo.cape_cin_grid(p3, T3, Td3)
    np.testing.assert_array_equal(cape_a, cape_b)


def test_grid_synthetic_field_vs_oracle_loop():
    """A (nlev, 6, 8) seeded field, checked column-by-column vs the oracle."""
    rng = np.random.RandomState(7)
    n = 60
    p = np.linspace(1000.0, 120.0, n)
    z = -np.log(p / 1000.0) * 7.5
    base_T = np.maximum(299.0 - 6.4 * z, 220.0)
    base_Td = np.maximum(293.0 - 5.2 * z, 198.0)
    T3 = np.empty((n, 6, 8))
    Td3 = np.empty((n, 6, 8))
    for y in range(6):
        for x in range(8):
            warm = rng.uniform(-2.0, 4.0)
            moist = rng.uniform(-4.0, 5.0)
            T3[:, y, x] = np.maximum(base_T + warm * np.exp(-z / 3.0), 220.0)
            Td3[:, y, x] = np.minimum(T3[:, y, x] - 0.5,
                                      base_Td + moist * np.exp(-z / 4.0))
    cape2d, cin2d = metpy_mojo.cape_cin_grid(p, T3, Td3)
    assert cape2d.shape == (6, 8)
    for y, x in [(0, 0), (2, 3), (5, 7), (3, 4)]:
        cape_o, cin_o = _oracle_cape_cin(p, T3[:, y, x], Td3[:, y, x], "surface")
        _assert_cape_close(float(cape2d[y, x]), cape_o)
        _assert_cin_close(float(cin2d[y, x]), cin_o)


def test_backend_selection():
    info = metpy_mojo.backend_info()
    if _expected_backend() == "native":
        assert info["native_available"], info
    else:
        assert not info["native_available"]
