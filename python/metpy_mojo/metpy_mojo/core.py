"""Parcel CAPE/CIN for standard atmospheric soundings, MetPy-compatible.

``cape_cin`` computes surface-based or most-unstable CAPE and CIN for one
column; ``cape_cin_grid`` does the same for every column of a gridded
(pressure-level, y, x, ...) field in one batched native call — the
3-D-field use case of MetPy issue #480. ``parcel_diagnostics`` adds the
LCL/LFC/EL pressures and parcel start level.

Computation runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back
to the vendored pure-Python reference implementation otherwise; both
backends implement the same algorithm and agree to ~1e-6 relative (see
``metpy_mojo/_reference.py`` for the documented formulas and parity
measurements against the MetPy oracle).

Units: pressure in hPa (strictly decreasing along each column),
temperature/dewpoint in K. Returns CAPE/CIN in J/kg; CIN is <= 0, and a
column with no LFC returns (0.0, 0.0) — mirroring the MetPy oracle.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from metpy_mojo import _reference
from metpy_mojo._native import (
    WHICH_MOST_UNSTABLE,
    WHICH_SURFACE,
    NativeUnavailable,
    cape_cin_batched,
)

__all__ = [
    "ParcelDiagnostics",
    "cape_cin",
    "cape_cin_grid",
    "parcel_diagnostics",
]

_WHICH_NAMES = {
    "surface": WHICH_SURFACE,
    "most_unstable": WHICH_MOST_UNSTABLE,
}


class ParcelDiagnostics(NamedTuple):
    """Full parcel result for one column (pressures in hPa, NaN if absent)."""

    cape: float
    cin: float
    lcl_pressure: float
    lfc_pressure: float
    el_pressure: float
    parcel_level: int  # index of the lifted parcel's start level


def _es_vapor_hpa(t_k: np.ndarray) -> np.ndarray:
    """Vectorized Bolton (1980) saturation vapor pressure for validation."""
    return 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))


def _which_code(which: str | int) -> int:
    if isinstance(which, str):
        try:
            return _WHICH_NAMES[which]
        except KeyError:
            raise ValueError(
                f"which must be one of {sorted(_WHICH_NAMES)} (got {which!r})"
            ) from None
    if which in (WHICH_SURFACE, WHICH_MOST_UNSTABLE):
        return int(which)
    raise ValueError(f"which must be 'surface' or 'most_unstable' (got {which!r})")


def _validate_columns(p: np.ndarray, t: np.ndarray, td: np.ndarray) -> None:
    """Fail early on out-of-scope columns (see the module docstring)."""
    ncols, nlev = p.shape
    if nlev < 2:
        raise ValueError(f"need at least 2 levels per column (got {nlev})")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t)) and np.all(np.isfinite(td))):
        raise ValueError("pressure/temperature/dewpoint must all be finite")
    if np.any(p <= 0.0):
        raise ValueError("pressure must be positive (hPa)")
    if np.any(np.diff(p, axis=1) >= 0.0):
        raise ValueError("pressure must be strictly decreasing along every column")
    # The mixing-ratio formulas require es < p at every level; beyond that
    # the column is outside the standard-atmosphere scope of this package.
    if np.any(_es_vapor_hpa(t) >= p) or np.any(_es_vapor_hpa(td) >= p):
        raise ValueError(
            "saturation vapor pressure exceeds ambient pressure at some level; "
            "the column is outside the standard-atmosphere scope"
        )
    del ncols


def _compute(p: np.ndarray, t: np.ndarray, td: np.ndarray, which: int,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated (ncols, nlev) batch on the active backend."""
    try:
        return cape_cin_batched(p, t, td, which, _reference.MU_DEPTH_HPA)
    except NativeUnavailable:
        return _reference.cape_cin_columns(p, t, td, which, _reference.MU_DEPTH_HPA)


def _parcel_start_levels(p: np.ndarray, t: np.ndarray, td: np.ndarray,
                         which: int) -> np.ndarray:
    """Per-column start index (0 for surface; theta-e argmax for MU)."""
    if which == WHICH_SURFACE:
        return np.zeros(p.shape[0], dtype=np.int64)
    return np.array([
        _reference._parcel_start_index(p[c], t[c], td[c], which,
                                       _reference.MU_DEPTH_HPA)
        for c in range(p.shape[0])
    ], dtype=np.int64)


def cape_cin(
    pressure,
    temperature,
    dewpoint,
    which: str | int = "surface",
) -> tuple[float, float]:
    """CAPE and CIN (J/kg) for one column of a standard sounding.

    ``pressure`` (hPa, strictly decreasing), ``temperature`` and
    ``dewpoint`` (K) are 1-D array-likes of equal length (>= 2). ``which``
    is ``"surface"`` (parcel lifted from the first level) or
    ``"most_unstable"`` (parcel with the highest equivalent potential
    temperature within 300 hPa of the surface). CIN is <= 0; a column
    with no LFC returns ``(0.0, 0.0)``.
    """
    code = _which_code(which)
    p = np.atleast_2d(np.asarray(pressure, dtype=np.float64))
    t = np.atleast_2d(np.asarray(temperature, dtype=np.float64))
    td = np.atleast_2d(np.asarray(dewpoint, dtype=np.float64))
    if p.shape != t.shape or p.shape != td.shape:
        raise ValueError(
            f"pressure, temperature and dewpoint must have the same shape "
            f"(got {p.shape}, {t.shape}, {td.shape})"
        )
    if p.shape[0] != 1:
        raise ValueError("cape_cin takes a single column; use cape_cin_grid for fields")
    _validate_columns(p, t, td)
    cape, cin, _, _, _ = _compute(p, t, td, code)
    return float(cape[0]), float(cin[0])


def parcel_diagnostics(
    pressure,
    temperature,
    dewpoint,
    which: str | int = "surface",
) -> ParcelDiagnostics:
    """CAPE/CIN plus LCL/LFC/EL pressures (hPa) and the parcel start level.

    Same input contract as :func:`cape_cin`. LFC/EL are NaN when they do
    not exist (no LFC => cape = cin = 0; no EL crossing => CAPE integrates
    to the profile top).
    """
    code = _which_code(which)
    p = np.atleast_2d(np.asarray(pressure, dtype=np.float64))
    t = np.atleast_2d(np.asarray(temperature, dtype=np.float64))
    td = np.atleast_2d(np.asarray(dewpoint, dtype=np.float64))
    if p.shape != t.shape or p.shape != td.shape:
        raise ValueError(
            f"pressure, temperature and dewpoint must have the same shape "
            f"(got {p.shape}, {t.shape}, {td.shape})"
        )
    if p.shape[0] != 1:
        raise ValueError(
            "parcel_diagnostics takes a single column; use cape_cin_grid for fields"
        )
    _validate_columns(p, t, td)
    cape, cin, lclp, lfcp, elp = _compute(p, t, td, code)
    level = _parcel_start_levels(p, t, td, code)[0]
    return ParcelDiagnostics(
        cape=float(cape[0]),
        cin=float(cin[0]),
        lcl_pressure=float(lclp[0]),
        lfc_pressure=float(lfcp[0]),
        el_pressure=float(elp[0]),
        parcel_level=int(level),
    )


def cape_cin_grid(
    pressure,
    temperature,
    dewpoint,
    which: str | int = "surface",
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE/CIN (J/kg) for every column of a gridded pressure-level field.

    ``temperature`` and ``dewpoint`` are array-likes shaped ``(nlev, ...)``
    (level first; any trailing grid dimensions). ``pressure`` (hPa) is
    either 1-D of length ``nlev`` (shared levels, the common case) or the
    same shape as ``temperature``. Returns ``(cape, cin)`` float64 arrays
    shaped ``(...)``.
    """
    code = _which_code(which)
    t = np.asarray(temperature, dtype=np.float64)
    td = np.asarray(dewpoint, dtype=np.float64)
    if t.shape != td.shape:
        raise ValueError(
            f"temperature and dewpoint must have the same shape (got {t.shape}, {td.shape})"
        )
    if t.ndim < 1:
        raise ValueError("temperature must have at least one (level) axis")
    nlev = t.shape[0]
    grid_shape = t.shape[1:]
    ncols = int(np.prod(grid_shape)) if grid_shape else 1
    t2 = np.ascontiguousarray(t.reshape(nlev, ncols).T)
    td2 = np.ascontiguousarray(td.reshape(nlev, ncols).T)

    p = np.asarray(pressure, dtype=np.float64)
    if p.ndim == 1:
        if p.shape[0] != nlev:
            raise ValueError(
                f"1-D pressure has {p.shape[0]} levels but temperature has {nlev}"
            )
        p2 = np.broadcast_to(p[:, None], (nlev, ncols))
        p2 = np.ascontiguousarray(p2.T)
    else:
        if p.shape != t.shape:
            raise ValueError(
                f"pressure must be 1-D (nlev,) or match temperature shape "
                f"(got {p.shape} vs {t.shape})"
            )
        p2 = np.ascontiguousarray(p.reshape(nlev, ncols).T)

    _validate_columns(p2, t2, td2)
    cape, cin, _, _, _ = _compute(p2, t2, td2, code)
    return cape.reshape(grid_shape), cin.reshape(grid_shape)
