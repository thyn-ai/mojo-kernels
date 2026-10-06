"""Public grid API: MO wavefunctions and electron densities on 3-D grids.

Both entry points evaluate contracted Cartesian Gaussian basis functions
(cclib's ``gbasis``) on a regular 3-D grid and return float64 NumPy arrays
whose layout matches cclib's ``Volume.data`` exactly: shape
``(nx, ny, nz)``, C-order, element ``[i, j, k]`` at coordinate
``(origin[0]+i*step[0], origin[1]+j*step[1], origin[2]+k*step[2])``.

Units follow cclib: ``atomcoords``, ``origin`` and ``step`` are in
Angstrom; Gaussian exponents in ``gbasis`` are in bohr^-2 (as parsed by
cclib); amplitudes are in atomic units. Evaluation runs on the native Mojo
kernel when available and on the vendored NumPy reference otherwise; both
backends share basis flattening and normalization in ``cclib_mojo._basis``.
"""

from __future__ import annotations

import math

import numpy as np

from cclib_mojo import _reference
from cclib_mojo._basis import (
    BOHR2ANG,
    COORDINATE_SANE_MAX,
    INTERMEDIATE_MAGNITUDE_MAX,
    BasisArrays,
    flatten_gbasis,
)
from cclib_mojo._native import (
    MODE_DENSITY,
    MODE_WAVEFUNCTION,
    NativeUnavailable,
    _load,
)

_INTERMEDIATE_LOG10_MAX = math.log10(INTERMEDIATE_MAGNITUDE_MAX)


class GridError(ValueError):
    """Malformed grid or MO-coefficient input (structured, fail-fast)."""


def _validate_vector3(name: str, value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise GridError(f"{name} must be a 3-vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise GridError(f"{name} contains non-finite values")
    return arr


def _validate_shape(shape: object) -> tuple[int, int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise GridError(f"shape must be a 3-tuple (nx, ny, nz), got {shape!r}")
    out = []
    for dim in shape:
        if isinstance(dim, bool) or int(dim) != dim or int(dim) <= 0:
            raise GridError(f"shape entries must be positive integers, got {shape!r}")
        out.append(int(dim))
    return out[0], out[1], out[2]


def _validate_grid_coordinates(
    origin: np.ndarray, step: np.ndarray, shape: tuple[int, int, int]
) -> None:
    """Every grid corner must stay inside the documented coordinate domain."""
    for i, (o, st, n) in enumerate(zip(origin, step, shape)):
        end = o + (n - 1) * st
        if abs(o) > COORDINATE_SANE_MAX or abs(end) > COORDINATE_SANE_MAX:
            raise GridError(
                f"grid axis {i} spans [{o!r}, {end!r}] Angstrom; every grid "
                f"coordinate must be inside "
                f"[-{COORDINATE_SANE_MAX!r}, {COORDINATE_SANE_MAX!r}] Angstrom"
            )


def _grid_axes(
    origin: np.ndarray, step: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-point coordinates in bohr, one vector per axis.

    Mirrors cclib's ``getGrid`` (cclib/method/volume.py): points are
    ``origin + i*step`` in Angstrom, divided element-wise by cclib's
    bohr->Angstrom constant 0.5291772109.
    """
    nx, ny, nz = shape
    ax = (origin[0] + np.arange(nx, dtype=np.float64) * step[0]) / BOHR2ANG
    ay = (origin[1] + np.arange(ny, dtype=np.float64) * step[1]) / BOHR2ANG
    az = (origin[2] + np.arange(nz, dtype=np.float64) * step[2]) / BOHR2ANG
    return ax, ay, az


def _validate_coeff(coeff: object, n_bf: int) -> np.ndarray:
    arr = np.asarray(coeff, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise GridError(f"coeff must be an (n_mo, n_bf) array, got ndim={arr.ndim}")
    if arr.shape[1] != n_bf:
        raise GridError(
            f"coeff has {arr.shape[1]} columns but the basis expands to "
            f"{n_bf} Cartesian basis functions"
        )
    if arr.shape[0] == 0:
        raise GridError("coeff contains zero MO rows")
    if not np.all(np.isfinite(arr)):
        raise GridError("coeff contains non-finite values")
    return arr


def _intermediate_log10_magnitudes(
    basis: BasisArrays,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    coeff2d: np.ndarray,
) -> np.ndarray:
    """Per basis function: log10 of a bound on every intermediate product.

    Each grid term ``c N_c (x-cx)^l (y-cy)^m (z-cz)^n w_p exp(-alpha_p r^2)``
    is a chain of float64 products that the two backends associate
    differently (the kernel forms ``((c N_c) x^l) y^m`` first, the fallback
    the polynomial times the contraction ``sum_p w_p exp(...)``). Since
    ``exp(...) <= 1``, every partial product in either order is at most

        max(1, |c| N_c) * max(1, |x-cx|)^l * max(1, |y-cy|)^m
            * max(1, |z-cz|)^n * max(1, sum_p |w_p|)

    with ``|c|`` the largest coefficient over the evaluated MO rows and each
    distance the farthest grid coordinate. The bound is returned as a sum of
    log10 terms so that computing it cannot overflow. Functions whose
    coefficient is exactly zero in every row are skipped by both backends
    (cclib's rule) and build no product: their bound is 0 (= log10 1).
    """
    coeff_peak = np.max(np.abs(coeff2d), axis=0)
    active = coeff_peak > 0.0
    # The placeholder 1.0 keeps log10 away from the skipped zeros.
    log10_coeff_norm = np.maximum(
        0.0, np.log10(np.where(active, coeff_peak, 1.0)) + np.log10(basis.bf_norm)
    )

    log10_polynomial = np.zeros(basis.n_bf, dtype=np.float64)
    centers = (basis.center_x, basis.center_y, basis.center_z)
    powers = (basis.powers_l, basis.powers_m, basis.powers_n)
    for axis, center, power in zip(axes, centers, powers):
        # The farthest grid coordinate from a center is one of the axis ends.
        reach = np.maximum(np.abs(axis.min() - center), np.abs(axis.max() - center))
        log10_polynomial += power * np.log10(np.maximum(reach, 1.0))

    # log10(max(1, sum_p |w_p|)), summed relative to max(1, max_p |w_p|) so
    # that every ratio is <= 1 and the per-function sum cannot overflow.
    abs_w = np.abs(basis.prim_w)
    starts = basis.offsets[:-1]
    w_unit = np.maximum(np.maximum.reduceat(abs_w, starts), 1.0)
    prims_per_function = np.diff(basis.offsets)
    w_rel_sum = np.add.reduceat(abs_w / np.repeat(w_unit, prims_per_function), starts)
    log10_weights = np.maximum(
        0.0,
        np.log10(w_unit) + np.log10(np.maximum(w_rel_sum, np.finfo(np.float64).tiny)),
    )

    return np.where(active, log10_coeff_norm + log10_polynomial + log10_weights, 0.0)


def _validate_intermediate_magnitudes(
    basis: BasisArrays,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    coeff2d: np.ndarray,
) -> None:
    """No grid term may build a product above ``INTERMEDIATE_MAGNITUDE_MAX``.

    Past the double range the backends' product orders overflow at different
    stages (inf on one side, inf * 0 = NaN on the other), so such inputs have
    no well-defined result and are rejected before evaluation.
    """
    log10_bound = _intermediate_log10_magnitudes(basis, axes, coeff2d)
    worst = int(np.argmax(log10_bound))
    if log10_bound[worst] > _INTERMEDIATE_LOG10_MAX:
        powers = (
            int(basis.powers_l[worst]),
            int(basis.powers_m[worst]),
            int(basis.powers_n[worst]),
        )
        raise GridError(
            f"basis function {worst} (Cartesian powers {powers}) builds "
            f"intermediate products up to ~1e{log10_bound[worst]:.1f} on this "
            f"grid (|MO coefficient| x contracted norm x polynomial x "
            f"primitive weights); every term must stay at or below "
            f"{INTERMEDIATE_MAGNITUDE_MAX!r} so that no evaluation order "
            f"overflows"
        )


def _eval(
    gbasis: object,
    atomcoords: object,
    coeff: object,
    origin: object,
    step: object,
    shape: object,
    mo_index: int | None,
) -> np.ndarray:
    basis = flatten_gbasis(gbasis, atomcoords)
    coeff2d = _validate_coeff(coeff, basis.n_bf)
    origin3 = _validate_vector3("origin", origin)
    step3 = _validate_vector3("step", step)
    if not np.all(step3 > 0.0):
        raise GridError(f"step entries must be positive, got {tuple(step3)}")
    shape3 = _validate_shape(shape)
    _validate_grid_coordinates(origin3, step3, shape3)

    if mo_index is not None:
        if isinstance(mo_index, bool) or int(mo_index) != mo_index:
            raise GridError(f"mo_index must be an integer, got {mo_index!r}")
        mo_index = int(mo_index)
        if not 0 <= mo_index < coeff2d.shape[0]:
            raise GridError(
                f"mo_index {mo_index} out of range for {coeff2d.shape[0]} MO rows"
            )
        coeff2d = coeff2d[mo_index : mo_index + 1]
        mode = MODE_WAVEFUNCTION
    else:
        # Sum of squared MOs over every provided row (psi^2 for one row).
        mode = MODE_DENSITY

    axes = _grid_axes(origin3, step3, shape3)
    _validate_intermediate_magnitudes(basis, axes, coeff2d)
    try:
        flat = _native_eval(basis, axes, coeff2d, mode)
    except NativeUnavailable:
        flat = _reference.eval_grid(basis, *axes, coeff2d, mode)
    return flat.reshape(shape3)


def _native_eval(basis, axes, coeff2d: np.ndarray, mode: int) -> np.ndarray:
    """Native path; raises NativeUnavailable so the caller can fall back."""
    from cclib_mojo import _native

    _load()  # fail fast here if the kernel cannot be used at all
    return _native.eval_grid(basis, *axes, coeff2d, mode)


def wavefunction_on_grid(
    gbasis: object,
    atomcoords: object,
    mocoeffs: object,
    origin: object,
    step: object,
    shape: object,
) -> np.ndarray:
    """Evaluate one molecular orbital's amplitude on a 3-D grid.

    Equivalent to cclib's ``cclib.method.volume.wavefunction``: the result
    equals that function's ``Volume.data`` for the same ``gbasis``,
    ``atomcoords`` (Angstrom), ``mocoeffs`` (1-D, length n_bf), and grid
    (Angstrom). Values are signed amplitudes in atomic units.
    """
    coeff = np.asarray(mocoeffs, dtype=np.float64)
    if coeff.ndim != 1:
        raise GridError(
            f"mocoeffs for wavefunction_on_grid must be 1-D, got shape {coeff.shape}"
        )
    return _eval(gbasis, atomcoords, coeff, origin, step, shape, mo_index=0)


def density_on_grid(
    gbasis: object,
    atomcoords: object,
    coeff: object,
    origin: object,
    step: object,
    shape: object,
    mo_index: int | None = None,
) -> np.ndarray:
    """Evaluate electron density (or one MO) on a 3-D grid.

    ``coeff`` is an (n_mo, n_bf) array of MO coefficients (a 1-D array is
    promoted to a single row). With ``mo_index=None`` the result is
    ``sum_mo |psi_mo(r)|^2`` — cclib's ``electrondensity_spin`` semantics;
    for a closed-shell total density pass the occupied spatial MOs and
    multiply by 2, exactly as cclib's ``electrondensity`` does. With an
    integer ``mo_index`` the result is that MO's signed amplitude instead
    (cclib's ``wavefunction`` semantics).
    """
    return _eval(gbasis, atomcoords, coeff, origin, step, shape, mo_index)
