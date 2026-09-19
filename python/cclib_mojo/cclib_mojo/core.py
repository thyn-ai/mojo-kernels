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

import numpy as np

from cclib_mojo import _reference
from cclib_mojo._basis import BOHR2ANG, BasisError, flatten_gbasis
from cclib_mojo._native import (
    MODE_DENSITY,
    MODE_WAVEFUNCTION,
    NativeUnavailable,
    _load,
)


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
