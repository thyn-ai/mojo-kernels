"""Public API: ASE-shaped ``primitive_neighbor_list`` / ``neighbor_list``.

Behavioral contract (pinned against ASE 3.26.0, the pip oracle):

* ``quantities`` is a subset of ``"ijSdD"`` (else ``ValueError``).
* ``pbc`` must be a length-3 array-like (a scalar raises ``TypeError``).
* ``cutoff`` is a float (global cutoff), a dict keyed by element *pairs*
  (atomic numbers or symbols, either order, e.g. ``{("C", "H"): 1.3}``;
  pairs missing from the dict get cutoff 0), or a per-atom array of radii
  (pair cutoff = r_i + r_j). Dict cutoffs require ``numbers``.
* The distance test is strict: pairs at exactly the cutoff are excluded.
* Self images ``(i, i, S != 0)`` are always included; ``(i, i, 0)`` is
  included iff ``self_interaction=True``.
* ``bothways=True`` (ASE's module-level behavior): every ordered
  ``(i, j, S)`` is reported. ``bothways=False`` applies the classic
  ``ase.neighborlist.NeighborList`` reduction, verified against that class:
  keep ``(i, j, S)`` iff lex(S) > 0 (first nonzero component positive) or
  ``S == 0 and i < j``; ``(i, i, 0)`` iff ``self_interaction``.
* Shift vectors refer to the *original* positions:
  ``D = positions[j] - positions[i] + S @ cell``.
* ``max_nbins`` caps the bin-grid memory (results are unchanged).
* A singular cell with any periodic axis raises ``ValueError`` (the oracle
  produces undefined output there). A singular cell with no periodic axis
  is treated as a Cartesian system with S = 0, matching the oracle.
* Output order is fully sorted by ``(i, j, S)`` — deterministic. The
  oracle sorts by ``i`` only and documents pair order as not guaranteed;
  comparisons should be order-insensitive.
"""

from __future__ import annotations

import numpy as np

from ase_mojo import _native, _reference
from ase_mojo._symbols import atomic_number

_VALID_QUANTITIES = frozenset("ijSdD")
_DUMMY_MATRIX = np.zeros((1, 1), dtype=np.float64)
_DUMMY_RADII = np.zeros(1, dtype=np.float64)


def _validate_pbc(pbc) -> np.ndarray:
    arr = np.asarray(pbc)
    if arr.shape != (3,):
        raise TypeError(
            f"pbc must be a length-3 array-like of booleans, got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr.astype(np.int32))


def _validate_positions(positions, cell: np.ndarray, use_scaled_positions: bool) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (n, 3), got {pos.shape}"
        )
    if use_scaled_positions:
        pos = pos @ cell
    return np.ascontiguousarray(pos, dtype=np.float64)


def _validate_cell(cell) -> np.ndarray:
    c = np.asarray(cell, dtype=np.float64)
    if c.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {c.shape}")
    return np.ascontiguousarray(c)


def _resolve_cutoff(cutoff, numbers, n: int):
    """Resolve the user cutoff into (mode, types, n_types, matrix, radii)."""
    if isinstance(cutoff, dict):
        if numbers is None:
            raise TypeError("numbers are required when cutoff is a dict")
        z = np.asarray(numbers, dtype=np.int64).ravel()
        if z.shape[0] != n:
            raise ValueError(
                f"numbers must have one atomic number per atom ({n}), got {z.shape[0]}"
            )
        species = sorted(set(int(v) for v in z))
        index = {v: k for k, v in enumerate(species)}
        types = np.array([index[int(v)] for v in z], dtype=np.int32)
        matrix = np.zeros((len(species), len(species)), dtype=np.float64)
        for key, value in cutoff.items():
            try:
                k0, k1 = key
            except (TypeError, ValueError):
                raise TypeError(
                    "cutoff dict keys must be (element, element) pairs, "
                    f"got {key!r}"
                ) from None
            z0, z1 = atomic_number(k0), atomic_number(k1)
            if z0 not in index or z1 not in index:
                continue  # species absent from this system: entry is inert
            a, b = index[z0], index[z1]
            matrix[a, b] = matrix[b, a] = float(value)
        return _native.MODE_MATRIX, types, len(species), matrix, _DUMMY_RADII

    arr = np.asarray(cutoff, dtype=np.float64)
    if arr.ndim == 0:
        types = np.zeros(n, dtype=np.int32)
        matrix = np.array([[float(arr)]], dtype=np.float64)
        return _native.MODE_MATRIX, types, 1, matrix, _DUMMY_RADII
    radii = arr.ravel().astype(np.float64)
    if radii.shape[0] != n:
        raise ValueError(
            f"per-atom cutoff array must have one radius per atom ({n}), "
            f"got {radii.shape[0]}"
        )
    types = np.zeros(n, dtype=np.int32)
    return _native.MODE_RADII, types, 1, _DUMMY_MATRIX, np.ascontiguousarray(radii)


def _build_ijs(
    pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff,
    numbers,
    self_interaction: bool,
    bothways: bool,
    max_nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = pos.shape[0]
    mode, types, n_types, matrix, radii = _resolve_cutoff(cutoff, numbers, n)
    types = np.ascontiguousarray(types, dtype=np.int32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    try:
        return _native.build_ijs(
            pos, cell, pbc, mode, types, n_types, matrix, radii,
            self_interaction, bothways, max_nbins,
        )
    except _native.NativeUnavailable:
        return _reference.build_ijs(
            pos, cell, pbc, mode, types, n_types, matrix, radii,
            self_interaction, bothways, max_nbins,
        )


def _check_geometry(cell: np.ndarray, pbc: np.ndarray) -> None:
    if pbc.any() and abs(float(np.linalg.det(cell))) < 1e-300:
        raise ValueError(
            "cell is singular but periodic boundary conditions require an "
            "invertible cell"
        )


def _assemble(quantities: str, i: np.ndarray, j: np.ndarray, S: np.ndarray,
              pos: np.ndarray, cell: np.ndarray) -> tuple:
    if i.shape[0] > 1:
        order = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
        i, j, S = i[order], j[order], S[order]
    results = []
    for q in quantities:
        if q == "i":
            results.append(i)
        elif q == "j":
            results.append(j)
        elif q == "S":
            results.append(S)
        elif q == "D":
            results.append(pos[j] - pos[i] + S @ cell if i.shape[0] else np.empty((0, 3)))
        elif q == "d":
            if i.shape[0]:
                D = pos[j] - pos[i] + S @ cell
                results.append(np.sqrt(np.einsum("ij,ij->i", D, D)))
            else:
                results.append(np.empty(0, dtype=np.float64))
    return tuple(results)


def primitive_neighbor_list(
    quantities: str,
    pbc,
    cell,
    positions,
    cutoff,
    numbers=None,
    self_interaction: bool = False,
    use_scaled_positions: bool = False,
    max_nbins: float = 1000000.0,
    *,
    bothways: bool = True,
) -> tuple:
    """Compute a neighbor list, drop-in-shaped after ASE's primitive version.

    See the module docstring for the full behavioral contract.
    """
    for q in quantities:
        if q not in _VALID_QUANTITIES:
            raise ValueError("Unsupported quantity specified.")
    pbc_arr = _validate_pbc(pbc)
    cell_arr = _validate_cell(cell)
    pos = _validate_positions(positions, cell_arr, use_scaled_positions)
    _check_geometry(cell_arr, pbc_arr)
    nbins_cap = int(max_nbins)
    i, j, S = _build_ijs(
        pos, cell_arr, pbc_arr, cutoff, numbers,
        bool(self_interaction), bool(bothways), nbins_cap,
    )
    return _assemble(quantities, i, j, S, pos, cell_arr)


def neighbor_list(
    quantities: str,
    a,
    cutoff,
    self_interaction: bool = False,
    max_nbins: float = 1000000.0,
    *,
    bothways: bool = True,
) -> tuple:
    """Compute a neighbor list for an ASE ``Atoms``-shaped object.

    ``a`` is duck-typed: it needs ``pbc``, ``cell``, ``positions`` (or
    ``get_positions()``) and ``numbers`` (or ``get_atomic_numbers()``).
    ASE itself is not imported.
    """
    pos = a.get_positions() if hasattr(a, "get_positions") else a.positions
    numbers = (
        a.get_atomic_numbers() if hasattr(a, "get_atomic_numbers") else a.numbers
    )
    return primitive_neighbor_list(
        quantities,
        a.pbc,
        a.cell,
        pos,
        cutoff,
        numbers=numbers,
        self_interaction=self_interaction,
        max_nbins=max_nbins,
        bothways=bothways,
    )
