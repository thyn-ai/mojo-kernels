"""Basis-set validation, normalization, and flattening — shared by both backends.

Converts cclib's ``gbasis`` (per-atom, per-shell primitive lists, as parsed
from a quantum-chemistry logfile) into the flat typed arrays the Mojo kernel
and the NumPy fallback both consume. Because this module is the single source
of basis data for the native and fallback paths, the two backends can never
disagree about normalization constants or centers.

Conventions (all verified against cclib + PyQuante, see README):

- Shell expansion order follows cclib's ``sym2powerlist`` verbatim
  (cclib/method/volume.py): S, then P (x, y, z), then Cartesian D
  (xx, yy, zz, xy, yz, xz), then Cartesian F
  (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).
- Centers are converted Angstrom -> bohr with cclib's own constant
  (``convertor(x, "Angstrom", "bohr")`` = x * 1.8897261245,
  cclib/parser/utils.py). Grid axes are converted separately in
  ``cclib_mojo.core`` (cclib divides by 0.5291772109 there — the two
  constants are not exact reciprocals, and we mirror cclib exactly).
- Primitive norms follow THO eq. 2.2 exactly as PyQuante's
  ``PGBF.normalize`` computes them; the contracted norm is
  ``1/sqrt(<g|g>)`` with the self-overlap from THO eq. 2.12, in PyQuante's
  operation order (PyQuante 1.6.5, R. P. Muller, BSD licensed).
- Exponents stay in bohr^-2 exactly as parsed (cclib never converts them).

The math below is textbook Gaussian-basis algebra (Taketa, Huzinaga,
O-ohata, J. Phys. Soc. Jap. 21, 2313 (1966)); operation order mirrors the
published PyQuante reference so results agree to the last few ulps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# cclib/method/volume.py `sym2powerlist`, verbatim: the Cartesian power
# triples (lx, ly, lz) each shell symbol expands into, in cclib's order.
SYM2POWERS: dict[str, list[tuple[int, int, int]]] = {
    "S": [(0, 0, 0)],
    "P": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "D": [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)],
    "F": [
        (3, 0, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 1, 1),
        (1, 0, 2),
        (0, 3, 0),
        (0, 2, 1),
        (0, 1, 2),
        (0, 0, 3),
    ],
}

# cclib/parser/utils.py `convertor`, Angstrom -> bohr (used for centers).
ANG2BOHR = 1.8897261245
# cclib/parser/utils.py `convertor`, bohr -> Angstrom (cclib's getGrid
# divides Angstrom grid coordinates by this value).
BOHR2ANG = 0.5291772109


class BasisError(ValueError):
    """Malformed basis-set or geometry input (structured, fail-fast)."""


@dataclass(frozen=True)
class BasisArrays:
    """Flat typed arrays describing every contracted Cartesian basis function.

    Primitive ``w`` folds the contraction coefficient and the primitive
    norm: w_p = coef_p * N_p. ``bf_norm`` is the contracted norm N_c, so
    bf(r) = bf_norm * (x-cx)^l (y-cy)^m (z-cz)^n * sum_p w_p exp(-alpha_p r^2).
    """

    offsets: np.ndarray  # int64 [n_bf + 1], CSR row offsets into alpha/w
    powers_l: np.ndarray  # int32 [n_bf]
    powers_m: np.ndarray  # int32 [n_bf]
    powers_n: np.ndarray  # int32 [n_bf]
    center_x: np.ndarray  # float64 [n_bf], bohr
    center_y: np.ndarray  # float64 [n_bf], bohr
    center_z: np.ndarray  # float64 [n_bf], bohr
    bf_norm: np.ndarray  # float64 [n_bf]
    prim_alpha: np.ndarray  # float64 [n_prims], bohr^-2
    prim_w: np.ndarray  # float64 [n_prims]
    n_bf: int
    n_prims: int


def _fact2(n: int) -> int:
    """Double factorial n!! = n*(n-2)*...*1; fact2(n) = 1 for n <= 0."""
    val = 1
    while n > 0:
        val *= n
        n -= 2
    return val


def primitive_norm(alpha: float, l: int, m: int, n: int) -> float:
    """Normalization of a primitive Cartesian Gaussian, THO eq. 2.2.

    Same expression (and operation order) as PyQuante 1.6.5
    ``PGBF.normalize``.
    """
    L = l + m + n
    return math.sqrt(
        pow(2, 2 * L + 1.5)
        * pow(alpha, L + 1.5)
        / _fact2(2 * l - 1)
        / _fact2(2 * m - 1)
        / _fact2(2 * n - 1)
        / pow(math.pi, 1.5)
    )


def _binomial_prefactor(s: int, ia: int, ib: int, xpa: float, xpb: float) -> float:
    """THO binomial expansion term (PyQuante pyints.binomial_prefactor)."""
    total = 0.0
    for t in range(s + 1):
        if s - ia <= t <= ib:
            total += (
                math.comb(ia, s - t)
                * math.comb(ib, t)
                * pow(xpa, ia - s + t)
                * pow(xpb, ib - t)
            )
    return total


def _overlap_1d(l1: int, l2: int, pax: float, pbx: float, gamma: float) -> float:
    """THO eq. 2.12 one-dimensional overlap (PyQuante pyints.overlap_1D)."""
    total = 0.0
    for i in range(1 + (l1 + l2) // 2):
        total += (
            _binomial_prefactor(2 * i, l1, l2, pax, pbx)
            * _fact2(2 * i - 1)
            / pow(2 * gamma, i)
        )
    return total


def _overlap(
    alpha1: float,
    powers: tuple[int, int, int],
    a_center: tuple[float, float, float],
    alpha2: float,
    b_center: tuple[float, float, float],
) -> float:
    """Unnormalized overlap of two primitives, THO eq. 2.12.

    Both primitives carry the same Cartesian powers (they belong to one
    contracted function). Mirrors PyQuante pyints.overlap operation order.
    """
    l1, m1, n1 = powers
    rab2 = (
        pow(a_center[0] - b_center[0], 2)
        + pow(a_center[1] - b_center[1], 2)
        + pow(a_center[2] - b_center[2], 2)
    )
    gamma = alpha1 + alpha2
    px = (alpha1 * a_center[0] + alpha2 * b_center[0]) / gamma
    py = (alpha1 * a_center[1] + alpha2 * b_center[1]) / gamma
    pz = (alpha1 * a_center[2] + alpha2 * b_center[2]) / gamma
    pre = pow(math.pi / gamma, 1.5) * math.exp(-alpha1 * alpha2 * rab2 / gamma)
    wx = _overlap_1d(l1, l1, px - a_center[0], px - b_center[0], gamma)
    wy = _overlap_1d(m1, m1, py - a_center[1], py - b_center[1], gamma)
    wz = _overlap_1d(n1, n1, pz - a_center[2], pz - b_center[2], gamma)
    return pre * wx * wy * wz


def _contracted_norm(
    prims: list[tuple[float, float]],
    pnorms: list[float],
    powers: tuple[int, int, int],
    center: tuple[float, float, float],
) -> float:
    """N_c = 1/sqrt(<g|g>) over normalized primitives (PyQuante order)."""
    S = 0.0
    for (ai, ci), ni in zip(prims, pnorms):
        for (aj, cj), nj in zip(prims, pnorms):
            S += ci * cj * (ni * nj * _overlap(ai, powers, center, aj, center))
    if S <= 0.0 or not math.isfinite(S):
        raise BasisError(
            f"contracted basis function with powers {powers} has non-positive "
            f"self-overlap {S!r}; check exponents and coefficients"
        )
    return 1.0 / math.sqrt(S)


def _validate_shell(atom_index: int, shell: object) -> tuple[str, list[tuple[float, float]]]:
    if not (isinstance(shell, (list, tuple)) and len(shell) == 2):
        raise BasisError(
            f"gbasis[{atom_index}]: each shell must be a (symbol, primitives) "
            f"pair, got {shell!r}"
        )
    sym, prims = shell
    if not isinstance(sym, str) or sym not in SYM2POWERS:
        raise BasisError(
            f"gbasis[{atom_index}]: unsupported shell symbol {sym!r}; "
            f"supported shells match cclib: {sorted(SYM2POWERS)}"
        )
    if not isinstance(prims, (list, tuple)) or len(prims) == 0:
        raise BasisError(f"gbasis[{atom_index}] shell {sym!r}: empty primitive list")
    out: list[tuple[float, float]] = []
    for prim in prims:
        if not (isinstance(prim, (list, tuple)) and len(prim) == 2):
            raise BasisError(
                f"gbasis[{atom_index}] shell {sym!r}: each primitive must be an "
                f"(exponent, coefficient) pair, got {prim!r}"
            )
        alpha, coef = float(prim[0]), float(prim[1])
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise BasisError(
                f"gbasis[{atom_index}] shell {sym!r}: exponent must be finite "
                f"and > 0, got {alpha!r}"
            )
        if not math.isfinite(coef):
            raise BasisError(
                f"gbasis[{atom_index}] shell {sym!r}: coefficient must be "
                f"finite, got {coef!r}"
            )
        out.append((alpha, coef))
    return sym, out


def flatten_gbasis(gbasis: object, atomcoords: object) -> BasisArrays:
    """Validate cclib-style inputs and build the flat kernel arrays.

    ``gbasis`` is cclib's per-atom list of (symbol, primitives) shells;
    ``atomcoords`` is an (n_atoms, 3) array in Angstrom (cclib convention;
    converted to bohr here with cclib's constant).
    """
    coords = np.asarray(atomcoords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise BasisError(
            f"atomcoords must have shape (n_atoms, 3), got {coords.shape}"
        )
    if not np.all(np.isfinite(coords)):
        raise BasisError("atomcoords contains non-finite values")
    if not isinstance(gbasis, (list, tuple)) or len(gbasis) == 0:
        raise BasisError("gbasis must be a non-empty per-atom list of shells")
    if len(gbasis) != coords.shape[0]:
        raise BasisError(
            f"len(gbasis) ({len(gbasis)}) != n_atoms ({coords.shape[0]})"
        )

    offsets = [0]
    powers_l: list[int] = []
    powers_m: list[int] = []
    powers_n: list[int] = []
    center_x: list[float] = []
    center_y: list[float] = []
    center_z: list[float] = []
    bf_norm: list[float] = []
    prim_alpha: list[float] = []
    prim_w: list[float] = []

    for atom_index, atom_shells in enumerate(gbasis):
        if not isinstance(atom_shells, (list, tuple)) or len(atom_shells) == 0:
            raise BasisError(f"gbasis[{atom_index}]: no shells")
        # cclib's pyquante2 bridge: convertor(atom, "Angstrom", "bohr").
        center = (
            float(coords[atom_index, 0]) * ANG2BOHR,
            float(coords[atom_index, 1]) * ANG2BOHR,
            float(coords[atom_index, 2]) * ANG2BOHR,
        )
        for shell in atom_shells:
            sym, prims = _validate_shell(atom_index, shell)
            for powers in SYM2POWERS[sym]:
                l, m, n = powers
                pnorms = [primitive_norm(alpha, l, m, n) for alpha, _ in prims]
                cnorm = _contracted_norm(prims, pnorms, powers, center)
                powers_l.append(l)
                powers_m.append(m)
                powers_n.append(n)
                center_x.append(center[0])
                center_y.append(center[1])
                center_z.append(center[2])
                bf_norm.append(cnorm)
                for (alpha, coef), pnorm in zip(prims, pnorms):
                    prim_alpha.append(alpha)
                    prim_w.append(coef * pnorm)
                offsets.append(len(prim_alpha))

    n_bf = len(bf_norm)
    if n_bf == 0:
        raise BasisError("gbasis expands to zero basis functions")
    return BasisArrays(
        offsets=np.asarray(offsets, dtype=np.int64),
        powers_l=np.asarray(powers_l, dtype=np.int32),
        powers_m=np.asarray(powers_m, dtype=np.int32),
        powers_n=np.asarray(powers_n, dtype=np.int32),
        center_x=np.asarray(center_x, dtype=np.float64),
        center_y=np.asarray(center_y, dtype=np.float64),
        center_z=np.asarray(center_z, dtype=np.float64),
        bf_norm=np.asarray(bf_norm, dtype=np.float64),
        prim_alpha=np.asarray(prim_alpha, dtype=np.float64),
        prim_w=np.asarray(prim_w, dtype=np.float64),
        n_bf=n_bf,
        n_prims=len(prim_alpha),
    )
