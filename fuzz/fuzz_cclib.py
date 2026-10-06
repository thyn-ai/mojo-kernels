"""Differential fuzzing for cclib-mojo: gaussgrid kernel vs NumPy fallback vs
the PyQuante reference.

Every input decodes to one grid evaluation -- a cclib-style ``gbasis`` (one
or two atoms, S/P/D/F shells, one to three primitives each), Angstrom
geometry, an MO coefficient matrix and a small regular grid -- which is
then evaluated through the public API (``density_on_grid`` /
``wavefunction_on_grid``) and compared against:

* the native Mojo kernel and the vendored NumPy fallback, called directly
  on the same flattened basis (``_native.eval_grid`` / ``_reference.eval_grid``),
* the PyQuante 1.6.5 amplitude path transcribed in ``tests/pyquante1_oracle.py``,
  for in-domain inputs.

Parity is asserted against a condition-aware forward-error bound (see
"Comparison" below): every grid value is a sum of terms, and two backends
that evaluate the same sum in different orders can only differ by a small
multiple of machine epsilon times the *sum of the absolute values* of those
terms. The bound therefore scales with that magnitude rather than with the
result, so an input whose primitives or MO coefficients cancel by many
orders of magnitude is held to the same ulp-level standard as a physical
one instead of failing on the rounding residue both backends legitimately
leave there. For well-conditioned inputs the two scales coincide, and the
bound is far tighter than the documented 1e-10 relative tolerance of
``tests/test_gaussgrid_differential.py``.

Two flavours of numbers are generated: in-domain (exponents 0.03..100
bohr^-2, coefficients and geometry of order one, steps 0.05..1.5 Angstrom)
and raw IEEE doubles (anything: negative, NaN, infinite, subnormal, 1e300).
On top of that, one of seven structural defects may be injected (atom count
mismatch, wrong coefficient width, zero grid dimension, negative step,
out-of-range ``mo_index``, unsupported shell, empty primitive list); the
contract under test is that malformed input raises ``BasisError`` or
``GridError`` (both ``ValueError`` subclasses) and nothing else.

Run modes (see ``_harness.py``)::

    pixi run -e fuzz fuzz-cclib -- -max_total_time=60   # atheris, Linux x86_64
    pixi run fuzz-regression-cclib                       # replay fuzz/corpus/cclib
    python fuzz/fuzz_cclib.py --regression path/to/crash-file

Known divergences are listed in ``KNOWN_ISSUES`` below, each tied to an
open GitHub issue and reproduced by a ``known-issue-*.bin`` seed.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

try:
    import atheris
except ImportError:  # regression replay, or a platform without an atheris wheel
    atheris = None

import numpy as np

from _harness import (
    FUZZ_DIR,
    REPO_ROOT,
    ByteCursor,
    ByteWriter,
    Divergence,
    KnownIssue,
    main,
)
from _harness import replay_seed as _replay_seed

if atheris is not None:
    with atheris.instrument_imports(include=["cclib_mojo"]):
        import cclib_mojo
else:
    import cclib_mojo

from cclib_mojo import BasisError, GridError, _native, _reference, core
from cclib_mojo._basis import INTERMEDIATE_MAGNITUDE_MAX, SYM2POWERS, BasisArrays, flatten_gbasis

# The PyQuante 1.6.5 transcription used by the differential suite (a test
# oracle, not part of any package). Optional: the differential between the
# kernel and the fallback needs no oracle.
sys.path.insert(0, str(REPO_ROOT / "tests"))
try:
    import pyquante1_oracle as oracle
except ImportError:
    oracle = None

NAME = "cclib"
CORPUS_DIR = FUZZ_DIR / "corpus" / NAME

SHELLS = ("S", "P", "D", "F")
MAX_ATOMS = 2
MAX_SHELLS = 2
MAX_PRIMS = 3
MAX_MO = 3
MAX_DIM = 4

# Structural defects a caller could plausibly produce (index = defect id).
DEFECTS = (
    None,
    "atom-count-mismatch",
    "coefficient-width",
    "zero-grid-dimension",
    "negative-step",
    "mo-index-out-of-range",
    "unsupported-shell",
    "empty-primitive-list",
)

Shell = tuple[str, tuple[tuple[float, float], ...]]


@dataclass(frozen=True)
class Case:
    raw_numbers: bool  # exponents/coefficients/geometry/grid are raw doubles
    defect: int  # index into DEFECTS
    gbasis: tuple[tuple[Shell, ...], ...]
    atomcoords: tuple[tuple[float, float, float], ...]
    coeff: tuple[tuple[float, ...], ...]  # (n_mo, n_bf)
    mo_index: int | None  # None: density of every row; else that MO's amplitude
    origin: tuple[float, float, float]
    step: tuple[float, float, float]
    shape: tuple[int, int, int]

    @property
    def n_bf(self) -> int:
        return sum(len(SYM2POWERS[sym]) for shells in self.gbasis for sym, _ in shells)

    def describe(self) -> str:
        return (
            f"defect={DEFECTS[self.defect]!r} gbasis={self.gbasis!r} atomcoords={self.atomcoords!r} "
            f"coeff={self.coeff!r} mo_index={self.mo_index!r} origin={self.origin!r} "
            f"step={self.step!r} shape={self.shape!r}"
        )


def _in_domain_exponent(u: float) -> float:
    return 10.0 ** (u * 3.5 - 1.5)  # 0.03 .. 100 bohr^-2


def decode(data: bytes) -> Case:
    """Total decoder: any byte string is one grid evaluation (see ``ByteCursor``)."""
    cur = ByteCursor(data)
    flags = cur.u8()
    raw = bool(flags & 1)
    sprinkle_zeros = bool(flags & 2)
    wavefunction = bool(flags & 4)

    def number(lo: float, hi: float) -> float:
        return cur.f64() if raw else lo + cur.unit() * (hi - lo)

    gbasis: list[tuple[Shell, ...]] = []
    atomcoords: list[tuple[float, float, float]] = []
    for _ in range(cur.int_in(1, MAX_ATOMS)):
        shells: list[Shell] = []
        for _ in range(cur.int_in(1, MAX_SHELLS)):
            sym = cur.choice(SHELLS)
            prims = tuple(
                (
                    cur.f64() if raw else _in_domain_exponent(cur.unit()),
                    number(-1.0, 1.0),
                )
                for _ in range(cur.int_in(1, MAX_PRIMS))
            )
            shells.append((sym, prims))
        gbasis.append(tuple(shells))
        atomcoords.append((number(-2.0, 2.0), number(-2.0, 2.0), number(-2.0, 2.0)))

    n_bf = sum(len(SYM2POWERS[sym]) for shells in gbasis for sym, _ in shells)
    n_mo = cur.int_in(1, MAX_MO)
    coeff: list[tuple[float, ...]] = []
    for _ in range(n_mo):
        row = []
        for _ in range(n_bf):
            value = number(-1.0, 1.0)
            if sprinkle_zeros and cur.u8() < 64:
                value = 0.0  # cclib's "skip exactly-zero coefficients" rule
            row.append(value)
        coeff.append(tuple(row))
    mo_index = cur.int_in(0, n_mo - 1) if wavefunction else None

    origin = (number(-3.0, 3.0), number(-3.0, 3.0), number(-3.0, 3.0))
    step = (number(0.05, 1.5), number(0.05, 1.5), number(0.05, 1.5))
    shape = (cur.int_in(1, MAX_DIM), cur.int_in(1, MAX_DIM), cur.int_in(1, MAX_DIM))
    # About one input in four carries one structural defect; an exhausted
    # (all-zero) input never does, so short inputs still reach the kernels.
    roll = cur.u8()
    defect = 1 + (roll - 192) % (len(DEFECTS) - 1) if roll >= 192 else 0
    return Case(
        raw_numbers=raw,
        defect=defect,
        gbasis=tuple(gbasis),
        atomcoords=tuple(atomcoords),
        coeff=tuple(coeff),
        mo_index=mo_index,
        origin=origin,
        step=step,
        shape=shape,
    )


def encode(case: Case) -> bytes:
    """Exact inverse of ``decode`` for hand-crafted (minimised) seeds.

    Exact for raw numbers; in-domain values are quantised to the 16-bit grid
    of ``ByteCursor.unit`` (fine for reproducers, which use raw numbers).
    """
    w = ByteWriter()
    raw = case.raw_numbers
    flags = (1 if raw else 0) | (4 if case.mo_index is not None else 0)
    w.u8(flags)  # bit 2 (sprinkle zeros) is never set: zeros are written literally

    def number(v: float, lo: float, hi: float) -> None:
        if raw:
            w.f64(v)
        else:
            w.unit((v - lo) / (hi - lo))

    w.int_in(len(case.gbasis), 1, MAX_ATOMS)
    for shells, coords in zip(case.gbasis, case.atomcoords):
        w.int_in(len(shells), 1, MAX_SHELLS)
        for sym, prims in shells:
            w.choice(SHELLS.index(sym), SHELLS)
            w.int_in(len(prims), 1, MAX_PRIMS)
            for alpha, coef in prims:
                if raw:
                    w.f64(alpha)
                else:
                    w.unit((math.log10(alpha) + 1.5) / 3.5)
                number(coef, -1.0, 1.0)
        for c in coords:
            number(c, -2.0, 2.0)
    w.int_in(len(case.coeff), 1, MAX_MO)
    for row in case.coeff:
        if len(row) != case.n_bf:
            raise ValueError("coefficient rows must have n_bf entries; inject width defects via `defect`")
        for value in row:
            number(value, -1.0, 1.0)
    if case.mo_index is not None:
        w.int_in(case.mo_index, 0, len(case.coeff) - 1)
    for o in case.origin:
        number(o, -3.0, 3.0)
    for s in case.step:
        number(s, 0.05, 1.5)
    for dim in case.shape:
        w.int_in(dim, 1, MAX_DIM)
    w.u8(192 + case.defect - 1 if case.defect else 0)
    return w.bytes()


# ---------------------------------------------------------------------------
# Known divergences (each tied to an open issue and a known-issue-*.bin seed)
# ---------------------------------------------------------------------------

KNOWN_ISSUES: tuple[KnownIssue, ...] = ()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _call_args(case: Case) -> dict:
    """Public-API arguments with the case's structural defect applied."""
    gbasis = [[(sym, [list(p) for p in prims]) for sym, prims in shells] for shells in case.gbasis]
    atomcoords = np.array(case.atomcoords, dtype=np.float64)
    coeff = np.array(case.coeff, dtype=np.float64)
    step = list(case.step)
    shape = list(case.shape)
    mo_index = case.mo_index
    defect = DEFECTS[case.defect]
    if defect == "atom-count-mismatch":
        atomcoords = np.vstack([atomcoords, [[0.1, 0.2, 0.3]]])
    elif defect == "coefficient-width":
        coeff = np.hstack([coeff, np.ones((coeff.shape[0], 1))])
    elif defect == "zero-grid-dimension":
        shape[1] = 0
    elif defect == "negative-step":
        step[2] = -abs(step[2]) if step[2] == step[2] else -1.0
    elif defect == "mo-index-out-of-range":
        mo_index = coeff.shape[0]
    elif defect == "unsupported-shell":
        gbasis[0].append(("G", [[1.0, 1.0]]))
    elif defect == "empty-primitive-list":
        gbasis[-1].append(("S", []))
    return {
        "gbasis": gbasis,
        "atomcoords": atomcoords,
        "coeff": coeff,
        "origin": list(case.origin),
        "step": step,
        "shape": tuple(shape),
        "mo_index": mo_index,
    }


def _public_api(args: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate through the public API; also via wavefunction_on_grid for one MO."""
    out = cclib_mojo.density_on_grid(
        args["gbasis"],
        args["atomcoords"],
        args["coeff"],
        args["origin"],
        args["step"],
        args["shape"],
        mo_index=args["mo_index"],
    )
    wf = None
    if args["mo_index"] is not None:
        wf = cclib_mojo.wavefunction_on_grid(
            args["gbasis"],
            args["atomcoords"],
            args["coeff"][args["mo_index"]],
            args["origin"],
            args["step"],
            args["shape"],
        )
    return out, wf


def backend_inputs(args: dict) -> tuple[BasisArrays, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, int]:
    """What the public API hands either backend for ``args`` (an accepted
    input): the flattened basis, the grid axes in bohr, the evaluated MO
    rows and the mode."""
    basis = flatten_gbasis(args["gbasis"], args["atomcoords"])
    coeff2d = np.asarray(args["coeff"], dtype=np.float64)
    if args["mo_index"] is not None:
        coeff2d = coeff2d[args["mo_index"] : args["mo_index"] + 1]
        mode = _reference.MODE_WAVEFUNCTION
    else:
        mode = _reference.MODE_DENSITY
    axes = core._grid_axes(  # the same axes the public API builds
        np.asarray(args["origin"], dtype=np.float64),
        np.asarray(args["step"], dtype=np.float64),
        tuple(args["shape"]),
    )
    return basis, axes, coeff2d, mode


# --- The bound: condition-aware forward error -------------------------------
#
# Every grid value is built from terms, one per (MO row, basis function b
# with a non-zero coefficient c_b, primitive p of b):
#
#     t = c_b N_b (x-cx)^l (y-cy)^m (z-cz)^n w_p exp(-x_t),   x_t = alpha_p |r - c|^2
#
# An MO amplitude is psi = sum_t t; the density is sum over MO rows of psi^2.
# The backends evaluate the same exact sum in different orders: the kernel
# accumulates every term of a row in one fused multiply-add chain and splits
# exp(-x_t) into three per-axis factors; the fallback sums each contraction
# before scaling it and evaluates exp(-x_t) once. Their rounding errors
# therefore scale with S = sum_t |t|, not with |psi|. When opposite-sign
# primitives or MO coefficients cancel, |psi| is orders of magnitude below S
# and the rounding residue of the large terms legitimately differs between
# the two orders. Example: one S shell [(1.0, 1e8), (1.0, -1e8), (0.5, 1.0)].
# The fallback's two rounded products of the opposite pair cancel to exactly
# 0; the kernel's fused multiply-add keeps the rounding error of the first
# product (below u times the pair's magnitude), up to ~6e-9 of the value the
# third primitive leaves.
#
# First-order forward-error analysis (Higham, "Accuracy and Stability of
# Numerical Algorithms", 2nd ed., sec. 3.1 and 4.2) bounds each backend's
# error in psi by
#
#     (K - 1 + M) u S  +  sum_g |sum_{t in g} t| e(x_g)  +  K * (underflow loss per term)
#
# with u = 2^-53 the unit roundoff, K the number of terms summed for the row,
# M = PRODUCT_ROUNDINGS the roundings in one term's product chain and e(x)
# the relative error of that backend's exp(-x), including the rounding of its
# argument. The exp error enters per group g of terms that share a centre
# and an exponent: each backend computes their exp factor once, from the same
# operands, so the group's terms carry the same exp error and it scales with
# the group's net sum (an exactly opposite primitive pair cancels its exp
# error too). Summing the two backends' bounds, with eps = 2u = 2.2e-16:
#
#     |psi_A - psi_B| <= (K + M) eps S  +  sum_g |sum_{t in g} t| (e_A(x_g) + e_B(x_g))
#                        +  K * UNDERFLOW_PER_TERM
#
# The (K + M) eps term is the summation and product rounding: a small
# multiple of machine epsilon times the number of terms. Each side's exp
# error is modelled as e(x) <= floor + slope * x (ExpError below). In density
# mode psi_A^2 - psi_B^2 = (psi_A - psi_B)(psi_A + psi_B) with
# |psi_A| + |psi_B| <= 2 S + D_psi, plus the rounding of the squares and of the
# sum over MO rows. term_scales computes S, the group sums and K by repeating
# the fallback's evaluation term by term.
#
# Apart from the factor 2 on the kernel's measured exp slope and the 3-ulp
# allowance for NumPy's exp (KERNEL_EXP, LIBM_EXP), every constant below is a
# rounding count or a measurement. A defect as small as a 1e-6 relative
# error in one primitive coefficient exceeds the bound by orders of
# magnitude on well-conditioned inputs (tests/test_fuzz_regression_cclib.py
# checks that).

UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2  # u = 2^-53
# Roundings in one term's product chain on either backend, counted for
# L = l + m + n <= 3. Kernel: c*N_c, *x^l, *y^m, *w_p, *exp_x, *exp_y,
# exp_z*z^n, the product inside the accumulating FMA, plus at most 2 inside
# the powers = 10. Fallback: c*N_c, *(poly*contraction), poly*contraction,
# two products in poly, w_p*exp, at most 2 in the powers = 8. The PyQuante
# oracle: norm*coef, three products with the powers, *exp, *N_c, *c, at most
# 2 in the powers = 9.
PRODUCT_ROUNDINGS = 10


@dataclass(frozen=True)
class ExpError:
    """One side's relative error in exp(-x), x = alpha_p |r - c|^2 >= 0,
    including the rounding of x itself: e(x) <= floor + slope * x."""

    floor: float
    slope: float


# Python's math.exp (the platform libm) is faithful, <= 1 ulp = 2u; NumPy
# may dispatch float64 exp to its own AVX-512 implementation, which is
# accurate to a few ulp rather than faithful, so the floor allows 3 ulp. The
# argument alpha * (dx^2 + dy^2 + dz^2) carries <= 5u relative rounding (a
# square taken with pow is itself only faithful), i.e. <= 5u x absolute.
LIBM_EXP = ExpError(floor=6 * UNIT_ROUNDOFF, slope=5 * UNIT_ROUNDOFF)
# The kernel's std.math.exp is not correctly rounded. Measured through the
# kernel on the pinned toolchain (Mojo 1.1.0, macOS arm64) over 2.5M
# arguments in [0, 708.3]: relative error <= 2.83e-13 * round(x / ln 2) + 4u
# (the 4u includes the measurement's own roundings), the signature of a range
# reduction whose ln 2 is 2.8e-13 off. Since round(x / ln 2) is 0 below
# ln 2 / 2 and at most 2 x / ln 2 above it, one factor's error is
# <= 8.2e-13 x + 4u; the kernel multiplies three per-axis factors whose
# arguments sum to x (and rounds those arguments, 2u x): <= 8.2e-13 x + 12u.
# The slope used is 1.6e-12, twice that worst case (the large-x slope
# actually measured is 4.08e-13), because the Linux x86_64 build the nightly
# fuzzes could not be measured the same way. Below x = ln 2 / 2 the kernel's
# error is a few u, so near the centres the bound stays at the summation term.
KERNEL_EXP = ExpError(floor=16 * UNIT_ROUNDOFF, slope=1.6e-12)
# Underflow. The kernel's exp flushes results below ~1.6e-308 to 0 (from
# x ~ 708.76 on; NumPy returns subnormals down to x ~ 745), and a product
# chain that passes through the subnormal range loses up to 2^-1075 per
# rounding. Either loss is multiplied by at most the term's other factors,
# which validation caps at INTERMEDIATE_MAGNITUDE_MAX (cclib_mojo.core), so
# one term on one side loses less than DBL_MIN * 1e140 ~ 2.2e-168; two sides.
UNDERFLOW_PER_TERM = 2 * np.finfo(np.float64).tiny * INTERMEDIATE_MAGNITUDE_MAX
# Density mode: the squares and the sum over MO rows lose up to 2^-1075 per
# rounding when they fall into the subnormal range.
SUBNORMAL_QUANTUM = np.finfo(np.float64).smallest_subnormal


@dataclass(frozen=True)
class TermScales:
    """What each MO row's rounding error scales with, per grid point."""

    magnitude: np.ndarray  # (n_mo, n_points): S = sum_t |t|
    exp_group: np.ndarray  # (n_mo, n_points): sum_g |sum_{t in g} t|
    exp_group_weighted: np.ndarray  # (n_mo, n_points): sum_g |sum_{t in g} t| x_g
    n_terms: np.ndarray  # (n_mo,): K, the (function, primitive) terms summed for the row


def term_scales(basis: BasisArrays, axes, coeff2d: np.ndarray) -> TermScales:
    """The inputs of the bound, from the fallback's evaluation term by term.

    Each term is formed in the fallback's order, ``(c N_c) (poly (w_p
    exp(-x)))``, and only for inputs the public API accepted, so every
    partial product stays under ``INTERMEDIATE_MAGNITUDE_MAX``. Terms are
    grouped by (centre, exponent): their exp factor is the same float in
    either backend.
    """
    ax, ay, az = axes
    grid_shape = (ax.shape[0], ay.shape[0], az.shape[0])
    n_mo = coeff2d.shape[0]
    magnitude = np.zeros((n_mo, *grid_shape), dtype=np.float64)
    n_terms = np.zeros(n_mo, dtype=np.int64)
    # Per MO row: (centre, exponent) -> [signed sum of the group's terms, x_g].
    groups: list[dict[tuple[float, float, float, float], list[np.ndarray]]] = [{} for _ in range(n_mo)]
    for b in range(basis.n_bf):
        rows = np.flatnonzero(coeff2d[:, b] != 0.0)  # both backends skip exact zeros
        if rows.size == 0:
            continue
        centre = (float(basis.center_x[b]), float(basis.center_y[b]), float(basis.center_z[b]))
        dx3 = (ax - centre[0])[:, None, None]
        dy3 = (ay - centre[1])[None, :, None]
        dz3 = (az - centre[2])[None, None, :]
        r2 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3
        poly = (
            np.power(dx3, basis.powers_l[b])
            * np.power(dy3, basis.powers_m[b])
            * np.power(dz3, basis.powers_n[b])
        )
        o0, o1 = int(basis.offsets[b]), int(basis.offsets[b + 1])
        for p in range(o0, o1):
            alpha = float(basis.prim_alpha[p])
            x = alpha * r2
            weighted_poly = poly * (basis.prim_w[p] * np.exp(-x))
            for mo in rows:
                term = (coeff2d[mo, b] * basis.bf_norm[b]) * weighted_poly
                magnitude[mo] += np.abs(term)
                group = groups[mo].setdefault((*centre, alpha), [np.zeros(grid_shape), x])
                group[0] += term
            n_terms[rows] += 1
    exp_group = np.zeros_like(magnitude)
    exp_group_weighted = np.zeros_like(magnitude)
    for mo in range(n_mo):
        for net, x in groups[mo].values():
            # net is the group sum as rounded here; the difference from the
            # exact sum (< K u S) times e(x) < 3e-9 is far inside the
            # (K + M) eps S term. net != 0 only where exp(-x) != 0, so x < 746.
            exp_group[mo] += np.abs(net)
            exp_group_weighted[mo] += np.abs(net) * x
    n_points = int(np.prod(grid_shape))
    return TermScales(
        magnitude=magnitude.reshape(n_mo, n_points),
        exp_group=exp_group.reshape(n_mo, n_points),
        exp_group_weighted=exp_group_weighted.reshape(n_mo, n_points),
        n_terms=n_terms,
    )


def amplitude_tolerance(scales: TermScales, exp_a: ExpError, exp_b: ExpError) -> np.ndarray:
    """(n_mo, n_points): the bound on |psi_A - psi_B| for each MO row."""
    eps = np.finfo(np.float64).eps
    k = scales.n_terms[:, None].astype(np.float64)
    return (
        (k + PRODUCT_ROUNDINGS) * eps * scales.magnitude
        + (exp_a.floor + exp_b.floor) * scales.exp_group
        + (exp_a.slope + exp_b.slope) * scales.exp_group_weighted
        + k * UNDERFLOW_PER_TERM
    )


def tolerance(scales: TermScales, mode: int, exp_a: ExpError, exp_b: ExpError) -> np.ndarray:
    """(n_points,): the bound on |A - B| for the evaluated quantity, A and B
    evaluating exp(-x) with the errors ``exp_a`` and ``exp_b``."""
    psi_tol = amplitude_tolerance(scales, exp_a, exp_b)
    if mode == _reference.MODE_WAVEFUNCTION:
        return psi_tol[0]
    eps = np.finfo(np.float64).eps
    s = scales.magnitude
    n_mo = s.shape[0]
    return (
        np.sum(psi_tol * (2.0 * s + psi_tol), axis=0)
        + (n_mo + 1) * eps * np.sum((s + psi_tol) ** 2, axis=0)
        + 2 * (n_mo + 1) * SUBNORMAL_QUANTUM
    )


def _close_mask(actual: np.ndarray, expected: np.ndarray, tol: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (
            (np.abs(actual - expected) <= tol)
            | (actual == expected)
            | (np.isnan(actual) & np.isnan(expected))
        )


def assert_close(what: str, actual: np.ndarray, expected: np.ndarray, tol: np.ndarray, case: Case | None) -> None:
    """Raise ``Divergence`` naming every grid point outside ``tol``."""
    ok = _close_mask(actual, expected, tol)
    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        described = case.describe() if case is not None else "(direct comparison)"
        raise Divergence(
            f"{what} differ at grid points {bad.tolist()}: actual={actual[bad]} expected={expected[bad]} "
            f"|diff|={np.abs(actual[bad] - expected[bad])} bound={tol[bad]}"
            f"\n  case: {described}"
        )


def evaluate(case: Case) -> str | None:
    """Run one case through the public API and both backends; classify."""
    args = _call_args(case)
    defect = DEFECTS[case.defect]
    try:
        out, wf = _public_api(args)
    except (BasisError, GridError):
        return None  # documented, structured rejection
    if defect is not None:
        raise Divergence(
            f"malformed input ({defect}) was accepted instead of raising\n  case: {case.describe()}"
        )

    shape = args["shape"]
    for label, arr in (("density_on_grid", out), ("wavefunction_on_grid", wf)):
        if arr is None:
            continue
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float64 or arr.shape != shape:
            raise Divergence(
                f"{label}: expected float64{shape}, got {type(arr).__name__} "
                f"{getattr(arr, 'dtype', '')}{getattr(arr, 'shape', '')}"
            )
        if not arr.flags.c_contiguous:
            raise Divergence(f"{label}: result is not C-contiguous (cclib Volume.data layout)")
    if wf is not None and not np.array_equal(out, wf, equal_nan=True):
        raise Divergence(
            f"density_on_grid(mo_index={args['mo_index']}) != wavefunction_on_grid on the "
            f"same MO\n  case: {case.describe()}"
        )

    # Both backends on the identical flattened basis and axes.
    basis, axes, coeff2d, mode = backend_inputs(args)
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    scales = term_scales(basis, axes, coeff2d)
    flat = out.reshape(-1)
    outcome: str | None = None

    if cclib_mojo.native_available():
        native = _native.eval_grid(basis, *axes, coeff2d, mode)
        assert_close(
            "native kernel vs fallback",
            native,
            fallback,
            tolerance(scales, mode, KERNEL_EXP, LIBM_EXP),
            case,
        )
        if not np.array_equal(flat, native, equal_nan=True):
            raise Divergence(
                "public API result is not the native kernel's output bit-for-bit\n  case: "
                + case.describe()
            )
    elif not np.array_equal(flat, fallback, equal_nan=True):
        raise Divergence(
            "public API result is not the fallback's output bit-for-bit\n  case: " + case.describe()
        )

    if oracle is not None and not case.raw_numbers:
        # The PyQuante transcription is only meaningful (and only robust)
        # for physically sensible inputs.
        if mode == _reference.MODE_WAVEFUNCTION:
            ref = oracle.wavefunction(
                args["gbasis"], args["atomcoords"], coeff2d[0],
                tuple(args["origin"]), tuple(args["step"]), shape,
            )
        else:
            ref = oracle.electrondensity_spin(
                args["gbasis"], args["atomcoords"], coeff2d,
                tuple(args["origin"]), tuple(args["step"]), shape,
            )
        assert_close(
            "fallback vs PyQuante oracle",
            fallback,
            ref.reshape(-1),
            tolerance(scales, mode, LIBM_EXP, LIBM_EXP),
            case,
        )
    return outcome


def test_one_input(data: bytes) -> str | None:
    """Fuzz target: decode, evaluate, and classify the outcome.

    Returns None (parity / documented rejection), a known-issue key, or
    raises ``Divergence`` for anything else -- including any exception type
    the package does not document for these inputs.
    """
    case = decode(data)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # overflow/invalid warnings are the IEEE behaviour under test
        try:
            return evaluate(case)
        except Divergence:
            raise
        except Exception as exc:
            raise Divergence(
                f"undocumented {type(exc).__name__}: {exc}\n  case: {case.describe()}"
            ) from exc


def native_available() -> bool:
    return cclib_mojo.native_available()


def replay_seed(seed: Path) -> str | None:
    """Replay one corpus file (pytest entry point).

    ``known-issue-*`` seeds must reproduce their issue whenever the native
    kernel is loadable; without it a native-vs-fallback divergence cannot
    reproduce, so the seed is only required to replay without a divergence.
    """
    return _replay_seed(seed, test_one_input, KNOWN_ISSUES, strict=native_available())


def _banner() -> str:
    info = cclib_mojo.backend_info()
    backend = (
        f"native kernel: {info['native_source']}"
        if info["native_available"]
        else f"native kernel UNAVAILABLE ({info['error']}); only fallback-vs-oracle checks are live"
    )
    ref = "PyQuante oracle: tests/pyquante1_oracle.py" if oracle else "PyQuante oracle: not importable"
    return f"{backend}\n{ref}"


def _require_native() -> str | None:
    info = cclib_mojo.backend_info()
    return None if info["native_available"] else str(info["error"])


if __name__ == "__main__":
    sys.exit(
        main(
            NAME,
            test_one_input,
            KNOWN_ISSUES,
            CORPUS_DIR,
            banner=_banner(),
            require_native=_require_native,
        )
    )
