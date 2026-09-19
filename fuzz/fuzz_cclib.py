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

Parity is asserted at the documented tolerance (1e-10 relative, 1e-12
absolute; ``tests/test_gaussgrid_differential.py``). The relative term is
taken against the *conditioning* of the sum -- the sum of the absolute
values of the contributions -- rather than against the result alone, so an
input whose coefficients cancel by twenty orders of magnitude is held to the
same ulp-level standard as a physical one instead of failing on the
amplified rounding noise both backends legitimately produce there. For
well-conditioned inputs the two scales coincide.

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
from cclib_mojo._basis import SYM2POWERS, flatten_gbasis

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

# Documented tolerance (tests/test_gaussgrid_differential.py RTOL / ATOL).
RTOL = 1e-10
ATOL = 1e-12

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

# Magnitudes inside these windows never overflow or underflow the THO
# normalisation arithmetic for any supported shell. Exponents: the F-shell
# bound is alpha^4.5 < 1.8e308, i.e. alpha < ~1e68, and the tiny-alpha bound
# comes from (pi/(2 alpha))^1.5 and (4 alpha)^3 in the self-overlap.
# Coordinates: for D/F shells the product centre P = (a1 A + a2 B)/(a1 + a2)
# can round one ulp away from a centre of magnitude ~1e170 or more, and
# (P - A)^2 in the binomial prefactor then overflows. Anything outside is
# physically meaningless (basis sets span ~1e-3..1e6 bohr^-2; the observable
# universe is ~1e37 Angstrom across).
EXPONENT_SANE_MIN = 1e-100
EXPONENT_SANE_MAX = 1e60
COORDINATE_SANE_MAX = 1e100  # Angstrom


def _extreme_magnitude(case: Case) -> bool:
    exponents = (
        alpha
        for shells in case.gbasis
        for _, prims in shells
        for alpha, _ in prims
        if math.isfinite(alpha) and alpha > 0.0
    )
    coordinates = (c for atom in case.atomcoords for c in atom if math.isfinite(c))
    return any(not (EXPONENT_SANE_MIN <= alpha <= EXPONENT_SANE_MAX) for alpha in exponents) or any(
        abs(c) > COORDINATE_SANE_MAX for c in coordinates
    )


def _prefactor_overflows(basis, axes, coeff2d: np.ndarray) -> bool:
    """True when |c_b| * N_b * |w_p| * |x-cx|^l |y-cy|^m |z-cz|^n exceeds the
    double range at some grid point for some basis function and primitive.

    The kernel multiplies coefficient, contracted norm, polynomial and
    primitive weight before the Gaussian factors; the fallback multiplies
    polynomial and contraction (weight times exponential) first. When that
    prefactor is inf and the exponential has underflowed to 0 the two orders
    give inf * 0 = NaN versus 0. Unreachable for physical inputs (it needs
    |c * N * w * r^L| beyond ~1e308) -- the third symptom of the missing
    magnitude validation tracked in the issue above.
    """
    ax, ay, az = axes
    with np.errstate(all="ignore"):
        cmax = np.abs(coeff2d).max(axis=0)
        for b in range(basis.n_bf):
            o0, o1 = int(basis.offsets[b]), int(basis.offsets[b + 1])
            wmax = np.abs(basis.prim_w[o0:o1]).max()
            dx = np.abs(ax - basis.center_x[b]) ** basis.powers_l[b]
            dy = np.abs(ay - basis.center_y[b]) ** basis.powers_m[b]
            dz = np.abs(az - basis.center_z[b]) ** basis.powers_n[b]
            poly = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
            if not np.all(np.isfinite(cmax[b] * basis.bf_norm[b] * wmax * poly)):
                return True
    return False


ISSUE_EXTREME_MAGNITUDE = KnownIssue(
    key="extreme-magnitude",
    url="https://github.com/thyn-ai/mojo-kernels/issues/16",
    title=(
        "cclib-mojo: extreme magnitudes are not validated: exponents beyond ~1e68 or "
        "below ~1e-100 and coordinates beyond ~1e170 Angstrom raise OverflowError/"
        "ZeroDivisionError instead of BasisError, and a coefficient*norm*weight*r^L prefactor "
        "beyond the double range makes the kernel return inf*0 = NaN where the fallback returns 0"
    ),
    applies=_extreme_magnitude,
)

KNOWN_ISSUES: tuple[KnownIssue, ...] = (ISSUE_EXTREME_MAGNITUDE,)


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


def _conditioning(basis, axes, coeff2d: np.ndarray, mode: int) -> np.ndarray:
    """Per grid point: the sum of |contribution| that the result was built
    from -- the scale on which rounding differences between two summation
    orders live. Wavefunction: sum_b |c_b bf_b(r)|; density: sum_mo of that
    squared (>= the value itself, so this only relaxes under cancellation)."""
    n_bf = basis.n_bf
    magnitudes = np.zeros((n_bf, int(np.prod([a.shape[0] for a in axes]))))
    for b in range(n_bf):
        one_hot = np.zeros((1, n_bf))
        one_hot[0, b] = 1.0
        magnitudes[b] = np.abs(_reference.eval_grid(basis, *axes, one_hot, _reference.MODE_WAVEFUNCTION))
    scale = np.zeros(magnitudes.shape[1])
    for row in coeff2d:
        t = np.abs(row) @ magnitudes
        scale += t * t if mode == _reference.MODE_DENSITY else t
    return scale


def _close_mask(actual: np.ndarray, expected: np.ndarray, scale: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        tol = ATOL + RTOL * np.maximum(np.abs(expected), scale)
        return (
            (np.abs(actual - expected) <= tol)
            | (actual == expected)
            | (np.isnan(actual) & np.isnan(expected))
        )


def _assert_close(what: str, actual: np.ndarray, expected: np.ndarray, scale: np.ndarray, case: Case) -> None:
    ok = _close_mask(actual, expected, scale)
    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        raise Divergence(
            f"{what} differ at grid points {bad.tolist()}: actual={actual[bad]} expected={expected[bad]}"
            f"\n  case: {case.describe()}"
        )


def evaluate(case: Case) -> str | None:
    """Run one case through the public API and both backends; classify."""
    args = _call_args(case)
    defect = DEFECTS[case.defect]
    try:
        out, wf = _public_api(args)
    except (BasisError, GridError):
        return None  # documented, structured rejection
    except (OverflowError, ZeroDivisionError) as exc:
        if ISSUE_EXTREME_MAGNITUDE.applies(case):
            return ISSUE_EXTREME_MAGNITUDE.key
        raise Divergence(
            f"undocumented {type(exc).__name__} from the public API: {exc}\n  case: {case.describe()}"
        ) from exc
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
    basis = flatten_gbasis(args["gbasis"], args["atomcoords"])
    coeff2d = np.asarray(args["coeff"], dtype=np.float64)
    if args["mo_index"] is not None:
        coeff2d = coeff2d[args["mo_index"] : args["mo_index"] + 1]
        mode = _reference.MODE_WAVEFUNCTION
    else:
        mode = _reference.MODE_DENSITY
    axes = core._grid_axes(  # noqa: SLF001 -- the same axes the public API builds
        np.asarray(args["origin"], dtype=np.float64), np.asarray(args["step"], dtype=np.float64), shape
    )
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    scale = _conditioning(basis, axes, coeff2d, mode)
    flat = out.reshape(-1)
    outcome: str | None = None

    if cclib_mojo.native_available():
        native = _native.eval_grid(basis, *axes, coeff2d, mode)
        ok = _close_mask(native, fallback, scale)
        if not np.all(ok):
            # Known shape: every disagreement has a non-finite value on at
            # least one side, and the prefactor really does overflow here.
            non_finite = ~np.isfinite(native) | ~np.isfinite(fallback)
            if np.all(ok | non_finite) and _prefactor_overflows(basis, axes, coeff2d):
                outcome = ISSUE_EXTREME_MAGNITUDE.key
            else:
                _assert_close("native kernel vs fallback", native, fallback, scale, case)
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
        _assert_close("fallback vs PyQuante oracle", fallback, ref.reshape(-1), scale, case)
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
