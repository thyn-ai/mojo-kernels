"""Replay the cclib fuzzing seed corpus inside the normal differential suite.

Each file under ``fuzz/corpus/cclib/`` is one scenario for
``fuzz/fuzz_cclib.py`` (see that module for what is compared). Run by
``scripts/test_all_cclib.sh`` on both backends: the native pass is the real
differential and requires every ``known-issue-*`` reproducer to still
reproduce its issue; the forced-fallback pass checks the fallback against
the PyQuante oracle, the public-API contracts and the validation contract
on the same seeds (a native-vs-fallback divergence cannot reproduce
without the kernel).
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import cclib_mojo
import fuzz_cclib as harness  # fuzz/ is on sys.path via tests/conftest.py
import gaussgrid_fixtures as fx
from _harness import Divergence, expected_issue_key
from cclib_mojo import _native, _reference, core
from cclib_mojo._basis import INTERMEDIATE_MAGNITUDE_MAX, flatten_gbasis

SEEDS = sorted(harness.CORPUS_DIR.glob("*.bin"))
CANCELLING_SEEDS = sorted(harness.CORPUS_DIR.glob("regression-cancelling-contraction-*.bin"))

requires_native = pytest.mark.skipif(
    not cclib_mojo.native_available(),
    reason="the native-vs-fallback residue needs the native kernel",
)


def test_seed_corpus_is_present():
    assert SEEDS, f"no seeds under {harness.CORPUS_DIR}; run fuzz/seed_corpus.py cclib"
    for issue in harness.KNOWN_ISSUES:
        assert any(expected_issue_key(s) == issue.key for s in SEEDS), (
            f"known issue {issue.key!r} has no reproducer seed"
        )


@pytest.mark.parametrize("seed", SEEDS, ids=lambda p: p.name)
def test_seed_replays_without_divergence(seed: Path):
    harness.replay_seed(seed)


def test_in_domain_boundary_values_are_accepted():
    """Values exactly at the validation domain boundary must not reject."""
    for alpha in (1e-100, 1e60):
        cclib_mojo.density_on_grid(
            [[("S", [(alpha, 1.0)])]],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0]]),
            origin=(-1.0, -1.0, -1.0),
            step=(1.0, 1.0, 1.0),
            shape=(2, 2, 2),
        )
    cclib_mojo.density_on_grid(
        [[("S", [(1.0, 1.0)])]],
        np.array([[1e100, 0.0, 0.0]]),
        np.array([[1.0]]),
        origin=(-1.0, 0.0, 0.0),
        step=(1.0, 1.0, 1.0),
        shape=(2, 2, 2),
    )


def test_in_domain_raw_seeds_do_not_overflow():
    """The random in-domain generator must never hit an arithmetic exception.

    Issue #16's known issue is removed because validation now rejects the
    out-of-domain class; this pins that in-domain cases still evaluate cleanly.
    """
    case = harness.Case(
        raw_numbers=False,
        defect=0,
        gbasis=(
            (
                ("S", ((1.0, 0.5), (10.0, 0.5))),
                ("P", ((0.5, 1.0),)),
            ),
        ),
        atomcoords=((0.0, 0.0, 0.0),),
        coeff=((1.0, 0.5, -0.3, 0.2),),
        mo_index=None,
        origin=(-1.0, -1.0, -1.0),
        step=(0.5, 0.5, 0.5),
        shape=(3, 3, 3),
    )
    assert harness.test_one_input(harness.encode(case)) is None


# The unit the nightly fuzz run 37260359272 (job "atheris differential
# fuzzing (cclib)") stopped on, verbatim from its log: exponents and
# coordinates inside the #16 windows, but |c| N_c reaches ~1e350 (py) and
# ~1e333 (pz), so the two evaluation orders overflow at different stages
# (kernel -inf, fallback -inf x 0 = NaN). It must be a documented rejection.
NIGHTLY_INTERMEDIATE_OVERFLOW = harness.Case(
    raw_numbers=True,
    defect=0,
    gbasis=(
        (
            (
                "P",
                (
                    (2.21420213728226e-52, 2.21420213728226e-52),
                    (2.2299208288013415e-52, 2.21420213728226e-52),
                ),
            ),
            (
                "P",
                (
                    (1.398043286095683e-76, 1.398043286095289e-76),
                    (1.398043286095289e-76, 1.398043286095289e-76),
                ),
            ),
        ),
    ),
    atomcoords=((4.0133397585694736e-57, 2.215018708925204e-52, 2.09414631903e-311),),
    coeff=(
        (
            1.3980433366019128e-76,
            4.0133397585694736e-57,
            2.215018708925204e-52,
            2.09414631903e-311,
            1.2677189948137588e275,
            -3.1594776358597076e257,
        ),
    ),
    mo_index=0,
    origin=(-6.48769282486076e-62, 1.2989442504e-314, 5.627320053137508e-249),
    step=(4.24329425326203e-274, 2.524356568153254e-29, 1.444878500878187e-309),
    shape=(1, 2, 2),
)


def test_nightly_intermediate_overflow_unit_is_rejected():
    args = harness._call_args(NIGHTLY_INTERMEDIATE_OVERFLOW)
    with pytest.raises(cclib_mojo.GridError, match=r"basis function 4 .* up to ~1e350"):
        harness._public_api(args)
    assert harness.test_one_input(harness.encode(NIGHTLY_INTERMEDIATE_OVERFLOW)) is None


# ---------------------------------------------------------------------------
# The comparator (fuzz_cclib.py, "Comparison"): condition-aware, not permissive
# ---------------------------------------------------------------------------

# Bound used by fuzz_cclib.evaluate for native kernel vs fallback.
EXP_PAIR = (harness.KERNEL_EXP, harness.LIBM_EXP)
# The documented result-relative tolerance (tests/test_gaussgrid_differential.py).
RESULT_RTOL, RESULT_ATOL = 1e-10, 1e-12


def test_cancelling_contraction_seeds_are_present():
    assert len(CANCELLING_SEEDS) == 3, "run fuzz/seed_corpus.py cclib"


@requires_native
@pytest.mark.parametrize("seed", CANCELLING_SEEDS, ids=lambda p: p.name)
def test_cancelling_contractions_need_the_condition_aware_bound(seed: Path):
    """The backends' residues differ by more than the documented
    result-relative tolerance, and stay inside the bound built from the
    terms' magnitudes (the harness's former comparator flagged all three)."""
    case = harness.decode(seed.read_bytes())
    basis, axes, coeff2d, mode = harness.backend_inputs(harness._call_args(case))
    native = _native.eval_grid(basis, *axes, coeff2d, mode)
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    diff = np.abs(native - fallback)
    assert np.any(diff > RESULT_RTOL * np.abs(fallback) + RESULT_ATOL)
    tol = harness.tolerance(harness.term_scales(basis, axes, coeff2d), mode, *EXP_PAIR)
    assert np.all(diff <= tol)
    assert harness.test_one_input(seed.read_bytes()) is None


def _fixture_args(fixture: str, n_mo: int, mo_index: int | None) -> dict:
    gbasis, atomcoords = getattr(fx, fixture)()
    n_bf = flatten_gbasis(gbasis, atomcoords).n_bf
    return {
        "gbasis": gbasis,
        "atomcoords": np.asarray(atomcoords, dtype=np.float64),
        "coeff": fx.seeded_coeffs(seed=61, n_mo=n_mo, n_bf=n_bf),
        "origin": [-2.0, -2.0, -2.0],
        "step": [0.8, 0.8, 0.8],
        "shape": (6, 6, 6),
        "mo_index": mo_index,
    }


SELF_TEST_INPUTS = {
    "h2o-sto3g-density": lambda: _fixture_args("h2o_sto3g", n_mo=3, mo_index=None),
    "carbon-spdf-amplitude": lambda: _fixture_args("carbon_sto3g_df", n_mo=2, mo_index=1),
    "cancelling-contraction": lambda: harness._call_args(
        harness.decode((harness.CORPUS_DIR / "regression-cancelling-contraction-1.bin").read_bytes())
    ),
}


def _with_weight(basis, prim: int, weight: float):
    w = basis.prim_w.copy()
    w[prim] = weight
    return dataclasses.replace(basis, prim_w=w)


def _significant_primitives(basis, axes, coeff2d) -> list[int]:
    """Primitives whose terms reach 1e-3 of the summation scale somewhere."""
    total = harness.term_scales(basis, axes, coeff2d).magnitude.sum(axis=0)
    out = []
    for p in range(basis.n_prims):
        only_p = dataclasses.replace(basis, prim_w=np.where(np.arange(basis.n_prims) == p, basis.prim_w, 0.0))
        share = harness.term_scales(only_p, axes, coeff2d).magnitude.sum(axis=0)
        if np.any(share >= 1e-3 * total):
            out.append(p)
    return out


@pytest.mark.parametrize("defect", ["coefficient-off-by-1e-6", "term-dropped"])
@pytest.mark.parametrize("name", sorted(SELF_TEST_INPUTS))
def test_comparator_flags_an_injected_defect(name: str, defect: str):
    """Harness self-test: a copy of the fallback with one primitive weight off
    by 1e-6 relative, or with one term dropped, must fail the comparison the
    harness applies to native vs fallback -- for every primitive whose terms
    reach 1e-3 of the summation scale anywhere on the grid. The unmodified
    fallback must pass the same comparison."""
    basis, axes, coeff2d, mode = harness.backend_inputs(SELF_TEST_INPUTS[name]())
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    actual = _native.eval_grid(basis, *axes, coeff2d, mode) if cclib_mojo.native_available() else fallback
    tol = harness.tolerance(harness.term_scales(basis, axes, coeff2d), mode, *EXP_PAIR)
    harness.assert_close("unmodified fallback", actual, fallback, tol, None)

    primitives = _significant_primitives(basis, axes, coeff2d)
    assert primitives, "the self-test needs at least one significant primitive"
    for p in primitives:
        weight = basis.prim_w[p] * (1.0 + 1e-6) if defect == "coefficient-off-by-1e-6" else 0.0
        defective = _reference.eval_grid(_with_weight(basis, p, weight), *axes, coeff2d, mode)
        with pytest.raises(Divergence, match="differ at grid points"):
            harness.assert_close(f"{defect} at primitive {p}", actual, defective, tol, None)


def test_cancelling_contraction_defects_are_flagged_despite_the_wider_bound():
    """At the centre of the cancelling seed S is ~3e8 x |psi|, so the bound is
    several 1e-7 of the result there -- wider than the documented 1e-10 -- yet 1e-6
    on a cancelling weight or the loss of the surviving primitive are still
    divergences."""
    args = SELF_TEST_INPUTS["cancelling-contraction"]()
    basis, axes, coeff2d, mode = harness.backend_inputs(args)
    assert basis.prim_w[0] == -basis.prim_w[1]  # the exactly opposite pair
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    tol = harness.tolerance(harness.term_scales(basis, axes, coeff2d), mode, *EXP_PAIR)
    assert np.max(tol / np.abs(fallback)) > 1e-7  # the cancellation is real
    for p, weight in ((0, basis.prim_w[0] * (1.0 + 1e-6)), (2, 0.0)):
        defective = _reference.eval_grid(_with_weight(basis, p, weight), *axes, coeff2d, mode)
        with pytest.raises(Divergence):
            harness.assert_close("defective fallback", fallback, defective, tol, None)


# ---------------------------------------------------------------------------
# Defects on cancelling inputs, in density and amplitude mode
# ---------------------------------------------------------------------------

# Each cancelling seed in both modes: (seed number, mo_index), where None is
# the density of every MO row and an index is that row's amplitude.
CANCELLING_VARIANTS = {
    "1-density": (1, None),
    "1-amplitude": (1, 0),
    "2-density": (2, None),
    "2-amplitude-mo0": (2, 0),
    "2-amplitude-mo1": (2, 1),
    "3-density": (3, None),
    "3-amplitude": (3, 0),
}

CANCELLING_DEFECTS = (
    "primitive-sign-flipped",
    "norm-times-sqrt3",
    "angular-power-plus-1",
    "angular-power-minus-1",
    "primitive-dropped",
    "coefficient-off-by-1e-6",
)


def _cancelling_case(variant: str) -> harness.Case:
    number, mo_index = CANCELLING_VARIANTS[variant]
    seed = harness.CORPUS_DIR / f"regression-cancelling-contraction-{number}.bin"
    return dataclasses.replace(harness.decode(seed.read_bytes()), mo_index=mo_index)


def _with_norm(basis, function: int, norm: float):
    n = basis.bf_norm.copy()
    n[function] = norm
    return dataclasses.replace(basis, bf_norm=n)


def _with_power_step(basis, function: int, axis: int, step: int):
    powers = [basis.powers_l.copy(), basis.powers_m.copy(), basis.powers_n.copy()]
    powers[axis][function] += step
    return dataclasses.replace(basis, powers_l=powers[0], powers_m=powers[1], powers_n=powers[2])


def _mutants(basis, coeff2d: np.ndarray, defect: str) -> list[tuple[str, object]]:
    """Every placement of ``defect`` on a function with a non-zero coefficient."""
    out = []
    for b in (int(b) for b in np.flatnonzero(np.any(coeff2d != 0.0, axis=0))):
        prims = range(int(basis.offsets[b]), int(basis.offsets[b + 1]))
        powers = (basis.powers_l[b], basis.powers_m[b], basis.powers_n[b])
        if defect == "primitive-sign-flipped":
            out += [(f"primitive {p}", _with_weight(basis, p, -basis.prim_w[p])) for p in prims]
        elif defect == "primitive-dropped":
            out += [(f"primitive {p}", _with_weight(basis, p, 0.0)) for p in prims]
        elif defect == "coefficient-off-by-1e-6":
            out += [(f"primitive {p}", _with_weight(basis, p, basis.prim_w[p] * (1.0 + 1e-6))) for p in prims]
        elif defect == "norm-times-sqrt3":
            out.append((f"function {b}", _with_norm(basis, b, basis.bf_norm[b] * math.sqrt(3.0))))
        elif defect in ("angular-power-plus-1", "angular-power-minus-1"):
            step = 1 if defect == "angular-power-plus-1" else -1
            out += [
                (f"function {b} axis {axis}", _with_power_step(basis, b, axis, step))
                for axis in range(3)
                if powers[axis] + step >= 0
            ]
        else:
            raise ValueError(f"unknown defect {defect!r}")
    return out


def _flagged(actual: np.ndarray, expected: np.ndarray, tol: np.ndarray) -> bool:
    try:
        harness.assert_close("defect", actual, expected, tol, None)
    except Divergence:
        return True
    return False


def _cancelling_defect_params():
    for variant, (number, _) in CANCELLING_VARIANTS.items():
        for defect in CANCELLING_DEFECTS:
            if number == 1 and defect == "angular-power-minus-1":
                continue  # seed 1 is one S shell: there is no power to lower
            yield pytest.param(variant, defect, id=f"{variant}-{defect}")


@pytest.mark.parametrize("variant,defect", list(_cancelling_defect_params()))
def test_comparator_flags_defects_on_cancelling_inputs(variant: str, defect: str):
    """Harness self-test on inputs whose primitives cancel (S >> |psi|).

    Every placement of the defect that changes the fallback's result at all
    must fail the comparison the harness applies to native vs fallback, both
    when the defect sits in the kernel's arrays and when it sits in the
    fallback (whose amplitudes then also centre the density bound). The
    former density bound, which bounded |psi_A + psi_B| by 2 S, let 10 of the
    17 density-mode cases here pass."""
    basis, axes, coeff2d, mode = harness.backend_inputs(harness._call_args(_cancelling_case(variant)))
    backend = _native if cclib_mojo.native_available() else _reference
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    actual = backend.eval_grid(basis, *axes, coeff2d, mode)
    scales = harness.term_scales(basis, axes, coeff2d)
    tol = harness.tolerance(scales, mode, *EXP_PAIR)
    harness.assert_close("unmodified backends", actual, fallback, tol, None)

    checked, missed = 0, []
    for site, mutant in _mutants(basis, coeff2d, defect):
        defective_fallback = _reference.eval_grid(mutant, *axes, coeff2d, mode)
        if np.array_equal(defective_fallback, fallback):
            # Invisible in the result itself: flipping the sign of the only
            # surviving primitive of seed 1 flips psi, and a one-row density
            # is psi^2. No comparator can see it; it is not a defect there.
            continue
        checked += 1
        defective_backend = backend.eval_grid(mutant, *axes, coeff2d, mode)
        if not _flagged(defective_backend, fallback, tol):
            missed.append(f"{site} (kernel side)")
        defective_scales = dataclasses.replace(
            scales, amplitude=harness.fallback_amplitudes(mutant, axes, coeff2d)
        )
        defective_tol = harness.tolerance(defective_scales, mode, *EXP_PAIR)
        if not _flagged(actual, defective_fallback, defective_tol):
            missed.append(f"{site} (fallback side)")
    assert checked, f"{defect} changes nothing on {variant}"
    assert not missed, f"{defect} on {variant} passed the comparator at {missed}"


@pytest.mark.parametrize("number", [1, 2, 3])
def test_density_bound_is_centred_on_the_fallbacks_amplitudes(number: int):
    """The density bound is sum_mo delta_mo (2 |psi_mo| + delta_mo) plus a
    few ulps of the density, with psi_mo the amplitudes the fallback squares:

    - those amplitudes reproduce the fallback's density bit for bit;
    - the bound stays a small fraction of the density even where S >> |psi|
      (the former 2 S-based bound was 1.9 on seed 2, at a peak density of 1.0).
    """
    case = _cancelling_case(f"{number}-density")
    basis, axes, coeff2d, mode = harness.backend_inputs(harness._call_args(case))
    assert mode == _reference.MODE_DENSITY
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    scales = harness.term_scales(basis, axes, coeff2d)
    squared = np.zeros_like(fallback)
    for psi in scales.amplitude:
        squared += psi * psi
    assert np.array_equal(squared, fallback)
    tol = harness.tolerance(scales, mode, *EXP_PAIR)
    assert np.max(scales.magnitude) > 1e5 * np.max(np.abs(scales.amplitude))  # S >> |psi|: it cancels
    assert np.max(tol) < 1e-5 * np.max(fallback)


@pytest.mark.parametrize("variant", sorted(CANCELLING_VARIANTS))
def test_cancelling_seeds_replay_clean_in_both_modes(variant: str):
    assert harness.test_one_input(harness.encode(_cancelling_case(variant))) is None


@pytest.mark.parametrize("mo_index", [None, 0], ids=["density", "amplitude"])
def test_inputs_just_below_the_intermediate_bound_pass_the_comparator(mo_index: int | None):
    """psi ~ 7e139 and a density ~ 5e279, just below INTERMEDIATE_MAGNITUDE_MAX:
    the bound itself stays finite and native vs fallback stays inside it."""
    gbasis = ((("S", ((1.0, 1.0),)),),)
    norm = flatten_gbasis([[("S", [(1.0, 1.0)])]], [[0.0, 0.0, 0.0]]).bf_norm[0]
    case = harness.Case(
        raw_numbers=True,
        defect=0,
        gbasis=gbasis,
        atomcoords=((0.0, 0.0, 0.0),),
        coeff=((INTERMEDIATE_MAGNITUDE_MAX / norm * (1.0 - 1e-6),),),
        mo_index=mo_index,
        origin=(-1.0, -1.0, -1.0),
        step=(1.0, 1.0, 1.0),
        shape=(3, 3, 3),
    )
    basis, axes, coeff2d, mode = harness.backend_inputs(harness._call_args(case))
    fallback = _reference.eval_grid(basis, *axes, coeff2d, mode)
    assert np.max(np.abs(fallback)) > (1e279 if mo_index is None else 1e139)
    tol = harness.tolerance(harness.term_scales(basis, axes, coeff2d), mode, *EXP_PAIR)
    assert np.all(np.isfinite(tol))
    assert harness.test_one_input(harness.encode(case)) is None


# ---------------------------------------------------------------------------
# The exp-group invariant behind the bound's exp term
# ---------------------------------------------------------------------------
#
# fuzz_cclib scales each backend's exp allowance with the net sum of a group
# of terms (exp_group_key: one centre, one exponent). That is sound only if,
# within one backend, every term of a group gets a bit-identical
# exp(-alpha |r - c|^2) at every grid point, although neither backend
# evaluates it once per group. This checks it through the backends' outputs:
# each (function, primitive) term is evaluated alone, with unit norm, weight
# and MO coefficient. Where every axis on which the function has a non-zero
# power lies a power of two (or 0) from the centre, the angular factor is a
# power of two and psi / angular factor is the term's exp factor bit for bit
# (the kernel's EX * EY * EZ, NumPy's np.exp), so members with different
# powers (S, px, py, pz of an SP shell, d and f components) are compared;
# other axes carry inexact cclib grid coordinates, so the exp arguments are
# rounded. Members with the same powers (two coincident atoms) are compared
# on psi itself, on any grid.
#
# Limit: a kernel change whose exp-argument rounding depends on the angular
# power (e.g. exp(-(a*(d*d))) only where l > 0) passes. On the dyadic axes
# every association of a*d*d is exact, and on inexact axes only same-power
# members are compared, which carry the same change. Catching it would need
# exp factors of different-power members compared at inexact coordinates.

# A carbon STO-3G atom (its second S and its P shell share three exponents)
# with d and f polarisation, and a second atom at the same place whose P and
# D shells reuse those three exponents.
EXP_GROUP_GBASIS = [
    fx.STO3G_C + [fx.D_POLARIZATION_C, fx.F_POLARIZATION_C],
    [fx.STO3G_C[2], ("D", fx.STO3G_C[2][1])],
]
DYADIC_CENTRE = (0.75, -1.25, 0.5)  # bohr
DYADIC_OFFSETS = (  # bohr, per axis; three lengths for different SIMD tails
    (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
    (-1.0, -0.5, 0.0, 0.25, 2.0),
    (-4.0, -1.0, -0.25, 0.0, 1.0, 2.0),
)
# geometry -> (axes whose offsets are powers of two, groups mixing powers)
EXP_GROUP_GEOMETRIES = {
    # atoms at (0, 0, 0) and (-0, 0, -0): one group per exponent across both
    "signed-zero-centres": ((True, True, True), 5),
    "dyadic-centre": ((True, True, True), 5),
    "x-dyadic": ((True, False, False), 3),  # S, px, dxx share the SP exponents
    "y-dyadic": ((False, True, False), 3),
    "z-dyadic": ((False, False, True), 3),
    "cclib-grid": ((False, False, False), 0),
}
EXP_GROUP_BACKENDS = [
    pytest.param(_native, marks=requires_native, id="native"),
    pytest.param(_reference, id="fallback"),
]


def _exp_group_input(geometry: str):
    """(basis with unit norms, axes in bohr) for one test geometry."""
    dyadic, _ = EXP_GROUP_GEOMETRIES[geometry]
    if geometry == "signed-zero-centres":
        basis = flatten_gbasis(EXP_GROUP_GBASIS, np.array([[0.0, 0.0, 0.0], [-0.0, 0.0, -0.0]]))
        centre = (0.0, 0.0, 0.0)
    else:
        basis = flatten_gbasis(EXP_GROUP_GBASIS, np.array([[0.31, -0.17, 0.42], [0.31, -0.17, 0.42]]))
        centre = DYADIC_CENTRE
        if any(dyadic):  # offsets that are powers of two need a dyadic centre
            names = ("center_x", "center_y", "center_z")
            basis = dataclasses.replace(basis, **{name: np.full(basis.n_bf, c) for name, c in zip(names, centre)})
    inexact = core._grid_axes(np.array([-1.3, -1.1, -0.9]), np.array([0.37, 0.41, 0.29]), (9, 7, 5))
    axes = tuple(
        c + np.array(offsets) if exact else grid
        for c, offsets, exact, grid in zip(centre, DYADIC_OFFSETS, dyadic, inexact)
    )
    return dataclasses.replace(basis, bf_norm=np.ones(basis.n_bf)), axes


def _term_alone(backend, basis, axes, b: int, p: int) -> np.ndarray:
    coeff = np.zeros((1, basis.n_bf))
    coeff[0, b] = 1.0
    weights = np.where(np.arange(basis.n_prims) == p, 1.0, 0.0)
    alone = dataclasses.replace(basis, prim_w=weights)
    return backend.eval_grid(alone, *axes, coeff, _reference.MODE_WAVEFUNCTION)


def _powers(basis, b: int) -> tuple[int, int, int]:
    return int(basis.powers_l[b]), int(basis.powers_m[b]), int(basis.powers_n[b])


def _angular_factor(basis, axes, b: int) -> np.ndarray:
    l, m, n = _powers(basis, b)  # noqa: E741
    dx = axes[0] - basis.center_x[b]
    dy = axes[1] - basis.center_y[b]
    dz = axes[2] - basis.center_z[b]
    return (dx[:, None, None] ** l * dy[None, :, None] ** m * dz[None, None, :] ** n).reshape(-1)


@pytest.mark.parametrize("backend", EXP_GROUP_BACKENDS)
@pytest.mark.parametrize("geometry", sorted(EXP_GROUP_GEOMETRIES))
def test_exp_group_members_share_one_exp_value(backend, geometry: str):
    dyadic, expected_mixed = EXP_GROUP_GEOMETRIES[geometry]
    basis, axes = _exp_group_input(geometry)
    groups = defaultdict(list)
    for b in range(basis.n_bf):
        for p in range(int(basis.offsets[b]), int(basis.offsets[b + 1])):
            groups[harness.exp_group_key(basis, b, p)].append((b, p))
    tiny = np.finfo(np.float64).tiny
    compared = 0
    mixed = 0
    for key, members in groups.items():
        # Values that must be bit-identical, by kind: the exp factor itself,
        # or psi of members with the same powers.
        by_kind = defaultdict(list)
        for b, p in members:
            psi = _term_alone(backend, basis, axes, b, p)
            powers = _powers(basis, b)
            if all(exact or power == 0 for exact, power in zip(dyadic, powers)):
                angular = _angular_factor(basis, axes, b)
                exact = (angular != 0.0) & (np.abs(psi) >= tiny)  # psi / angular is exact here
                factor = np.where(exact, psi / np.where(exact, angular, 1.0), np.nan)
                by_kind["exp factor"].append(((b, p), powers, factor))
            else:
                by_kind[powers].append(((b, p), powers, psi))
        for kind, values in by_kind.items():
            mixed += kind == "exp factor" and len({powers for _, powers, _ in values}) > 1
            first_term, _, first = values[0]
            for term, _, value in values[1:]:
                both = ~np.isnan(first) & ~np.isnan(value)
                compared += int(np.count_nonzero(both & (first != 0.0)))
                assert np.array_equal(first[both].view(np.uint64), value[both].view(np.uint64)), (
                    f"{kind} differs within exp group {key}: terms {first_term} and {term}"
                )
    assert compared > 500, "the comparison must cover non-zero exp factors"
    assert mixed == expected_mixed
    if geometry == "signed-zero-centres":  # the group spans both signs of zero
        assert np.signbit(basis.center_x).any() and not np.signbit(basis.center_x).all()
