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
from pathlib import Path

import numpy as np
import pytest

import cclib_mojo
import fuzz_cclib as harness  # fuzz/ is on sys.path via tests/conftest.py
import gaussgrid_fixtures as fx
from _harness import Divergence, expected_issue_key
from cclib_mojo import _native, _reference
from cclib_mojo._basis import flatten_gbasis

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
