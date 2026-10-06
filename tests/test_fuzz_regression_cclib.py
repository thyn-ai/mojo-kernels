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

from pathlib import Path

import numpy as np
import pytest

import cclib_mojo
import fuzz_cclib as harness  # fuzz/ is on sys.path via tests/conftest.py
from _harness import expected_issue_key

SEEDS = sorted(harness.CORPUS_DIR.glob("*.bin"))


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
