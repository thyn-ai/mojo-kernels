"""Replay the bm25 fuzzing seed corpus inside the normal differential suite.

Each file under ``fuzz/corpus/bm25/`` is one scenario for
``fuzz/fuzz_bm25.py`` (see that module for what is compared). Run by
``scripts/test_all.sh`` on both backends: the native pass is the real
differential and requires every ``known-issue-*`` reproducer to still
reproduce its issue; the forced-fallback pass checks the fallback against
the ``rank_bm25`` oracle and the exception contracts on the same seeds
(a native-vs-fallback divergence cannot reproduce without the kernel).

The tests after the replay pin the ``ill-conditioned-norm`` known issue:
the exact unit the coverage-guided run found, the shape rule that classifies
it, and the proof that in-domain parameters can never reach it.
"""

from __future__ import annotations

import base64
import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

import fuzz_bm25 as harness  # fuzz/ is on sys.path via tests/conftest.py
from _harness import Divergence, expected_issue_key

SEEDS = sorted(harness.CORPUS_DIR.glob("*.bin"))

# Verbatim the unit libFuzzer wrote on #54 (fuzz workflow job 106379599687,
# crash-6c4af46e8b111987242e1f3829f815e3c6801368), the Base64 the job log
# printed. Before the ill-conditioned-norm classification it raised
#   Divergence: native kernel != fallback for query ['w02', 'w02', 'w02'] at
#   documents [0, 1, 2]: native=[2.10031216e+157 ...] fallback=[2.10031008e+157 ...]
ILL_CONDITIONED_UNIT = base64.b64decode(
    "AgABAADi4uLi4uLi4uLi4uLi4uLi4uLi4uJi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi"
    "4uLi4uLi4uLi4uLi4gAAAAABQAAAAAAAAOg//////wEAAAAA///v/wMAAQABAAAAAQABAAAAAgAB"
    "AAAAAwABAQAAAAAAAAFAAAAAAADiJQEAAAAA4gDoP////wAAAUAAAAAAAADoP/8BAAEAAAACAAAC"
    "AAA="
)
# The two score vectors that job printed for the first query.
ILL_CONDITIONED_NATIVE = np.array([2.10031216e157] * 3)
ILL_CONDITIONED_FALLBACK = np.array([2.10031008e157] * 3)

# The sibling a 60 s run of the same workflow command found on the fix
# branch (linux/amd64 container, crash-caa7aef187b44ab87d25718c8a90ac6e95ace895):
# the same parameters (one byte of b differs) on a corpus where the query
# term is posted in one document of four, so one query shows both shapes:
#   native=[1.43393053e+169 1.43393053e+169 1.43393053e+169 1.56669398e+158]
#   fallback=[nan nan nan 1.56669185e+158]
COMBINED_UNIT = base64.b64decode(
    "AgABAADi4uLi4uJ+4uLi4uLi4uLi4uLi4uJiAwMDAwMDAwMDAwMDAwMDAwcDAwMDAwMDAwMDAwMDAw"
    "MDAwMDAwMDAAAAAAAAAAABAAA="
)
COMBINED_NATIVE = np.array([1.43393053e169] * 3 + [1.56669398e158])
COMBINED_FALLBACK = np.array([np.nan] * 3 + [1.56669185e158])


def test_seed_corpus_is_present():
    assert SEEDS, f"no seeds under {harness.CORPUS_DIR}; run fuzz/seed_corpus.py bm25"
    for issue in harness.KNOWN_ISSUES:
        assert any(expected_issue_key(s) == issue.key for s in SEEDS), (
            f"known issue {issue.key!r} has no reproducer seed"
        )


@pytest.mark.parametrize("seed", SEEDS, ids=lambda p: p.name)
def test_seed_replays_without_divergence(seed: Path):
    harness.replay_seed(seed)


def test_ill_conditioned_unit_decodes_to_the_reported_case():
    case = harness.decode(ILL_CONDITIONED_UNIT)
    assert harness.VARIANT_NAMES[case.variant] == "BM25Plus"
    assert case.raw_params
    assert (case.k1, case.b, case.third) == (
        -2.227377823252691e168,
        -2.227377823277027e168,
        2.227377823277027e168,
    )
    assert case.corpus == (("w02", "w02", "w02"),) * 3
    assert case.queries == (
        ("w02", "w02", "w02"),
        ("w02", "w00", "w00", "zzz-unseen"),
        ("w00",),
    )


def test_ill_conditioned_unit_has_a_residue_normaliser():
    """Exact arithmetic gives ``1 - b + b * 3 / 3 == 1`` at every document;
    IEEE-754 gives 0, because the 1 is below one ulp of b."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    one_minus_b = 1.0 - case.b
    length_term = case.b * 3.0 / 3.0
    assert one_minus_b == -case.b
    assert one_minus_b + length_term == 0.0
    assert harness._amplification(np.float64(0.0), one_minus_b, length_term) == np.inf
    assert harness._ill_conditioned_documents(case, list(case.queries[0])).tolist() == [True] * 3
    assert harness.ISSUE_ILL_CONDITIONED_NORM.applies(case)


def test_ill_conditioned_unit_is_classified():
    case = harness.decode(ILL_CONDITIONED_UNIT)
    query = list(case.queries[0])
    # The two vectors the job printed are 9.9e6 tolerances apart ...
    assert not harness._close(ILL_CONDITIONED_NATIVE, ILL_CONDITIONED_FALLBACK).any()
    assert (
        harness._compare_backends(case, query, ILL_CONDITIONED_NATIVE, ILL_CONDITIONED_FALLBACK)
        == harness.ISSUE_ILL_CONDITIONED_NORM.key
    )
    # ... and the live replay reaches the same classification with the kernel
    # loaded; without it the fallback is compared with itself and agrees.
    expected = harness.ISSUE_ILL_CONDITIONED_NORM.key if harness.native_available() else None
    assert harness.test_one_input(ILL_CONDITIONED_UNIT) == expected


def test_ill_conditioned_shape_excuses_only_the_cancelled_documents():
    """Lengths 3, 3, 3, 2, 4 keep avgdl at 3: the first three normalisers
    cancel to 0, the last two are -b/3 and b/3 (amplification 3 and 4) and
    must still agree."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    mixed = dataclasses.replace(case, corpus=case.corpus + (("w02", "w02"), ("w02",) * 4))
    query = ["w02"]
    assert harness._ill_conditioned_documents(mixed, query).tolist() == [True] * 3 + [False] * 2
    fallback = np.array([2.1e157, 2.1e157, 2.1e157, 5.0, 7.0])
    native = fallback.copy()
    native[:3] *= 1 + 1e-6  # the residue rounds differently on the two sides
    assert (
        harness._compare_backends(mixed, query, native, fallback)
        == harness.ISSUE_ILL_CONDITIONED_NORM.key
    )
    native[3] += 1.0  # a well-conditioned document disagreeing is a finding
    with pytest.raises(Divergence):
        harness._compare_backends(mixed, query, native, fallback)


def test_both_shapes_in_one_query_are_classified():
    """Normaliser 0 at every document: 0/0 (degenerate-nan) where the term is
    absent, residue (ill-conditioned-norm) where it is present. Every
    position matches one shape; the outcome is the ill-conditioned key."""
    case = harness.decode(COMBINED_UNIT)
    assert harness.VARIANT_NAMES[case.variant] == "BM25Plus"
    assert case.corpus == (("w03",) * 4,) * 3 + (("w03", "w03", "w00", "w00"),)
    assert case.queries == (("w00",) * 4,)
    query = list(case.queries[0])
    assert harness.ISSUE_DEGENERATE_NAN.applies(case)
    assert harness._ill_conditioned_documents(case, query).tolist() == [False] * 3 + [True]
    assert (
        harness._compare_backends(case, query, COMBINED_NATIVE, COMBINED_FALLBACK)
        == harness.ISSUE_ILL_CONDITIONED_NORM.key
    )
    # Each shape alone still classifies as itself ...
    nan_part = COMBINED_NATIVE.copy()
    nan_part[3] = COMBINED_FALLBACK[3]
    assert (
        harness._compare_backends(case, query, nan_part, COMBINED_FALLBACK)
        == harness.ISSUE_DEGENERATE_NAN.key
    )
    residue_part = COMBINED_FALLBACK.copy()
    residue_part[3] = COMBINED_NATIVE[3]
    assert (
        harness._compare_backends(case, query, residue_part, COMBINED_FALLBACK)
        == harness.ISSUE_ILL_CONDITIONED_NORM.key
    )
    # ... and a NaN on the kernel's side matches neither.
    with pytest.raises(Divergence):
        harness._compare_backends(case, query, COMBINED_FALLBACK, COMBINED_NATIVE)
    expected = harness.ISSUE_ILL_CONDITIONED_NORM.key if harness.native_available() else None
    assert harness.test_one_input(COMBINED_UNIT) == expected


def test_raw_parameters_without_cancellation_do_not_apply():
    """The same parameters on a corpus with no document of average length:
    every normaliser is +-b/3, nothing cancels, the issue does not apply."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    uneven = dataclasses.replace(case, corpus=(("w02", "w02"), ("w02",) * 4))
    assert not harness._ill_conditioned_documents(uneven, ["w02"]).any()
    assert not harness.ISSUE_ILL_CONDITIONED_NORM.applies(uneven)


def test_in_domain_parameters_never_cancel():
    """b in [0, 1], k1 in [0, 5], epsilon/delta in [0, 3] (the in-domain
    generator's ranges): every summand of the normaliser and of the outer
    denominators is >= 0, so max|summand| <= |sum|, the amplification is at
    most 1, and ILL_CONDITIONED_AMPLIFICATION (1e13) is out of reach for
    every variant and every corpus shape."""
    rng = random.Random(20260921)
    b_grid = [0.0, 1.0, 0.5, 1.0 - 2.0**-53, 2.0**-1074] + [rng.random() for _ in range(10)]
    k1_grid = [0.0, 5.0, 1.5] + [rng.uniform(0.0, 5.0) for _ in range(2)]
    third_grid = [0.0, 3.0, 1.0] + [rng.uniform(0.0, 3.0) for _ in range(2)]
    lengths_grid = [(3, 3, 3), (0, 1, 24), (1,), tuple(rng.randint(0, 24) for _ in range(8))]
    checked = 0
    for variant in range(len(harness.VARIANT_NAMES)):
        for b in b_grid:
            for k1 in k1_grid:
                for third in third_grid:
                    for lengths in lengths_grid:
                        doc_len = np.array(lengths, dtype=np.float64)
                        avgdl = doc_len.mean()
                        length_term = b * doc_len / avgdl
                        amplification = harness._amplification(
                            (1.0 - b) + length_term, 1.0 - b, length_term
                        )
                        assert np.all(amplification <= 1.0), (b, lengths, amplification)
                        case = harness.Case(
                            variant=variant,
                            raw_params=False,
                            k1=k1,
                            b=b,
                            third=third,
                            vocab_size=1,
                            with_unicode=False,
                            corpus=tuple(("w00",) * n for n in lengths),
                            queries=(("w00",),),
                        )
                        assert not harness._ill_conditioned_documents(case, ["w00"]).any(), (
                            case.describe()
                        )
                        assert not harness.ISSUE_ILL_CONDITIONED_NORM.applies(case)
                        checked += 1
    assert checked == len(harness.VARIANT_NAMES) * len(b_grid) * len(k1_grid) * len(
        third_grid
    ) * len(lengths_grid)
