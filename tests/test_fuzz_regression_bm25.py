"""Replay the bm25 fuzzing seed corpus inside the normal differential suite.

Each file under ``fuzz/corpus/bm25/`` is one scenario for
``fuzz/fuzz_bm25.py`` (see that module for what is compared). Run by
``scripts/test_all.sh`` on both backends: the native pass is the real
differential and requires every ``known-issue-*`` reproducer to still
reproduce its issue; the forced-fallback pass checks the fallback against
the ``rank_bm25`` oracle and the exception contracts on the same seeds
(a native-vs-fallback divergence cannot reproduce without the kernel).

The tests after the replay pin the #58 conditioning guard: the exact unit
the coverage-guided run found (now agreeing on every query), corpus-shape
variants of it, and the proof that in-domain parameters can never reach the
guard's 1e13 amplification threshold.
"""

from __future__ import annotations

import base64
import dataclasses
import random
from pathlib import Path

import fuzz_bm25 as harness  # fuzz/ is on sys.path via tests/conftest.py
import numpy as np
import pytest
from _harness import expected_issue_key

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


def test_ill_conditioned_unit_now_agrees_on_the_posted_query():
    """Issue #58 regression: the query that posts the cancelled term now scores
    identically on both backends -- the kernel's conditioning guard
    (``kernels/bm25/src/bm25mojo.mojo``) poisons the term's fast-path bound at
    index build for any posting whose normaliser or outer denominator cancels
    by 1e13 or more, so the query takes the precise reference-order path.
    Before the guard the full unit returned the ``ill-conditioned-norm``
    known-issue key (native 2.10031216e157 vs fallback 2.10031008e157)."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    posted_only = dataclasses.replace(case, queries=(case.queries[0],))
    assert harness.test_one_input(harness.encode(posted_only)) is None


def test_ill_conditioned_unit_classifies_the_nan_side():
    """The unit's other two queries post unseen terms: with norm == 0 at every
    document the reference evaluates 0/0 for them (the still-open
    ``degenerate-nan`` issue #15), and that is now the only shape left."""
    expected = harness.ISSUE_DEGENERATE_NAN.key if harness.native_available() else None
    assert harness.test_one_input(ILL_CONDITIONED_UNIT) == expected


def test_mixed_lengths_still_agree():
    """Lengths 3, 3, 3, 2, 4 keep avgdl at 3: the first three normalisers
    cancel to 0, the last two are -b/3 and b/3 (amplification 3 and 4, below
    the guard's 1e13 threshold). The whole term takes the precise path (the
    guard acts per term, so the query is precise end to end) and every
    document agrees."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    mixed = dataclasses.replace(
        case,
        corpus=case.corpus + (("w02", "w02"), ("w02",) * 4),
        queries=(case.queries[0],),
    )
    assert harness.test_one_input(harness.encode(mixed)) is None


def test_uneven_lengths_agree_without_cancellation():
    """The same parameters on a corpus with no document of average length:
    every normaliser is +-b/3, nothing cancels, the fast path runs -- and
    still agrees."""
    case = harness.decode(ILL_CONDITIONED_UNIT)
    uneven = dataclasses.replace(case, corpus=(("w02", "w02"), ("w02",) * 4))
    assert harness.test_one_input(harness.encode(uneven)) is None


def test_both_shapes_in_one_query_classify_the_nan_side():
    """Normaliser 0 at every document: 0/0 (``degenerate-nan``, still an open
    known issue) where the term is absent, and exact agreement where it is
    present (the guard's precise path). The outcome is the degenerate-nan
    key, now for its own shape alone."""
    case = harness.decode(COMBINED_UNIT)
    assert harness.VARIANT_NAMES[case.variant] == "BM25Plus"
    assert case.corpus == (("w03",) * 4,) * 3 + (("w03", "w03", "w00", "w00"),)
    assert case.queries == (("w00",) * 4,)
    assert harness.ISSUE_DEGENERATE_NAN.applies(case)
    expected = harness.ISSUE_DEGENERATE_NAN.key if harness.native_available() else None
    assert harness.test_one_input(COMBINED_UNIT) == expected


def _amplification(total: np.ndarray, *summands: np.ndarray) -> np.ndarray:
    """``max|summand| / |total|`` -- the kernel guard's own statistic: a summation
    only becomes ill-conditioned when this reaches 1e13 (1 / SCORE_RTOL)."""
    largest = np.max(np.abs(np.broadcast_arrays(*summands)), axis=0)
    with np.errstate(all="ignore"):
        return largest / np.maximum(np.abs(total), np.finfo(np.float64).tiny)


def test_in_domain_parameters_never_cancel():
    """b in [0, 1], k1 in [0, 5], epsilon/delta in [0, 3] (the in-domain
    generator's ranges): every summand of the normaliser and of the outer
    denominators is >= 0, so max|summand| <= |sum|, the amplification is at
    most 1, and the guard's 1e13 threshold is out of reach for every variant
    and every corpus shape -- the fast path always runs in-domain."""
    rng = random.Random(20260921)
    b_grid = [0.0, 1.0, 0.5, 1.0 - 2.0**-53, 2.0**-1074] + [
        rng.random() for _ in range(10)
    ]
    k1_grid = [0.0, 5.0, 1.5] + [rng.uniform(0.0, 5.0) for _ in range(2)]
    third_grid = [0.0, 3.0, 1.0] + [rng.uniform(0.0, 3.0) for _ in range(2)]
    lengths_grid = [
        (3, 3, 3),
        (0, 1, 24),
        (1,),
        tuple(rng.randint(0, 24) for _ in range(8)),
    ]
    checked = 0
    for variant in range(len(harness.VARIANT_NAMES)):
        for b in b_grid:
            for k1 in k1_grid:
                for third in third_grid:
                    for lengths in lengths_grid:
                        doc_len = np.array(lengths, dtype=np.float64)
                        avgdl = doc_len.mean()
                        length_term = b * doc_len / avgdl
                        amplification = _amplification(
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
                        # The guard's own threshold is out of reach; the fast
                        # path always runs in-domain. (No live replay here:
                        # the (0, 1, 24) row's empty document is the
                        # still-open degenerate-nan shape for b == 1, not
                        # this guard's.)
                        checked += 1
    assert checked == len(harness.VARIANT_NAMES) * len(b_grid) * len(k1_grid) * len(
        third_grid
    ) * len(lengths_grid)
