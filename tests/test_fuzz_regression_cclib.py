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

import pytest

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
