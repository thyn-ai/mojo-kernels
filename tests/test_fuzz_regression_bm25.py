"""Replay the bm25 fuzzing seed corpus inside the normal differential suite.

Each file under ``fuzz/corpus/bm25/`` is one scenario for
``fuzz/fuzz_bm25.py`` (see that module for what is compared). Run by
``scripts/test_all.sh`` on both backends: the native pass is the real
differential; the forced-fallback pass still checks the fallback against
the ``rank_bm25`` oracle and the exception contracts, but skips the
``known-issue-*`` reproducers, which need the kernel to reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fuzz"))

import fuzz_bm25 as harness  # noqa: E402
from _harness import expected_issue_key  # noqa: E402

SEEDS = sorted(harness.CORPUS_DIR.glob("*.bin"))


def test_seed_corpus_is_present():
    assert SEEDS, f"no seeds under {harness.CORPUS_DIR}; run fuzz/seed_corpus.py bm25"
    for issue in harness.KNOWN_ISSUES:
        assert any(expected_issue_key(s) == issue.key for s in SEEDS), (
            f"known issue {issue.key!r} has no reproducer seed"
        )


@pytest.mark.parametrize("seed", SEEDS, ids=lambda p: p.name)
def test_seed_replays_without_divergence(seed: Path):
    if expected_issue_key(seed) is not None and not harness.native_available():
        pytest.skip("known-issue reproducers are native-vs-fallback divergences; needs the kernel")
    harness.replay_seed(seed)
