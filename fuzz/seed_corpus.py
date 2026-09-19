"""(Re)generate the checked-in seed corpora under fuzz/corpus/.

Two kinds of seed per harness:

* ``random-NN.bin`` -- fixed-seed pseudo-random byte strings of varying
  length (``random.Random(base + NN).randbytes(...)``). Because every
  harness decodes any byte string into a valid case, these are already
  diverse scenarios; libFuzzer uses them as the starting population.
* ``known-issue-<key>-<n>.bin`` -- hand-minimised reproducers for the open
  divergences listed in each harness's ``KNOWN_ISSUES``, encoded with the
  harness's ``encode``. Generation asserts that each one still decodes to
  the intended case and still reproduces its issue, so this script doubles
  as the check that a reproducer is minimal *and* live.

Usage (from the repo root, inside the pixi environment, kernels built)::

    PYTHONPATH=python/bm25_mojo  python fuzz/seed_corpus.py bm25
    PYTHONPATH=python/cclib_mojo python fuzz/seed_corpus.py cclib

Regenerating is deterministic; a clean ``git status`` afterwards means the
corpus is what this script says it is.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

from _harness import FUZZ_DIR

RANDOM_SEEDS = 24
RANDOM_BASE = {"bm25": 1000, "cclib": 5000}


def _random_blobs(name: str) -> dict[str, bytes]:
    out = {}
    for n in range(RANDOM_SEEDS):
        rng = random.Random(RANDOM_BASE[name] + n)
        out[f"random-{n:02d}.bin"] = rng.randbytes(rng.randint(16, 768))
    return out


def _bm25_reproducers(harness) -> dict[str, bytes]:
    """Minimal cases for KnownIssue 'degenerate-nan' (one per trigger)."""
    Case = harness.Case
    two_docs = (("w00",), ("w01",))
    cases = {
        # k1 == 0: the unposted document divides 0 by 0 in the reference.
        "known-issue-degenerate-nan-1.bin": Case(
            variant=0, raw_params=False, k1=0.0, b=0.75, third=0.25,
            vocab_size=2, with_unicode=False, corpus=two_docs, queries=(("w00",),),
        ),
        # b == 1 with an empty document: norm = 1 - b + b * 0 / avgdl == 0.
        "known-issue-degenerate-nan-2.bin": Case(
            variant=0, raw_params=False, k1=1.5, b=1.0, third=0.25,
            vocab_size=1, with_unicode=False, corpus=(("w00",), ()), queries=(("w00",),),
        ),
        # BM25L with k1 + delta == 0.
        "known-issue-degenerate-nan-3.bin": Case(
            variant=1, raw_params=False, k1=0.0, b=0.75, third=0.0,
            vocab_size=2, with_unicode=False, corpus=two_docs, queries=(("w00",),),
        ),
        # BM25Plus with k1 == 0: the floor is finite, the reference is NaN.
        "known-issue-degenerate-nan-4.bin": Case(
            variant=2, raw_params=False, k1=0.0, b=0.75, third=1.0,
            vocab_size=2, with_unicode=False, corpus=two_docs, queries=(("w00",),),
        ),
        # Non-finite k1 (raw parameters): NaN on posted docs on both sides,
        # NaN vs 0 on unposted ones.
        "known-issue-degenerate-nan-5.bin": Case(
            variant=0, raw_params=True, k1=float("nan"), b=0.75, third=0.25,
            vocab_size=2, with_unicode=False, corpus=two_docs, queries=(("w00",),),
        ),
        # Non-finite epsilon: a term in more than half the corpus gets its
        # negative idf floored to epsilon * average_idf == NaN on both sides;
        # the one document without it is NaN * 0 in the reference and 0 in
        # the kernel (found by the first coverage-guided run on Linux x86_64).
        "known-issue-degenerate-nan-6.bin": Case(
            variant=0, raw_params=True, k1=1.5, b=0.75, third=float("nan"),
            vocab_size=2, with_unicode=False,
            corpus=(("w00", "w01"), ("w00",), ("w01",)), queries=(("w00",),),
        ),
    }
    return {name: harness.encode(case) for name, case in cases.items()}


def _cclib_reproducers(harness) -> dict[str, bytes]:
    """Minimal cases for KnownIssue 'extreme-magnitude' (one per exception path)."""
    Case = harness.Case

    def case(sym: str, alpha: float, x: float = 0.0) -> "harness.Case":
        return Case(
            raw_numbers=True, defect=0,
            gbasis=(((sym, ((alpha, 1.0),)),),),
            atomcoords=((x, 0.0, 0.0),),
            coeff=((1.0,) * len(harness.SYM2POWERS[sym]),),
            mo_index=None,
            origin=(-1.0, -1.0, -1.0), step=(1.0, 1.0, 1.0), shape=(2, 2, 2),
        )

    cases = {
        "known-issue-extreme-magnitude-1.bin": case("F", 1e70),  # pow(alpha, 4.5) overflows
        "known-issue-extreme-magnitude-2.bin": case("S", 1e210),  # pow(alpha, 1.5) overflows
        "known-issue-extreme-magnitude-3.bin": case("S", 1e-210),  # (pi/gamma)^1.5 overflows
        "known-issue-extreme-magnitude-4.bin": case("F", 1e-120),  # (2 gamma)^i underflows to 0 -> 0 division
        # Found by the first coverage-guided run on Linux x86_64: a D shell
        # centred at |x| ~ 1e292 Angstrom with a small exponent; the product
        # centre rounds away from the atom and (P - A)^2 overflows.
        "known-issue-extreme-magnitude-5.bin": case("D", 2.6571366763582966e-22, -1.941414049957967e292),
        # Found by the third coverage-guided run: coefficient 1e150 on a P
        # function evaluated 1e170 Angstrom away. |c N x| overflows to inf
        # while exp(-alpha r^2) underflows to 0; the kernel's product order
        # gives inf * 0 = NaN, the fallback's gives 0.
        "known-issue-extreme-magnitude-6.bin": Case(
            raw_numbers=True, defect=0,
            gbasis=((("P", ((1.0, 1.0),)),),),
            atomcoords=((0.0, 0.0, 0.0),),
            coeff=((1e150, 0.0, 0.0),),
            mo_index=None,
            origin=(1e170, 0.0, 0.0), step=(1.0, 1.0, 1.0), shape=(1, 1, 1),
        ),
        # Same mechanism through the primitive weight (found by the first
        # workflow run on the pull request): an S function (polynomial 1)
        # with an in-window exponent whose c * N * w product overflows.
        "known-issue-extreme-magnitude-7.bin": Case(
            raw_numbers=True, defect=0,
            gbasis=((("S", ((1e6, 1.0),)),),),
            atomcoords=((0.0, 0.0, 0.0),),
            coeff=((1e305,),),
            mo_index=None,
            origin=(1e170, 0.0, 0.0), step=(1.0, 1.0, 1.0), shape=(1, 1, 1),
        ),
        # Order matters for IEEE overflow (found by the fifth Linux run): with
        # a tiny primitive weight (alpha ~ 1e-77) c * N * x overflows first,
        # so the kernel's ((c N) x) w is inf while (c N w) x would be finite.
        "known-issue-extreme-magnitude-8.bin": Case(
            raw_numbers=True, defect=0,
            gbasis=((("P", ((1e-77, 1.0),)),),),
            atomcoords=((0.0, 0.0, 0.0),),
            coeff=((1e150, 0.0, 0.0),),
            mo_index=None,
            origin=(1e170, 0.0, 0.0), step=(1.0, 1.0, 1.0), shape=(1, 1, 1),
        ),
    }
    return {name: harness.encode(c) for name, c in cases.items()}


def build(name: str) -> None:
    harness = importlib.import_module(f"fuzz_{name}")
    corpus_dir: Path = FUZZ_DIR / "corpus" / name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    seeds = _random_blobs(name)
    reproducers = {"bm25": _bm25_reproducers, "cclib": _cclib_reproducers}[name](harness)

    # Every reproducer must round-trip through decode and reproduce its issue.
    for filename, data in reproducers.items():
        expected_key = filename[len("known-issue-") :].rsplit("-", 1)[0]
        assert harness.encode(harness.decode(data)) == data, f"{filename}: encode/decode round trip"
        outcome = harness.test_one_input(data)
        assert outcome == expected_key, f"{filename}: expected outcome {expected_key!r}, got {outcome!r}"
    seeds.update(reproducers)

    stale = {p.name for p in corpus_dir.iterdir()} - set(seeds)
    for filename in sorted(stale):
        (corpus_dir / filename).unlink()
        print(f"removed stale {filename}")
    for filename, data in sorted(seeds.items()):
        (corpus_dir / filename).write_bytes(data)
    print(f"{name}: wrote {len(seeds)} seeds to {corpus_dir.relative_to(FUZZ_DIR.parent)} "
          f"({len(reproducers)} known-issue reproducers)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in RANDOM_BASE:
        print(f"usage: {sys.argv[0]} {{bm25|cclib}}", file=sys.stderr)
        sys.exit(2)
    build(sys.argv[1])
