# Fuzzing

Differential fuzzing for the kernels in this repository. Each harness draws
a scenario from an arbitrary byte string, runs it through the native Mojo
kernel, the vendored fallback and (where one is installed) the reference
package, and fails on any disagreement beyond the tolerance the package
documents -- or on any exception the package does not document. The same
harnesses run coverage-guided under [atheris](https://github.com/google/atheris)
and, without any fuzzer, as a replay of the checked-in seed corpora inside
the normal test suites.

| Harness | Compares | Contract |
|---|---|---|
| `fuzz_bm25.py` | `bm25_mojo` native kernel vs `BM25._reference_scores` vs `rank_bm25` 0.2.2 | kernel vs fallback scores within 1e-8 absolute (`tests/conftest.py`) plus 1e-13 relative (the relative term only matters once out-of-domain parameters push scores past ~1e5, where 1e-8 is below one ulp), fallback bit-identical to `rank_bm25`, `get_top_n` order equal up to ties, construction raises `ZeroDivisionError` exactly when `rank_bm25` does (empty corpus; all-empty documents under Okapi), wrong-length `documents` asserts |
| `fuzz_cclib.py` | `cclib_mojo` native kernel vs `_reference.eval_grid` vs the PyQuante 1.6.5 transcription in `tests/pyquante1_oracle.py` | grids within 1e-10 relative / 1e-12 absolute (`tests/test_gaussgrid_differential.py`) measured against the conditioning of the sum, public API bit-identical to the active backend, malformed input raises `BasisError`/`GridError` and nothing else |
| `typescript/fuse-mojo/tests/parity.property.test.js` | `@fuse-mojo/core` vs published Fuse.js 7.1.0 ([fast-check](https://fast-check.dev)) | identical `refIndex` order, scores within 1e-9, identical match spans (`tests/helpers.cjs` `compareResults`) |

Scenario generation is *total*: `_harness.ByteCursor` reads zeros past the
end of the input, so every byte string libFuzzer produces is a valid case
and nothing is rejected before it reaches the code under test. Both kinds of
number are generated -- in-domain values (BM25 `k1` in [0, 5], `b` in [0, 1];
Gaussian exponents 0.03..100 bohr^-2, geometry of order one) and raw IEEE
doubles (negative, NaN, infinite, subnormal, 1e300) -- together with empty
documents, Unicode tokens, empty collections and structurally malformed
input (wrong coefficient width, zero grid dimension, unsupported shell, ...).

## Running

```bash
# Replay the seed corpora (any platform; also part of `pixi run test` and
# `pixi run test-cclib` via tests/test_fuzz_regression_*.py):
pixi run fuzz-regression

# Coverage-guided search (Linux x86_64; atheris has no macOS wheel and Apple's
# clang has no libFuzzer). libFuzzer flags pass through after `--`:
pixi run -e fuzz fuzz-bm25  -- -max_total_time=60
pixi run -e fuzz fuzz-cclib -- -max_total_time=60 -jobs=4

# fast-check parity properties for fuse-mojo, native then forced fallback:
FC_NUM_RUNS=5000 pixi run fuzz-fuse
```

`.github/workflows/fuzz.yml` runs all three on every pull request with a
bounded budget (60 s per atheris harness, 2,000 fast-check scenarios per
property) and nightly with a larger one (20 minutes, 50,000 scenarios).
Coverage-guided runs read the seeds from `fuzz/corpus/<name>/` and write
new inputs and crash artifacts to `.fuzz-work/` (git-ignored), never into
the checked-in corpus.

## When a run fails

An atheris run stops at the first divergence and writes the input to
`.fuzz-work/crash-<sha1>` (uploaded as a workflow artifact in CI). Replay
and minimise it, then keep the minimal input as a seed:

```bash
PYTHONPATH=python/bm25_mojo pixi run python fuzz/fuzz_bm25.py --regression .fuzz-work/crash-...
pixi run -e fuzz fuzz-bm25 -- -minimize_crash=1 -exact_artifact_path=.fuzz-work/minimal .fuzz-work/crash-...
cp .fuzz-work/minimal fuzz/corpus/bm25/<descriptive-name>.bin
```

A failing fast-check property prints the shrunk counterexample together
with the seed and path that reproduce it; replay with
`FC_SEED=<seed> FC_PATH=<path> npm run test:property` in
`typescript/fuse-mojo`.

A divergence is either a bug to fix (the seed then guards the fix as a
regression test) or, when the fix needs a maintainer decision, a **known
issue**: open a GitHub issue, add a `KnownIssue` entry to the harness whose
predicate names exactly the input class and whose comparison accepts exactly
the observed shape of the disagreement, and check the minimised reproducer in
as `fuzz/corpus/<name>/known-issue-<key>-<n>.bin`. The replay then requires
that seed to *keep* reproducing the issue, so the entry cannot outlive the
bug: fixing it fails the replay until the seed and the entry are removed and
the issue is closed in the same change. Nothing is ever excluded from the
generators.

### Open known issues

| Key | Harness | Issue |
|---|---|---|
| `degenerate-nan` | `fuzz_bm25.py` | [#15](https://github.com/thyn-ai/mojo-kernels/issues/15) -- kernel scores unposted documents 0 where `rank_bm25` yields NaN (`k1 == 0`, `b == 1` with empty documents, `b > 1`, non-finite parameters) |
| `extreme-magnitude` | `fuzz_cclib.py` | [#16](https://github.com/thyn-ai/mojo-kernels/issues/16) -- magnitudes are not validated: exponents beyond ~1e68 or below ~1e-100 and coordinates beyond ~1e170 Angstrom raise `OverflowError`/`ZeroDivisionError` instead of `BasisError`; intermediates at the edge of the double range (coefficient x norm x weight x r^L on the kernel's side, polynomial x contraction on the fallback's) make the two evaluation orders disagree: inf x 0 = NaN on one side, 0 on the other |

## Layout

```
fuzz/_harness.py        ByteCursor/ByteWriter, KnownIssue, regression replay, CLI (atheris or --regression)
fuzz/fuzz_bm25.py       bm25-mojo harness: decode/encode, KNOWN_ISSUES, test_one_input
fuzz/fuzz_cclib.py      cclib-mojo harness
fuzz/seed_corpus.py     regenerates fuzz/corpus/ deterministically and checks every reproducer
fuzz/corpus/<name>/     random-NN.bin (fixed-seed byte strings) + known-issue-<key>-<n>.bin
tests/test_fuzz_regression_{bm25,cclib}.py   pytest replay, one test per seed, both backends
typescript/fuse-mojo/tests/parity.property.test.js   fast-check properties (node --test)
```

`python fuzz/seed_corpus.py bm25` (with `PYTHONPATH=python/bm25_mojo`, inside
the pixi environment, kernel built) rewrites the bm25 corpus from its fixed
seeds and asserts that each `known-issue-*` seed round-trips through
`encode`/`decode` and still reproduces its issue; `cclib` likewise. A clean
`git status` afterwards means the corpus is what the script says it is.
