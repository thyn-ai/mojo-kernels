# difflib-mojo

**[difflib](https://docs.python.org/3/library/difflib.html)'s
`SequenceMatcher` and `get_close_matches`, accelerated by a Mojo kernel** —
the standard library's own fuzzy-matching workhorse, with a vendored
pure-Python fallback. Prebuilt per-platform binaries mean **no Mojo
toolchain is ever required** on an end user's machine, and every output —
ratios, matching blocks, opcodes, ranked close-match lists **including tie
order** — is **bit-identical to CPython 3.12's difflib** on both backends.

It works two ways:

1. **Drop-in** — `difflib_mojo.SequenceMatcher` mirrors
   `difflib.SequenceMatcher` (CPython 3.12 semantics:
   `ratio`/`real_quick_ratio`/`quick_ratio`/`find_longest_match`/
   `get_matching_blocks`/`get_opcodes`/`get_grouped_opcodes`), and
   `difflib_mojo.get_close_matches` mirrors `difflib.get_close_matches`,
   same arguments, same results, same error messages.
2. **Batched** — `get_close_matches_batch(words, candidates, ...)` scores
   the whole words × candidates grid in one kernel call (with an optional
   `key=` transform, e.g. `key=str.casefold`), and `ratio_batch(pairs)`
   vectorizes `ratio()` over independent pairs.

## Install

```
pip install difflib-mojo
```

Per-platform wheels (macOS arm64, Linux x86_64) carry the native kernel,
self-contained (the Mojo runtime is vendored into the wheel; nothing to
compile, no absolute rpaths). On any other platform — including Windows —
the same wheel API runs on the vendored pure-Python engine, silently and
correctly. There is no sdist: a source tarball cannot rebuild the native
library.

## Quickstart

```python
import difflib_mojo

# Drop-in SequenceMatcher (constructor signature like difflib's):
s = difflib_mojo.SequenceMatcher(lambda x: x == " ",
                                 "private Thread currentThread;",
                                 "private volatile Thread currentThread;")
s.ratio()                # 0.865..., identical to difflib's
s.get_matching_blocks()  # [Match(a=0, b=0, size=8), Match(a=8, b=17, size=21), Match(a=29, b=38, size=0)]
s.get_opcodes()          # [('equal', 0, 8, 0, 8), ('insert', 8, 8, 8, 17), ('equal', 8, 29, 17, 38)]

# Drop-in get_close_matches (identical ranking and tie order):
difflib_mojo.get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
# ['apple', 'ape']

# Batched: 64 words x 4000 candidates in one kernel call; key= scores a
# transformed form (case-insensitive here) but returns the originals:
difflib_mojo.get_close_matches_batch(
    ["colour", "flavour"], ["color", "flavor", "COLOR", "savor"],
    n=2, cutoff=0.6, key=str.casefold)
# [['color', 'COLOR'], ['flavor']]
```

## Benchmark

Measured with `benchmarks/bench_difflib.py` in this repository
(`PYTHONPATH=python/difflib_mojo python benchmarks/bench_difflib.py` to
reproduce). Correctness is gated before every timing run: outputs must be
bit-identical to stdlib difflib or the benchmark aborts. Median of 5 runs,
single-threaded. Environment: **Apple M4 Max, macOS 26.6.2 arm64, Python
3.12.14, numpy 2.5.3, Mojo 1.1.0**, 2026-09-20.

Warm steady-state, per call:

| workload | stdlib difflib | difflib-mojo (native) | speedup |
|---|---:|---:|---:|
| W1 `ratio()` 12415+12513 chars, program text | 0.1270 s | 0.0024 s | **53.7×** |
| W2 `ratio()` 100k+100k chars (5k-codepoint alphabet, 2000 edits) | 0.7275 s | 0.0357 s | **20.4×** |
| W3 `get_close_matches` 1 word × 2000 candidates (20–80 chars) | 0.0855 s | 0.0053 s | **16.1×** |
| W4 `get_close_matches_batch` 48 words × 2000 candidates | 2.5574 s | 0.1521 s | **16.8×** |
| W5 `find_longest_match()` 100k × 100k (hot loop, isolated) | 0.1775 s | 0.0076 s | **23.4×** |

Forced pure-Python fallback (same workloads, vs the same stdlib baseline):

| workload | stdlib difflib | difflib-mojo (engine) | ratio |
|---|---:|---:|---:|
| W1 `ratio()` program text | 0.1274 s | 0.1798 s | 0.7× |
| W2 `ratio()` 100k pair | 0.7546 s | 0.9774 s | 0.8× |
| W3 `get_close_matches` 1 × 2000 | 0.0893 s | 0.1128 s | 0.8× |
| W4 batch 48 × 2000 | 2.5913 s | 3.3611 s | 0.8× |
| W5 `find_longest_match()` 100k × 100k | 0.1966 s | 0.2206 s | 0.9× |

The fallback's job is silent correctness, not speed: it runs at 0.7–0.9×
stdlib (same algorithmic complexity, a somewhat leaner inner loop than the
reference's dict formulation), so on platforms without the kernel you are
never slower than the standard library you would have used anyway — within
a factor of ~1.4.

Cold first call (fresh process: import + first `ratio()` + first
`get_close_matches`, median of 5 launches):

| variant | cold time |
|---|---:|
| stdlib difflib | 5.90 ms |
| difflib-mojo native | 184.68 ms |
| difflib-mojo fallback | 150.76 ms |

Cold start is dominated by the one-time numpy import (~145 ms of it — the
kernel's dlopen + first FFI adds ~35 ms on top of the fallback's); it is a
per-process cost, irrelevant to the long-running fuzzy-matching services
this package targets, and is reported here for completeness.

Why the gap is so large: stdlib's `find_longest_match` interprets a Python
dict-based DP row per (i, matching j) — hundreds of millions of interpreter
steps on multi-KB inputs. The Mojo kernel keeps the same algorithm (same
tie-breaking, same junk extension, same autojunk heuristic) but runs it
compiled: a CSR index of b built once per sequence, generation-stamped
double-buffered DP rows (no per-row or per-call resets), and one FFI call
per whole operation — batch scoring amortizes it across an entire
words × candidates grid.

## How it works

```
pip install difflib-mojo
        │
        ▼
difflib_mojo (thin Python wrapper)
        │  encodes str -> int32 codepoints (UTF-32-LE, surrogate-safe),
        │  applies the isjunk callable per unique element and the autojunk
        │  popularity rule (len(b) >= 200, count > len(b)//100 + 1),
        │  hands the kernel per-position masks
        ▼
libdifflibmojo.dylib / .so           (Mojo kernel, C ABI v1)
        │  difflibmojo_matching_blocks / _find_longest_match /
        │  _quick_ratio / _close_matches_batch
        ▼
raw triples / scores — sorted, collapsed, and ranked in Python with the
reference's exact rules (tuple-order ties, heapq.nlargest semantics)
```

- **Batch-shaped C ABI**: one call computes all matching blocks of a pair,
  or scores every (word, candidate) pair with the three-tier
  `real_quick_ratio` → `quick_ratio` → `ratio` filter; FFI overhead is
  per-call, not per-element.
- **Reference-matching arithmetic**: ratios are `2.0*M/T` in IEEE-754
  float64 with the reference's operation order (and its
  `length == 0 -> 1.0` rule), so floats compare bit-identical.
- **ABI handshake**: the wrapper checks `difflibmojo_abi_version()` before
  calling; a mismatch falls back cleanly.
- **Zero-dependency at runtime**: numpy is the only hard dependency
  (codepoint arrays shared with the kernel).

## Fallback semantics

There is no Windows Mojo toolchain today, and a shared library can always
go missing — so the wrapper **falls back to a vendored pure-Python engine**
(`difflib_mojo/_fallback.py`, clean-room, standard library only):

- Resolution order: `$DIFFLIB_MOJO_NATIVE_LIB` → the library bundled in
  the wheel → the repo development build output.
- `DIFFLIB_MOJO_DISABLE_NATIVE=1` forces the fallback (the test suite runs
  this way as its second pass).
- The engine shares the kernel's architecture (CSR index + salted
  double-buffered DP rows) rather than the reference's dict formulation;
  the differential suite asserts it is output-identical to stdlib on every
  fixture, so the fallback is **silently correct**, just slower.
- Non-`str` sequences (lists of hashables, etc.) always run on the engine —
  the kernel accepts int32 codepoints only — and match stdlib too.
- Inspect what's active: `difflib_mojo.backend_info()` and
  `difflib_mojo.native_available()`.
- Wheels are **per-platform** (`py3-none-macosx_*_arm64`,
  `py3-none-manylinux_*_x86_64`) and **wheel-only**. Each wheel is
  **self-contained**: `delocate` (macOS) / `auditwheel repair` (Linux)
  vendor the Mojo runtime libraries and rewrite load paths to be
  wheel-relative. (Redistribution terms for Modular's runtime binaries
  should be confirmed with Modular before any public release.) A pure
  `py3-none-any` fallback wheel can be produced with
  `DIFFLIB_MOJO_ALLOW_PURE_WHEEL=1`.

## Differential tests

```
PYTHONPATH=python/difflib_mojo bash scripts/test_all_difflib.sh
# builds nothing itself — run `bash kernels/difflib/build.sh` first —
# then runs the suite twice: once native, once DIFFLIB_MOJO_DISABLE_NATIVE=1
```

The suite (`tests/test_difflib_*.py`) compares difflib-mojo against the
running interpreter's stdlib difflib with **zero tolerance** on
deterministic seeded fixtures: random grids over 8 alphabets (sizes
0–1500), empty strings, identical/substring/reversed pairs, long common
prefixes, unicode (astral planes, combining marks, lone surrogates),
junk-heavy inputs with three junk callables, the autojunk on/off ×
popularity-threshold matrix around len(b) = 200, the all-popular-purged
regime, 100KB strings (sparse alphabet, 2000-edit copy, tiny alphabet),
a ~12KB program-text diff, `find_longest_match` windows, non-str
sequences, caching/identity semantics, `get_close_matches` grids with
forced score ties (tie order verified), error messages, and the batch
APIs with `key=str.casefold` / `key=lambda` and duplicate forms.
34 tests pass per backend (68 per full run), and the benchmark asserts
the same bit-identity gate before every timing run.

## Scope and limitations

Honest list of what this package does **not** do:

- **Supported surface**: `SequenceMatcher` (all comparison methods),
  `get_close_matches`, `Match`, `IS_CHARACTER_JUNK`, `IS_LINE_JUNK`, plus
  the `get_close_matches_batch` / `ratio_batch` extensions. The rest of
  difflib's module API — `Differ`, `ndiff`, `unified_diff`, `context_diff`,
  `restore`, `diff_bytes`, `HtmlDiff` — is **not included** (those are
  line-oriented formatters, not the hot loop this kernel targets).
- **Sequences**: the native kernel accelerates `str` inputs (any
  codepoint, including lone surrogates). Arbitrary hashable sequences work
  on the pure-Python engine on both backends, identical to stdlib.
- **Introspection attributes** `b2j` / `fullbcount` are computed lazily on
  first read (the reference computes them eagerly) and are read-only:
  **assigning to or mutating** `b2j`, `bjunk`, `bpopular`, or `fullbcount`
  after `set_seq2` has no effect on results (the reference honors such
  mutation; its own docs tell users not to rely on it). Read values are
  identical to the reference's.
- `find_longest_match` **validates its window** and raises `ValueError`
  for out-of-range arguments; the reference's behavior is undefined there
  (nonsense triples or `IndexError` depending on values).
- `get_close_matches` **materializes** `possibilities` once (the reference
  iterates lazily); results are identical for any finite iterable.
- Single-threaded kernel (determinism first); multithreading is future
  work and would multiply the batch speedup.
- Results are verified against **CPython 3.12** difflib (the interpreter
  this package is tested with). Other Python versions' difflib may differ
  in edge behavior; the differential suite always compares against the
  difflib of the interpreter running it.

## Citing

If you use difflib-mojo in academic work, please cite this package:

> difflib-mojo: Mojo-accelerated difflib SequenceMatcher and
> get_close_matches. thyn-ai, 2026.
> https://github.com/thyn-ai/mojo-kernels
> (citation file with DOI forthcoming)

The algorithm is Ratcliff and Obershelp's "gestalt pattern matching"
(*Dr. Dobb's Journal*, July 1988), as popularized by Python's difflib.

## License

Apache-2.0, © 2026 Algenta The kernel and wrapper are clean-room
implementations of the documented gestalt-pattern-matching behavior.
CPython's difflib is used only as the test/benchmark reference, never as a
runtime dependency.
