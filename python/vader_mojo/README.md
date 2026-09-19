# vader-mojo

Drop-in faster replacement for the [`vaderSentiment`](https://github.com/cjhutto/vaderSentiment)
package — VADER sentiment scoring powered by a **Mojo kernel**, with a vendored
pure-Python fallback everywhere the native library can't run.

Same `polarity_scores` contract, same scores. The Mojo kernel replicates the
complete VADER rule set: punctuation emphasis (exclamation flooding cap,
question-mark tiers), ALL-CAPS emphasis, booster/dampener degree modifiers,
the 3-word negation window, `but` contrast, `no`/`least`/`never so-this`/
`without doubt` special cases, sentiment-laden idioms (`the shit`, `bad ass`,
`yeah right`, ...), and emoji/emoticon handling.

- **Native backend** (macOS arm64, Linux x86_64/aarch64): the rules engine runs
  inside a Mojo shared library; measured **4.1x** faster steady-state (see
  Benchmarks).
- **Fallback backend** (Windows, or any host where the shared library is
  missing/broken): a vendored pure-Python implementation of the same rules.
  Results are identical; the switch is silent and automatic.
- Zero runtime dependencies. The lexicon data is vendored in the wheel.

```python
import vader_mojo

vader_mojo.polarity_scores("VADER is smart, handsome, and funny!")
# {'neg': 0.0, 'neu': 0.248, 'pos': 0.752, 'compound': 0.8439}

# or the drop-in class, same as vaderSentiment's:
from vader_mojo import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
analyzer.polarity_scores("Today SUX!")
# {'neg': 0.779, 'neu': 0.221, 'pos': 0.0, 'compound': -0.5461}

analyzer.backend          # "native" or "fallback"
vader_mojo.backend_info() # full diagnostics (resolver path, ABI version, ...)
```

## Install

```bash
pip install vader_mojo-<version>-py3-none-<platform>.whl
```

Platform wheels (macOS arm64, Linux x86_64/aarch64) carry the compiled kernel
plus its Mojo runtime libraries, repaired to be self-contained (delocate /
auditwheel): they work on machines with no Mojo toolchain installed. On
Windows, or if the shared library fails to load for any reason, the package
still works — on the fallback backend.

Environment variables:

- `VADER_MOJO_DISABLE_NATIVE=1` — force the pure-Python fallback (used by the
  differential test suite).
- `VADER_MOJO_NATIVE_LIB=/path/to/libvadermojo.{dylib,so}` — explicit kernel
  override for development.

## Score parity

Both backends are asserted, in CI on every push, to produce **exactly** the
reference package's output dict — `neg`/`neu`/`pos` rounded to 3 decimals and
`compound` to 4, compared with `==`, no tolerance — on:

- the 29 documented VADER demo + tricky sentences,
- 86 targeted cases (one per rule and edge path), and
- a 400-sentence seeded corpus mixing lexicon words, boosters, negators,
  idioms, emoticons, emojis (including ZWJ sequences, flags, keycaps, skin
  tones), Unicode-cased tokens (`É`, `TRÈS`, `ΩΩΩ`, `ДА`, `Ǆ`), Unicode
  whitespace, and lone surrogates.

Measured agreement on this machine (Apple M4 Max, 2026-09-19,
vaderSentiment 3.3.2): **exact dict equality on every one of the ~1,000
asserted sentences, on both backends.** Raw (pre-rounding) float64 scores
additionally agree to within 2e-3 — in practice bit-for-bit — because the
kernel evaluates IEEE-754 float64 in the reference's operation order and the
wrapper applies Python's `round()` to both backends.

Reference semantics are replicated deliberately, including the reference's
quirks (e.g. the operator-precedence behavior in the 3-word "never so/this"
arm, and the index-based mutation loop of the `but` check on duplicate
valences).

### Scope notes

- Casing follows Python's `str.isupper()`/`str.lower()` via embedded Unicode
  15.0 case tables (the reference interpreter's per-character ground truth),
  so CAPS emphasis behaves identically on non-ASCII tokens (`:-Þ`, `É`, ...).
- The only non-ASCII→ASCII lowercase mapping (U+212A KELVIN SIGN → `k`) is
  applied; every other non-ASCII code point lowercases to a non-ASCII form,
  which can never equal an ASCII lexicon/booster/negation key — so lookup
  outcomes are unchanged.
- The lexicon's two non-ASCII keys (`:-Þ`, `:Þ`) are unreachable in the
  reference itself (its per-token `.lower()` can never produce them) and are
  dropped from the native tables; emoji keys longer than one code point are
  likewise unreachable in its per-character lookup. Both drops are
  behavior-identical and covered by the differential suite.
- Windows: no Mojo toolchain builds there today, so Windows uses the fallback
  backend (identical scores, no native speedup).

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14,
Mojo 1.1.0, vaderSentiment 3.3.2 from PyPI), 2026-09-19. Corpus: 2,000 seeded
social-media-length sentences (3–30 tokens mixing lexicon words, boosters,
negators, emoticons, emojis, punctuation floods), generated locally by
`benchmarks/bench_vader.py` (seed 20260919). Correctness gate: exact dict
parity on a 100-sentence sample before any timing.

| regime | vaderSentiment | vader_mojo (native) | speedup |
|---|---:|---:|---:|
| cold: import + first call (ms, median of 5 fresh processes) | 28.4 | 33.2 | 0.85x |
| warm: ms/sentence (median of 5 runs, 2000 sentences) | 0.068 | 0.017 | 4.1x |
| warm: sentences/sec | 14,650 | 59,708 | 4.1x |

Cold-start is slightly slower than the reference (+4.8ms once per process):
the native analyzer build (7,520-entry lexicon + 1,893 emoji descriptions) and
`dlopen` cost more than the reference's Python dict construction. The vendored
fallback scores the same corpus at 0.062 ms/sentence (1.10x the reference —
context only, never the headline number).

Reproduce:

```bash
pixi run python benchmarks/bench_vader.py
```

## How it works

```
kernels/vader/src/vadermojo.mojo   # the Mojo rules engine (C ABI, float64)
python/vader_mojo/vader_mojo/
    _native.py                     # ctypes loader + ABI handshake + resolver
    _reference.py                  # vendored pure-Python implementation
    core.py                        # backend selection + reference rounding
    data/                          # vendored VADER lexicons (MIT, C.J. Hutto)
```

The ABI is versioned (`vadermojo_abi_version`): if the resolved shared library
reports an ABI the wrapper doesn't understand, the wrapper falls back instead
of risking wrong scores. The kernel returns raw float64 `neg`/`neu`/`pos`/
`compound`; `core.py` applies the reference's public rounding so results are
byte-identical on every platform. Scoring is stateless per call — one analyzer
handle is safe to share across threads.

## Development

```bash
bash kernels/vader/build.sh                 # build the Mojo kernel
pixi run bash scripts/test_all_vader.sh     # differential suite, both backends
pixi run bash python/vader_mojo/build_wheel.sh   # wheel + delocate/auditwheel repair
pixi run python benchmarks/bench_vader.py   # measured benchmark
```

The differential suite needs the oracle (`vaderSentiment==3.3.2`, pinned by
sha256); the test script downloads it once into `~/.cache/vader-oracle/`
(outside the pixi environment, which re-syncs and would wipe it).

## Attribution & licenses

- The sentiment rule set and lexicon data are **VADER**, created by
  **C.J. Hutto**: Hutto, C.J. & Gilbert, E.E. (2014), *VADER: A Parsimonious
  Rule-based Model for Sentiment Analysis of Social Media Text*, ICWSM-14.
  The vendored lexicon files ship under the MIT license — see
  [`vader_mojo/data/LICENSE.vaderSentiment.txt`](vader_mojo/data/LICENSE.vaderSentiment.txt).
- The Mojo kernel, Python wrapper, and fallback implementation are by
  **Algenta**, Apache-2.0.
- Wheels vendor Modular's Mojo runtime libraries (via delocate/auditwheel
  repair); their redistribution terms should be confirmed with Modular before
  any public release.
