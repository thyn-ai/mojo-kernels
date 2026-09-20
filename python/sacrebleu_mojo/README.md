# sacrebleu-mojo

A drop-in faster replacement for [`sacrebleu`](https://pypi.org/project/sacrebleu/)'s
corpus BLEU and chrF metrics, powered by a clean-room Mojo kernel — with a
vendored pure-Python fallback for platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import sacrebleu_mojo  # same call shapes as sacrebleu

bleu = sacrebleu_mojo.corpus_bleu(hypotheses, [reference_stream])
bleu.score        # matches sacrebleu.corpus_bleu(...).score
bleu.counts, bleu.totals, bleu.precisions, bleu.bp, bleu.sys_len, bleu.ref_len

chrf = sacrebleu_mojo.corpus_chrf(hypotheses, [reference_stream])
chrf.score        # matches sacrebleu.corpus_chrf(...).score
```

- **Same results**: `.score` matches the published `sacrebleu` package
  (tested against 2.5.1) within 1e-9 on both the native and fallback
  backends. BLEU is bit-exact in practice — counts, totals, precisions and
  the brevity penalty are asserted exactly equal by the differential suite;
  chrF is bit-exact in the large majority of cases and otherwise within a
  couple of ulps.
- **Same defaults**: BLEU with the mteval-v13a (`'13a'`) tokenizer and
  `'exp'` smoothing (plus `'floor'`/`'none'` and `use_effective_order`);
  chrF with `char_order=6`, `word_order=0` (0-4 supported), `beta=2`,
  `remove_whitespace=True`, and multi-reference best-per-sentence selection.
- **Much faster**: integer-exact n-gram statistics from a compiled Mojo
  kernel instead of nested Python `Counter` loops — see the benchmark table
  below (measured on this machine; full method in `benchmarks/bench_sacrebleu.py`).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `SACREBLEU_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `sacrebleu_mojo.backend_info()`.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.5,
numpy 2.5.3, Mojo 1.1.0, oracle sacrebleu 2.5.1) on 2026-09-19, with
synthetic hypothesis/reference corpora (10-40 token sentences from a
Zipf-ish 12k-term vocabulary; correctness to the oracle asserted within
1e-9 before timing — measured agreement was 0.0). "Cold" is the first call
after import; "warm" is the median of 5 repeated calls. Reproduce with
`PYTHONPATH=python/sacrebleu_mojo python benchmarks/bench_sacrebleu.py`.

BLEU (`corpus_bleu`, 13a tokenizer):

| pairs | sacrebleu cold (s) | sacrebleu_mojo cold (s) | sacrebleu warm (s) | sacrebleu_mojo warm (s) | cold speedup | warm speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.3258 | 0.1410 | 0.3825 | 0.1294 | 2.31x | 2.95x |
| 10,000 | 1.5659 | 0.5584 | 1.7333 | 0.5575 | 2.80x | 3.11x |
| 30,000 | 4.6475 | 1.7285 | 4.5162 | 1.5534 | 2.69x | 2.91x |

chrF (`corpus_chrf`):

| pairs | sacrebleu cold (s) | sacrebleu_mojo cold (s) | sacrebleu warm (s) | sacrebleu_mojo warm (s) | cold speedup | warm speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.9505 | 0.0738 | 1.0730 | 0.0713 | 12.88x | 15.04x |
| 10,000 | 5.2638 | 0.4947 | 4.2024 | 0.2980 | 10.64x | 14.10x |
| 30,000 | 14.2391 | 1.8352 | 20.4381 | 1.3486 | 7.76x | 15.15x |

BLEU's end-to-end speedup is bounded by the mteval-v13a tokenization, which
runs in Python on both sides (identical text in, identical tokens out); the
Mojo kernel computes the corpus n-gram statistics ~20x faster than the
reference loops. chrF's character n-gram matching is almost entirely inside
the kernel, hence the larger speedup. The pure-Python fallback is
correctness-first (for platforms without a native build), not tuned for
speed.

## Scope and limitations

- Supported: `corpus_bleu` / `corpus_chrf` with default tokenizers
  (`'13a'` for BLEU, none for chrF), BLEU smoothing methods `'exp'`
  (default) and `'floor'` and `use_effective_order`, chrF `char_order` 1-6,
  `word_order` 0-4, any numeric `beta`, `remove_whitespace` both values.
- Not supported (raises a clear error instead of guessing): BLEU tokenizers
  other than `'13a'`/`'none'`, chrF `char_order` outside [1, 6] or
  `word_order` outside [0, 4], and `eps_smoothing=True` (its exact reference
  behavior could not be pinned empirically; the default `False` is fully
  supported). Ragged inputs follow the oracle's zip-with-longest-stream
  truncation semantics.
- chrF sentence-level selection over multiple references is exact (ties keep
  the earliest reference, like the oracle).

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
