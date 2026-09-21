"""sacrebleu-mojo: drop-in faster replacements for the `sacrebleu` package's
BLEU and chrF corpus metrics.

Same call shapes, same scores — powered by a Mojo kernel where the platform
supports it (macOS arm64, Linux x86_64), with a vendored pure-Python
fallback everywhere else (including Windows).

    import sacrebleu_mojo

    bleu = sacrebleu_mojo.corpus_bleu(hyps, [refs])   # like sacrebleu.corpus_bleu
    bleu.score, bleu.counts, bleu.precisions, bleu.bp, bleu.sys_len, bleu.ref_len

    chrf = sacrebleu_mojo.corpus_chrf(hyps, [refs])   # like sacrebleu.corpus_chrf
    chrf.score

Set SACREBLEU_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from sacrebleu_mojo._native import backend_info, native_available
from sacrebleu_mojo.core import (
    BLEUScore,
    CHRFScore,
    corpus_bleu,
    corpus_chrf,
    tokenize_13a,
    word_tokens,
)

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "BLEUScore",
    "CHRFScore",
    "corpus_bleu",
    "corpus_chrf",
    "tokenize_13a",
    "word_tokens",
    "backend_info",
    "native_available",
    "__version__",
]
