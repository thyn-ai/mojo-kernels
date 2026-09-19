"""API-compatible BLEU and chrF metric functions, matching the `sacrebleu` package.

`sacrebleu_mojo.corpus_bleu(hyps, refs)` and `sacrebleu_mojo.corpus_chrf(hyps,
refs)` return score objects whose `.score` matches the published PyPI
`sacrebleu` package (differential suite tolerance: 1e-9; BLEU is bit-exact,
chrF is bit-exact in the large majority of cases and otherwise within a
couple of ulps).

Statistics run on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently fall back to
the vendored pure-Python reference implementation otherwise. Tokenization,
id encoding, and score assembly are shared between both backends, so results
are identical either way.

Out of scope (raises instead of guessing): BLEU tokenizers other than '13a'
and 'none', chrF with ``char_order`` outside [1, 6] or ``word_order`` outside
[0, 4], and ``eps_smoothing=True`` (its exact reference behavior is
undocumented and could not be pinned empirically; the default False is fully
supported).
"""

from __future__ import annotations

import array
import math
import re

import numpy as np

from sacrebleu_mojo import _native, _reference
from sacrebleu_mojo._native import NativeUnavailable

__all__ = [
    "BLEUScore",
    "CHRFScore",
    "corpus_bleu",
    "corpus_chrf",
    "tokenize_13a",
    "word_tokens",
]

MAX_ORDER = 4

# ---------------------------------------------------------------------------
# mteval-v13a language-independent tokenization (the Moses '13a' tokenizer,
# as published in the mteval-v13a.pl script).
# ---------------------------------------------------------------------------

_NORM_13A = [
    (re.compile(r"<skipped>"), ""),
    (re.compile(r"-\n"), ""),
    (re.compile(r"\n"), " "),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
]
_POST_13A = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),  # punctuation
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),  # ., unless after a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),  # ., unless before a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),  # dash after a digit
    (re.compile(r"\s+"), " "),
]


def tokenize_13a(line: str) -> str:
    """mteval-v13a language-independent tokenization."""
    for pattern, repl in _NORM_13A:
        line = pattern.sub(repl, line)
    line = " " + line + " "
    for pattern, repl in _POST_13A:
        line = pattern.sub(repl, line)
    return line.strip()


def _tokenize_13a_split(line: str) -> list[str]:
    """tokenize_13a(line).split() without the redundant final cleanup pass.

    The last 13a step collapses whitespace runs and strips the ends; `str.split`
    with no arguments already treats any whitespace run as one separator and
    ignores leading/trailing whitespace, so the token stream is identical.
    """
    for pattern, repl in _NORM_13A:
        line = pattern.sub(repl, line)
    line = " " + line + " "
    for pattern, repl in _POST_13A[:-1]:
        line = pattern.sub(repl, line)
    return line.split()


# ---------------------------------------------------------------------------
# chrF word tokenizer: per whitespace-delimited piece, a trailing
# non-alphanumeric char peels off as its own token (the glued remainder
# stays one token); otherwise a leading non-alphanumeric char peels off.
# Internal punctuation always stays glued. Peeling is not recursive.
# ---------------------------------------------------------------------------


def word_tokens(line: str) -> list[str]:
    out: list[str] = []
    for piece in line.split():
        if not piece:
            continue
        if not piece[-1].isalnum():
            core = piece[:-1]
            if core:
                out.append(core)
            out.append(piece[-1])
        elif not piece[0].isalnum():
            out.append(piece[0])
            core = piece[1:]
            if core:
                out.append(core)
        else:
            out.append(piece)
    return out


# ---------------------------------------------------------------------------
# Encoding helpers.
# ---------------------------------------------------------------------------


def _validate_inputs(hypotheses, references) -> None:
    if hypotheses is None or references is None:
        raise ValueError("hypotheses and references are required")
    if isinstance(hypotheses, (str, bytes)):
        raise ValueError("hypotheses must be a sequence of strings, not a single string")
    if isinstance(references, (str, bytes)):
        raise ValueError(
            "references must be a sequence of reference streams "
            "(e.g. [[ref1_sent1, ref1_sent2, ...], [ref2_sent1, ...]])"
        )
    if len(references) == 0:
        raise ValueError("references must contain at least one reference stream")
    for stream in references:
        if isinstance(stream, (str, bytes)):
            raise ValueError(
                "references must be a sequence of reference streams, not a "
                "sequence of strings (did you mean [[...]]?)"
            )


def _n_pairs(hypotheses, references) -> int:
    """Pair count under zip-with-longest-stream truncation semantics."""
    return min(len(hypotheses), max(len(s) for s in references))


class _Vocab(dict):
    """First-appearance-order string -> id encoder (C-speed via __missing__)."""

    def __missing__(self, key):
        idx = len(self)
        self[key] = idx
        return idx


def _encode_sentences(sentences, vocab: _Vocab, out_ids, out_off) -> None:
    get = vocab.__getitem__
    for sent in sentences:
        out_ids.extend(map(get, sent))
        out_off.append(len(out_ids))


# ---------------------------------------------------------------------------
# Score objects (attribute-compatible with sacrebleu's BLEUScore/CHRFScore).
# ---------------------------------------------------------------------------


class BLEUScore:
    """BLEU score, mirroring the public attributes of sacrebleu.BLEUScore."""

    def __init__(self, score, counts, totals, precisions, bp, sys_len, ref_len, backend):
        self.score = score
        self.counts = counts
        self.totals = totals
        self.precisions = precisions
        self.bp = bp
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.backend = backend  # additive diagnostic: "native" or "fallback"

    @property
    def ratio(self) -> float:
        return self.sys_len / self.ref_len if self.ref_len else 0.0

    def format(self, width: int = 2) -> str:
        precisions = "/".join(f"{p:.1f}" for p in self.precisions)
        return (
            f"BLEU = {self.score:.{width}f} {precisions} "
            f"(BP = {self.bp:.3f} ratio = {self.ratio:.3f} "
            f"hyp_len = {self.sys_len} ref_len = {self.ref_len})"
        )

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()


class CHRFScore:
    """chrF score, mirroring the public attributes of sacrebleu.CHRFScore."""

    def __init__(self, score, char_order, word_order, beta, backend):
        self.score = score
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta
        self.backend = backend  # additive diagnostic: "native" or "fallback"

    def format(self, width: int = 2) -> str:
        return f"chrF{self.beta}{'+' * self.word_order} = {self.score:.{width}f}"

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()


# ---------------------------------------------------------------------------
# BLEU score assembly (shared by both backends).
# ---------------------------------------------------------------------------


def _compute_bleu(correct, total, sys_len, ref_len, smooth_method, smooth_value,
                  use_effective_order):
    """BLEU score from sufficient statistics (percent precisions, NIST 'exp'
    smoothing, brevity penalty); mirrors the oracle's float operation order."""
    if sys_len == 0:
        bp = 1.0 if ref_len == 0 else 0.0
        return 0.0, [0.0] * MAX_ORDER, bp
    bp = math.exp(1 - ref_len / sys_len) if sys_len < ref_len else 1.0
    if smooth_method == "floor" and smooth_value is None:
        smooth_value = 0.1
    precisions = [0.0] * MAX_ORDER
    if correct[0] > 0:
        smooth = 1.0
        for n in range(1, MAX_ORDER + 1):
            t = total[n - 1]
            if t == 0:
                continue
            c = correct[n - 1]
            if c > 0:
                precisions[n - 1] = 100 * c / t
            elif smooth_method == "exp" and n > 1:
                smooth *= 2
                precisions[n - 1] = 100 / (smooth * t)
            elif smooth_method == "floor":
                precisions[n - 1] = 100 * smooth_value / t
    effective_order = MAX_ORDER
    if use_effective_order:
        effective_order = sum(1 for t in total if t > 0)
    relevant = precisions[:effective_order]
    if relevant and min(relevant) > 0:
        score = math.exp(sum(math.log(p) for p in relevant) / effective_order) * bp
    else:
        score = 0.0
    return score, precisions, bp


def _bleu_backend_stats(hyp_ids, hyp_off, ref_ids, ref_off, seg_index, n_pairs):
    """BLEU statistics on the native kernel with overflow-pair fallback."""
    stats = np.zeros(10, dtype=np.int64)

    def reference_pair(p: int) -> None:
        _reference.bleu_stats_range(
            hyp_ids, hyp_off, ref_ids, ref_off, seg_index, p, p + 1, stats
        )

    _native.bleu_stats(
        hyp_ids, hyp_off, ref_ids, ref_off, seg_index, 0, n_pairs, stats, reference_pair
    )
    return stats


# ---------------------------------------------------------------------------
# Public API: corpus_bleu.
# ---------------------------------------------------------------------------


def corpus_bleu(
    hypotheses,
    references,
    smooth_method: str = "exp",
    smooth_value=None,
    force: bool = False,  # noqa: ARG001 - accepted for API compatibility
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = False,
) -> BLEUScore:
    """Corpus BLEU with the default mteval-v13a tokenizer.

    Mirrors ``sacrebleu.corpus_bleu``: same defaults (smooth_method='exp',
    tokenize='13a'), same zip-with-longest-stream truncation for ragged
    inputs, same returned score attributes. ``.score`` matches the published
    PyPI package (differential tolerance 1e-9; bit-exact in practice).
    """
    _validate_inputs(hypotheses, references)
    if smooth_method not in ("exp", "floor", "none"):
        raise ValueError(f"unsupported smooth_method: {smooth_method!r}")
    if tokenize not in ("13a", "none"):
        raise ValueError(f"unsupported tokenizer: {tokenize!r} (supported: '13a', 'none')")
    tok = _tokenize_13a_split if tokenize == "13a" else (lambda s: s.split())

    n = _n_pairs(hypotheses, references)
    vocab = _Vocab()
    get = vocab.__getitem__
    hyp_ids = array.array("I")
    hyp_off = [0]
    ref_ids = array.array("I")
    ref_off = [0]
    seg_index = [0]
    for i in range(n):
        hyp = hypotheses[i]
        if lowercase:
            hyp = hyp.lower()
        hyp_ids.extend(map(get, tok(hyp)))
        hyp_off.append(len(hyp_ids))
        for stream in references:
            if i < len(stream):
                ref = stream[i]
                if lowercase:
                    ref = ref.lower()
                ref_ids.extend(map(get, tok(ref)))
                ref_off.append(len(ref_ids))
        seg_index.append(len(ref_off) - 1)

    hyp_ids_a = np.frombuffer(hyp_ids, dtype=np.dtype("<u4"))
    hyp_off_a = np.array(hyp_off, dtype=np.int64)
    ref_ids_a = np.frombuffer(ref_ids, dtype=np.dtype("<u4"))
    ref_off_a = np.array(ref_off, dtype=np.int64)
    seg_index_a = np.array(seg_index, dtype=np.int64)

    backend = "native"
    try:
        stats = _bleu_backend_stats(
            hyp_ids_a, hyp_off_a, ref_ids_a, ref_off_a, seg_index_a, n
        )
    except NativeUnavailable:
        backend = "fallback"
        stats = np.zeros(10, dtype=np.int64)
        # Plain Python lists index much faster than numpy scalars here.
        _reference.bleu_stats_range(
            hyp_ids_a.tolist(), hyp_off_a.tolist(), ref_ids_a.tolist(),
            ref_off_a.tolist(), seg_index_a.tolist(), 0, n, stats
        )

    correct = [int(stats[j]) for j in range(4)]
    total = [int(stats[4 + j]) for j in range(4)]
    sys_len = int(stats[8])
    ref_len = int(stats[9])
    score, precisions, bp = _compute_bleu(
        correct, total, sys_len, ref_len, smooth_method, smooth_value, use_effective_order
    )
    return BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len, backend)


# ---------------------------------------------------------------------------
# Public API: corpus_chrf.
# ---------------------------------------------------------------------------


def _chrf_backend_stats(hypc, hypc_off, refc, refc_off, seg_index,
                        hypw, hypw_off, refw, refw_off, n_pairs,
                        char_order, word_order, beta, orders):
    out_m = np.zeros(orders, dtype=np.int64)
    out_h = np.zeros(orders, dtype=np.int64)
    out_r = np.zeros(orders, dtype=np.int64)

    def reference_pair(p: int) -> None:
        _reference.chrf_stats_range(
            hypc, hypc_off, refc, refc_off, seg_index,
            hypw, hypw_off, refw, refw_off, p, p + 1,
            char_order, word_order, beta, out_m, out_h, out_r,
        )

    _native.chrf_stats(
        hypc, hypc_off, refc, refc_off, seg_index,
        hypw, hypw_off, refw, refw_off, 0, n_pairs,
        char_order, word_order, beta, out_m, out_h, out_r, reference_pair,
    )
    return out_m, out_h, out_r


def corpus_chrf(
    hypotheses,
    references,
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    """Corpus chrF (character n-grams plus optional word n-grams).

    Mirrors ``sacrebleu.corpus_chrf``: same defaults (char_order=6,
    word_order=0, beta=2, remove_whitespace=True), same per-sentence
    best-reference selection for multi-reference corpora, same returned
    score attributes. ``.score`` matches the published PyPI package
    (differential tolerance 1e-9).
    """
    _validate_inputs(hypotheses, references)
    if eps_smoothing:
        raise ValueError(
            "eps_smoothing=True is not supported: its exact reference "
            "behavior could not be pinned empirically; the default "
            "eps_smoothing=False is fully supported"
        )
    if not (1 <= char_order <= _native.MAX_CHAR_ORDER):
        raise ValueError(
            f"char_order must be in [1, {_native.MAX_CHAR_ORDER}], got {char_order}"
        )
    if not (0 <= word_order <= _native.MAX_WORD_ORDER):
        raise ValueError(
            f"word_order must be in [0, {_native.MAX_WORD_ORDER}], got {word_order}"
        )
    beta_f = float(beta)
    orders = char_order + word_order

    n = _n_pairs(hypotheses, references)
    vocab = _Vocab()
    get = vocab.__getitem__
    u4 = np.dtype("<u4")
    hypc = bytearray()
    hypc_off = [0]
    refc = bytearray()
    refc_off = [0]
    seg_index = [0]
    hypw = array.array("I")
    hypw_off = [0]
    refw = array.array("I")
    refw_off = [0]
    for i in range(n):
        hyp = hypotheses[i]
        chars = "".join(hyp.split()) if remove_whitespace else hyp
        hypc += chars.encode("utf-32-le")
        hypc_off.append(len(hypc) // 4)
        if word_order > 0:
            hypw.extend(map(get, word_tokens(hyp)))
            hypw_off.append(len(hypw))
        for stream in references:
            if i < len(stream):
                ref = stream[i]
                rchars = "".join(ref.split()) if remove_whitespace else ref
                refc += rchars.encode("utf-32-le")
                refc_off.append(len(refc) // 4)
                if word_order > 0:
                    refw.extend(map(get, word_tokens(ref)))
                    refw_off.append(len(refw))
        if word_order == 0:
            hypw_off.append(0)
            for stream in references:
                if i < len(stream):
                    refw_off.append(len(refw))
        seg_index.append(len(refc_off) - 1)

    hypc_a = np.frombuffer(bytes(hypc), dtype=u4)
    hypc_off_a = np.array(hypc_off, dtype=np.int64)
    refc_a = np.frombuffer(bytes(refc), dtype=u4)
    refc_off_a = np.array(refc_off, dtype=np.int64)
    seg_index_a = np.array(seg_index, dtype=np.int64)
    hypw_a = np.frombuffer(hypw, dtype=u4)
    hypw_off_a = np.array(hypw_off, dtype=np.int64)
    refw_a = np.frombuffer(refw, dtype=u4)
    refw_off_a = np.array(refw_off, dtype=np.int64)

    backend = "native"
    try:
        out_m, out_h, out_r = _chrf_backend_stats(
            hypc_a, hypc_off_a, refc_a, refc_off_a, seg_index_a,
            hypw_a, hypw_off_a, refw_a, refw_off_a, n,
            char_order, word_order, beta_f, orders,
        )
    except NativeUnavailable:
        backend = "fallback"
        out_m = np.zeros(orders, dtype=np.int64)
        out_h = np.zeros(orders, dtype=np.int64)
        out_r = np.zeros(orders, dtype=np.int64)
        # Plain Python lists index much faster than numpy scalars here.
        _reference.chrf_stats_range(
            hypc_a.tolist(), hypc_off_a.tolist(), refc_a.tolist(),
            refc_off_a.tolist(), seg_index_a.tolist(),
            hypw_a.tolist(), hypw_off_a.tolist(), refw_a.tolist(),
            refw_off_a.tolist(), 0, n,
            char_order, word_order, beta_f, out_m, out_h, out_r,
        )

    score = _reference.chrf_fscore(
        [int(v) for v in out_m], [int(v) for v in out_h], [int(v) for v in out_r], beta_f
    )
    return CHRFScore(score, char_order, word_order, beta, backend)
