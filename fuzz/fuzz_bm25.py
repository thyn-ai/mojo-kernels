"""Differential fuzzing for bm25-mojo: Mojo kernel vs fallback vs rank_bm25.

Every input decodes to one BM25 scenario -- a variant (Okapi, L, Plus), its
parameters, a small token corpus and a handful of queries -- which is then
scored three ways:

* the native Mojo kernel (``bm25_mojo.BM25*.get_scores``),
* the vendored pure-Python fallback on the *same* index
  (``BM25._reference_scores``), and
* the ``rank_bm25`` 0.2.2 oracle, when it is installed (it is, in the pixi
  environment).

The kernel and the fallback must agree within the documented parity
tolerance (1e-8 absolute, ``tests/conftest.py``); the fallback must be
bit-identical to rank_bm25; ``get_top_n`` must rank the same documents up
to ties inside that tolerance. Two flavours of parameters are generated:
in-domain values (``k1`` in [0, 5], ``b`` in [0, 1], ``epsilon``/``delta``
in [0, 3]) and raw IEEE doubles (anything: negative, huge, subnormal, NaN,
infinite), so the parameter-validation and IEEE-propagation behaviour is
fuzzed too. Corpora mix ASCII tokens, Unicode tokens (accents, CJK, emoji,
the empty string), empty documents and an empty corpus; queries mix known,
repeated and unseen terms.

Run modes (see ``_harness.py``)::

    pixi run -e fuzz fuzz-bm25 -- -max_total_time=60   # atheris, Linux x86_64
    pixi run fuzz-regression-bm25                       # replay fuzz/corpus/bm25
    python fuzz/fuzz_bm25.py --regression path/to/crash-file

Known divergences are listed in ``KNOWN_ISSUES`` below, each tied to an
open GitHub issue and reproduced by a ``known-issue-*.bin`` seed.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

try:
    import atheris
except ImportError:  # regression replay, or a platform without an atheris wheel
    atheris = None

import numpy as np

from _harness import (
    FUZZ_DIR,
    ByteCursor,
    ByteWriter,
    Divergence,
    KnownIssue,
    main,
)
from _harness import replay_seed as _replay_seed

if atheris is not None:
    # Coverage feedback for the wrapper and the fallback; numpy is already
    # imported above and is not instrumented.
    with atheris.instrument_imports(include=["bm25_mojo"]):
        import bm25_mojo
else:
    import bm25_mojo

try:
    import rank_bm25
except ImportError:  # the oracle is a test dependency, never a runtime one
    rank_bm25 = None

NAME = "bm25"
CORPUS_DIR = FUZZ_DIR / "corpus" / NAME

# Documented parity tolerance (tests/conftest.py SCORE_ATOL). The relative
# term only matters for out-of-domain parameters that blow scores up past
# ~1e5, where an absolute 1e-8 is below one ulp; at ~450 ulps it is still
# orders of magnitude below any formula or indexing error.
SCORE_ATOL = 1e-8
SCORE_RTOL = 1e-13

VARIANT_NAMES = ("BM25Okapi", "BM25L", "BM25Plus")
_THIRD_PARAM = ("epsilon", "delta", "delta")
_OURS = (bm25_mojo.BM25Okapi, bm25_mojo.BM25L, bm25_mojo.BM25Plus)
_ORACLE = (
    (rank_bm25.BM25Okapi, rank_bm25.BM25L, rank_bm25.BM25Plus)
    if rank_bm25 is not None
    else None
)

ASCII_TOKENS = tuple(f"w{i:02d}" for i in range(48))
# Accents, CJK, emoji (astral plane), a combining mark, the empty string, a
# lone space and a token containing a space: everything a tokenizer might
# hand over. Case is significant in rank_bm25 ("Apple" != "apple").
UNICODE_TOKENS = (
    "café",
    "naïve",
    "日本語",
    "😀",
    "Ω",
    "ß",
    "İ",
    "Apple",
    "apple",
    "́",
    "",
    " ",
    "a b",
)
UNSEEN_TOKEN = "zzz-unseen"

MAX_DOCS = 32
MAX_DOC_LEN = 24
MAX_QUERIES = 4
MAX_QUERY_LEN = 6


@dataclass(frozen=True)
class Case:
    variant: int  # index into VARIANT_NAMES
    raw_params: bool  # k1/b/third are raw doubles rather than in-domain values
    k1: float
    b: float
    third: float  # epsilon for Okapi, delta for L and Plus
    vocab_size: int  # ASCII vocabulary prefix length (1..48)
    with_unicode: bool
    corpus: tuple[tuple[str, ...], ...]
    queries: tuple[tuple[str, ...], ...]

    @property
    def kwargs(self) -> dict[str, float]:
        return {"k1": self.k1, "b": self.b, _THIRD_PARAM[self.variant]: self.third}

    def describe(self) -> str:
        return (
            f"{VARIANT_NAMES[self.variant]}({self.kwargs}) "
            f"corpus={list(map(list, self.corpus))!r} queries={list(map(list, self.queries))!r}"
        )


def _vocab(vocab_size: int, with_unicode: bool) -> tuple[str, ...]:
    vocab = ASCII_TOKENS[:vocab_size]
    return vocab + UNICODE_TOKENS if with_unicode else vocab


def decode(data: bytes) -> Case:
    """Total decoder: any byte string is one scenario (see ``ByteCursor``)."""
    cur = ByteCursor(data)
    variant = cur.int_in(0, 2)
    flags = cur.u8()
    raw_params = bool(flags & 1)
    allow_empty_docs = bool(flags & 2)
    with_unicode = bool(flags & 4)
    # One input in 32 (never an exhausted, all-zero input): the empty-corpus
    # ZeroDivisionError parity path.
    empty_corpus = (flags >> 3) == 0x1F
    if raw_params:
        k1, b, third = cur.f64(), cur.f64(), cur.f64()
    else:
        k1, b, third = cur.unit() * 5.0, cur.unit(), cur.unit() * 3.0
    vocab_size = cur.int_in(1, len(ASCII_TOKENS))
    vocab = _vocab(vocab_size, with_unicode)
    corpus: list[tuple[str, ...]] = []
    if not empty_corpus:
        for _ in range(cur.int_in(1, MAX_DOCS)):
            n_tokens = cur.int_in(0 if allow_empty_docs else 1, MAX_DOC_LEN)
            corpus.append(tuple(cur.choice(vocab) for _ in range(n_tokens)))
    queries: list[tuple[str, ...]] = []
    for _ in range(cur.int_in(1, MAX_QUERIES)):
        query = [cur.choice(vocab) for _ in range(cur.int_in(0, MAX_QUERY_LEN))]
        if cur.flag():
            query.append(UNSEEN_TOKEN)
        queries.append(tuple(query))
    return Case(
        variant=variant,
        raw_params=raw_params,
        k1=k1,
        b=b,
        third=third,
        vocab_size=vocab_size,
        with_unicode=with_unicode,
        corpus=tuple(corpus),
        queries=tuple(queries),
    )


def encode(case: Case) -> bytes:
    """Exact inverse of ``decode`` for hand-crafted (minimised) seeds.

    In-domain parameters are quantised to 16 bits, so ``decode(encode(c))``
    reproduces ``c`` exactly for raw parameters and for in-domain values on
    the 1/65535 grid (0.0, 1.0, 2.5, ... -- every value a reproducer needs).
    """
    vocab = _vocab(case.vocab_size, case.with_unicode)
    w = ByteWriter()
    w.int_in(case.variant, 0, 2)
    flags = (
        (1 if case.raw_params else 0)
        | (2 if any(len(doc) == 0 for doc in case.corpus) else 0)
        | (4 if case.with_unicode else 0)
        | (0 if case.corpus else 0xF8)
    )
    w.u8(flags)
    if case.raw_params:
        w.f64(case.k1).f64(case.b).f64(case.third)
    else:
        w.unit(case.k1 / 5.0).unit(case.b).unit(case.third / 3.0)
    w.int_in(case.vocab_size, 1, len(ASCII_TOKENS))
    allow_empty_docs = bool(flags & 2)
    if case.corpus:
        w.int_in(len(case.corpus), 1, MAX_DOCS)
        for doc in case.corpus:
            w.int_in(len(doc), 0 if allow_empty_docs else 1, MAX_DOC_LEN)
            for token in doc:
                w.choice(vocab.index(token), vocab)
    w.int_in(len(case.queries), 1, MAX_QUERIES)
    for query in case.queries:
        terms = list(query)
        unseen = bool(terms) and terms[-1] == UNSEEN_TOKEN
        if unseen:
            terms.pop()
        w.int_in(len(terms), 0, MAX_QUERY_LEN)
        for token in terms:
            w.choice(vocab.index(token), vocab)
        w.flag(unseen)
    return w.bytes()


# ---------------------------------------------------------------------------
# Known divergences (each tied to an open issue and a known-issue-*.bin seed)
# ---------------------------------------------------------------------------


def _degenerate_parameters(case: Case) -> bool:
    """Inputs for which rank_bm25 evaluates 0/0 for documents that do not
    contain a query term.

    rank_bm25 (and the vendored fallback) evaluate the per-document factor
    densely over the whole corpus, so a document with ``qf == 0`` computes
    ``0 * (k1 + 1) / (0 + k1 * norm)`` (Okapi/Plus) or ``0 / norm`` and
    ``0 / (k1 + 0 + delta)`` (L), with ``norm = 1 - b + b * dl / avgdl``.
    Whenever that denominator is exactly 0 or NaN the reference yields NaN;
    the kernel only visits posted documents, so it yields 0 (or the BM25Plus
    floor) there. That happens for ``k1 == 0`` (incl. underflow of
    ``k1 * norm``), ``b == 1`` with an empty document, any ``b > 1`` that
    zeroes ``norm``, non-finite parameters, and BM25L with ``k1 + delta == 0``.
    """
    if not all(math.isfinite(v) for v in (case.k1, case.b, case.third)):
        return True
    if not case.corpus:
        return False
    doc_len = np.array([len(doc) for doc in case.corpus], dtype=np.float64)
    avgdl = doc_len.sum() / doc_len.shape[0]
    if avgdl == 0:
        return False  # the wrapper itself routes avgdl == 0 to the fallback
    with np.errstate(all="ignore"):
        norm = 1.0 - case.b + case.b * doc_len / avgdl
        if case.variant == 1:  # BM25L
            ctd_den = norm
            outer_den = np.full_like(norm, case.k1 + case.third)
        else:  # Okapi / Plus
            ctd_den = np.ones_like(norm)
            outer_den = case.k1 * norm
    for den in (ctd_den, outer_den):
        if np.any(den == 0.0) or not np.all(np.isfinite(den)):
            return True
    return False


ISSUE_DEGENERATE_NAN = KnownIssue(
    key="degenerate-nan",
    url="https://github.com/thyn-ai/mojo-kernels/issues/15",
    title=(
        "bm25-mojo: kernel scores unposted documents 0 where rank_bm25 yields NaN "
        "(k1 == 0, b == 1 with empty documents, b > 1, non-finite parameters)"
    ),
    applies=_degenerate_parameters,
)

KNOWN_ISSUES: tuple[KnownIssue, ...] = (ISSUE_DEGENERATE_NAN,)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise parity: equal NaN/inf, else within atol + rtol * |b|."""
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(all="ignore"):
        finite_close = np.abs(a - b) <= SCORE_ATOL + SCORE_RTOL * np.abs(b)
    return both_nan | (a == b) | (np.isfinite(a) & np.isfinite(b) & finite_close)


def _check_score_array(scores: object, n_docs: int, what: str) -> None:
    if not isinstance(scores, np.ndarray):
        raise Divergence(f"{what}: expected np.ndarray, got {type(scores).__name__}")
    if scores.dtype != np.float64 or scores.shape != (n_docs,):
        raise Divergence(f"{what}: expected float64[{n_docs}], got {scores.dtype}{scores.shape}")


def _compare_backends(
    case: Case, query: list[str], native: np.ndarray, fallback: np.ndarray
) -> str | None:
    """None when the kernel matches the fallback; a KnownIssue key when the
    disagreement has exactly a documented shape; raises otherwise."""
    ok = _close(native, fallback)
    if ok.all():
        return None
    # Known shape: the reference is NaN where the kernel is finite, and every
    # other position agrees.
    nan_only = np.isnan(fallback) & ~np.isnan(native)
    if ISSUE_DEGENERATE_NAN.applies(case) and np.all(ok | nan_only):
        return ISSUE_DEGENERATE_NAN.key
    bad = np.flatnonzero(~ok)
    raise Divergence(
        f"native kernel != fallback for query {query!r} at documents {bad.tolist()}: "
        f"native={native[bad]} fallback={fallback[bad]}\n  case: {case.describe()}"
    )


def _same_index_attribute(a: object, b: object) -> bool:
    """Equality that treats NaN == NaN (a non-finite ``epsilon`` floors negative
    idf values to NaN on both sides; a plain ``==`` would report a difference)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_index_attribute(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_index_attribute(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_ranking_up_to_ties(ours: list, ref: list, scores: np.ndarray) -> bool:
    """Rankings may only differ where the reference scores tie within tolerance."""
    if len(ours) != len(ref):
        return False
    for a, b in zip(ours, ref):
        if a != b and not _close(np.array([scores[a]]), np.array([scores[b]]))[0]:
            return False
    return True


def _construct(cls, corpus: list[list[str]], kwargs: dict[str, float]):
    """Build an index, returning (instance, None) or (None, ZeroDivisionError).

    rank_bm25 raises ZeroDivisionError for an empty corpus (``avgdl``) and,
    for BM25Okapi, for a corpus whose documents are all empty (the
    ``average_idf`` over an empty vocabulary); bm25_mojo mirrors both
    (tests/test_differential.py::test_empty_corpus_raises_like_reference).
    Any other exception is undocumented and propagates as a Divergence.
    """
    try:
        return cls(corpus, **kwargs), None
    except ZeroDivisionError as exc:
        return None, exc


def evaluate(case: Case) -> str | None:
    """Run one scenario on every backend; return the known-issue key it hit, if any."""
    corpus = [list(doc) for doc in case.corpus]
    kwargs = case.kwargs
    ours_cls = _OURS[case.variant]
    oracle_cls = _ORACLE[case.variant] if _ORACLE is not None else None

    ours, ours_error = _construct(ours_cls, corpus, kwargs)
    oracle, oracle_error = (
        _construct(oracle_cls, corpus, kwargs) if oracle_cls is not None else (None, None)
    )
    if oracle_cls is not None and (ours_error is None) != (oracle_error is None):
        raise Divergence(
            f"construction differs from rank_bm25: bm25_mojo {ours_error!r}, "
            f"rank_bm25 {oracle_error!r}\n  case: {case.describe()}"
        )
    if ours_error is not None:
        if corpus and any(corpus):
            raise Divergence(
                f"ZeroDivisionError on a corpus with tokens: {ours_error}\n  case: {case.describe()}"
            )
        return None  # documented: empty corpus, or all-empty documents under BM25Okapi
    n_docs = len(corpus)
    documents = [f"d{i}" for i in range(n_docs)]
    outcome: str | None = None

    if oracle is not None:
        # Index attributes are shared by both of our backends; they must
        # match the oracle exactly (same idf, same lengths, same avgdl).
        for attr in ("corpus_size", "avgdl", "doc_len", "doc_freqs", "idf"):
            if not _same_index_attribute(getattr(ours, attr), getattr(oracle, attr)):
                raise Divergence(
                    f"index attribute {attr} differs from rank_bm25: "
                    f"{getattr(ours, attr)!r} vs {getattr(oracle, attr)!r}\n  case: {case.describe()}"
                )

    for query_t in case.queries:
        query = list(query_t)
        native = ours.get_scores(query)
        _check_score_array(native, n_docs, "get_scores")
        fallback = ours._reference_scores(query)  # noqa: SLF001 -- same index, fallback scorer
        _check_score_array(fallback, n_docs, "_reference_scores")
        hit = _compare_backends(case, query, native, fallback)
        outcome = outcome or hit

        if oracle is not None:
            ref = oracle.get_scores(query)
            if not np.array_equal(fallback, ref, equal_nan=True):
                bad = np.flatnonzero(~(np.isnan(fallback) & np.isnan(ref)) & (fallback != ref))
                raise Divergence(
                    f"fallback != rank_bm25 for query {query!r} at documents {bad.tolist()}: "
                    f"fallback={fallback[bad]} rank_bm25={ref[bad]}\n  case: {case.describe()}"
                )
            if hit is None and not np.isnan(ref).any():
                # Ranking parity, tolerant to ties within the score tolerance.
                for n in (1, 5, n_docs + 1):
                    top_ours = ours.get_top_n(query, documents, n=n)
                    top_ref = oracle.get_top_n(query, documents, n=n)
                    idx_ours = [documents.index(d) for d in top_ours]
                    idx_ref = [documents.index(d) for d in top_ref]
                    if not _same_ranking_up_to_ties(idx_ours, idx_ref, ref):
                        raise Divergence(
                            f"get_top_n(n={n}) ranks differently for query {query!r}: "
                            f"ours={top_ours} rank_bm25={top_ref}\n  case: {case.describe()}"
                        )

    if n_docs >= 2:
        # Documented fail-fast: a documents list of the wrong length asserts.
        try:
            ours.get_top_n(list(case.queries[0]), documents[:-1], n=1)
        except AssertionError:
            pass
        else:
            raise Divergence("get_top_n accepted a documents list of the wrong length")
    return outcome


def test_one_input(data: bytes) -> str | None:
    """Fuzz target: decode, evaluate, and classify the outcome.

    Returns None (parity), a known-issue key (documented open divergence),
    or raises ``Divergence`` for anything else -- including any exception
    type the package does not document for these inputs.
    """
    case = decode(data)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # 0/0 and overflow warnings from numpy are the reference's
        # behaviour under test, not noise to act on.
        warnings.simplefilter("ignore")
        try:
            return evaluate(case)
        except Divergence:
            raise
        except Exception as exc:
            raise Divergence(
                f"undocumented {type(exc).__name__}: {exc}\n  case: {case.describe()}"
            ) from exc


def native_available() -> bool:
    return bm25_mojo.native_available()


def replay_seed(seed: Path) -> str | None:
    """Replay one corpus file with the strict known-issue rules (pytest entry)."""
    return _replay_seed(seed, test_one_input, KNOWN_ISSUES)


def _banner() -> str:
    info = bm25_mojo.backend_info()
    backend = (
        f"native kernel: {info['native_source']}"
        if info["native_available"]
        else f"native kernel UNAVAILABLE ({info['error']}); only fallback-vs-oracle checks are live"
    )
    oracle = "rank_bm25 oracle: installed" if rank_bm25 else "rank_bm25 oracle: not installed"
    return f"{backend}\n{oracle}"


def _require_native() -> str | None:
    info = bm25_mojo.backend_info()
    return None if info["native_available"] else str(info["error"])


if __name__ == "__main__":
    sys.exit(
        main(
            NAME,
            test_one_input,
            KNOWN_ISSUES,
            CORPUS_DIR,
            banner=_banner(),
            require_native=_require_native,
        )
    )
