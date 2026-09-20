"""Differential tests: sacrebleu_mojo must match the published sacrebleu package.

Run twice by `scripts/test_all_sacrebleu.sh`: once against the native Mojo
kernel and once with SACREBLEU_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle within SCORE_ATOL = 1e-9
on `.score` everywhere (BLEU is bit-exact in practice: counts, totals,
precisions, and bp are asserted exactly equal).

The oracle is the published PyPI package (tested against sacrebleu 2.5.1).
Everything is generated locally from explicit seeds — no network, no
randomness without a fixed seed.
"""

from __future__ import annotations

import os
import random

import pytest
import sacrebleu as oracle

import sacrebleu_mojo
from sacrebleu_mojo import corpus_bleu, corpus_chrf

SCORE_ATOL = 1e-9  # documented tolerance; BLEU is bit-exact in practice

WORDS = (
    "the quick brown fox jumps over lazy dog a an of in on at to and or "
    "but not is are was were be been being have has had do does did will "
    "would shall should can could may might must hello world test case "
    "unicode café naïve résumé über straße 日本語 中文 한국어 α β γ δ "
    "don't it's o'clock state-of-the-art 3.14 2,500 100-200 mr. dr. "
    "end. next, stop; go! run? (paren) [brack] {brace} <tag> &amp; "
    "&quot;quoted&quot; #hash @mention $price %percent ^caret *star "
    "under_score hyphen-word slash/path back\\slash tilde~ pipe| "
    "a bb ccc dddd eeeee ffffff gg h iii "
    "(x) [y] {z} <w> 'q' \"v\" a'b c'd' 'e f' g' h'' i! j? k:l; m.n,o p-q "
    "r*s t/u v\\w x|y z~ `a` _b_ @c #d $e %f ^g &h ++ -- == += -= "
    "émile 東京 e=mc2 a.b.c x,y,z 1,000.50 20% 1st 2nd can't won't o'reilly"
).split()


def rand_sentence(rng, maxlen=25):
    n = rng.randint(0, maxlen)
    s = " ".join(rng.choice(WORDS) for _ in range(n))
    if rng.random() < 0.15:
        s = s.replace(" ", "  ", 1)
    if rng.random() < 0.1:
        s = "  " + s + " "
    if rng.random() < 0.05:
        s = s.replace(" ", "\t", 1)
    return s


def rand_corpus(rng, n_pairs, n_refs=1):
    hyps = [rand_sentence(rng) for _ in range(n_pairs)]
    refs = [[rand_sentence(rng) for _ in range(n_pairs)] for _ in range(n_refs)]
    return hyps, refs


def expected_backend() -> str:
    return "fallback" if os.environ.get("SACREBLEU_MOJO_DISABLE_NATIVE") == "1" else "native"


def assert_bleu_parity(hyps, refs, **kw):
    ob = oracle.corpus_bleu(hyps, refs, **kw)
    mb = corpus_bleu(hyps, refs, **kw)
    assert mb.backend == expected_backend()
    # BLEU agreement is bit-exact on all components.
    assert mb.score == ob.score
    assert mb.counts == ob.counts
    assert mb.totals == ob.totals
    assert mb.precisions == ob.precisions
    assert mb.bp == ob.bp
    assert mb.sys_len == ob.sys_len
    assert mb.ref_len == ob.ref_len
    return mb


def assert_chrf_parity(hyps, refs, **kw):
    oc = oracle.corpus_chrf(hyps, refs, **kw)
    mc = corpus_chrf(hyps, refs, **kw)
    assert mc.backend == expected_backend()
    assert mc.score == pytest.approx(oc.score, abs=SCORE_ATOL, rel=0)
    return mc


@pytest.mark.parametrize("seed", range(12))
def test_seeded_corpora_parity(seed):
    rng = random.Random(1000 + seed)
    n_pairs = rng.choice([1, 2, 3, 5, 8, 13, 21])
    n_refs = rng.choice([1, 1, 2, 3, 4])
    hyps, refs = rand_corpus(rng, n_pairs, n_refs)
    if n_refs > 1 and rng.random() < 0.4:
        refs[rng.randrange(n_refs)] = list(refs[rng.randrange(n_refs)])
    if rng.random() < 0.25:
        refs[0] = list(hyps)
    assert_bleu_parity(hyps, refs)
    assert_chrf_parity(hyps, refs)


@pytest.mark.parametrize("seed", range(4))
def test_bleu_smoothing_variants(seed):
    rng = random.Random(2000 + seed)
    hyps, refs = rand_corpus(rng, 9, 1)
    assert_bleu_parity(hyps, refs)  # default exp
    assert_bleu_parity(hyps, refs, smooth_method="floor")
    assert_bleu_parity(hyps, refs, smooth_method="floor", smooth_value=0.01)
    assert_bleu_parity(hyps, refs, smooth_method="floor", smooth_value=0.5)
    assert_bleu_parity(hyps, refs, smooth_method="none")
    assert_bleu_parity(hyps, refs, use_effective_order=True)


@pytest.mark.parametrize("seed", range(4))
def test_chrf_options(seed):
    rng = random.Random(3000 + seed)
    hyps, refs = rand_corpus(rng, 9, 1)
    assert_chrf_parity(hyps, refs)
    assert_chrf_parity(hyps, refs, word_order=1)
    assert_chrf_parity(hyps, refs, word_order=2)
    assert_chrf_parity(hyps, refs, word_order=3)
    assert_chrf_parity(hyps, refs, word_order=4)
    assert_chrf_parity(hyps, refs, remove_whitespace=False)
    assert_chrf_parity(hyps, refs, beta=1)
    assert_chrf_parity(hyps, refs, beta=3)
    assert_chrf_parity(hyps, refs, char_order=1)
    assert_chrf_parity(hyps, refs, char_order=3)


def test_multi_reference_corpora():
    rng = random.Random(42)
    for _ in range(6):
        n_pairs = rng.choice([2, 5, 11])
        n_refs = rng.choice([2, 3, 4])
        hyps, refs = rand_corpus(rng, n_pairs, n_refs)
        if rng.random() < 0.5:
            refs[rng.randrange(n_refs)] = list(refs[rng.randrange(n_refs)])
        assert_bleu_parity(hyps, refs)
        assert_chrf_parity(hyps, refs)
        assert_chrf_parity(hyps, refs, word_order=2)


def test_edge_cases():
    cases = [
        ([""], [[""]]),
        ([""], [["a"]]),
        (["a"], [[""]]),
        (["a"], [["a"]]),
        (["a b c d e f g h"], [["a b c d e f g h"]]),
        (["x" * 500], [["x" * 500]]),
        (["ab " * 100], [["ab " * 100]]),
        (["α"], [["α"]]),
        (["日本語 の 文章 です"], [["日本語 の 文章 です"]]),
        (["a"], [["a"], ["a"], ["a"]]),
        (["a b"], [["a"], ["a b"]]),  # uneven ref streams
        (["a b", "c d", "e f"], [["a b", "c d"]]),  # ragged: zip truncation
        (["unseen"], [["completely different"]]),
    ]
    for hyps, refs in cases:
        assert_bleu_parity(hyps, refs)
        assert_chrf_parity(hyps, refs)
        assert_chrf_parity(hyps, refs, word_order=2)


def test_tokenizer_nasties():
    hyps = [
        "Hello, world! It's 3.14 &amp; <skipped> end-\ncontinued [x] {y} 2-3.",
        '"quoted" (paren) [brack] {brace} <tag> &lt;tag&gt; 100-200 2,500 a.b.c',
        "don't can\'t won't o'clock rock 'n' roll a''b c'd' e' f'' g''' h'''' i",
        "C++ x++ a++b ++x x-- a--b x+= a+=b x|= a|=b p-q r*s t/u v\\w",
        "&quot;&quot;quoted&quot;&quot; &amp;&amp; %f $e @c #d ^g *star _b_ `a`",
    ]
    refs = [[s[::-1] if i % 2 else s for i, s in enumerate(hyps)]]
    assert_bleu_parity(hyps, refs)
    assert_chrf_parity(hyps, refs)
    assert_chrf_parity(hyps, refs, word_order=2)


def test_native_vs_fallback_bit_agreement():
    """The two backends must produce identical statistics on one corpus."""
    if os.environ.get("SACREBLEU_MOJO_DISABLE_NATIVE") == "1":
        pytest.skip("native run only")
    rng = random.Random(7)
    hyps, refs = rand_corpus(rng, 64, 3)
    native_b = corpus_bleu(hyps, refs)
    os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"] = "1"
    try:
        fallback_b = corpus_bleu(hyps, refs)
    finally:
        del os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"]
    assert fallback_b.backend == "fallback"
    assert native_b.backend == "native"
    assert native_b.score == fallback_b.score
    assert native_b.counts == fallback_b.counts
    assert native_b.totals == fallback_b.totals
    native_c = corpus_chrf(hyps, refs, word_order=2)
    os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"] = "1"
    try:
        fallback_c = corpus_chrf(hyps, refs, word_order=2)
    finally:
        del os.environ["SACREBLEU_MOJO_DISABLE_NATIVE"]
    assert native_c.score == fallback_c.score


def test_lowercase_parity():
    hyps = ["The Quick BROWN Fox!", "HELLO world 3.14"]
    refs = [["the quick brown fox!", "hello WORLD 3,14"]]
    ob = oracle.corpus_bleu(hyps, refs, lowercase=True)
    mb = corpus_bleu(hyps, refs, lowercase=True)
    assert mb.score == ob.score
    assert mb.counts == ob.counts


def test_output_object_shape():
    mb = corpus_bleu(["a b c"], [["a b c"]])
    assert isinstance(mb.score, float)
    assert isinstance(mb.counts, list) and isinstance(mb.counts[0], int)
    assert isinstance(mb.totals, list)
    assert isinstance(mb.precisions, list) and isinstance(mb.precisions[0], float)
    assert isinstance(mb.bp, float)
    assert isinstance(mb.sys_len, int) and isinstance(mb.ref_len, int)
    assert str(mb).startswith("BLEU = ")
    mc = corpus_chrf(["a b c"], [["a b c"]])
    assert isinstance(mc.score, float)
    assert (mc.char_order, mc.word_order, mc.beta) == (6, 0, 2)
    assert str(mc).startswith("chrF2 = ")
    mc2 = corpus_chrf(["a b c"], [["a b c"]], word_order=2)
    assert str(mc2).startswith("chrF2++ = ")


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [])
    with pytest.raises(ValueError):
        corpus_bleu(["a"], ["a"])  # refs not a sequence of streams
    with pytest.raises(ValueError):
        corpus_bleu("a", [["a"]])  # hyps not a sequence
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"]], smooth_method="bogus")
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"]], tokenize="zh")
    with pytest.raises(ValueError):
        corpus_chrf(["a"], [["a"]], eps_smoothing=True)
    with pytest.raises(ValueError):
        corpus_chrf(["a"], [["a"]], char_order=0)
    with pytest.raises(ValueError):
        corpus_chrf(["a"], [["a"]], char_order=7)
    with pytest.raises(ValueError):
        corpus_chrf(["a"], [["a"]], word_order=5)


def test_backend_selected():
    mb = corpus_bleu(["a"], [["a"]])
    assert mb.backend == expected_backend()
    info = sacrebleu_mojo.backend_info()
    assert info["disabled_by_env"] == (expected_backend() == "fallback")
    if expected_backend() == "native":
        assert info["native_available"] is True
