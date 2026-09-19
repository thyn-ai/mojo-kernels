"""Differential tests: vader_mojo must match vaderSentiment dict-for-dict.

Run twice by `scripts/test_all_vader.sh`: once against the native Mojo kernel
and once with VADER_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback). Both
backends must agree with the reference package (PyPI vaderSentiment 3.3.2) on
EVERY sentence — exact dict equality, no tolerance: neg/neu/pos are compared
after the reference's 3-decimal rounding and compound after its 4-decimal
rounding, applied by vader_mojo.core to both backends.

The oracle is the published PyPI package; scripts/test_all_vader.sh makes it
importable (downloads the sha256-pinned wheel if needed). The seeded corpus is
generated locally from fixed seeds — no network at test time beyond that
one-time oracle fetch.
"""

from __future__ import annotations

import os
import random

import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as RefAnalyzer

import vader_mojo
from vader_mojo import SentimentIntensityAnalyzer

SEED = 20260919
N_SEEDED = 400

# The documented VADER example sentences (Hutto & Gilbert's demo set) covering
# punctuation emphasis, ALL-CAPS emphasis, degree modifiers, contrastive
# conjunction 'but', negations, emoticons, and utf-8 emojis.
DEMO_SENTENCES = [
    "VADER is smart, handsome, and funny.",
    "VADER is smart, handsome, and funny!",
    "VADER is very smart, handsome, and funny.",
    "VADER is VERY SMART, handsome, and FUNNY.",
    "VADER is VERY SMART, handsome, and FUNNY!!!",
    "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
    "VADER is not smart, handsome, nor funny.",
    "The book was good.",
    "At least it isn't a horrible book.",
    "The book was only kind of good.",
    "The plot was good, but the characters are uncompelling and the dialog is not great.",
    "Today SUX!",
    "Today only kinda sux! But I'll get by, lol",
    "Make sure you :) or :D today!",
    "Catch utf-8 emoji such as 💘 and 💋 and 😁",
    "Not bad at all",
]

# The documented "tricky" sentences: special idioms and 'least' handling.
TRICKY_SENTENCES = [
    "Sentiment analysis has never been good.",
    "Sentiment analysis has never been this good!",
    "Most automated sentiment analysis tools are shit.",
    "With VADER, sentiment analysis is the shit!",
    "Other sentiment analysis tools can be quite bad.",
    "On the other hand, VADER is quite bad ass",
    "VADER is such a badass!",
    "Without a doubt, excellent idea.",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "Roger Dodger is one of the least compelling variations on this theme.",
    "Not such a badass after all.",
    "Without a doubt, an excellent idea.",
]

# One case per rule and edge path, including the degenerate inputs.
TARGETED_CASES = [
    # empty / whitespace / punctuation-only
    "", " ", "  \t\n  ", "!!!", "?????", "?!?!?!", "...", "foo",
    # exclamation flooding: 1..6 (cap at 4)
    "good!", "good!!", "good!!!", "good!!!!", "good!!!!!", "good!!!!!!",
    # question marks: 1 (no amp), 2..3 (scaled), 4+ (cap 0.96)
    "good?", "good??", "good???", "good????", "good?????",
    # caps emphasis and the all-caps differential
    "good", "GOOD", "GOOD bad", "good BAD", "ALL CAPS GOOD BAD", "MiXeD good",
    "Today SUX!", "sux", "SUX", "SUX SUX SUX",
    # booster / dampener words, incl. distance decay and caps boosters
    "very good", "so very good", "really very good", "uber good",
    "almost good", "barely good", "slightly good", "kinda good",
    "VERY good", "good VERY bad", "friggin awesome", "hella good",
    # kind of / sort of dampeners (main-loop and bi-gram paths)
    "kind of good", "sort of bad", "kind of sort of good", "The book was only kind of good.",
    # negation window: 1, 2, 3 words back
    "not good", "not really good", "not really very good", "isn't good",
    "can't win", "didn't work", "won't !!!", "never good", "nothing good",
    # 'never so/this' amplification and the start_i==2 precedence quirk
    "never so good", "never this good", "it is never so very good",
    "so good", "this good", "never bad", "so bad",
    # 'without doubt' exemption
    "without doubt good", "without a doubt good", "Without a doubt, an excellent idea.",
    # 'no' as adjacent negator vs stand-alone lexicon item
    "no good", "no good at all", "no no good", "good no bad", "no or good",
    "no nor good", "said no one", "no", "yes no yes",
    # 'least' negation vs 'at least' / 'very least'
    "least good", "at least good", "very least good", "the least bad",
    "At least it isn't a horrible book.",
    # contrastive conjunction 'but', incl. duplicates and multiple 'but's
    "good but bad", "bad but good", "good good good but bad bad bad",
    "good but good but bad", "good but", "but good", "BUT good bad",
    "The food was great but the service was horrible",
    # special-case idioms
    "the shit", "this is the shit", "the bomb", "bad ass", "badass",
    "yeah right", "to die for", "kiss of death", "beating heart", "bus stop",
    "quite bad ass", "such a badass",
    # emoticons (kept vs stripped) and emoji substitution
    ":)", ":(", ":D", ":-)", ":-(", ":p", ";)", ":/", "o.O",
    "Make sure you :) or :D today!", "great :)", "terrible :(",
    "💘", "💋 and 😁", "Catch utf-8 emoji such as 💘 and 💋 and 😁",
    "👍 good", "👎 bad", "💔", "🎉 party 🎉", "😴 boring 😴",
    # emoji forms the per-character lookup cannot match (ZWJ, VS16, flags,
    # keycaps, skin tones): they pass through as neutral characters
    "🏳️‍🌈 good", "👨‍👩‍👧‍👦 good", "🇺🇸", "#️⃣ bad", "👍🏻 good", "❤️ love",
    "☺️ happy", "👨‍⚕️",
    # the lexicon's two non-ASCII emoticon keys are unreachable in the
    # reference (its per-token .lower() can never produce them)
    ":-Þ", ":Þ", ":-Þ good", ":Þ bad", ":-þ good",
    # Unicode-cased tokens (CAPS checks must follow Python's str.isupper())
    "É good", "ÀÉÈ bad", "TRÈS GOOD", "ÜÖÄ sux", "ΩΩΩ good", "ДА bad",
    "ǄǄ good", "ß good", "ΣΟΦΟΣ bad",
    # U+212A KELVIN SIGN lowercases to ASCII 'k'
    "K", "K good", "K bad",
    # Unicode whitespace splitting/stripping
    "\xa0good\xa0", "good\u2003bad", "\tgood\nbad\r", "good\x0cbad",
    "good bad", "  padded  good  ",
    # lone surrogates pass through as neutral characters
    "\ud800 good", "bad \udfff", "\ud800",
    # mixed bag
    "not really very good at all", "kind of sort of good",
    "no no no", "but but but", "very very very good",
    "It was one of the worst movies I've seen, despite good reviews.",
    "Unbelievably bad acting!! Poor direction. VERY poor production.",
    "The movie was bad. Very bad movie. VERY BAD movie!",
]

_BOOSTERS = [
    "very", "really", "so", "SO", "uber", "friggin", "kind of", "sort of",
    "almost", "barely", "slightly", "REALLY", "utterly", "most",
]
_NEGATORS = [
    "not", "never", "no", "isn't", "can't", "didn't", "won't", "without",
    "nor", "nothing", "seldom", "hardly",
]
_PUNCT = ["!", "!!", "!!!", "!!!!", "!!!!!", "?", "??", "???", "????", "?!", "!?", ".", "...", ","]
_EMOTICONS = [":)", ":(", ":D", ":-)", ":-(", ":p", ":-Þ", ":Þ", ";)", ":/", "o.o", "lol", "xd"]
_EMOJIS = ["💘", "💋", "😁", "😢", "😡", "👍", "👎", "❤️", "💔", "🎉", "😴", "🤮",
           "🏳️‍🌈", "👨‍👩‍👧", "🇺🇸", "#️⃣"]
_IDIOMS = [
    "but", "BUT", "least", "at least", "very least", "the shit", "the bomb",
    "bad ass", "badass", "yeah right", "to die for", "kiss of death",
    "beating heart", "bus stop", "so", "this", "doubt", "kind", "of",
]
_UNI = ["É", "ÀÉÈ", "ΩΩΩ", "ДА", "TRÈS", "K", "İ", "ΣΟΦΟΣ", "ǅ", "ß", "é", "ÜÖÄ", "ǄǄ"]
_FILLER = ["zzz", "qwerty", "lorem", "123", "U.S.A", "can't-even"]


def _seeded_corpus() -> list[str]:
    from vader_mojo.core import _load_lexicon

    lex_words = list(_load_lexicon())
    rng = random.Random(SEED)
    corpus = []
    for _ in range(N_SEEDED):
        parts = []
        for _ in range(rng.randint(1, 25)):
            r = rng.random()
            if r < 0.42:
                parts.append(rng.choice(lex_words))
            elif r < 0.57:
                parts.append(rng.choice(_BOOSTERS))
            elif r < 0.67:
                parts.append(rng.choice(_NEGATORS))
            elif r < 0.75:
                parts.append(rng.choice(_IDIOMS))
            elif r < 0.82:
                parts.append(rng.choice(_EMOTICONS))
            elif r < 0.87:
                parts.append(rng.choice(_EMOJIS))
            elif r < 0.90:
                parts.append(rng.choice(_UNI))
            else:
                parts.append(rng.choice(_FILLER))
        s = " ".join(parts)
        if rng.random() < 0.5:
            s += rng.choice(_PUNCT)
        if rng.random() < 0.25:
            s = s.upper()
        if rng.random() < 0.15:
            s = "  " + s + "  "
        if rng.random() < 0.1:
            s = s.replace(" ", "  ")
        corpus.append(s)
    return corpus


ALL_SENTENCES = DEMO_SENTENCES + TRICKY_SENTENCES + TARGETED_CASES


@pytest.fixture(scope="module")
def ref():
    return RefAnalyzer()


@pytest.fixture(scope="module")
def ours():
    return SentimentIntensityAnalyzer()


@pytest.fixture(scope="module")
def seeded():
    return _seeded_corpus()


def _expected_backend() -> str:
    # scripts/test_all_vader.sh runs the suite once per backend.
    return "fallback" if os.environ.get("VADER_MOJO_DISABLE_NATIVE") == "1" else "native"


def test_backend_is_the_expected_one(ours):
    assert ours.backend == _expected_backend()


def test_demo_sentences_exact(ours, ref):
    for s in ALL_SENTENCES:
        assert ours.polarity_scores(s) == ref.polarity_scores(s), s


def test_seeded_corpus_exact(ours, ref, seeded):
    for s in seeded:
        assert ours.polarity_scores(s) == ref.polarity_scores(s), s


def test_module_level_api_matches_class(ref):
    # vader_mojo.polarity_scores uses a shared process-global analyzer.
    for s in ALL_SENTENCES[:40]:
        assert vader_mojo.polarity_scores(s) == ref.polarity_scores(s), s


def test_repeated_calls_are_deterministic(ours, ref, seeded):
    for s in seeded[:25]:
        first = ours.polarity_scores(s)
        assert ours.polarity_scores(s) == first == ref.polarity_scores(s), s


def test_raw_scores_match_before_rounding(ours, ref, seeded):
    # The kernel returns raw float64 scores; compare with higher precision to
    # catch last-ulp drift that the 3/4-digit rounding would hide.
    raw_ours = ours._native.polarity if ours._native is not None else ours._fallback.polarity
    for s in seeded[:50]:
        got = raw_ours(s)
        r = ref.polarity_scores(s)
        exp = (r["neg"], r["neu"], r["pos"], r["compound"])
        for g, e in zip(got, exp):
            assert g == pytest.approx(e, abs=2e-3), (s, got, exp)


def test_non_str_rejected(ours):
    with pytest.raises(TypeError):
        ours.polarity_scores(42)
