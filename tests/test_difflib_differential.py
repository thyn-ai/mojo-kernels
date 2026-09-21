"""Differential tests: difflib_mojo must be output-identical to stdlib difflib.

Run twice by scripts/test_all_difflib.sh: once against the native Mojo
kernel and once with DIFFLIB_MOJO_DISABLE_NATIVE=1 (forced pure-Python
engine). Both backends must agree with CPython 3.12's difflib exactly —
floats bit-identical, matching blocks / opcodes / ranked lists identical
including tie order (no tolerances anywhere in this suite).

Everything is generated locally from explicit seeds: no network, no
randomness without a fixed seed.
"""

from __future__ import annotations

import difflib
import keyword
import os
import random

import pytest

import difflib_mojo
from difflib_mojo import Match, SequenceMatcher

# ---------------------------------------------------------------------------
# Deterministic generators
# ---------------------------------------------------------------------------

ALPHABETS = [
    "ab",
    "abcde",
    "abcdefghij",
    "ab \t",
    "abc def\t\n",
    "aąbć🙂é",
    "αβγδε",
    "日本語の文字",
]


def _rand_str(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _expected_backend() -> str:
    # scripts/test_all_difflib.sh runs the suite once per backend.
    return "fallback" if os.environ.get("DIFFLIB_MOJO_DISABLE_NATIVE") == "1" else "native"


def _active_backend() -> str:
    return "native" if difflib_mojo.native_available() else "fallback"


def _assert_same_matcher(a, b, isjunk=None, autojunk=True):
    """Every public method of the two SequenceMatchers must agree exactly."""
    ref = difflib.SequenceMatcher(isjunk, a, b, autojunk)
    got = SequenceMatcher(isjunk, a, b, autojunk)
    assert got.ratio() == ref.ratio()
    assert got.real_quick_ratio() == ref.real_quick_ratio()
    assert got.quick_ratio() == ref.quick_ratio()
    assert [tuple(t) for t in got.get_matching_blocks()] == [
        tuple(t) for t in ref.get_matching_blocks()
    ]
    assert got.get_opcodes() == ref.get_opcodes()
    assert list(got.get_grouped_opcodes()) == list(ref.get_grouped_opcodes())
    assert tuple(got.find_longest_match()) == tuple(ref.find_longest_match())
    # introspection attributes (read side) must match too
    assert got.b2j == ref.b2j
    assert got.bjunk == ref.bjunk
    assert got.bpopular == ref.bpopular
    assert got.fullbcount == ref.fullbcount
    # Match namedtuples are field-compatible with difflib.Match
    assert got.get_matching_blocks()[0] == ref.get_matching_blocks()[0]


# ---------------------------------------------------------------------------
# Random grids
# ---------------------------------------------------------------------------


def test_backend_is_the_expected_one():
    assert _active_backend() == _expected_backend()


def test_random_pairs_small_grid():
    rng = random.Random(20260920)
    for _ in range(120):
        alphabet = rng.choice(ALPHABETS)
        a = _rand_str(rng, rng.randint(0, 120), alphabet)
        b = _rand_str(rng, rng.randint(0, 120), alphabet)
        isjunk = rng.choice([None, lambda x: x in " \t"])
        autojunk = rng.choice([True, False])
        _assert_same_matcher(a, b, isjunk, autojunk)


def test_random_pairs_medium_grid():
    rng = random.Random(424242)
    for _ in range(25):
        alphabet = rng.choice(ALPHABETS)
        a = _rand_str(rng, rng.randint(200, 1500), alphabet)
        b = _rand_str(rng, rng.randint(200, 1500), alphabet)
        _assert_same_matcher(a, b, rng.choice([None, lambda x: x == " "]), True)


def test_find_longest_match_windows():
    rng = random.Random(77)
    for _ in range(60):
        alphabet = rng.choice(ALPHABETS[:4])
        a = _rand_str(rng, rng.randint(0, 60), alphabet)
        b = _rand_str(rng, rng.randint(0, 60), alphabet)
        la, lb = len(a), len(b)
        alo, ahi = sorted(rng.randint(0, la) for _ in range(2))
        blo, bhi = sorted(rng.randint(0, lb) for _ in range(2))
        isjunk = rng.choice([None, lambda x: x == " "])
        ref = difflib.SequenceMatcher(isjunk, a, b)
        got = SequenceMatcher(isjunk, a, b)
        assert tuple(got.find_longest_match(alo, ahi, blo, bhi)) == tuple(
            ref.find_longest_match(alo, ahi, blo, bhi)
        ), (a, b, alo, ahi, blo, bhi)


def test_find_longest_match_window_validation():
    s = SequenceMatcher(None, "abc", "abc")
    with pytest.raises(ValueError):
        s.find_longest_match(-1, 2, 0, 3)
    with pytest.raises(ValueError):
        s.find_longest_match(0, 2, 2, 1)
    with pytest.raises(ValueError):
        s.find_longest_match(0, 4, 0, 3)


# ---------------------------------------------------------------------------
# Adversarial fixtures
# ---------------------------------------------------------------------------


def test_empty_strings():
    for a, b in [("", ""), ("", "abc"), ("abc", ""), ("a", ""), ("", "a")]:
        _assert_same_matcher(a, b)
        _assert_same_matcher(a, b, lambda x: True)
    assert SequenceMatcher(None, "", "").ratio() == 1.0
    assert SequenceMatcher(None, "", "").quick_ratio() == 1.0
    assert SequenceMatcher(None, "", "").real_quick_ratio() == 1.0
    assert SequenceMatcher(None, "", "").get_matching_blocks() == [Match(0, 0, 0)]
    assert SequenceMatcher(None, "", "").get_opcodes() == []


def test_long_common_prefix_and_suffix():
    rng = random.Random(31337)
    for n in (100, 1000, 5000):
        prefix = _rand_str(rng, n, "abcdefghij")
        a = prefix + "X" + _rand_str(rng, 100, "abcdefghij")
        b = prefix + "Y" + _rand_str(rng, 100, "abcdefghij")
        _assert_same_matcher(a, b)
        _assert_same_matcher(a + "TAIL" * 10, b + "TAIL" * 10)


def test_identical_substring_reversed():
    rng = random.Random(555)
    a = _rand_str(rng, 500, "abcdefghij")
    _assert_same_matcher(a, a)
    _assert_same_matcher(a, a[::-1])
    _assert_same_matcher("abcdefgh", "cdef")
    _assert_same_matcher("abc" * 500, "abc" * 500 + "X")
    _assert_same_matcher("hello world " * 100, "world hello " * 100)


def test_unicode_adversarial():
    _assert_same_matcher(
        "héllo wörld 🙂🎉 café — naïve αβγ 日本語",
        "hello world 🙂🎉 cafe - naive αβγ 日本国",
    )
    _assert_same_matcher("🙂🎉🚀" * 30, "🙂🚀🎉" * 30)
    # lone surrogates must round-trip (surrogatepass encoding)
    _assert_same_matcher("\ud800lone\udfff", "\ud800lone\udffe")
    _assert_same_matcher("\udcff\udcff", "\udcffx")
    # combining marks: canonically different, codepoint-identical sequences
    _assert_same_matcher("ée" * 40, "ée" * 40)


def test_autojunk_boundary():
    """The popularity heuristic switches on at len(b) >= 200 and purges
    elements occurring more than len(b)//100 + 1 times — probe both sides
    of both thresholds."""
    rng = random.Random(200)
    for n in (198, 199, 200, 201, 202, 300, 500):
        for frac in (0.005, 0.01, 0.02, 0.5):
            a = "".join(
                "a" if rng.random() < frac else rng.choice("bcdef") for _ in range(n)
            )
            b = "".join(
                "a" if rng.random() < frac else rng.choice("bcdef") for _ in range(n)
            )
            _assert_same_matcher(a, b, None, True)
            _assert_same_matcher(a, b, None, False)
            _assert_same_matcher(a, b, lambda x: x == "b", True)
            _assert_same_matcher(a, b, lambda x: x == "b", False)


def test_junk_heavy():
    rng = random.Random(8642)
    for _ in range(30):
        a = _rand_str(rng, rng.randint(0, 300), "ab \t")
        b = _rand_str(rng, rng.randint(0, 300), "ab \t")
        for isjunk in (lambda x: x == " ", lambda x: x in " \t", difflib.IS_CHARACTER_JUNK):
            _assert_same_matcher(a, b, isjunk, rng.choice([True, False]))


def test_all_popular_purged():
    """With a tiny alphabet and autojunk on, every element is popular and
    the index is empty — matches come only from the extension phases."""
    rng = random.Random(606)
    a = _rand_str(rng, 2000, "ab")
    b = _rand_str(rng, 2000, "ab")
    _assert_same_matcher(a, b, None, True)
    _assert_same_matcher(a, b, None, False)
    _assert_same_matcher("x" * 1000, "x" * 1000, None, True)


def test_hundred_kb_strings():
    """100KB pairs: a large alphabet (sparse matches), an edited copy, and
    the autojunk-purged tiny alphabet."""
    rng = random.Random(100000)
    big_alphabet = [chr(0x4E00 + i) for i in range(5000)]
    a = "".join(rng.choice(big_alphabet) for _ in range(100_000))
    b = list(a)
    for _ in range(2000):  # 2000 scattered point edits
        b[rng.randrange(len(b))] = rng.choice(big_alphabet)
    _assert_same_matcher(a, "".join(b))

    c = _rand_str(rng, 100_000, "abcdefgh")
    d = _rand_str(rng, 100_000, "abcdefgh")
    _assert_same_matcher(c, d, None, True)  # autojunk purges everything
    # NOTE: autojunk=False on a tiny alphabet is quadratic for ANY correct
    # implementation (every element indexes ~len(b)/8 positions) — that
    # regime is covered at survivable sizes by test_autojunk_boundary.


def test_realistic_text_pair():
    """A ~12KB program-text pair with line-level edits (the classic diff
    workload), sized so the pure-Python oracle stays quick."""
    rng = random.Random(2026)
    words = [
        "def", "return", "import", "for", "while", "if", "else", "value",
        "index", "result", "data", "self", "none", "true", "false",
    ]
    lines_a = []
    for _ in range(1200):
        lines_a.append(
            "    " * rng.randint(0, 3)
            + " ".join(rng.choice(words) for _ in range(rng.randint(2, 8)))
        )
    lines_b = list(lines_a)
    for _ in range(80):  # insert, delete, tweak lines
        op = rng.randrange(3)
        i = rng.randrange(len(lines_b))
        if op == 0:
            lines_b.insert(i, "    " + " ".join(rng.choice(words) for _ in range(4)))
        elif op == 1 and len(lines_b) > 1:
            del lines_b[i]
        else:
            lines_b[i] = lines_b[i] + "  # edited"
    _assert_same_matcher("\n".join(lines_a), "\n".join(lines_b))
    _assert_same_matcher("\n".join(lines_a), "\n".join(lines_b), lambda x: x in " \t")


def test_non_str_sequences():
    """Arbitrary hashable sequences work on the engine on BOTH backends
    (the kernel only accepts str) and must match the reference."""
    rng = random.Random(909)
    for _ in range(40):
        vocab = list(range(rng.randint(2, 12)))
        a = [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        b = [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        _assert_same_matcher(a, b)
    _assert_same_matcher([1, 2, 3, 4, 5], [1, 2, 9, 4, 5, 6])
    _assert_same_matcher(list("abc def"), list("abc daf"), lambda x: x == " ")
    # tuples of strings, with a junk callable over elements
    a = [("x", 1), ("y", 2), ("x", 1), ("z", 3)]
    b = [("x", 1), ("z", 3), ("y", 2)]
    _assert_same_matcher(a, b)


def test_caching_and_identity_semantics():
    """set_seq1/set_seq2 short-circuit on identity; blocks/opcodes caches
    fill and invalidate exactly like the reference's."""
    a, b = "abcdef", "abcxef"
    s = SequenceMatcher(None, a, b)
    assert s.matching_blocks is None and s.opcodes is None
    blocks = s.get_matching_blocks()
    assert s.get_matching_blocks() is blocks  # cached, same object
    opcodes = s.get_opcodes()
    assert s.get_opcodes() is opcodes
    s.set_seq1(a)  # identical object: no invalidation
    assert s.get_matching_blocks() is blocks
    s.set_seq1("abcdef".join([]) if False else "".join(["abc", "def"]))
    assert s.matching_blocks is None  # equal but not identical: recompute
    blocks2 = s.get_matching_blocks()
    assert blocks2 == blocks and blocks2 is not blocks  # recomputed, same value
    s.set_seq2(b)  # identical object: no invalidation this time
    assert s.get_matching_blocks() is blocks2
    s.set_seq2("new value")
    assert s.matching_blocks is None and s.opcodes is None
    assert s.fullbcount == {"n": 1, "e": 2, "w": 1, " ": 1, "v": 1, "a": 1, "l": 1, "u": 1}


def test_docstring_examples():
    assert SequenceMatcher(None, " abcd", "abcd abcd").find_longest_match(0, 5, 0, 9) == Match(
        0, 4, 5
    )
    assert SequenceMatcher(lambda x: x == " ", " abcd", "abcd abcd").find_longest_match(
        0, 5, 0, 9
    ) == Match(1, 0, 4)
    assert SequenceMatcher(None, "ab", "c").find_longest_match(0, 2, 0, 1) == Match(0, 0, 0)
    assert SequenceMatcher(None, "abcd", "bcde").ratio() == 0.75
    assert SequenceMatcher(None, "abcd", "bcde").quick_ratio() == 0.75
    assert SequenceMatcher(None, "abcd", "bcde").real_quick_ratio() == 1.0
    s = SequenceMatcher(lambda x: x == " ", "private Thread currentThread;",
                        "private volatile Thread currentThread;")
    assert round(s.ratio(), 3) == 0.866


# ---------------------------------------------------------------------------
# get_close_matches
# ---------------------------------------------------------------------------


def test_get_close_matches_grid_with_ties():
    """Small alphabet + duplicate candidates force exact score ties; the
    ranked lists (order included) must be identical."""
    rng = random.Random(1313)
    vocab = [_rand_str(rng, rng.randint(2, 8), "abc") for _ in range(300)]
    vocab += vocab[:40]
    tie_cases = 0
    for _ in range(200):
        word = _rand_str(rng, rng.randint(2, 8), "abc")
        n = rng.choice([1, 2, 3, 5, 10, 100])
        cutoff = rng.choice([0.0, 0.25, 0.5, 0.6, 0.66, 0.75, 0.9, 1.0])
        want = difflib.get_close_matches(word, vocab, n, cutoff)
        got = difflib_mojo.get_close_matches(word, vocab, n, cutoff)
        assert got == want, (word, n, cutoff, got, want)
        # count ties among passing scores to prove ties are exercised
        s = difflib.SequenceMatcher()
        s.set_seq2(word)
        scores = []
        for x in vocab:
            s.set_seq1(x)
            if (
                s.real_quick_ratio() >= cutoff
                and s.quick_ratio() >= cutoff
                and s.ratio() >= cutoff
            ):
                scores.append(s.ratio())
        tie_cases += len(scores) - len(set(scores))
    assert tie_cases > 100, "fixture no longer exercises tie order"


def test_get_close_matches_unicode_and_mixed():
    rng = random.Random(747)
    vocab = [_rand_str(rng, rng.randint(3, 12), "abcdefghéü🙂") for _ in range(500)]
    for _ in range(60):
        word = _rand_str(rng, rng.randint(3, 12), "abcdefghéü🙂")
        n = rng.choice([1, 3, 7])
        cutoff = rng.choice([0.3, 0.6, 0.8])
        assert difflib_mojo.get_close_matches(
            word, vocab, n, cutoff
        ) == difflib.get_close_matches(word, vocab, n, cutoff)


def test_get_close_matches_docstring_examples():
    assert difflib_mojo.get_close_matches(
        "appel", ["ape", "apple", "peach", "puppy"]
    ) == ["apple", "ape"]
    assert difflib_mojo.get_close_matches("wheel", keyword.kwlist) == ["while"]
    assert difflib_mojo.get_close_matches("Apple", keyword.kwlist) == []
    assert difflib_mojo.get_close_matches("accept", keyword.kwlist) == ["except"]


def test_get_close_matches_errors_and_edges():
    for bad_n in (0, -1, -100):
        with pytest.raises(ValueError, match=r"n must be > 0"):
            difflib_mojo.get_close_matches("x", ["x"], bad_n)
    for bad_cutoff in (-0.1, 1.1, 2.0):
        with pytest.raises(ValueError, match=r"cutoff must be in \[0.0, 1.0\]"):
            difflib_mojo.get_close_matches("x", ["x"], 3, bad_cutoff)
    # exact stdlib error strings
    with pytest.raises(ValueError) as exc:
        difflib_mojo.get_close_matches("x", ["x"], 0)
    assert str(exc.value) == "n must be > 0: 0"
    with pytest.raises(ValueError) as exc:
        difflib_mojo.get_close_matches("x", ["x"], 3, 1.5)
    assert str(exc.value) == "cutoff must be in [0.0, 1.0]: 1.5"
    # empties
    assert difflib_mojo.get_close_matches("", ["", "a"]) == difflib.get_close_matches(
        "", ["", "a"]
    )
    assert difflib_mojo.get_close_matches("abc", []) == []
    assert difflib_mojo.get_close_matches("", []) == []
    assert difflib_mojo.get_close_matches("", [""], 1, 0.0) == [""]
    # possibilities given as a generator (the reference iterates lazily)
    gen = (x for x in ["ape", "apple", "peach"])
    assert difflib_mojo.get_close_matches("appel", gen) == ["apple", "ape"]


def test_get_close_matches_non_str():
    words = [[1, 2, 3], [1, 2, 4], [9, 9, 9], [1, 2, 3, 4]]
    assert difflib_mojo.get_close_matches(
        [1, 2, 3], words, 2, 0.5
    ) == difflib.get_close_matches([1, 2, 3], words, 2, 0.5)


# ---------------------------------------------------------------------------
# Batch APIs (difflib_mojo extensions)
# ---------------------------------------------------------------------------


def test_get_close_matches_batch_equals_per_word():
    rng = random.Random(21)
    vocab = [_rand_str(rng, rng.randint(3, 10), "abcdef") for _ in range(400)]
    words = [_rand_str(rng, rng.randint(3, 10), "abcdef") for _ in range(30)]
    for n, cutoff in [(1, 0.5), (3, 0.6), (5, 0.75), (10, 0.0)]:
        batch = difflib_mojo.get_close_matches_batch(words, vocab, n, cutoff)
        per_word = [difflib.get_close_matches(w, vocab, n, cutoff) for w in words]
        assert batch == per_word


def _expected_with_key(word, cands, key, n, cutoff):
    """What key= must return: run the reference on the transformed forms,
    then map winning forms back to originals, first-unused occurrence first
    (the reference's stable order among full (ratio, form) duplicates)."""
    forms = [key(x) for x in cands]
    want_forms = difflib.get_close_matches(key(word), forms, n, cutoff)
    used = [False] * len(cands)
    out = []
    for f in want_forms:
        for i, x in enumerate(forms):
            if not used[i] and x == f:
                used[i] = True
                out.append(cands[i])
                break
    return out


def test_get_close_matches_batch_key_casefold():
    """key= scores transformed forms but returns the originals; with ties,
    the tie order is by scored form (the reference's tuple order), mapped
    back stably. Duplicate forms under key= are allowed."""
    rng = random.Random(22)
    cands = [_rand_str(rng, rng.randint(4, 9), "abcdEF") for _ in range(200)]
    words = [_rand_str(rng, 6, "abcdEF") for _ in range(30)]
    got = difflib_mojo.get_close_matches_batch(words, cands, 3, 0.5, key=str.casefold)
    for w, want_list in zip(words, got):
        assert want_list == _expected_with_key(w, cands, str.casefold, 3, 0.5)


def test_get_close_matches_batch_key_lambda_and_duplicate_forms():
    cands = ["Apple", "APPLE", "apple", "Banana", "BANANA", "cherry"]
    key = lambda s: s.casefold()  # noqa: E731
    got = difflib_mojo.get_close_matches_batch(["aple", "BANANA"], cands, 5, 0.0, key=key)
    assert got[0] == _expected_with_key("aple", cands, key, 5, 0.0)
    assert got[1] == _expected_with_key("BANANA", cands, key, 5, 0.0)
    # duplicate 'apple' forms map back in first-seen order
    assert got[0][:2] == ["Apple", "APPLE"]
    # validation happens once, same messages
    with pytest.raises(ValueError):
        difflib_mojo.get_close_matches_batch(["x"], ["x"], 0)
    with pytest.raises(ValueError):
        difflib_mojo.get_close_matches_batch(["x"], ["x"], 3, 1.5)
    # empty word list / empty pool
    assert difflib_mojo.get_close_matches_batch([], ["a"], 3, 0.5) == []
    assert difflib_mojo.get_close_matches_batch(["a"], [], 3, 0.5) == [[]]


def test_ratio_batch():
    rng = random.Random(23)
    pairs = [
        (_rand_str(rng, rng.randint(0, 200), "abcde"), _rand_str(rng, rng.randint(0, 200), "abcde"))
        for _ in range(40)
    ]
    pairs += [("", ""), ("", "x"), ("abc", "abc")]
    got = difflib_mojo.ratio_batch(pairs)
    want = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in pairs]
    assert got == want


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_junk_helpers():
    assert difflib_mojo.IS_CHARACTER_JUNK(" ")
    assert difflib_mojo.IS_CHARACTER_JUNK("\t")
    assert not difflib_mojo.IS_CHARACTER_JUNK("x")
    assert difflib_mojo.IS_CHARACTER_JUNK(" ", ws=" ") == difflib.IS_CHARACTER_JUNK(" ", ws=" ")
    for line in ["", "   ", "# comment", "  # x", "code", " code # tail"]:
        assert difflib_mojo.IS_LINE_JUNK(line) == difflib.IS_LINE_JUNK(line)


def test_match_namedtuple_compat():
    m = difflib_mojo.Match(1, 2, 3)
    r = difflib.Match(1, 2, 3)
    assert m == r
    assert (m.a, m.b, m.size) == (1, 2, 3)
    assert tuple(m) == (1, 2, 3)
