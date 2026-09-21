"""Drop-in difflib.SequenceMatcher / get_close_matches, Mojo-accelerated.

The public surface mirrors CPython 3.12's difflib: the SequenceMatcher
class (ratio, real_quick_ratio, quick_ratio, find_longest_match,
get_matching_blocks, get_opcodes, get_grouped_opcodes), the module-level
get_close_matches with identical ranking and tie order, and the Match /
IS_CHARACTER_JUNK / IS_LINE_JUNK helpers. On top of that, difflib_mojo adds
vectorized batch entry points (get_close_matches_batch, ratio_batch) for
the "one word vs many candidates" shape that dominates real workloads.

Backend selection: with the native Mojo kernel available and both
sequences being str, methods run on the kernel (sequences cross the FFI as
int32 codepoints, with the isjunk/autojunk classification of b's elements
precomputed as per-position masks). Everything else — no kernel, or
arbitrary hashable sequences — runs on the vendored pure-Python engine
(difflib_mojo._fallback), which the differential suite proves identical to
the reference on both paths. Set DIFFLIB_MOJO_DISABLE_NATIVE=1 to force
the engine.
"""

from __future__ import annotations

import re
from collections import namedtuple
from heapq import nlargest as _nlargest
from types import GenericAlias

import numpy as np

from difflib_mojo import _fallback, _native

Match = namedtuple("Match", "a b size")

# The reference's autojunk rule: popular elements occur more than
# len(b)//100 + 1 times, and only when len(b) >= 200.
_AUTOJUNK_MIN_LEN = 200


def _encode(s: str) -> np.ndarray:
    """str -> int32 codepoint array (one element per character, exactly the
    sequence semantics the reference sees; surrogatepass keeps lone
    surrogates lossless)."""
    if s.isascii():
        return np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int32)
    return np.frombuffer(s.encode("utf-32-le", "surrogatepass"), dtype="<i4")


def _encode_many(strs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten many strings into one codepoint array + CSR offsets."""
    n = len(strs)
    off = np.zeros(n + 1, dtype=np.int64)
    for i, s in enumerate(strs):
        off[i + 1] = off[i] + len(s)
    if all(s.isascii() for s in strs):
        flat = np.frombuffer("".join(strs).encode("ascii"), dtype=np.uint8).astype(
            np.int32
        )
        return flat, off
    parts = [_encode(s) for s in strs]
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    return flat, off


class SequenceMatcher:
    """A drop-in for difflib.SequenceMatcher (CPython 3.12 semantics).

    Compares pairs of sequences; with str inputs and the native kernel
    present, the hot loops run in compiled code. See the module docstring
    for backend rules and the package README for the (narrow) documented
    differences in introspection attributes.
    """

    def __init__(self, isjunk=None, a="", b="", autojunk=True):
        """Construct a SequenceMatcher (same arguments as difflib's)."""
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a, b)

    # -- sequence setup -------------------------------------------------

    def set_seqs(self, a, b):
        """Set the two sequences to be compared."""
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        """Set the first sequence; the second is not changed."""
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None
        self._a_cp = _encode(a) if isinstance(a, str) else None

    def set_seq2(self, b):
        """Set the second sequence and recompute its derived data."""
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self._fullbcount_cache = None
        self._b2j_cache = None
        self._engine = None
        self._chain_b()

    def _chain_b(self):
        """Classify b's elements (junk / autojunk-popular) and, for str b,
        precompute the codepoint array and per-position masks the native
        kernel consumes."""
        b = self.b
        self._b_cp = None
        self._b_junk_mask = None
        self._b_allowed_mask = None
        n = len(b)
        if isinstance(b, str):
            cp = _encode(b)
            self._b_cp = cp
            uniq, counts = np.unique(cp, return_counts=True)
            uniq_l = uniq.tolist()
            junk_cps: set[int] = set()
            if self.isjunk is not None:
                for c in uniq_l:
                    if self.isjunk(chr(c)):
                        junk_cps.add(c)
            pop_cps: set[int] = set()
            if self.autojunk and n >= _AUTOJUNK_MIN_LEN:
                ntest = n // 100 + 1
                for c, k in zip(uniq_l, counts.tolist()):
                    if c not in junk_cps and k > ntest:
                        pop_cps.add(c)
            self.bjunk = {chr(c) for c in junk_cps}
            self.bpopular = {chr(c) for c in pop_cps}
            if junk_cps:
                self._b_junk_mask = np.isin(
                    cp, np.array(sorted(junk_cps), dtype=np.int32)
                ).astype(np.uint8)
            else:
                self._b_junk_mask = np.zeros(n, dtype=np.uint8)
            excluded = junk_cps | pop_cps
            if excluded:
                self._b_allowed_mask = (
                    ~np.isin(cp, np.array(sorted(excluded), dtype=np.int32))
                ).astype(np.uint8)
            else:
                self._b_allowed_mask = np.ones(n, dtype=np.uint8)
        else:
            uniq_elts = list(dict.fromkeys(b))
            bjunk = (
                {e for e in uniq_elts if self.isjunk(e)}
                if self.isjunk is not None
                else set()
            )
            bpopular: set = set()
            if self.autojunk and n >= _AUTOJUNK_MIN_LEN:
                counts: dict = {}
                for elt in b:
                    counts[elt] = counts.get(elt, 0) + 1
                ntest = n // 100 + 1
                bpopular = {
                    e for e in uniq_elts if e not in bjunk and counts[e] > ntest
                }
            self.bjunk = bjunk
            self.bpopular = bpopular

    # -- introspection attributes (read-only; see README) ----------------

    @property
    def b2j(self):
        """The reference's element -> ascending-positions map of b, minus
        junk and autojunk-popular elements. Built on first access."""
        if self._b2j_cache is None:
            b2j: dict = {}
            for i, elt in enumerate(self.b):
                b2j.setdefault(elt, []).append(i)
            for elt in self.bjunk:
                b2j.pop(elt, None)
            for elt in self.bpopular:
                b2j.pop(elt, None)
            self._b2j_cache = b2j
        return self._b2j_cache

    @property
    def fullbcount(self):
        """The reference's element -> count map of b (junk included), built
        on first access."""
        if self._fullbcount_cache is None:
            fullbcount: dict = {}
            for elt in self.b:
                fullbcount[elt] = fullbcount.get(elt, 0) + 1
            self._fullbcount_cache = fullbcount
        return self._fullbcount_cache

    # -- backend plumbing -------------------------------------------------

    def _use_native(self) -> bool:
        return (
            self._a_cp is not None
            and self._b_cp is not None
            and _native.native_available()
        )

    def _engine_bindex(self) -> _fallback.BIndex:
        generic = not (isinstance(self.a, str) and isinstance(self.b, str))
        eng = self._engine
        if eng is None or eng[0] is not generic:
            eng = (
                generic,
                _fallback.BIndex(self.b, self.bjunk, self.bpopular, generic=generic),
            )
            self._engine = eng
        return eng[1]

    # -- the reference's public API ---------------------------------------

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None) -> Match:
        """Find the longest matching block in a[alo:ahi] and b[blo:bhi],
        with the reference's tie breaking and junk-extension rules.

        Unlike the reference, out-of-range windows raise ValueError (the
        reference's behavior is undefined there — it can return nonsense
        triples or raise IndexError depending on the values)."""
        a, b = self.a, self.b
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        if not (0 <= alo <= ahi <= len(a)) or not (0 <= blo <= bhi <= len(b)):
            raise ValueError(
                "find_longest_match window out of range: "
                f"alo={alo!r} ahi={ahi!r} blo={blo!r} bhi={bhi!r} "
                f"for len(a)={len(a)} len(b)={len(b)}"
            )
        if self._use_native():
            return Match(
                *_native.find_longest_match(
                    self._a_cp,
                    self._b_cp,
                    self._b_junk_mask,
                    self._b_allowed_mask,
                    alo,
                    ahi,
                    blo,
                    bhi,
                )
            )
        bindex = self._engine_bindex()
        return Match(*_fallback.find_longest_match(a, b, bindex, alo, ahi, blo, bhi))

    def get_matching_blocks(self) -> list:
        """Return the list of matching (i, j, k) Match triples, terminated
        by the (len(a), len(b), 0) sentinel — identical to the reference."""
        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)
        if self._use_native():
            raw = _native.matching_blocks(
                self._a_cp, self._b_cp, self._b_junk_mask, self._b_allowed_mask
            )
            triples = _fallback.collapse_blocks(
                [(int(t[0]), int(t[1]), int(t[2])) for t in raw], la, lb
            )
        else:
            bindex = self._engine_bindex()
            triples = _fallback.matching_blocks(self.a, self.b, bindex)
        self.matching_blocks = [Match(i, j, k) for i, j, k in triples]
        return self.matching_blocks

    def get_opcodes(self) -> list:
        """Return the list of (tag, i1, i2, j1, j2) edit opcodes turning
        a into b — identical to the reference."""
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            tag = ""
            if i < ai and j < bj:
                tag = "replace"
            elif i < ai:
                tag = "delete"
            elif j < bj:
                tag = "insert"
            if tag:
                answer.append((tag, i, ai, j, bj))
            i, j = ai + size, bj + size
            # the matching-block list ends with a size-0 sentinel
            if size:
                answer.append(("equal", ai, i, bj, j))
        return answer

    def get_grouped_opcodes(self, n=3):
        """Isolate change clusters with up to n elements of context —
        identical to the reference (including its in-place adjustment of
        the first/last cached opcodes)."""
        codes = self.get_opcodes()
        if not codes:
            codes = [("equal", 0, 1, 0, 1)]
        # Fixup leading and trailing groups if they show no changes.
        if codes[0][0] == "equal":
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2 - n), i2, max(j1, j2 - n), j2
        if codes[-1][0] == "equal":
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)

        nn = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            # End the current group and start a new one whenever there is a
            # large range with no changes.
            if tag == "equal" and i2 - i1 > nn:
                group.append((tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)))
                yield group
                group = []
                i1, j1 = max(i1, i2 - n), max(j1, j2 - n)
            group.append((tag, i1, i2, j1, j2))
        if group and not (len(group) == 1 and group[0][0] == "equal"):
            yield group

    def ratio(self) -> float:
        """2.0*M / T over the matching blocks — identical to the reference."""
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        return _fallback.calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self) -> float:
        """Multiset-intersection upper bound — identical to the reference."""
        if self._use_native():
            return _native.quick_ratio(self._a_cp, self._b_cp)
        bindex = self._engine_bindex()
        return _fallback.quick_ratio(self.a, self.b, bindex)

    def real_quick_ratio(self) -> float:
        """Length-only upper bound — identical to the reference."""
        la, lb = len(self.a), len(self.b)
        return _fallback.calculate_ratio(min(la, lb), la + lb)

    __class_getitem__ = classmethod(GenericAlias)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def _validate_close_args(n, cutoff) -> None:
    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))


def _rank_scored(scored, cand_forms, cands, n) -> list:
    """Rank passing candidates exactly like the reference: descending by
    (ratio, scored form), first-seen order among full duplicates.

    ``scored`` is a list of (ratio, candidate_index). heapq.nlargest(n,
    iterable) is documented equivalent to sorted(iterable, reverse=True)[:n];
    the index-based sort below is that equivalent, stable, and maps winning
    forms back to their original candidates (which differ from the scored
    forms when a key= transform is in play)."""
    order = sorted(
        range(len(scored)),
        key=lambda k: (scored[k][0], cand_forms[scored[k][1]]),
        reverse=True,
    )
    return [cands[scored[k][1]] for k in order[:n]]


def _close_matches_engine(word_forms, cand_forms, cands, n, cutoff) -> list:
    """The reference's per-candidate filter loop on the pure-Python engine.
    ``word_forms``/``cand_forms`` are what gets scored; ``cands`` (same
    length as ``cand_forms``) is what gets returned."""
    out = []
    for wf in word_forms:
        scored = []
        s = SequenceMatcher()
        s.set_seq2(wf)
        for ci, xf in enumerate(cand_forms):
            s.set_seq1(xf)
            if (
                s.real_quick_ratio() >= cutoff
                and s.quick_ratio() >= cutoff
                and s.ratio() >= cutoff
            ):
                scored.append((s.ratio(), ci))
        out.append(_rank_scored(scored, cand_forms, cands, n))
    return out


def _close_matches_native(word_forms, cand_forms, cands, n, cutoff) -> list:
    """The same filter loop on the native batch kernel: one FFI call scores
    every (word, candidate) pair; ranking (descending by (ratio, scored
    form), the reference's exact tie order) stays in Python."""
    w_flat, w_off = _encode_many(word_forms)
    c_flat, c_off = _encode_many(cand_forms)
    passes, ratios = _native.close_matches_batch(
        w_flat, w_off, c_flat, c_off, float(cutoff), True
    )
    out = []
    nc = len(cand_forms)
    for wi in range(len(word_forms)):
        row_pass, row_ratio = passes[wi], ratios[wi]
        scored = [
            (float(row_ratio[ci]), ci) for ci in range(nc) if row_pass[ci]
        ]
        out.append(_rank_scored(scored, cand_forms, cands, n))
    return out


def _close_matches(word_forms, cand_forms, cands, n, cutoff) -> list:
    if (
        all(isinstance(w, str) for w in word_forms)
        and all(isinstance(x, str) for x in cand_forms)
        and _native.native_available()
    ):
        return _close_matches_native(word_forms, cand_forms, cands, n, cutoff)
    return _close_matches_engine(word_forms, cand_forms, cands, n, cutoff)


def get_close_matches(word, possibilities, n=3, cutoff=0.6) -> list:
    """Drop-in for difflib.get_close_matches: the best (at most n) matches
    among possibilities, most similar first, with identical ranking and tie
    order. possibilities is materialized once (the reference iterates it
    lazily; results are the same for any finite iterable)."""
    _validate_close_args(n, cutoff)
    cands = list(possibilities)
    return _close_matches([word], cands, cands, n, cutoff)[0]


def get_close_matches_batch(words, possibilities, n=3, cutoff=0.6, *, key=None) -> list:
    """Vectorized get_close_matches over many words against one candidate
    pool: returns one ranked list per word, each identical to what
    get_close_matches(word, possibilities, n, cutoff) would return.

    ``key`` (optional) maps words and candidates to the form that is
    scored — e.g. ``key=str.casefold`` for case-insensitive matching; the
    returned matches are still the original candidate objects. This is an
    extension the reference does not have."""
    _validate_close_args(n, cutoff)
    word_list = list(words)
    cands = list(possibilities)
    if key is not None:
        word_forms = [key(w) for w in word_list]
        cand_forms = [key(x) for x in cands]
    else:
        word_forms, cand_forms = word_list, cands
    return _close_matches(word_forms, cand_forms, cands, n, cutoff)


def ratio_batch(pairs) -> list:
    """ratio() for an iterable of (a, b) sequence pairs, with the
    reference's default SequenceMatcher settings (autojunk on, no junk
    callable). Returns one float per pair."""
    out = []
    for a, b in pairs:
        s = SequenceMatcher(None, a, b)
        out.append(s.ratio())
    return out


_WS = " \t"


def IS_CHARACTER_JUNK(ch, ws=_WS) -> bool:
    """Return True iff ch is ignorable whitespace (the reference's helper)."""
    return ch in ws


_LINE_JUNK_PAT = re.compile(r"\s*(#\s*)?$")


def IS_LINE_JUNK(line, pat=_LINE_JUNK_PAT) -> bool:
    """Return True iff line is blank or a lone comment (the reference's helper)."""
    return pat.match(line) is not None
