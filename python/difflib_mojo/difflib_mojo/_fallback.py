"""Vendored pure-Python Ratcliff/Obershelp engine.

A fresh implementation of the gestalt-pattern-matching algorithm that
difflib.SequenceMatcher documents: the longest junk-free contiguous match
with earliest-start tie breaking, two-phase junk extension, recursive
divide-and-conquer into matching blocks, and the multiset-intersection
quick ratio. It shares the native Mojo kernel's architecture (a CSR index
of b over small integer element ids, generation-stamped double-buffered DP
rows) rather than the reference's dict formulation, and the differential
suite asserts it is output-identical to the reference on every fixture —
including junk callables, the autojunk popularity heuristic and tie order.

Used when the native kernel is unavailable or disabled, and for non-str
sequences (the kernel accepts int32 codepoints only). Element ids: for str
inputs the id is the codepoint; for arbitrary hashable sequences a dense
per-pair mapping is assigned (b first, then a-only elements).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def calculate_ratio(matches: int, length: int) -> float:
    """2.0*M / T, or 1.0 when both sequences are empty (the reference rule)."""
    if length:
        return 2.0 * matches / length
    return 1.0


class BIndex:
    """CSR index of the b sequence plus DP scratch, built once per b.

    ``full[c]`` counts every occurrence of element id c in b (junk or not,
    for the quick-ratio multiset intersection). ``off``/``pos`` index only
    allowed positions (not junk, not autojunk-popular), ascending per id —
    the iteration order the reference's b2j lists produce. ``junk[j]`` marks
    junk positions for the extension phases. ``jlen*``/``stamp*`` are the
    salted parity-swapped row buffers; ``avail*`` the salted quick-ratio
    bookkeeping. Salts mean no buffer is ever cleared between rows or calls.
    """

    __slots__ = (
        "b_ids",
        "id_of",
        "off",
        "pos",
        "full",
        "junk",
        "jlen",
        "stamp",
        "salt",
        "avail",
        "avail_stamp",
        "avail_salt",
    )

    def __init__(
        self,
        b: Sequence,
        bjunk: set,
        bpopular: set,
        generic: bool = False,
    ) -> None:
        # generic=True forces the element->id mapping even for str b (used
        # when the other sequence of the pair is not a str).
        if not generic and isinstance(b, str):
            b_ids = [ord(c) for c in b]
            self.id_of = None
        else:
            id_of: dict = {}
            b_ids = []
            append = b_ids.append
            for elt in b:
                idx = id_of.get(elt)
                if idx is None:
                    idx = len(id_of)
                    id_of[elt] = idx
                append(idx)
            self.id_of = id_of
        self.b_ids = b_ids
        lb = len(b_ids)
        n_ids = (max(b_ids) + 1) if b_ids else 0

        full = [0] * (n_ids + 1)
        for c in b_ids:
            full[c] += 1

        if bjunk or bpopular:
            junk = [1 if elt in bjunk else 0 for elt in b]
            allowed = [0 if (elt in bjunk or elt in bpopular) else 1 for elt in b]
        else:
            junk = [0] * lb
            allowed = [1] * lb

        off = [0] * (n_ids + 2)
        for j in range(lb):
            if allowed[j]:
                off[b_ids[j] + 1] += 1
        for c in range(n_ids + 1):
            off[c + 1] += off[c]

        pos = [0] * lb
        cursor = off[: n_ids + 1]
        for j in range(lb):
            if allowed[j]:
                c = b_ids[j]
                pos[cursor[c]] = j
                cursor[c] += 1

        self.off = off
        self.pos = pos
        self.full = full
        self.junk = junk
        self.jlen = [[0] * lb, [0] * lb]
        self.stamp = [[0] * lb, [0] * lb]
        self.salt = 0
        self.avail = [0] * (n_ids + 1)
        self.avail_stamp = [0] * (n_ids + 1)
        self.avail_salt = 0

    def ids_for(self, a: Sequence) -> list[int]:
        """Element ids of a in this index's id space (b first, then a-only)."""
        if self.id_of is None:
            return [ord(c) for c in a]  # a is a str (guaranteed by the caller)
        id_of = self.id_of
        out = []
        append = out.append
        for elt in a:
            idx = id_of.get(elt)
            if idx is None:
                idx = len(id_of)
                id_of[elt] = idx
            append(idx)
        return out


def find_longest_match_ids(
    a_ids: list[int], bindex: BIndex, alo: int, ahi: int, blo: int, bhi: int
) -> tuple[int, int, int]:
    """Longest junk-free match in a[alo:ahi] vs b[blo:bhi].

    Tie breaking matches the reference exactly: a strictly longer match
    wins; among equals the one found first scanning i ascending and, per i,
    j ascending. Then the two extension phases (non-junk, then junk).
    """
    off, pos, junk = bindex.off, bindex.pos, bindex.junk
    b_ids = bindex.b_ids
    jlen, stamp = bindex.jlen, bindex.stamp
    n_ids = len(off) - 2
    base = bindex.salt
    besti, bestj, bestsize = alo, blo, 0

    for i in range(alo, ahi):
        ci = a_ids[i]
        if ci > n_ids:
            continue
        rowstamp = base + (i - alo) + 1
        buf = (i - alo) & 1
        cur, prev = stamp[buf], stamp[buf ^ 1]
        curv, prevv = jlen[buf], jlen[buf ^ 1]
        for p in range(off[ci], off[ci + 1]):
            j = pos[p]
            if j < blo:
                continue
            if j >= bhi:
                break
            k = 1
            if j > 0 and prev[j - 1] == rowstamp - 1:
                k = prevv[j - 1] + 1
            curv[j] = k
            cur[j] = rowstamp
            if k > bestsize:
                besti, bestj, bestsize = i - k + 1, j - k + 1, k

    # This call's salt window can never collide with a later call.
    bindex.salt = base + (ahi - alo) + 1

    # Extend the best by non-junk elements on each end (autojunk-popular
    # elements are absent from the index but still join a match here) ...
    while (
        besti > alo
        and bestj > blo
        and not junk[bestj - 1]
        and a_ids[besti - 1] == b_ids[bestj - 1]
    ):
        besti -= 1
        bestj -= 1
        bestsize += 1
    while (
        besti + bestsize < ahi
        and bestj + bestsize < bhi
        and not junk[bestj + bestsize]
        and a_ids[besti + bestsize] == b_ids[bestj + bestsize]
    ):
        bestsize += 1

    # ... then suck up matching junk adjacent to the interesting match.
    while (
        besti > alo
        and bestj > blo
        and junk[bestj - 1]
        and a_ids[besti - 1] == b_ids[bestj - 1]
    ):
        besti -= 1
        bestj -= 1
        bestsize += 1
    while (
        besti + bestsize < ahi
        and bestj + bestsize < bhi
        and junk[bestj + bestsize]
        and a_ids[besti + bestsize] == b_ids[bestj + bestsize]
    ):
        bestsize += 1

    return besti, bestj, bestsize


def collapse_blocks(
    raw_triples: Iterable[tuple[int, int, int]], la: int, lb: int
) -> list[tuple[int, int, int]]:
    """Sort discovery-order triples, collapse adjacent ones, append the
    (la, lb, 0) sentinel — the reference's post-processing, verbatim in
    observable behavior."""
    triples = sorted(raw_triples)
    non_adjacent: list[tuple[int, int, int]] = []
    i1 = j1 = k1 = 0
    for i2, j2, k2 in triples:
        if i1 + k1 == i2 and j1 + k1 == j2:
            k1 += k2
        else:
            if k1:
                non_adjacent.append((i1, j1, k1))
            i1, j1, k1 = i2, j2, k2
    if k1:
        non_adjacent.append((i1, j1, k1))
    non_adjacent.append((la, lb, 0))
    return non_adjacent


def matching_blocks_ids(a_ids: list[int], bindex: BIndex) -> list[tuple[int, int, int]]:
    """All matching blocks (collapsed, with sentinel) for the id pair."""
    la, lb = len(a_ids), len(bindex.b_ids)
    stack = [(0, la, 0, lb)]
    raw: list[tuple[int, int, int]] = []
    while stack:
        alo, ahi, blo, bhi = stack.pop()
        i, j, k = find_longest_match_ids(a_ids, bindex, alo, ahi, blo, bhi)
        if k:
            raw.append((i, j, k))
            if alo < i and blo < j:
                stack.append((alo, i, blo, j))
            if i + k < ahi and j + k < bhi:
                stack.append((i + k, ahi, j + k, bhi))
    return collapse_blocks(raw, la, lb)


def quick_matches_ids(a_ids: list[int], bindex: BIndex) -> int:
    """Multiset-intersection cardinality of a and b (quick_ratio's M)."""
    full, avail, avail_stamp = bindex.full, bindex.avail, bindex.avail_stamp
    n_ids = len(full) - 1
    bindex.avail_salt += 1
    salt = bindex.avail_salt
    matches = 0
    for c in a_ids:
        if c > n_ids:
            continue  # not in b at all: consumes nothing, matches nothing
        numb = avail[c] if avail_stamp[c] == salt else full[c]
        avail_stamp[c] = salt
        avail[c] = numb - 1
        if numb > 0:
            matches += 1
    return matches


def find_longest_match(
    a: Sequence,
    b: Sequence,
    bindex: BIndex,
    alo: int,
    ahi: int,
    blo: int,
    bhi: int,
) -> tuple[int, int, int]:
    """Sequence-level wrapper around the id engine."""
    return find_longest_match_ids(bindex.ids_for(a), bindex, alo, ahi, blo, bhi)


def matching_blocks(a: Sequence, b: Sequence, bindex: BIndex) -> list[tuple[int, int, int]]:
    """Sequence-level wrapper around the id engine."""
    return matching_blocks_ids(bindex.ids_for(a), bindex)


def quick_ratio(a: Sequence, b: Sequence, bindex: BIndex) -> float:
    """quick_ratio() for the pair, identical to the reference."""
    return calculate_ratio(quick_matches_ids(bindex.ids_for(a), bindex), len(a) + len(b))
