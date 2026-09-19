"""Vendored pure-Python fallback: the same detection algorithm as the native
kernel, implemented with the standard library only.

This is a clean-room implementation of the reference algorithm's documented
procedure (n-gram extraction with script normalization + the seeded sampling
estimator), written against the same data tables the native kernel consumes
(``_data.Profiles``). Because it runs on CPython's own ``random.Random`` and
float arithmetic in the reference operation order, its output is bit-identical
to the seeded reference detector; the differential suite asserts that on every
corpus text.
"""

from __future__ import annotations

import random
import re

from langdetect_mojo._data import Profiles

ALPHA_DEFAULT = 0.5
ALPHA_WIDTH = 0.05
ITERATION_LIMIT = 1000
CONV_THRESHOLD = 0.99999
BASE_FREQ = 10000
N_TRIAL = 7
MAX_TEXT_LENGTH = 10000
_SP = 0x20

# The reference's preprocessing patterns; only ASCII classes, so replicating
# them with stdlib re is exact by construction.
_URL_RE = re.compile(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}")
_MAIL_RE = re.compile(r"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}")


def _normalize_cp(p: Profiles, cp: int) -> int:
    """Per-script normalization (reference NGram.normalize)."""
    if cp <= 0x7F:  # Basic Latin
        if cp < 0x41 or (0x5A < cp < 0x61) or cp > 0x7A:
            return _SP
        return cp
    if cp <= 0xFF:  # Latin-1 Supplement
        if cp in p.latin1:
            return _SP
        return cp
    if 0x180 <= cp <= 0x24F:  # Latin Extended-B (Romanian comma-below)
        if cp == 0x219:
            return 0x15F
        if cp == 0x21B:
            return 0x163
        return cp
    if 0x2000 <= cp <= 0x206F:  # General Punctuation
        return _SP
    if 0x600 <= cp <= 0x6FF:  # Arabic (Farsi yeh => Arabic yeh)
        if cp == 0x6CC:
            return 0x64A
        return cp
    if 0x1E00 <= cp <= 0x1EFF:  # Latin Extended Additional (Vietnamese)
        if cp >= 0x1EA0:
            return 0x1EC3
        return cp
    if 0x3040 <= cp <= 0x309F:  # Hiragana
        return 0x3042
    if 0x30A0 <= cp <= 0x30FF:  # Katakana
        return 0x30A2
    if (0x3100 <= cp <= 0x312F) or (0x31A0 <= cp <= 0x31BF):  # Bopomofo
        return 0x3105
    if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
        lo, hi = 0, len(p.cjk_pairs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            frm, to = p.cjk_pairs[mid]
            if cp < frm:
                hi = mid - 1
            elif cp > frm:
                lo = mid + 1
            else:
                return to
        return cp
    if 0xAC00 <= cp <= 0xD7AF:  # Hangul Syllables
        return 0xAC00
    return cp


def _normalize_vi(p: Profiles, cps: list[int]) -> list[int]:
    """Map (alphabet, combining diacritic) pairs to precomposed codepoints."""
    out: list[int] = []
    i, n = 0, len(cps)
    while i < n:
        cp = cps[i]
        try:
            a = p.vi_alpha.index(cp)
        except ValueError:
            a = -1
        if a >= 0 and i + 1 < n:
            try:
                d = p.vi_dmark.index(cps[i + 1])
            except ValueError:
                d = -1
            if d >= 0:
                out.append(p.vi_rows[d * len(p.vi_alpha) + a])
                i += 2
                continue
        out.append(cp)
        i += 1
    return out


def _extract_ngrams(p: Profiles, text: str) -> list[int]:
    """Reference NGram state machine + Detector._extract_ngrams."""
    cps = _normalize_vi(p, [ord(c) for c in text])
    ids: list[int] = []
    grams = [_SP]
    capital = False
    key_to_kid = p.key_to_kid
    for cp in cps:
        ch = _normalize_cp(p, cp)
        last = grams[-1]
        if last == _SP:
            grams = [_SP]
            capital = False
            if ch == _SP:
                continue  # add_char returns early; emission below yields nothing
        elif len(grams) >= 3:
            grams = grams[1:]
        grams.append(ch)
        if chr(ch).isupper():
            if chr(last).isupper():
                capital = True
        else:
            capital = False
        if capital:
            continue
        for n in (1, 2, 3):
            if len(grams) < n:
                break
            w = grams[-n:]
            if w == [_SP]:
                continue
            key = 0
            for k, c in enumerate(w):
                key |= c << (21 * k)
            kid = key_to_kid.get(key)
            if kid is not None:
                ids.append(kid)
    return ids


def detect_block(p: Profiles, text: str, seed: int) -> tuple[int, list[float] | None]:
    """Full pipeline; returns (status, langprob|None) like the native kernel."""
    text = _URL_RE.sub(" ", text)
    text = _MAIL_RE.sub(" ", text)

    # collapse consecutive spaces, capped at max_text_length codepoints
    buf: list[str] = []
    pre = 0
    for i in range(min(len(text), MAX_TEXT_LENGTH)):
        ch = text[i]
        if ch != " " or pre != " ":
            buf.append(ch)
        pre = ch
    text = "".join(buf)

    # cleaning_text: drop Latin-range chars when heavily outnumbered
    latin = non_latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x41 <= cp <= 0x7A:
            latin += 1
        elif cp >= 0x300 and not (0x1E00 <= cp <= 0x1EFF):
            non_latin += 1
    if latin * 2 < non_latin:
        text = "".join(ch for ch in text if not (0x41 <= ord(ch) <= 0x7A))

    ids = _extract_ngrams(p, text)
    if not ids:
        return 1, None

    n = p.n_langs
    rng = random.Random(seed)
    langprob = [0.0] * n
    for _trial in range(N_TRIAL):
        prob = [1.0 / n] * n
        alpha = ALPHA_DEFAULT + rng.gauss(0.0, 1.0) * ALPHA_WIDTH
        i = 0
        while True:
            kid = rng.choice(ids)
            weight = alpha / BASE_FREQ
            addv = [weight] * n
            for e in range(p.kid_off[kid], p.kid_off[kid + 1]):
                addv[p.ent_lang[e]] = weight + p.ent_prob[e]
            for j in range(n):
                prob[j] *= addv[j]
            if i % 5 == 0:
                sump = sum(prob)
                maxp = 0.0
                for j in range(n):
                    pj = prob[j] / sump
                    if maxp < pj:
                        maxp = pj
                    prob[j] = pj
                if maxp > CONV_THRESHOLD or i >= ITERATION_LIMIT:
                    break
            i += 1
        for j in range(n):
            langprob[j] += prob[j] / N_TRIAL
    return 0, langprob
