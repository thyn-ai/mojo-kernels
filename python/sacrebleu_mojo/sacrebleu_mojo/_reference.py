"""Vendored pure-Python reference statistics for sacrebleu-mojo.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``SACREBLEU_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of
the published metric definitions (Papineni et al. 2002 BLEU sufficient
statistics with clipped multi-reference n-gram counts; Popovic 2015 chrF
character/word n-gram precision-recall with per-sentence best-reference
selection), written to be observably identical to the widely used
``sacrebleu`` package: the differential suite asserts score agreement within
1e-9 against the published PyPI package on both backends (BLEU agreement is
bit-exact; chrF is bit-exact in the large majority of cases and otherwise
within a couple of ulps).

Only the counting lives here; tokenization, encoding, and score assembly are
shared with the native path in ``sacrebleu_mojo.core``, so the two backends
can never disagree about anything but raw counting speed.
"""

from __future__ import annotations

from collections import Counter

MAX_ORDER = 4


def _ngram_counts(ids, start: int, length: int, n: int) -> Counter:
    if length < n:
        return Counter()
    return Counter(tuple(ids[start + i : start + i + n]) for i in range(length - n + 1))


def bleu_stats_range(
    hyp_ids,
    hyp_off,
    ref_ids,
    ref_off,
    seg_index,
    i0: int,
    i1: int,
    stats,
) -> None:
    """Accumulate BLEU sufficient statistics for pairs [i0, i1) into ``stats``.

    ``stats`` is a mutable int sequence of length 10: correct[4], total[4],
    sys_len, ref_len. Mirrors the native kernel pair-for-pair.
    """
    for p in range(i0, i1):
        hs, he = hyp_off[p], hyp_off[p + 1]
        hl = he - hs
        s0, s1 = seg_index[p], seg_index[p + 1]
        if s1 <= s0:
            continue  # pair with no references: contributes nothing
        ref_lens = [int(ref_off[s + 1] - ref_off[s]) for s in range(s0, s1)]
        stats[8] += hl
        stats[9] += min(ref_lens, key=lambda rl: (abs(rl - hl), rl))
        for n in range(1, MAX_ORDER + 1):
            hyp_counts = _ngram_counts(hyp_ids, hs, hl, n)
            if hyp_counts:
                stats[4 + (n - 1)] += sum(hyp_counts.values())
            ref_max: dict = {}
            for s in range(s0, s1):
                rs = int(ref_off[s])
                rl = int(ref_off[s + 1] - ref_off[s])
                for k, v in _ngram_counts(ref_ids, rs, rl, n).items():
                    if v > ref_max.get(k, 0):
                        ref_max[k] = v
            correct = 0
            for k, v in hyp_counts.items():
                correct += min(v, ref_max.get(k, 0))
            stats[n - 1] += correct


def chrf_fscore(m, hh, rr, beta: float) -> float:
    """Sentence-level chrF F-score (x100), shared op order with the kernel."""
    np_ = 0
    sp = 0.0
    sr = 0.0
    for mj, hj, rj in zip(m, hh, rr):
        if hj > 0 and rj > 0:
            np_ += 1
            sp += mj / hj
            sr += mj / rj
    if np_ == 0:
        return 0.0
    avg_p = sp / np_
    avg_r = sr / np_
    beta_sq = beta**2
    denom = beta_sq * avg_p + avg_r
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * avg_p * avg_r / denom * 100


def chrf_stats_range(
    hypc,
    hypc_off,
    refc,
    refc_off,
    seg_index,
    hypw,
    hypw_off,
    refw,
    refw_off,
    i0: int,
    i1: int,
    char_order: int,
    word_order: int,
    beta: float,
    out_m,
    out_h,
    out_r,
) -> None:
    """Accumulate chrF per-order (match, hyp_total, ref_total) for [i0, i1).

    Per sentence pair, the reference with the highest sentence-level F score
    (ties keep the earliest) is selected; its statistics are accumulated with
    the rule: both totals > 0 -> accumulate (m, h, r); only ref_total > 0 ->
    accumulate ref_total; otherwise accumulate nothing. Mirrors the kernel.
    """
    orders = char_order + word_order
    for p in range(i0, i1):
        hcs, hce = hypc_off[p], hypc_off[p + 1]
        hcl = hce - hcs
        hws, hwe = hypw_off[p], hypw_off[p + 1]
        hwl = hwe - hws
        s0, s1 = seg_index[p], seg_index[p + 1]
        if s1 <= s0:
            continue  # pair with no references: contributes nothing
        # hypothesis n-gram counts per order (shared across references)
        hyp_counts = []
        for n in range(1, char_order + 1):
            hyp_counts.append(_ngram_counts(hypc, hcs, hcl, n))
        for n in range(1, word_order + 1):
            hyp_counts.append(_ngram_counts(hypw, hws, hwl, n))
        best_f = None
        best = None
        for s in range(s0, s1):
            rcs = int(refc_off[s])
            rcl = int(refc_off[s + 1] - refc_off[s])
            rws = int(refw_off[s])
            rwl = int(refw_off[s + 1] - refw_off[s])
            m = [0] * orders
            hh = [0] * orders
            rr = [0] * orders
            for j, n in enumerate(range(1, char_order + 1)):
                hc = hyp_counts[j]
                rc = _ngram_counts(refc, rcs, rcl, n)
                hh[j] = max(0, hcl - n + 1)
                rr[j] = max(0, rcl - n + 1)
                if hh[j] > 0 and rr[j] > 0:
                    m[j] = sum(min(v, rc.get(k, 0)) for k, v in hc.items())
            for idx, n in enumerate(range(1, word_order + 1)):
                j = char_order + idx
                hc = hyp_counts[j]
                rc = _ngram_counts(refw, rws, rwl, n)
                hh[j] = max(0, hwl - n + 1)
                rr[j] = max(0, rwl - n + 1)
                if hh[j] > 0 and rr[j] > 0:
                    m[j] = sum(min(v, rc.get(k, 0)) for k, v in hc.items())
            f = chrf_fscore(m, hh, rr, beta)
            if best_f is None or f > best_f:
                best_f = f
                best = (m, hh, rr)
        m, hh, rr = best
        for j in range(orders):
            if hh[j] > 0 and rr[j] > 0:
                out_m[j] += m[j]
                out_h[j] += hh[j]
                out_r[j] += rr[j]
            elif rr[j] > 0:
                out_r[j] += rr[j]
