"""Vendored pure-Python (NumPy) reference: dithering + p-value spectrum.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``ELEPHANT_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of
the published algorithms:

  * spike dithering as used for SPADE surrogate generation (Torre et al.,
    Front. Comput. Neurosci. 7:132, 2013): independent uniform displacement
    in [-dither, +dither), optionally shrunken so dithered spikes keep a
    refractory distance min(given, smallest ISI) from their neighbours;
  * the p-value spectrum of pattern signatures: per (size, duration)
    column of the per-surrogate maximal-occurrence matrix, a unit-bin
    histogram from ``min_occ`` to the column maximum, reverse-cumulated
    and divided by ``n_surr``.

Both functions mirror the reference semantics operation for operation
(elephant's ``spike_train_surrogates.dither_spikes``,
``conversion.BinnedSpikeTrain`` with ``tolerance=None``, and
``spade._get_pvalue_spec``), so the p-value spectrum is bit-exact against
the reference and the dithering is distributionally equivalent (RNG
sequence parity across implementations is impossible by construction —
the contract is same-seed reproducibility within a backend and
distributional equivalence across backends, asserted by the differential
suite).
"""

from __future__ import annotations

import numpy as np

METHOD_PLAIN = 0
METHOD_REFRACTORY = 1

EDGES_DROP = 0
EDGES_CLAMP = 1


def dither_reference(
    spike_counts: np.ndarray,
    spike_times: np.ndarray,
    t_start: float,
    t_stop: float,
    bin_size: float,
    n_bins: int,
    dither: float,
    refractory_period: float,
    method: int,
    edges_mode: int,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    """NumPy implementation of the dither kernel; same contract as native.

    Returns the C-order uint8 buffer (n_surrogates, n_trains, n_bins) with
    occupied bins set to 1. Deterministic for a given ``seed`` (PCG64);
    the seed does not need to (and cannot) reproduce any other
    implementation's RNG sequence.
    """
    rng = np.random.default_rng(seed)
    n_trains = int(spike_counts.shape[0])
    out = np.zeros((n_surrogates, n_trains, n_bins), dtype=np.uint8)
    base = 0
    for i in range(n_trains):
        n = int(spike_counts[i])
        times = spike_times[base : base + n]
        base += n
        if method == METHOD_PLAIN:
            if n == 0:
                continue
            for s in range(n_surrogates):
                tp = times + 2.0 * dither * rng.random(n) - dither
                if edges_mode == EDGES_DROP:
                    tp = tp[(tp > t_start) & (tp < t_stop)]
                else:
                    tp = np.clip(tp, t_start, t_stop)
                # BinnedSpikeTrain (tolerance=None): truncation; spikes whose
                # bin equals n_bins (landing on t_stop) are discarded.
                b = ((tp - t_start) / bin_size).astype(np.int64)
                b = b[(b >= 0) & (b < n_bins)]
                out[s, i, b] = 1
        else:
            if n >= 2:
                refr = min(refractory_period, float(np.min(np.diff(times))))
            else:
                refr = refractory_period
            for s in range(n_surrogates):
                cur = times.copy()
                for idx in rng.permutation(n):
                    spike = cur[idx]
                    prev_spike = cur[idx - 1] if idx > 0 else t_start - refr
                    next_spike = cur[idx + 1] if idx < n - 1 else t_stop + refr
                    prev_dither = min(dither, spike - prev_spike - refr)
                    next_dither = min(dither, next_spike - spike - refr)
                    cur[idx] = spike + (prev_dither + next_dither) * rng.random() - prev_dither
                b = ((cur - t_start) / bin_size).astype(np.int64)
                b = b[(b >= 0) & (b < n_bins)]
                out[s, i, b] = 1
    return out


def pvalue_spec_reference(
    max_occs: np.ndarray,
    min_spikes: int,
    min_occ: int,
    n_surr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy implementation of the p-value spectrum; same contract as native.

    ``max_occs`` is the float64 C-order cube (n_surr, n_sizes, winlen); for
    the 2-D spectrum '#' pass winlen == 1 (and dur is emitted as 0). The
    per-column histogram uses unit bins from ``min_occ`` to the column
    maximum, the latter truncated to int16 exactly like the reference;
    p-values are the reverse cumulated counts divided by ``n_surr``.
    """
    n_sizes = int(max_occs.shape[1])
    winlen = int(max_occs.shape[2])
    sizes: list[int] = []
    occs_out: list[int] = []
    durs: list[int] = []
    pvals: list[float] = []
    for size_id in range(n_sizes):
        for dur in range(winlen):
            col = max_occs[:, size_id, dur]
            # Reference arithmetic: the column maximum is truncated to int16
            # before the bin edges are built (np.arange(min_occ, max + 2)).
            upper = np.max(col).astype(np.int16) + 2
            counts, edges = np.histogram(col, bins=np.arange(min_occ, upper))
            pvalues = np.cumsum(counts[::-1])[::-1] / n_surr
            for occ_id, occ in enumerate(edges[:-1].astype(np.uint16)):
                sizes.append(min_spikes + size_id)
                occs_out.append(int(occ))
                durs.append(dur)
                pvals.append(float(pvalues[occ_id]))
    return (
        np.asarray(sizes, dtype=np.int32),
        np.asarray(occs_out, dtype=np.int32),
        np.asarray(durs, dtype=np.int32),
        np.asarray(pvals, dtype=np.float64),
    )
