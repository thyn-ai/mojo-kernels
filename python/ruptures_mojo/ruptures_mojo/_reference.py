"""Vendored pure-Python change-point detection (fallback path).

This is the fallback used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``RUPTURES_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
textbook offline change-point detection algorithms (Truong et al., "Selective
review of offline change point detection methods", Signal Processing 167
(2020)) written to be *observably identical* to the published ``ruptures``
package (PyPI ruptures 1.1.10) on 1-D signals with the ``l2`` and ``l1``
cost models: identical candidate enumeration, identical float64 operation
order, identical tie-breaking, so the returned breakpoint sets are integer
equal, not merely close.

Exactness contract, mirroring ruptures semantics:

- the l2 segment cost of ``signal[start:end]`` (length ``m``) is
  ``signal[start:end].var() * m`` evaluated with NumPy's two-pass
  pairwise-summation variance (this module calls NumPy the same way, so the
  values are bit-identical to ruptures');
- the l1 segment cost is ``abs(segment - median(segment)).sum()`` with the
  NumPy median (middle value for odd ``m``, ``(a + b) / 2`` of the two middle
  values for even ``m``);
- dynamic programming (Dynp) minimizes the total cost over partitions whose
  breakpoints are multiples of ``jump`` with segments of at least ``min_size``
  samples; ties resolve to the smallest candidate last breakpoint (first
  minimum in ascending scan order);
- PELT minimizes ``sum(costs) + pen * n_bkps`` with the same admissibility,
  pruning a candidate start ``t`` once its partial total exceeds the current
  optimum plus ``pen``; ties resolve to the smallest candidate start;
- binary segmentation (Binseg) greedily splits the segment with the largest
  cost gain; within a segment, gain ties resolve to the *largest* candidate
  breakpoint (tuple ``max``), across segments ties resolve to the *leftmost*
  segment (first maximum).

The effective ``min_size`` is ``max(min_size, cost_min_size)`` with
``cost_min_size == 1`` for l2 and ``2`` for l1, as in ruptures.
"""

from __future__ import annotations

import math

import numpy as np

# Per-model minimum segment sizes baked into the cost functions (ruptures
# CostL2.min_size == 1, CostL1.min_size == 2).
_COST_MIN_SIZE = {"l2": 1, "l1": 2}


class BadSegmentationParameters(Exception):
    """Raised when no partition is possible for the given parameters.

    Mirrors ``ruptures.exceptions.BadSegmentationParameters``.
    """


def effective_min_size(model: str, min_size: int) -> int:
    """The min_size the estimator actually uses for this cost model."""
    return max(min_size, _COST_MIN_SIZE[model])


def sanity_check(n_samples: int, n_bkps: int, jump: int, min_size: int) -> bool:
    """True if a partition of n_samples into n_bkps+1 segments is possible.

    Mirrors ``ruptures.utils.sanity_check``: breakpoints land on multiples of
    ``jump`` and every segment holds at least ``min_size`` samples.
    """
    n_adm_bkps = n_samples // jump  # number of admissible breakpoints
    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * math.ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True


class _CostL2:
    """Least squared deviation: var(segment) * len(segment), NumPy-exact."""

    __slots__ = ("signal",)

    def __init__(self, signal: np.ndarray) -> None:
        self.signal = signal

    def error(self, start: int, end: int) -> float:
        m = end - start
        return float(self.signal[start:end].var() * m)


class _CostL1:
    """Least absolute deviation: sum |x - median(x)| over the segment."""

    __slots__ = ("signal",)

    def __init__(self, signal: np.ndarray) -> None:
        self.signal = signal

    def error(self, start: int, end: int) -> float:
        sub = self.signal[start:end]
        med = np.median(sub)
        return float(np.abs(sub - med).sum())


def make_cost(model: str, signal: np.ndarray):
    """Cost object for `model` ("l2" or "l1") over a 1-D float64 signal."""
    if model == "l2":
        return _CostL2(signal)
    if model == "l1":
        return _CostL1(signal)
    raise ValueError(f"unsupported cost model: {model!r} (supported: 'l2', 'l1')")


def _check_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize the input to a contiguous 1-D float64 array.

    Accepts (n,) and (n, 1) inputs (the two shapes ruptures treats
    identically for these costs). Raises TypeError/ValueError otherwise.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(
            f"signal must be 1-D (shape (n,) or (n, 1)); got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal must contain only finite values (no NaN or inf)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _dynp_dp(cost, n_samples: int, n_bkps: int, jump: int, min_size: int) -> list[int]:
    """Exact optimal partition by dynamic programming (Dynp semantics).

    G[k][t] = minimum total cost of partitioning signal[0:t] into k + 1
    admissible segments (k breakpoints). The recurrence and its tie-breaking
    (first minimum over ascending candidate last breakpoints) reproduce
    ruptures' memoized recursion; G values are sums of segment costs in
    left-to-right signal order, bit-identical to ruptures' dict-value sums.
    """
    # Grid of candidate segment ends: multiples of jump, plus n_samples.
    grid = [t for t in range(0, n_samples + 1, jump)]
    if grid[-1] != n_samples:
        grid.append(n_samples)
    index_of = {t: i for i, t in enumerate(grid)}
    n_grid = len(grid)

    INF = math.inf
    # G[k][i]: optimal total cost for signal[0:grid[i]] with k breakpoints.
    G = [[INF] * n_grid for _ in range(n_bkps + 1)]
    parent = [[-1] * n_grid for _ in range(n_bkps + 1)]

    cost_cache: dict[tuple[int, int], float] = {}

    def seg_cost(a: int, b: int) -> float:
        key = (a, b)
        val = cost_cache.get(key)
        if val is None:
            val = cost.error(a, b)
            cost_cache[key] = val
        return val

    for k in range(0, n_bkps + 1):
        for i, t in enumerate(grid):
            if k == 0:
                # A single segment: only state needed is full-length prefixes
                # that can serve as left parts of deeper recursions.
                if t >= min_size:
                    G[0][i] = seg_cost(0, t)
                continue
            best = INF
            best_b = -1
            # Candidate last breakpoints: nonzero multiples of jump below t,
            # ascending. Admissibility mirrors ruptures exactly: the closed-
            # form sanity check for the left subproblem, and >= min_size
            # samples for the right segment.
            for b in range(jump, t, jump):
                if t - b < min_size:
                    continue
                if not sanity_check(b, k - 1, jump, min_size):
                    continue
                left = G[k - 1][index_of[b]]
                total = left + seg_cost(b, t)
                if total < best:  # strict: first (smallest b) wins ties
                    best = total
                    best_b = b
            G[k][i] = best
            parent[k][i] = best_b

    i_end = index_of[n_samples]
    if parent[n_bkps][i_end] < 0:
        raise BadSegmentationParameters
    bkps = []
    k, i = n_bkps, i_end
    while k > 0:
        b = parent[k][i]
        bkps.append(b)
        k, i = k - 1, index_of[b]
    bkps.reverse()
    bkps.append(n_samples)
    return bkps


def dynp_detect(
    signal: np.ndarray,
    n_bkps: int,
    min_size: int = 2,
    jump: int = 5,
    model: str = "l2",
) -> list[int]:
    """Optimal segmentation with exactly `n_bkps` breakpoints (Dynp)."""
    sig = _check_signal(signal)
    n_samples = sig.shape[0]
    min_size = effective_min_size(model, min_size)
    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")
    if not isinstance(n_bkps, int) or n_bkps < 0:
        raise ValueError(f"n_bkps must be a non-negative int, got {n_bkps!r}")
    if not sanity_check(n_samples, n_bkps, jump, min_size):
        raise BadSegmentationParameters
    if n_bkps == 0:
        return [n_samples]
    cost = make_cost(model, sig)
    return _dynp_dp(cost, n_samples, n_bkps, jump, min_size)


def _pelt_seg(cost, n_samples: int, pen: float, jump: int, min_size: int) -> list[int]:
    """PELT recursion; totals are left-to-right sums, ties keep smallest t."""
    # F[t]: optimal total cost (penalties included) for signal[0:t].
    F: dict[int, float] = {0: 0.0}
    parent: dict[int, int] = {}
    admissible: list[int] = []

    ind = [k for k in range(0, n_samples, jump) if k >= min_size]
    ind.append(n_samples)
    for bkp in ind:
        # Add the point that becomes admissible at this iteration.
        new_adm_pt = math.floor((bkp - min_size) / jump)
        new_adm_pt *= jump
        admissible.append(new_adm_pt)

        best_total = math.inf
        best_t = -1
        totals: dict[int, float] = {}
        for t in admissible:
            left = F.get(t)
            if left is None:  # no partition of signal[0:t]; dropped below
                continue
            total = left + (cost.error(t, bkp) + pen)
            totals[t] = total
            if total < best_total:  # strict: first (smallest t) wins ties
                best_total = total
                best_t = t
        if best_t < 0:
            # Mirrors ruptures raising ValueError on min() of an empty set
            # (only reachable with pen < 0 pruning everything).
            raise ValueError("min() iterable argument is empty")
        F[bkp] = best_total
        parent[bkp] = best_t
        # Prune: keep starts whose candidate total is within pen of optimum.
        admissible = [t for t in admissible if t in totals and totals[t] <= best_total + pen]

    bkps = []
    t = n_samples
    while parent[t] > 0:
        bkps.append(parent[t])
        t = parent[t]
    bkps.reverse()
    bkps.append(n_samples)
    return bkps


def pelt_detect(
    signal: np.ndarray,
    pen: float,
    min_size: int = 2,
    jump: int = 5,
    model: str = "l2",
) -> list[int]:
    """Penalized change-point detection (PELT)."""
    sig = _check_signal(signal)
    n_samples = sig.shape[0]
    min_size = effective_min_size(model, min_size)
    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")
    if not sanity_check(n_samples, 0, jump, min_size):
        raise BadSegmentationParameters
    cost = make_cost(model, sig)
    return _pelt_seg(cost, n_samples, float(pen), jump, min_size)


def _single_bkp(cost, start: int, end: int, jump: int, min_size: int):
    """Best split of signal[start:end]: (bkp, gain) or (None, 0).

    Gain ties resolve to the largest candidate breakpoint (tuple max).
    """
    segment_cost = cost.error(start, end)
    best_gain = None
    best_bkp = None
    for bkp in range(start, end, jump):
        if bkp - start >= min_size and end - bkp >= min_size:
            gain = segment_cost - cost.error(start, bkp) - cost.error(bkp, end)
            # tuple (gain, bkp) max: strict > on gain; on tie larger bkp wins
            if best_gain is None or gain > best_gain or (
                gain == best_gain and bkp > best_bkp
            ):
                best_gain = gain
                best_bkp = bkp
    if best_bkp is None:
        return None, 0
    return best_bkp, best_gain


def _binseg_seg(cost, n_samples: int, n_bkps: int, jump: int, min_size: int) -> list[int]:
    bkps = [n_samples]
    memo: dict[tuple[int, int], tuple] = {}
    while True:
        new_bkps = []
        for start, end in zip([0] + bkps, bkps):
            key = (start, end)
            hit = memo.get(key)
            if hit is None:
                hit = _single_bkp(cost, start, end, jump, min_size)
                memo[key] = hit
            new_bkps.append(hit)
        # max by gain, first maximum wins: leftmost segment on ties.
        bkp, gain = new_bkps[0]
        for cand_bkp, cand_gain in new_bkps[1:]:
            if cand_gain > gain:
                bkp, gain = cand_bkp, cand_gain
        if bkp is None:  # all configurations explored
            break
        if len(bkps) - 1 >= n_bkps:
            break
        bkps.append(bkp)
        bkps.sort()
    return sorted(bkps)


def binseg_detect(
    signal: np.ndarray,
    n_bkps: int,
    min_size: int = 2,
    jump: int = 5,
    model: str = "l2",
) -> list[int]:
    """Greedy binary segmentation with exactly `n_bkps` breakpoints (Binseg)."""
    sig = _check_signal(signal)
    n_samples = sig.shape[0]
    min_size = effective_min_size(model, min_size)
    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")
    if not isinstance(n_bkps, int) or n_bkps < 0:
        raise ValueError(f"n_bkps must be a non-negative int, got {n_bkps!r}")
    if not sanity_check(n_samples, n_bkps, jump, min_size):
        raise BadSegmentationParameters
    cost = make_cost(model, sig)
    return _binseg_seg(cost, n_samples, n_bkps, jump, min_size)
