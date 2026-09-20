"""Vendored pure-Python fallback for the nuScenes detection matching passes.

Semantics are identical to the native Mojo kernel (and to the greedy matching
inside nuscenes-devkit's detection evaluation): predictions of one class, in
confidence-sorted order (descending score, ties by descending original index),
are matched against the nearest still-unmatched ground-truth box of the same
class in the same sample, once per distance threshold. All arithmetic is plain
IEEE-754 float64 in the same operation order, so this fallback agrees with the
native backend element-wise.
"""

from __future__ import annotations

import math

import numpy as np

BOX_STRIDE = 8
OUT_STRIDE = 6

_NAN = float("nan")


def _angle_diff(x: float, y: float, period: float) -> float:
    """Smallest signed difference from angle y to angle x, modulo period."""
    half = period / 2.0
    diff = (x - y + half) % period - half
    if diff > math.pi:
        diff = diff - (2 * math.pi)
    return diff


def reference_match(
    *,
    n_classes: int,
    n_samples: int,
    gt_class_offsets: np.ndarray,
    gt_sample_offsets: np.ndarray,
    gt_vals: np.ndarray,
    gt_attr: np.ndarray,
    pred_class_offsets: np.ndarray,
    pred_sample: np.ndarray,
    pred_vals: np.ndarray,
    pred_attr: np.ndarray,
    periods: np.ndarray,
    dist_ths: np.ndarray,
) -> np.ndarray:
    """Pure-Python twin of the native kernel's `nuscenesevalmojo_match`."""
    # Work on plain Python lists/floats: float64 values convert exactly, and
    # scalar Python loops keep the reference operation order.
    gvals = [float(v) for v in np.asarray(gt_vals, dtype=np.float64).ravel()]
    pvals = [float(v) for v in np.asarray(pred_vals, dtype=np.float64).ravel()]
    gattr = [int(v) for v in np.asarray(gt_attr, dtype=np.int32).ravel()]
    pattr = [int(v) for v in np.asarray(pred_attr, dtype=np.int32).ravel()]
    psample = [int(v) for v in np.asarray(pred_sample, dtype=np.int32).ravel()]
    gcls = [int(v) for v in np.asarray(gt_class_offsets, dtype=np.int64).ravel()]
    pcls = [int(v) for v in np.asarray(pred_class_offsets, dtype=np.int64).ravel()]
    gcsr = [int(v) for v in np.asarray(gt_sample_offsets, dtype=np.int64).ravel()]
    periods = [float(v) for v in np.asarray(periods, dtype=np.float64).ravel()]
    ths = [float(v) for v in np.asarray(dist_ths, dtype=np.float64).ravel()]

    n_gt_total = len(gattr)
    n_pred_total = len(pattr)
    out = np.empty((len(ths), n_pred_total, OUT_STRIDE), dtype=np.float64)
    taken = bytearray(n_gt_total)

    for c in range(n_classes):
        gt_c0, gt_c1 = gcls[c], gcls[c + 1]
        pred_c0, pred_c1 = pcls[c], pcls[c + 1]
        period = periods[c]
        csr_base = c * (n_samples + 1)
        for t, dist_th in enumerate(ths):
            for g in range(gt_c0, gt_c1):
                taken[g] = 0
            for p in range(pred_c0, pred_c1):
                s = psample[p]
                pb = p * BOX_STRIDE
                px, py = pvals[pb], pvals[pb + 1]
                min_dist = math.inf
                best = -1
                for g in range(gcsr[csr_base + s], gcsr[csr_base + s + 1]):
                    if taken[g]:
                        continue
                    gb = g * BOX_STRIDE
                    dx = px - gvals[gb]
                    dy = py - gvals[gb + 1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < min_dist:
                        min_dist = d
                        best = g
                row = out[t, p]
                if best < 0 or not min_dist < dist_th:
                    row[0] = 0.0
                    row[1:] = _NAN
                    continue
                taken[best] = 1
                gb = best * BOX_STRIDE
                row[0] = 1.0
                row[1] = min_dist
                dvx = pvals[pb + 6] - gvals[gb + 6]
                dvy = pvals[pb + 7] - gvals[gb + 7]
                row[2] = math.sqrt(dvx * dvx + dvy * dvy)
                mw = min(pvals[pb + 2], gvals[gb + 2])
                ml = min(pvals[pb + 3], gvals[gb + 3])
                mh = min(pvals[pb + 4], gvals[gb + 4])
                inter = mw * ml * mh
                vol_pred = pvals[pb + 2] * pvals[pb + 3] * pvals[pb + 4]
                vol_gt = gvals[gb + 2] * gvals[gb + 3] * gvals[gb + 4]
                row[3] = 1.0 - inter / (vol_gt + vol_pred - inter)
                row[4] = abs(_angle_diff(gvals[gb + 5], pvals[pb + 5], period))
                ga = gattr[best]
                if ga < 0:
                    row[5] = _NAN
                else:
                    row[5] = 1.0 if ga != pattr[p] else 0.0
    return out
