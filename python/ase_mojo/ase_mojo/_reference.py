"""Vendored pure-Python/NumPy reference implementation of the neighbor search.

This is the correctness oracle and the Windows/unsupported-platform backend.
It implements the same linked-cell algorithm as the Mojo kernel (see
kernels/ase_neighborlist/src/aseneighborlistmojo.mojo for the full
derivation), vectorized differently: instead of a per-atom Python loop, the
candidate (atom, bin-member) pairs of each bin-offset are produced by one
vectorized equi-join between the atoms' target-bin keys and the sorted bin
table — 27-ish NumPy joins per call instead of O(atoms x offsets) Python
iterations.

Bit-compatibility with the native kernel: the squared-distance predicate is
evaluated with the same float64 operations in the same association order
(dr0 = df0*c00 + df1*c10 + df2*c20, then dd = dr0^2 + dr1^2 + dr2^2, strict
<), so native and fallback emit bit-identical (i, j, S) sets — not just
sets within a tolerance.
"""

from __future__ import annotations

import numpy as np

from ase_mojo._native import MODE_MATRIX, MODE_RADII

_NBINS_SAFETY_CEILING = 1 << 31


def _geometry(
    positions: np.ndarray, cell: np.ndarray, pbc_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inv-or-identity, metric, heights) — mirrors the native kernel."""
    det = float(np.linalg.det(cell))
    if abs(det) < 1e-300:
        if pbc_b.any():
            raise ValueError(
                "invalid geometry: the cell is singular on a periodic axis"
            )
        return np.eye(3), np.eye(3), np.ones(3)
    inv = np.linalg.inv(cell)
    h = 1.0 / np.linalg.norm(inv, axis=0)
    return inv, cell, h


def _bin_grid(
    fw: np.ndarray,
    pbc_b: np.ndarray,
    h: np.ndarray,
    cmax: float,
    max_nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis (nbins, widths, origins, K) — mirrors the native kernel."""
    nbins = np.ones(3, dtype=np.int64)
    widths = np.ones(3, dtype=np.float64)
    origins = np.zeros(3, dtype=np.float64)
    for d in range(3):
        if pbc_b[d]:
            nbins[d] = max(1, int(np.floor(h[d] / cmax)))
            widths[d] = 1.0 / nbins[d]
        else:
            lo = float(fw[:, d].min())
            hi = float(fw[:, d].max())
            rng = hi - lo
            origins[d] = lo
            if rng > 0.0:
                nbins[d] = max(1, int(np.floor(rng * h[d] / cmax)))
                widths[d] = rng / nbins[d]
            else:
                nbins[d] = 1
                widths[d] = 1.0
    ceiling = max_nbins if max_nbins > 0 else _NBINS_SAFETY_CEILING
    product = int(nbins[0] * nbins[1] * nbins[2])
    if product > _NBINS_SAFETY_CEILING:
        raise ValueError("bin grid exceeds the safety ceiling")
    rounds = 0
    while product > ceiling and rounds < 64:
        s = (float(ceiling) / float(product)) ** (1.0 / 3.0)
        nbins = np.maximum(1, np.floor(nbins * s).astype(np.int64))
        for d in range(3):
            if pbc_b[d]:
                widths[d] = 1.0 / nbins[d]
            else:
                rng = float(fw[:, d].max() - fw[:, d].min())
                if rng > 0.0:
                    widths[d] = rng / nbins[d]
        product = int(nbins[0] * nbins[1] * nbins[2])
        rounds += 1
    if product > _NBINS_SAFETY_CEILING or product <= 0:
        raise ValueError("bin grid exceeds the safety ceiling")
    kx = np.floor(cmax / (h * widths)).astype(np.int64) + 1
    return nbins, widths, origins, kx


def _assign_bins(
    fw: np.ndarray,
    pbc_b: np.ndarray,
    nbins: np.ndarray,
    widths: np.ndarray,
    origins: np.ndarray,
) -> np.ndarray:
    """Per-atom bin coordinates (n, 3) — mirrors the native kernel."""
    n = fw.shape[0]
    coords = np.zeros((n, 3), dtype=np.int64)
    for d in range(3):
        if pbc_b[d]:
            b = np.floor(fw[:, d] * nbins[d]).astype(np.int64)
        elif float(fw[:, d].max() - fw[:, d].min()) > 0.0:
            b = np.floor((fw[:, d] - origins[d]) / widths[d]).astype(np.int64)
        else:
            b = np.zeros(n, dtype=np.int64)
        coords[:, d] = np.clip(b, 0, nbins[d] - 1)
    return coords


def build_ijs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff_mode: int,
    atom_types: np.ndarray,
    n_types: int,
    cut_matrix: np.ndarray,
    radii: np.ndarray,
    self_interaction: bool,
    bothways: bool,
    max_nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (i, j, S) with the same contract as the native kernel."""
    n = positions.shape[0]
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )

    pbc_b = pbc.astype(bool)
    inv, metric, h = _geometry(positions, cell, pbc_b)

    # Fractional coordinates; wrap periodic axes, keep the wrap integers.
    fw = positions @ inv
    kk = np.zeros((n, 3), dtype=np.int64)
    for d in range(3):
        if pbc_b[d]:
            fl = np.floor(fw[:, d])
            kk[:, d] = fl.astype(np.int64)
            fw[:, d] = fw[:, d] - fl

    if cutoff_mode == MODE_MATRIX:
        cmax = float(cut_matrix.max()) if cut_matrix.size else 0.0
    elif cutoff_mode == MODE_RADII:
        cmax = float(2.0 * radii.max()) if n else 0.0
    else:
        raise ValueError(f"unknown cutoff mode {cutoff_mode}")
    if not (cmax > 0.0):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )

    nbins, widths, origins, kx = _bin_grid(fw, pbc_b, h, cmax, max_nbins)
    coords = _assign_bins(fw, pbc_b, nbins, widths, origins)
    linear = (coords[:, 2] * nbins[1] + coords[:, 1]) * nbins[0] + coords[:, 0]

    # CSR over bins (atoms in index order within each bin).
    atom_order = np.argsort(linear, kind="stable")
    nb_total = int(nbins[0] * nbins[1] * nbins[2])
    starts = np.zeros(nb_total + 1, dtype=np.int64)
    np.add.at(starts, linear[atom_order] + 1, 1)
    starts = np.cumsum(starts)

    c00, c01, c02 = metric[0]
    c10, c11, c12 = metric[1]
    c20, c21, c22 = metric[2]

    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    out_s: list[np.ndarray] = []

    for m0 in range(-kx[0], kx[0] + 1):
        s0 = coords[:, 0] + m0
        if pbc_b[0]:
            q0 = np.floor_divide(s0, nbins[0])
            t0 = s0 - q0 * nbins[0]
            ok0 = np.ones(n, dtype=bool)
        else:
            q0 = np.zeros(n, dtype=np.int64)
            t0 = s0
            ok0 = (t0 >= 0) & (t0 < nbins[0])
        for m1 in range(-kx[1], kx[1] + 1):
            s1 = coords[:, 1] + m1
            if pbc_b[1]:
                q1 = np.floor_divide(s1, nbins[1])
                t1 = s1 - q1 * nbins[1]
                ok1 = ok0
            else:
                q1 = np.zeros(n, dtype=np.int64)
                t1 = s1
                ok1 = ok0 & (t1 >= 0) & (t1 < nbins[1])
            for m2 in range(-kx[2], kx[2] + 1):
                s2 = coords[:, 2] + m2
                if pbc_b[2]:
                    q2 = np.floor_divide(s2, nbins[2])
                    t2 = s2 - q2 * nbins[2]
                    ok = ok1
                else:
                    q2 = np.zeros(n, dtype=np.int64)
                    t2 = s2
                    ok = ok1 & (t2 >= 0) & (t2 < nbins[2])
                if not ok.any():
                    continue
                a_idx = np.nonzero(ok)[0]
                tbin = (t2[ok] * nbins[1] + t1[ok]) * nbins[0] + t0[ok]
                # Equi-join atoms (target bin tbin) against the CSR table.
                seg_start = starts[tbin]
                seg_len = starts[tbin + 1] - seg_start
                total = int(seg_len.sum())
                if total == 0:
                    continue
                a_rep = np.repeat(a_idx, seg_len)
                seg_off = np.repeat(seg_start, seg_len)
                within = np.arange(total, dtype=np.int64) - np.repeat(
                    np.cumsum(seg_len) - seg_len, seg_len
                )
                b_rep = atom_order[seg_off + within]

                df0 = fw[b_rep, 0] + q0[a_rep].astype(np.float64) - fw[a_rep, 0]
                df1 = fw[b_rep, 1] + q1[a_rep].astype(np.float64) - fw[a_rep, 1]
                df2 = fw[b_rep, 2] + q2[a_rep].astype(np.float64) - fw[a_rep, 2]
                # Same op order as the native kernel: bit-identical dd.
                dr0 = df0 * c00 + df1 * c10 + df2 * c20
                dr1 = df0 * c01 + df1 * c11 + df2 * c21
                dr2 = df0 * c02 + df1 * c12 + df2 * c22
                dd = dr0 * dr0 + dr1 * dr1 + dr2 * dr2
                if cutoff_mode == MODE_MATRIX:
                    rc = cut_matrix[atom_types[a_rep], atom_types[b_rep]]
                else:
                    rc = radii[a_rep] + radii[b_rep]
                mask = (rc > 0.0) & (dd < rc * rc)
                if not mask.any():
                    continue
                a_sel = a_rep[mask]
                b_sel = b_rep[mask]
                s_out0 = q0[a_sel] - kk[b_sel, 0] + kk[a_sel, 0]
                s_out1 = q1[a_sel] - kk[b_sel, 1] + kk[a_sel, 1]
                s_out2 = q2[a_sel] - kk[b_sel, 2] + kk[a_sel, 2]

                self_zero = (a_sel == b_sel) & (s_out0 == 0) & (s_out1 == 0) & (s_out2 == 0)
                if bothways:
                    emit = ~self_zero
                    if self_interaction:
                        emit = emit | self_zero
                else:
                    lexpos = (s_out0 > 0) | (
                        (s_out0 == 0) & ((s_out1 > 0) | ((s_out1 == 0) & (s_out2 > 0)))
                    )
                    zero_s = (s_out0 == 0) & (s_out1 == 0) & (s_out2 == 0)
                    emit = (~self_zero) & (lexpos | (zero_s & (a_sel < b_sel)))
                    if self_interaction:
                        emit = emit | self_zero
                keep = np.nonzero(emit)[0]
                if keep.size == 0:
                    continue
                out_i.append(a_sel[keep])
                out_j.append(b_sel[keep])
                out_s.append(
                    np.stack([s_out0[keep], s_out1[keep], s_out2[keep]], axis=1)
                )

    if not out_i:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )
    return (
        np.concatenate(out_i).astype(np.int64),
        np.concatenate(out_j).astype(np.int64),
        np.concatenate(out_s).astype(np.int64),
    )
