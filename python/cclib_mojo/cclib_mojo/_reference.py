"""Vendored pure-Python (NumPy-vectorized) reference grid evaluator.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``CCLIB_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
textbook contracted-Cartesian-Gaussian amplitude (Taketa, Huzinaga, O-ohata,
J. Phys. Soc. Jap. 21, 2313 (1966)):

    bf(r) = N_c * (x-cx)^l (y-cy)^m (z-cz)^n * sum_p w_p exp(-alpha_p |r-c|^2)

written to match the published PyQuante reference semantics (same primitive
and contracted normalization, same zero-coefficient skipping rule as
cclib). Both backends consume the same ``_basis.BasisArrays``, so they can
never disagree about centers, powers, or normalization constants; the
differential suite asserts both against PyQuante within 1e-10 relative.

Only the grid evaluation lives here — basis validation and normalization
are shared with the native path in ``cclib_mojo._basis``.
"""

from __future__ import annotations

import numpy as np

MODE_WAVEFUNCTION = 0
MODE_DENSITY = 1


def eval_grid(
    basis,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    coeff: np.ndarray,
    mode: int,
) -> np.ndarray:
    """Evaluate MO amplitude(s) or the summed density on the whole grid.

    Same contract as ``cclib_mojo._native.eval_grid``: returns the C-order
    raveled float64 grid of shape (nx*ny*nz,). ``coeff`` is (n_mo, n_bf).
    """
    if mode not in (MODE_WAVEFUNCTION, MODE_DENSITY):
        raise ValueError(f"unknown mode {mode}")
    if mode == MODE_WAVEFUNCTION and coeff.shape[0] != 1:
        raise ValueError("wavefunction mode requires exactly one MO row")

    nx, ny, nz = ax.shape[0], ay.shape[0], az.shape[0]
    out = np.zeros((nx, ny, nz), dtype=np.float64)
    for mo in range(coeff.shape[0]):
        crow = coeff[mo]
        psi = np.zeros((nx, ny, nz), dtype=np.float64)
        for b in range(basis.n_bf):
            c = crow[b]
            if c == 0.0:
                # cclib's `if abs(mocoeffs[bs]) > 0.0` rule: functions with
                # an exactly zero coefficient contribute exactly zero.
                continue
            dx = ax - basis.center_x[b]
            dy = ay - basis.center_y[b]
            dz = az - basis.center_z[b]
            dx3 = dx[:, None, None]
            dy3 = dy[None, :, None]
            dz3 = dz[None, None, :]
            r2 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3
            o0, o1 = int(basis.offsets[b]), int(basis.offsets[b + 1])
            contraction = np.zeros((nx, ny, nz), dtype=np.float64)
            for p in range(o0, o1):
                contraction += basis.prim_w[p] * np.exp(-basis.prim_alpha[p] * r2)
            psi += (
                c
                * basis.bf_norm[b]
                * (
                    np.power(dx3, basis.powers_l[b])
                    * np.power(dy3, basis.powers_m[b])
                    * np.power(dz3, basis.powers_n[b])
                    * contraction
                )
            )
        if mode == MODE_WAVEFUNCTION:
            out = psi
        else:
            out += psi * psi
    return out.reshape(nx * ny * nz)
