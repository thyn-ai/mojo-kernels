"""Vendored pure-Python reference for Konno-Ohmachi spectral smoothing.

This is the fallback path used when the native Mojo kernel is unavailable
(unsupported platform, missing shared library, ABI mismatch, or
``OBSPY_MOJO_DISABLE_NATIVE=1``). It is a clean-room implementation of the
published smoothing window (Konno & Ohmachi, 1998, BSSA 88(1):228-241),

    W(f, fc) = (sin(x)/x)^4   with   x = b * log10(f / fc),

written to be observably identical to the widely used `obspy` package
(function ``obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing``):
same defaults, same branch decisions, same edge semantics. The exact
behavioral contract was established black-box (probing the oracle's inputs
and outputs, never its source):

  * two evaluation paths, chosen exactly like the oracle:
      - loop path: per-center fused window+reduction; used for 1-D input
        with ``count == 1``, whenever ``enforce_no_matrix`` is set, or when
        the window matrix estimate exceeds ``max_memory_usage`` MB
        (estimated as ``n_freqs**2 * dtype.itemsize / 1e6``);
      - matrix path: build W, optionally row-normalize, take the
        ``count``-th matrix power, and reduce with one matmul; used
        otherwise. The two paths give measurably different results (the
        loop path normalizes each center's window by its own sum; the
        matrix path row-normalizes the sample-by-center matrix), and both
        are reproduced faithfully.
  * ``normalize=True`` divides by the window sum (loop path: normalize the
    window first, then reduce; matrix path: row-normalize W);
    ``normalize=False`` is the plain weighted sum.
  * ``count`` is the number of filter applications, clamped to at least 1.
  * Samples at exactly ``f == 0`` carry zero weight; centers at exactly
    ``f == 0`` pass the input value through unchanged (structurally, via an
    identity row/column, on the matrix path). Negative frequencies poison
    the output with NaN through the log10 domain, exactly like the oracle.
  * float32 input is computed entirely in float32; mixed spectra/frequency
    dtypes are both cast to float64 (the caller warns, mirroring the oracle).
"""

from __future__ import annotations

import numpy as np

# Centers are processed in blocks of this many columns so the loop path never
# materializes the full O(n^2) window matrix (its raison d'etre), while the
# per-center arithmetic stays vectorized NumPy like the oracle's own loop.
_LOOP_BLOCK = 512


def window_matrix(frequencies: np.ndarray, bandwidth: float) -> np.ndarray:
    """Raw window matrix W[i, j] = W(f_i, f_j) (fallback computation).

    Applies the limiting value 1 wherever f_i == f_j and zeroes the rows of
    exact-zero sample frequencies. Zero-center columns come out NaN here
    (sin(+-inf)); `normalize_window_matrix` zeroes them, mirroring the
    oracle's structural identity handling.
    """
    f = frequencies
    with np.errstate(all="ignore"):
        x = bandwidth * np.log10(f[:, None] / f[None, :])
        w = (np.sin(x) / x) ** 4
    w[f[:, None] == f[None, :]] = 1.0
    w[f == 0.0, :] = 0.0
    return np.asarray(w, dtype=f.dtype)


def normalize_window_matrix(
    w: np.ndarray, frequencies: np.ndarray, normalize: bool
) -> np.ndarray:
    """Shared post-processing of a raw window matrix (both backends).

    Zero-center columns are zeroed, rows are normalized to sum to 1 when
    `normalize` is set (rows whose sum is zero stay zero), and the
    zero-frequency diagonal is set to 1, making zero-frequency centers an
    identity transform that survives matrix powers.
    """
    zero = frequencies == 0.0
    out = np.array(w, copy=True)  # never mutate the caller's buffer
    out[:, zero] = 0.0
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = out.sum(axis=1)
            np.divide(out, row_sums[:, None], out=out, where=(row_sums != 0)[:, None])
    out[zero, zero] = 1.0
    return out


def smoothing_matrix(
    frequencies: np.ndarray, bandwidth: float, normalize: bool
) -> np.ndarray:
    """The oracle's effective smoothing matrix (fallback computation)."""
    return normalize_window_matrix(window_matrix(frequencies, bandwidth), frequencies, normalize)


def apply_matrix(spectra: np.ndarray, matrix: np.ndarray, count: int) -> np.ndarray:
    """Matrix-path reduction: spectra (n_spectra, n) @ matrix**count."""
    if count <= 1:
        return spectra @ matrix
    return spectra @ np.linalg.matrix_power(matrix, count)


def smooth_loop(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    bandwidth: float,
    count: int,
    normalize: bool,
) -> np.ndarray:
    """Loop-path smoothing (fallback computation), one spectrum per row.

    Per application and per center j: w = W(f, f_j) over all samples with
    the f == f_j limiting value and the f == 0 sample mask; normalize=True
    divides the window by its sum before reducing (the oracle's op order);
    centers at exactly f == 0 pass the current value through unchanged.
    """
    n_spectra, n_freqs = spectra.shape
    zero_f = frequencies == 0.0
    cur = np.array(spectra, copy=True)
    out = np.empty_like(cur)
    for _ in range(count):
        for start in range(0, n_freqs, _LOOP_BLOCK):
            stop = min(start + _LOOP_BLOCK, n_freqs)
            fc = frequencies[start:stop]
            with np.errstate(all="ignore"):
                x = bandwidth * np.log10(frequencies[:, None] / fc[None, :])
                w = (np.sin(x) / x) ** 4
                w[frequencies[:, None] == fc[None, :]] = 1.0
                w[zero_f, :] = 0.0
                if normalize:
                    w = w / w.sum(axis=0)[None, :]
                block = (w[None, :, :] * cur[:, :, None]).sum(axis=1)
            out[:, start:stop] = block
        # Zero-frequency centers pass through unchanged (also discards the
        # NaN their empty windows produced above).
        out[:, zero_f] = cur[:, zero_f]
        cur, out = out, cur
    return cur
