"""Konno-Ohmachi spectral smoothing, API-compatible with ObsPy.

``konno_ohmachi_smoothing`` mirrors
``obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing`` (obspy 1.5.x):
same signature, same defaults (``bandwidth=40``, ``count=1``,
``enforce_no_matrix=False``, ``max_memory_usage=512``, ``normalize=False``),
same validation errors, same loop-vs-matrix branch decision, and the same
results on both backends (differential-tested; see the package README for
the documented tolerance and measured agreement).

Smoothing runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back
to the vendored pure-NumPy implementation otherwise. Both backends share
validation and branch selection in this module, so they can never disagree
about anything but the last few ulps of transcendental evaluation.
"""

from __future__ import annotations

import operator
import warnings

import numpy as np

from obspy_mojo import _native, _reference
from obspy_mojo._native import NativeUnavailable

__all__ = ["konno_ohmachi_smoothing"]

_VALID_DTYPES = (np.float32, np.float64)


def _validate_inputs(spectra, frequencies):
    """Mirror the oracle's input validation, in its order.

    Attribute access on ``.dtype`` comes first, so plain lists fail with
    AttributeError exactly like the oracle; then dtype membership, then the
    mixed-dtype warning + cast to float64.
    """
    if spectra.dtype not in _VALID_DTYPES:
        raise ValueError("`spectra` needs to have a dtype of float32/64.")
    if frequencies.dtype not in _VALID_DTYPES:
        raise ValueError("`frequencies` needs to have a dtype of float32/64.")
    if spectra.dtype != frequencies.dtype:
        warnings.warn(
            "`frequencies` and `spectra` should have the same dtype. "
            "It will be changed to np.float64 for both."
        )
        spectra = spectra.astype(np.float64)
        frequencies = frequencies.astype(np.float64)
    return spectra, frequencies


def _uses_matrix_path(spectra, frequencies, count, enforce_no_matrix, max_memory_usage):
    """The oracle's loop-vs-matrix branch decision, with its memory estimate.

    The estimate (probed black-box at exact thresholds) counts the window
    matrix, the frequency vector and two scalars per first-axis entry of the
    input spectra, in mebibytes, and the comparison is strict::

        (n_freqs**2 + n_freqs + 2 * spectra.shape[0]) * dtype.itemsize / 2**20
            < max_memory_usage
    """
    if enforce_no_matrix:
        return False
    if spectra.ndim == 1 and not count > 1:
        return False
    n_freqs = frequencies.shape[0]
    estimated_mb = (
        (n_freqs**2 + n_freqs + 2 * spectra.shape[0])
        * spectra.dtype.itemsize
        / 1048576
    )
    return estimated_mb < max_memory_usage


def _check_compatible_lengths(spectra, frequencies):
    """Length agreement, checked before any native call (the fallback and the
    oracle surface the same ValueError class from NumPy broadcasting/matmul)."""
    if spectra.shape[-1] != frequencies.shape[0]:
        raise ValueError(
            "operands could not be broadcast together with shapes "
            f"{spectra.shape[-1:]} ({frequencies.shape[0]},)"
        )


def konno_ohmachi_smoothing(
    spectra,
    frequencies,
    bandwidth=40,
    count=1,
    enforce_no_matrix=False,
    max_memory_usage=512,
    normalize=False,
):
    """Smooth one spectrum per row with the Konno-Ohmachi smoothing window.

    Drop-in compatible with
    ``obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing``; see the
    package docstring for the mirrored contract. Returns a new
    ``np.ndarray`` of the same shape and dtype as ``spectra``.
    """
    spectra, frequencies = _validate_inputs(spectra, frequencies)
    use_matrix = _uses_matrix_path(
        spectra, frequencies, count, enforce_no_matrix, max_memory_usage
    )
    # `count` is the number of filter applications. The oracle clamps it to
    # at least one and rejects non-integers where they are used
    # (range()/matrix_power); operator.index reproduces both behaviors.
    applications = operator.index(count)
    if applications < 1:
        applications = 1
    _check_compatible_lengths(spectra, frequencies)

    n_freqs = frequencies.shape[0]
    single = spectra.ndim == 1
    if use_matrix:
        # Flatten leading dimensions (the oracle treats any (..., n) input as
        # one spectrum per row of a flat 2-D view). np.prod keeps n_freqs == 0
        # unambiguous, where reshape(-1, 0) would refuse to infer the lead.
        lead = int(np.prod(spectra.shape[:-1], dtype=np.int64))
        flat_in = spectra.reshape(lead, n_freqs)
        flat_freqs = np.ascontiguousarray(frequencies)
        try:
            w = _native.window_matrix(flat_freqs, bandwidth)
        except NativeUnavailable:
            matrix = _reference.smoothing_matrix(frequencies, bandwidth, normalize)
        else:
            matrix = _reference.normalize_window_matrix(w, flat_freqs, normalize)
        flat_out = _reference.apply_matrix(flat_in, matrix, applications)
        return flat_out.reshape(spectra.shape)

    # Loop path. The oracle's loop only supports 1-D/2-D input; mirror its
    # failure for higher-dimensional spectra.
    if spectra.ndim > 2:
        raise IndexError(
            f"index {spectra.shape[1]} is out of bounds for axis 1 "
            f"with size {spectra.shape[1]}"
        )
    rows = spectra.reshape(1, n_freqs) if single else spectra
    rows_in = np.ascontiguousarray(rows)
    freqs_in = np.ascontiguousarray(frequencies)
    try:
        out = _native.smooth_loop(rows_in, freqs_in, bandwidth, applications, normalize)
    except NativeUnavailable:
        out = _reference.smooth_loop(rows_in, freqs_in, bandwidth, applications, normalize)
    return out.reshape(n_freqs) if single else out
