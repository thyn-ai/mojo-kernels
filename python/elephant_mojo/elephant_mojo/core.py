"""Public API: SPADE surrogate dithering and the p-value spectrum.

``dither`` generates binned surrogate spike trains the way SPADE's
significance evaluation consumes them (uniform spike dithering, with an
optional refractory constraint, followed by binning); ``pvalue_spectrum``
turns the per-surrogate maximal pattern-occurrence matrix into the
p-value spectrum of pattern signatures.

Both entry points run on the native Mojo kernel when available and on the
vendored NumPy reference otherwise; validation and flattening are shared
in this module, so the backends cannot disagree about inputs. The
p-value spectrum is deterministic and bit-exact against the reference on
both backends. Dithering is inherently stochastic: same-seed output is
reproducible within a backend, and distributions match the reference
implementation within tight statistical tolerances (RNG *sequence* parity
across implementations is impossible — see the package README for the
documented parity split).
"""

from __future__ import annotations

import os

import numpy as np

from elephant_mojo import _reference
from elephant_mojo._native import (
    EDGES_CLAMP,
    EDGES_DROP,
    METHOD_PLAIN,
    METHOD_REFRACTORY,
    NativeUnavailable,
    _load,
)

METHOD_DITHER_SPIKES = "dither_spikes"
METHOD_REFRACTORY_PERIOD = "dither_spikes_with_refractory_period"

_METHOD_IDS = {
    METHOD_DITHER_SPIKES: METHOD_PLAIN,
    METHOD_REFRACTORY_PERIOD: METHOD_REFRACTORY,
}


class DitherError(ValueError):
    """Malformed dither input (structured, fail-fast)."""


class PValueSpectrumError(ValueError):
    """Malformed p-value-spectrum input (structured, fail-fast)."""


def _validate_positive_float(name: str, value: object, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise DitherError(f"{name} must be a real number, got {value!r}")
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise DitherError(f"{name} must be a real number, got {value!r}") from None
    if not np.isfinite(v):
        raise DitherError(f"{name} must be finite, got {v}")
    if allow_zero:
        if v < 0.0:
            raise DitherError(f"{name} must be >= 0, got {v}")
    elif v <= 0.0:
        raise DitherError(f"{name} must be > 0, got {v}")
    return v


def _validate_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DitherError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < minimum:
        raise DitherError(f"{name} must be >= {minimum}, got {v}")
    return v


def _flatten_spiketrains(
    spiketrains: object, t_start: float, t_stop: float
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the trains and flatten them to (counts, concatenated times)."""
    if not isinstance(spiketrains, (list, tuple)) or len(spiketrains) == 0:
        raise DitherError("spiketrains must be a non-empty list of 1-D arrays")
    counts = np.empty(len(spiketrains), dtype=np.int64)
    parts: list[np.ndarray] = []
    for i, train in enumerate(spiketrains):
        arr = np.asarray(train, dtype=np.float64)
        if arr.ndim != 1:
            raise DitherError(
                f"spiketrains[{i}] must be a 1-D array, got ndim={arr.ndim}"
            )
        if arr.size:
            if not np.all(np.isfinite(arr)):
                raise DitherError(f"spiketrains[{i}] contains non-finite times")
            if arr[0] < t_start or arr[-1] > t_stop:
                raise DitherError(
                    f"spiketrains[{i}] has spikes outside [{t_start}, {t_stop}]"
                )
            if np.any(np.diff(arr) < 0.0):
                raise DitherError(f"spiketrains[{i}] must be sorted ascending")
        counts[i] = arr.size
        parts.append(arr)
    total = int(counts.sum())
    if total:
        times = np.concatenate(parts).astype(np.float64, copy=False)
    else:
        times = np.empty(0, dtype=np.float64)
    return counts, np.ascontiguousarray(times)


def _resolve_seed(seed: object) -> int:
    """Explicit seed -> uint64; None -> OS entropy (documented escape hatch)."""
    if seed is None:
        return int.from_bytes(os.urandom(8), "little")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise DitherError(f"seed must be a non-negative integer or None, got {seed!r}")
    s = int(seed)
    if s < 0:
        raise DitherError(f"seed must be non-negative, got {s}")
    return s & 0xFFFFFFFFFFFFFFFF


def dither(
    spiketrains: object,
    bin_size: object,
    dither: object,
    n_surrogates: object = 1,
    *,
    t_start: object = 0.0,
    t_stop: object = None,
    method: str = METHOD_DITHER_SPIKES,
    edges: bool = True,
    refractory_period: object = None,
    seed: object = None,
) -> np.ndarray:
    """Generate binned surrogates by spike dithering.

    Equivalent to the surrogate generation behind SPADE's p-value spectrum
    (``dither_spikes`` followed by binning): every spike is displaced by an
    independent uniform offset in ``[-dither, +dither)`` and the result is
    binned onto ``n_bins = int((t_stop - t_start) / bin_size)`` bins
    (truncation, mirroring ``BinnedSpikeTrain(tolerance=None)``; a spike
    landing exactly on ``t_stop`` is discarded).

    Parameters
    ----------
    spiketrains : list of array-like
        Spike times per train (sorted ascending, within [t_start, t_stop]),
        in the same time unit as ``bin_size``, ``dither``, ``t_start`` and
        ``t_stop``.
    bin_size : float
        Bin width (> 0).
    dither : float
        Dither half-width (>= 0).
    n_surrogates : int
        Number of surrogates per train (>= 1).
    t_start : float
        Left recording edge (default 0.0).
    t_stop : float
        Right recording edge (required, > t_start).
    method : {'dither_spikes', 'dither_spikes_with_refractory_period'}
        Plain dither (default) or refractory-constrained dither.
    edges : bool
        Plain dither only: True drops spikes dithered outside
        ``(t_start, t_stop)`` (the reference default); False clamps them to
        the range ends (a spike clamped onto ``t_stop`` is then discarded
        by the binning, exactly like the reference pipeline).
    refractory_period : float
        Required for ``method='dither_spikes_with_refractory_period'``: the
        dither range of each spike is shrunk so it cannot enter the
        effective refractory period ``min(refractory_period, smallest ISI)``
        of its (current) neighbours. Must be None for the plain method.
    seed : int or None
        Seed for the generator. An explicit seed makes the call
        reproducible on the same backend; None draws OS entropy (the one
        non-deterministic input). Sequences differ between the native and
        NumPy backends by construction.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_surrogates, n_trains, n_bins) — the
        binned occupancy, i.e. the ``to_bool_array()`` form SPADE's pattern
        mining consumes.
    """
    if t_stop is None:
        raise DitherError("t_stop is required (no default is assumed)")
    t_start_f = _validate_positive_float("t_start", t_start, allow_zero=True)
    t_stop_f = _validate_positive_float("t_stop", t_stop, allow_zero=True)
    if not t_start_f < t_stop_f:
        raise DitherError(f"t_start must be smaller than t_stop, got {t_start_f} >= {t_stop_f}")
    bin_size_f = _validate_positive_float("bin_size", bin_size, allow_zero=False)
    dither_f = _validate_positive_float("dither", dither, allow_zero=True)
    n_surrogates_i = _validate_int("n_surrogates", n_surrogates, 1)
    # BinnedSpikeTrain(tolerance=None) derives the bin count by truncation.
    n_bins = int((t_stop_f - t_start_f) / bin_size_f)
    if n_bins < 1:
        raise DitherError(
            f"(t_stop - t_start) / bin_size truncates to {n_bins} bins; "
            "need at least 1"
        )
    if not isinstance(method, str) or method not in _METHOD_IDS:
        raise DitherError(
            f"method must be one of {sorted(_METHOD_IDS)}, got {method!r}"
        )
    if not isinstance(edges, bool):
        raise DitherError(f"edges must be a bool, got {edges!r}")
    method_id = _METHOD_IDS[method]
    if method_id == METHOD_REFRACTORY:
        if refractory_period is None:
            raise DitherError(
                "refractory_period is required for "
                "method='dither_spikes_with_refractory_period'"
            )
        refr_f = _validate_positive_float(
            "refractory_period", refractory_period, allow_zero=False
        )
    else:
        if refractory_period is not None:
            raise DitherError(
                "refractory_period must be None for method='dither_spikes'"
            )
        refr_f = 0.0
    edges_mode = EDGES_DROP if edges else EDGES_CLAMP
    seed_u64 = _resolve_seed(seed)
    counts, times = _flatten_spiketrains(spiketrains, t_start_f, t_stop_f)

    try:
        _load()  # fail fast here if the kernel cannot be used at all
        from elephant_mojo import _native

        out = _native.dither_native(
            counts,
            times,
            t_start_f,
            t_stop_f,
            bin_size_f,
            n_bins,
            dither_f,
            refr_f,
            method_id,
            edges_mode,
            n_surrogates_i,
            seed_u64,
        )
    except NativeUnavailable:
        out = _reference.dither_reference(
            counts,
            times,
            t_start_f,
            t_stop_f,
            bin_size_f,
            n_bins,
            dither_f,
            refr_f,
            method_id,
            edges_mode,
            n_surrogates_i,
            seed_u64,
        )
    return out.astype(bool, copy=False)


def pvalue_spectrum(
    max_occs: object,
    min_spikes: object,
    max_spikes: object,
    min_occ: object,
    n_surr: object = None,
    winlen: object = 1,
    spectrum: str = "#",
) -> list[list]:
    """Compute the p-value spectrum from maximal pattern occurrences.

    Equivalent to the deterministic tail of SPADE's surrogate evaluation:
    ``max_occs[surrogate, size, duration]`` holds, for every surrogate,
    the highest occurrence count of any mined pattern of that size (and
    duration) — as produced per surrogate from the mined concepts (see the
    README recipe). For each (pattern size, duration) column the
    occurrence histogram over unit bins ``[min_occ, ..., max + 1]`` is
    reverse-cumulated and divided by ``n_surr``, giving the p-value of
    every signature ``(size, occurrence[, duration])`` in the reference
    emission order (size, then duration, then occurrence ascending).

    Parameters
    ----------
    max_occs : array-like
        Float matrix of shape ``(n_surr, max_spikes - min_spikes + 1)`` for
        ``spectrum='#'``, or ``(n_surr, n_sizes, winlen)`` for
        ``spectrum='3d#'``. Values are occurrence counts (integral).
    min_spikes, max_spikes, min_occ : int
        Signature bounds; ``max_spikes >= min_spikes >= 1``,
        ``min_occ >= 0``.
    n_surr : int or None
        Number of surrogates the spectrum was collected from (the p-value
        denominator). Defaults to ``max_occs.shape[0]``, which is what the
        reference pipeline passes.
    winlen : int
        Window length; only meaningful for ``spectrum='3d#'`` (for '#' the
        reference forces it to 1, and so does this function).
    spectrum : {'#', '3d#'}
        2-D (size, occurrence) or 3-D (size, occurrence, duration)
        spectrum.

    Returns
    -------
    list of list
        ``[pattern_size, pattern_occ, p_value]`` entries for '#', or
        ``[pattern_size, pattern_occ, pattern_dur, p_value]`` for '3d#'.
        Deterministic and bit-exact against the reference on both
        backends.
    """
    if spectrum not in ("#", "3d#"):
        raise PValueSpectrumError(f"Invalid spectrum: '{spectrum}'")

    def _vint(name: str, value: object, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PValueSpectrumError(f"{name} must be an integer, got {value!r}")
        v = int(value)
        if v < minimum:
            raise PValueSpectrumError(f"{name} must be >= {minimum}, got {v}")
        return v

    min_spikes_i = _vint("min_spikes", min_spikes, 1)
    max_spikes_i = _vint("max_spikes", max_spikes, 1)
    if max_spikes_i < min_spikes_i:
        raise PValueSpectrumError(
            f"max_spikes ({max_spikes_i}) must be >= min_spikes ({min_spikes_i})"
        )
    min_occ_i = _vint("min_occ", min_occ, 0)
    n_sizes = max_spikes_i - min_spikes_i + 1

    arr = np.asarray(max_occs, dtype=np.float64)
    if spectrum == "#":
        # The reference forces winlen to 1 for the 2-D spectrum.
        winlen_i = 1
        if arr.ndim != 2:
            raise PValueSpectrumError(
                f"max_occs for spectrum '#' must be 2-D (n_surr, n_sizes), "
                f"got ndim={arr.ndim}"
            )
        arr = arr[:, :, None]
    else:
        winlen_i = _vint("winlen", winlen, 1)
        if arr.ndim != 3:
            raise PValueSpectrumError(
                f"max_occs for spectrum '3d#' must be 3-D "
                f"(n_surr, n_sizes, winlen), got ndim={arr.ndim}"
            )
        if arr.shape[2] != winlen_i:
            raise PValueSpectrumError(
                f"max_occs duration axis has length {arr.shape[2]} but "
                f"winlen={winlen_i}"
            )
    if arr.shape[1] != n_sizes:
        raise PValueSpectrumError(
            f"max_occs size axis has length {arr.shape[1]} but "
            f"max_spikes - min_spikes + 1 = {n_sizes}"
        )
    if arr.shape[0] < 1:
        raise PValueSpectrumError("max_occs must contain at least one surrogate")
    if not np.all(np.isfinite(arr)):
        raise PValueSpectrumError("max_occs contains non-finite values")
    if n_surr is None:
        n_surr_i = int(arr.shape[0])
    else:
        n_surr_i = _vint("n_surr", n_surr, 1)
    arr = np.ascontiguousarray(arr)

    try:
        _load()
        from elephant_mojo import _native

        sizes, occs, durs, pvals = _native.pvalue_spec_native(
            arr, min_spikes_i, min_occ_i, n_surr_i
        )
    except NativeUnavailable:
        sizes, occs, durs, pvals = _reference.pvalue_spec_reference(
            arr, min_spikes_i, min_occ_i, n_surr_i
        )

    entries: list[list] = []
    if spectrum == "#":
        for k in range(sizes.shape[0]):
            entries.append([int(sizes[k]), int(occs[k]), float(pvals[k])])
    else:
        for k in range(sizes.shape[0]):
            entries.append(
                [int(sizes[k]), int(occs[k]), int(durs[k]), float(pvals[k])]
            )
    return entries
