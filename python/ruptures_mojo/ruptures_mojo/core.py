"""Change-point detection API-compatible with the `ruptures` package.

`detect(signal, method, ...)` runs Dynp (exact dynamic programming), Pelt
(penalized), or Binseg (binary segmentation) on a 1-D signal with the `l2`
or `l1` cost model, and returns the same sorted breakpoint list (integer
indices, always ending with n_samples) as the matching `ruptures` estimator:

    ruptures.Dynp(model=model, min_size=min_size, jump=jump).fit_predict(signal, n_bkps)
    ruptures.Pelt(model=model, min_size=min_size, jump=jump).fit_predict(signal, pen)
    ruptures.Binseg(model=model, min_size=min_size, jump=jump).fit_predict(signal, n_bkps)

Detection runs on the native Mojo kernel when its shared library is available
(macOS arm64 / Linux x86_64 wheels) and transparently falls back to the
vendored pure-Python implementation otherwise. Both backends enumerate
candidates, order float64 operations, and break ties identically, so the
returned breakpoint sets are integer equal on either path; the differential
test suite asserts exactly that against `ruptures` itself.
"""

from __future__ import annotations

import numbers

from ruptures_mojo import _native, _reference
from ruptures_mojo._native import NativeUnavailable
from ruptures_mojo._reference import BadSegmentationParameters

__all__ = [
    "BadSegmentationParameters",
    "detect",
    "last_backend",
]

_METHODS = {"dynp": _native.METHOD_DYNP, "pelt": _native.METHOD_PELT, "binseg": _native.METHOD_BINSEG}
_MODELS = {"l2": _native.MODEL_L2, "l1": _native.MODEL_L1}

# Diagnostic: which backend served the most recent detect() call.
_backend_last = "fallback"


def last_backend() -> str:
    """Which backend served the most recent detect(): "native" or "fallback"."""
    return _backend_last


def _set_backend(name: str) -> str:
    global _backend_last
    _backend_last = name
    return name


def _normalize_method(method) -> int:
    if isinstance(method, str):
        key = method.strip().lower()
        if key in _METHODS:
            return _METHODS[key]
    raise ValueError(
        f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
    )


def _normalize_model(model) -> int:
    if isinstance(model, str):
        key = model.strip().lower()
        if key in _MODELS:
            return _MODELS[key]
    raise ValueError(
        f"unknown cost model {model!r}; supported models: {sorted(_MODELS)}"
    )


def _normalize_n_bkps(n_bkps, method: str) -> int:
    if n_bkps is None:
        raise TypeError(f"detect(method={method!r}) requires n_bkps")
    if isinstance(n_bkps, numbers.Integral):
        k = int(n_bkps)
    else:
        raise ValueError(f"n_bkps must be a non-negative int, got {n_bkps!r}")
    if k < 0:
        raise ValueError(f"n_bkps must be a non-negative int, got {n_bkps!r}")
    return k


def _validate_common(signal, model, min_size, jump):
    sig = _reference._check_signal(signal)
    if not isinstance(min_size, numbers.Integral):
        raise ValueError(f"min_size must be an int, got {min_size!r}")
    min_size = int(min_size)
    if not isinstance(jump, numbers.Integral):
        raise ValueError(f"jump must be an int, got {jump!r}")
    jump = int(jump)
    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")
    eff_min_size = _reference.effective_min_size(model, min_size)
    return sig, min_size, jump, eff_min_size


def _run_native(sig, method_id, model_id, min_size, jump, n_bkps, pen):
    """Native dispatch; returns a breakpoint list, "fallback", or raises."""
    result = _native.native_detect(
        sig, method_id, model_id, min_size, jump, n_bkps, pen
    )
    if isinstance(result, list):
        return result
    if result == _native.STATUS_BAD_SEGMENTATION:
        raise BadSegmentationParameters
    if result == _native.STATUS_EMPTY_CANDIDATES:
        # Matches Python's ValueError from min() over an empty candidate set
        # (reachable with pen < 0, as in ruptures).
        raise ValueError("min() iterable argument is empty")
    # Too large for the native path, allocation failure, or an internal
    # mismatch: the vendored implementation is always correct; use it.
    return "fallback"


def detect(
    signal,
    method,
    *,
    model="l2",
    min_size=2,
    jump=5,
    n_bkps=None,
    pen=None,
) -> list[int]:
    """Detect change points in a 1-D signal; return sorted breakpoint indices.

    Args:
        signal: array-like of shape (n,) or (n, 1), finite values.
        method: "dynp" (exact DP, needs n_bkps), "pelt" (needs pen), or
            "binseg" (needs n_bkps).
        model: cost model, "l2" (least squared deviation) or "l1" (least
            absolute deviation).
        min_size: minimum segment length in samples (as in ruptures, the
            effective value is max(min_size, 1) for l2 and max(min_size, 2)
            for l1).
        jump: candidate breakpoints are multiples of jump.
        n_bkps: number of breakpoints (dynp, binseg).
        pen: penalty value (pelt).

    Returns:
        list[int]: sorted breakpoint indices, always ending with n.

    Raises:
        BadSegmentationParameters: no partition exists for the parameters.
        ValueError: invalid parameters (same cases as ruptures).
    """
    global _backend_last
    method_id = _normalize_method(method)
    model_id = _normalize_model(model)
    model_name = "l1" if model_id == _native.MODEL_L1 else "l2"

    sig, min_size, jump, eff_min_size = _validate_common(signal, model_name, min_size, jump)
    n_samples = sig.shape[0]

    if method_id == _native.METHOD_PELT:
        if pen is None:
            raise TypeError("detect(method='pelt') requires pen")
        pen_value = float(pen)
        k = 0
    else:
        k = _normalize_n_bkps(n_bkps, method.strip().lower())
        pen_value = 0.0

    if not _reference.sanity_check(n_samples, k, jump, eff_min_size):
        raise BadSegmentationParameters

    try:
        result = _run_native(
            sig, method_id, model_id, min_size, jump, k, pen_value
        )
    except NativeUnavailable:
        result = "fallback"

    if result != "fallback":
        _set_backend("native")
        return result

    _set_backend("fallback")
    if method_id == _native.METHOD_DYNP:
        return _reference.dynp_detect(
            sig, k, min_size=min_size, jump=jump, model=model_name
        )
    if method_id == _native.METHOD_PELT:
        return _reference.pelt_detect(
            sig, pen_value, min_size=min_size, jump=jump, model=model_name
        )
    return _reference.binseg_detect(
        sig, k, min_size=min_size, jump=jump, model=model_name
    )
