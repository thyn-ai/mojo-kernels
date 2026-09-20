"""ctypes loader for the tamojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$TA_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``ta_mojo/_native/``
    3. the repository development build output ``kernels/ta/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``TA_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t tamojo_abi_version(void)
    int32_t tamojo_ema(const double* x, int64_t n, int64_t length,
                       int32_t adjust, int32_t sma, double* out)
    int32_t tamojo_rma(const double* x, int64_t n, int64_t length, double* out)
    int32_t tamojo_true_range(const double* h, const double* l,
                              const double* c, int64_t n, int64_t drift,
                              double* out)
    int32_t tamojo_macd(const double* x, int64_t n, int64_t fast, int64_t slow,
                        int64_t signal, double* out_macd, double* out_signal,
                        double* out_hist)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/ta/src/tamojo.mojo. A mismatch means the
# installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "TA_MOJO_NATIVE_LIB"
_ENV_DISABLE = "TA_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native tamojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libtamojo.dylib"
    if sys.platform.startswith("linux"):
        return "libtamojo.so"
    if sys.platform.startswith("win"):
        return "tamojo.dll"  # no Mojo toolchain builds this today
    return "libtamojo.so"


def _candidate_paths() -> list[tuple[str, str]]:
    """(source_label, path) candidates, in resolver order."""
    out: list[tuple[str, str]] = []
    override = os.environ.get(_ENV_LIB)
    if override:
        out.append((f"env {_ENV_LIB}", override))
    here = os.path.dirname(__file__)
    out.append(("bundled in wheel", os.path.join(here, "_native", _lib_basename())))
    out.append(
        (
            "repo-dev build output",
            os.path.abspath(
                os.path.join(here, "..", "..", "..", "kernels", "ta", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.tamojo_abi_version.argtypes = []
    lib.tamojo_abi_version.restype = ctypes.c_int32
    lib.tamojo_ema.argtypes = [
        f64p,  # x[n]
        ctypes.c_int64,  # n
        ctypes.c_int64,  # length
        ctypes.c_int32,  # adjust
        ctypes.c_int32,  # sma
        f64p,  # out[n]
    ]
    lib.tamojo_ema.restype = ctypes.c_int32
    lib.tamojo_rma.argtypes = [f64p, ctypes.c_int64, ctypes.c_int64, f64p]
    lib.tamojo_rma.restype = ctypes.c_int32
    lib.tamojo_true_range.argtypes = [
        f64p,  # high[n]
        f64p,  # low[n]
        f64p,  # close[n]
        ctypes.c_int64,  # n
        ctypes.c_int64,  # drift
        f64p,  # out[n]
    ]
    lib.tamojo_true_range.restype = ctypes.c_int32
    lib.tamojo_macd.argtypes = [
        f64p,  # x[n]
        ctypes.c_int64,  # n
        ctypes.c_int64,  # fast
        ctypes.c_int64,  # slow
        ctypes.c_int64,  # signal
        f64p,  # out_macd[n]
        f64p,  # out_signal[n]
        f64p,  # out_hist[n]
    ]
    lib.tamojo_macd.restype = ctypes.c_int32


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None


def _load() -> ctypes.CDLL:
    """Resolve, dlopen, and ABI-handshake the native kernel. Never caches failure."""
    global _LIB, _LIB_SOURCE, _LOAD_ERROR
    if os.environ.get(_ENV_DISABLE) == "1":
        raise NativeUnavailable(f"native kernel disabled by {_ENV_DISABLE}=1")
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        errors: list[str] = []
        for label, path in _candidate_paths():
            try:
                if not path or not os.path.exists(path):
                    continue
                try:
                    lib = ctypes.CDLL(path)
                except OSError as exc:
                    errors.append(f"{label} ({path}): {exc}")
                    continue
                try:
                    _bind_abi(lib)
                    abi = int(lib.tamojo_abi_version())
                except Exception as exc:  # missing/renamed symbol = wrong lib
                    errors.append(f"{label} ({path}): ABI not recognized: {exc}")
                    continue
                if abi != ABI_VERSION:
                    errors.append(
                        f"{label} ({path}): native ABI v{abi} != wrapper ABI v{ABI_VERSION}"
                    )
                    continue
                _LIB, _LIB_SOURCE = lib, f"{label} ({path})"
                _LOAD_ERROR = None
                return lib
            except OSError as exc:
                errors.append(f"{label}: {exc}")
        _LOAD_ERROR = "; ".join(errors) or "no native kernel found on any resolver path"
        raise NativeUnavailable(_LOAD_ERROR)


def native_available() -> bool:
    """True if the native kernel can compute right now. Never raises."""
    try:
        _load()
        return True
    except NativeUnavailable:
        return False


def backend_info() -> dict:
    """Diagnostics for the active backend. Never raises."""
    info = {
        "native_available": False,
        "native_source": None,
        "abi_version_expected": ABI_VERSION,
        "abi_version_native": None,
        "disabled_by_env": os.environ.get(_ENV_DISABLE) == "1",
        "platform": sys.platform,
        "error": None,
    }
    try:
        lib = _load()
    except NativeUnavailable as exc:
        info["error"] = str(exc)
        return info
    info["native_available"] = True
    info["native_source"] = _LIB_SOURCE
    info["abi_version_native"] = int(lib.tamojo_abi_version())
    return info


def _f64p(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _check_rc(rc: int, what: str) -> None:
    if rc != 0:
        raise NativeUnavailable(f"native {what} failed with status {rc}")


def ema(x: np.ndarray, length: int, adjust: bool, sma: bool) -> np.ndarray:
    """Native EMA into a fresh float64 array. Raises NativeUnavailable."""
    lib = _load()
    out = np.empty(x.shape[0], dtype=np.float64)
    rc = lib.tamojo_ema(
        _f64p(x),
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(length),
        ctypes.c_int32(1 if adjust else 0),
        ctypes.c_int32(1 if sma else 0),
        _f64p(out),
    )
    _check_rc(rc, "ema")
    return out


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """Native Wilder RMA into a fresh float64 array. Raises NativeUnavailable."""
    lib = _load()
    out = np.empty(x.shape[0], dtype=np.float64)
    rc = lib.tamojo_rma(
        _f64p(x), ctypes.c_int64(x.shape[0]), ctypes.c_int64(length), _f64p(out)
    )
    _check_rc(rc, "rma")
    return out


def true_range(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, drift: int
) -> np.ndarray:
    """Native true range into a fresh float64 array. Raises NativeUnavailable."""
    lib = _load()
    out = np.empty(h.shape[0], dtype=np.float64)
    rc = lib.tamojo_true_range(
        _f64p(h),
        _f64p(l),
        _f64p(c),
        ctypes.c_int64(h.shape[0]),
        ctypes.c_int64(drift),
        _f64p(out),
    )
    _check_rc(rc, "true_range")
    return out


def macd(
    x: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native MACD into three fresh float64 arrays. Raises NativeUnavailable."""
    lib = _load()
    n = x.shape[0]
    out_macd = np.empty(n, dtype=np.float64)
    out_signal = np.empty(n, dtype=np.float64)
    out_hist = np.empty(n, dtype=np.float64)
    rc = lib.tamojo_macd(
        _f64p(x),
        ctypes.c_int64(n),
        ctypes.c_int64(fast),
        ctypes.c_int64(slow),
        ctypes.c_int64(signal),
        _f64p(out_macd),
        _f64p(out_signal),
        _f64p(out_hist),
    )
    _check_rc(rc, "macd")
    return out_macd, out_signal, out_hist
