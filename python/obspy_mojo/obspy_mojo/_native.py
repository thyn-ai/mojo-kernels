"""ctypes loader for the konnomojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$OBSPY_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``obspy_mojo/_native/``
    3. the repository development build output ``kernels/obspy-konno/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``OBSPY_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t konnomojo_abi_version(void)
    int32_t konnomojo_smooth_loop_f64(const double* spectra,
                                      const double* freqs,
                                      int64_t n_spectra, int64_t n_freqs,
                                      double bandwidth, int64_t count,
                                      int32_t normalize, double* out)
    int32_t konnomojo_smooth_loop_f32(const float* spectra,
                                      const float* freqs,
                                      int64_t n_spectra, int64_t n_freqs,
                                      float bandwidth, int64_t count,
                                      int32_t normalize, float* out)
    int32_t konnomojo_window_matrix_f64(const double* freqs, int64_t n_freqs,
                                        double bandwidth, double* out_w)
    int32_t konnomojo_window_matrix_f32(const float* freqs, int64_t n_freqs,
                                        float bandwidth, float* out_w)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/obspy-konno/src/konnomojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back.
ABI_VERSION = 1

_ENV_LIB = "OBSPY_MOJO_NATIVE_LIB"
_ENV_DISABLE = "OBSPY_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native konnomojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libkonnomojo.dylib"
    if sys.platform.startswith("linux"):
        return "libkonnomojo.so"
    if sys.platform.startswith("win"):
        return "konnomojo.dll"  # no Mojo toolchain builds this today
    return "libkonnomojo.so"


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
                os.path.join(
                    here, "..", "..", "..", "kernels", "obspy-konno", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.konnomojo_abi_version.argtypes = []
    lib.konnomojo_abi_version.restype = ctypes.c_int32
    lib.konnomojo_smooth_loop_f64.argtypes = [
        f64p,  # spectra[n_spectra * n_freqs]
        f64p,  # freqs[n_freqs]
        ctypes.c_int64,  # n_spectra
        ctypes.c_int64,  # n_freqs
        ctypes.c_double,  # bandwidth
        ctypes.c_int64,  # count
        ctypes.c_int32,  # normalize
        f64p,  # out[n_spectra * n_freqs]
    ]
    lib.konnomojo_smooth_loop_f64.restype = ctypes.c_int32
    lib.konnomojo_smooth_loop_f32.argtypes = [
        f32p,
        f32p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,  # bandwidth
        ctypes.c_int64,
        ctypes.c_int32,
        f32p,
    ]
    lib.konnomojo_smooth_loop_f32.restype = ctypes.c_int32
    lib.konnomojo_window_matrix_f64.argtypes = [
        f64p,
        ctypes.c_int64,
        ctypes.c_double,
        f64p,
    ]
    lib.konnomojo_window_matrix_f64.restype = ctypes.c_int32
    lib.konnomojo_window_matrix_f32.argtypes = [
        f32p,
        ctypes.c_int64,
        ctypes.c_float,
        f32p,
    ]
    lib.konnomojo_window_matrix_f32.restype = ctypes.c_int32


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
                    abi = int(lib.konnomojo_abi_version())
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
    """True if the native kernel can smooth right now. Never raises."""
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
    info["abi_version_native"] = int(lib.konnomojo_abi_version())
    return info


def smooth_loop(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    bandwidth: float,
    count: int,
    normalize: bool,
) -> np.ndarray:
    """Fused window+reduction loop path on the native kernel.

    `spectra` must be C-contiguous (n_spectra, n_freqs) float32/float64 and
    `frequencies` C-contiguous (n_freqs,) of the same dtype; the caller owns
    validation. Returns a freshly allocated array of the same shape/dtype.
    """
    lib = _load()  # raises NativeUnavailable
    n_spectra, n_freqs = spectra.shape
    out = np.empty_like(spectra)
    if spectra.dtype == np.float64:
        ptr = ctypes.POINTER(ctypes.c_double)
        rc = lib.konnomojo_smooth_loop_f64(
            spectra.ctypes.data_as(ptr),
            frequencies.ctypes.data_as(ptr),
            ctypes.c_int64(n_spectra),
            ctypes.c_int64(n_freqs),
            ctypes.c_double(bandwidth),
            ctypes.c_int64(count),
            ctypes.c_int32(1 if normalize else 0),
            out.ctypes.data_as(ptr),
        )
    else:
        ptr = ctypes.POINTER(ctypes.c_float)
        rc = lib.konnomojo_smooth_loop_f32(
            spectra.ctypes.data_as(ptr),
            frequencies.ctypes.data_as(ptr),
            ctypes.c_int64(n_spectra),
            ctypes.c_int64(n_freqs),
            ctypes.c_float(bandwidth),
            ctypes.c_int64(count),
            ctypes.c_int32(1 if normalize else 0),
            out.ctypes.data_as(ptr),
        )
    if rc != 0:
        raise NativeUnavailable(f"native loop-path smoothing failed with status {rc}")
    return out


def window_matrix(frequencies: np.ndarray, bandwidth: float) -> np.ndarray:
    """Raw n x n Konno-Ohmachi window matrix from the native kernel.

    Row normalization, matrix powers, and the matmul reduction stay with the
    caller (NumPy/BLAS is already native-speed for those).
    """
    lib = _load()  # raises NativeUnavailable
    n = frequencies.shape[0]
    out_w = np.empty((n, n), dtype=frequencies.dtype)
    if frequencies.dtype == np.float64:
        ptr = ctypes.POINTER(ctypes.c_double)
        rc = lib.konnomojo_window_matrix_f64(
            frequencies.ctypes.data_as(ptr),
            ctypes.c_int64(n),
            ctypes.c_double(bandwidth),
            out_w.ctypes.data_as(ptr),
        )
    else:
        ptr = ctypes.POINTER(ctypes.c_float)
        rc = lib.konnomojo_window_matrix_f32(
            frequencies.ctypes.data_as(ptr),
            ctypes.c_int64(n),
            ctypes.c_float(bandwidth),
            out_w.ctypes.data_as(ptr),
        )
    if rc != 0:
        raise NativeUnavailable(f"native window-matrix build failed with status {rc}")
    return out_w
