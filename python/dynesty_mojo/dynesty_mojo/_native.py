"""ctypes loader for the dynestymojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$DYNESTY_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``dynesty_mojo/_native/``
    3. the repository development build output ``kernels/dynesty/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored NumPy reference implementation.
Set ``DYNESTY_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  dynestymojo_abi_version(void)
    void*    dynestymojo_rng_create(uint64_t seed)
    void     dynestymojo_rng_destroy(void* rng)
    int32_t  dynestymojo_fit_ellipsoid(int64_t ndim, int64_t n,
                                       const double* U, double enlarge,
                                       double* out_mean, double* out_chol,
                                       double* out_logvol)
    int64_t  dynestymojo_propose_batch(void* rng, int64_t ndim,
                                       const double* mean,
                                       const double* chol, int32_t mode,
                                       int64_t max_out, double* out)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import NamedTuple

import numpy as np

# Must equal ABI_VERSION in kernels/dynesty/src/dynestymojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back.
ABI_VERSION = 1

MODE_ELLIPSOID = 0
MODE_CUBE = 1

_ENV_LIB = "DYNESTY_MOJO_NATIVE_LIB"
_ENV_DISABLE = "DYNESTY_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native dynestymojo kernel could not be found, loaded, or verified."""


class EllipsoidFit(NamedTuple):
    """One bounding ellipsoid: {x : (x-mean)^T (chol chol^T)^-1 (x-mean) <= 1}."""

    mean: np.ndarray  # (ndim,)
    chol: np.ndarray  # (ndim, ndim) lower-triangular
    logvol: float


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libdynestymojo.dylib"
    if sys.platform.startswith("linux"):
        return "libdynestymojo.so"
    if sys.platform.startswith("win"):
        return "dynestymojo.dll"  # no Mojo toolchain builds this today
    return "libdynestymojo.so"


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
                    here, "..", "..", "..", "kernels", "dynesty", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.dynestymojo_abi_version.argtypes = []
    lib.dynestymojo_abi_version.restype = ctypes.c_int32
    lib.dynestymojo_rng_create.argtypes = [ctypes.c_uint64]
    lib.dynestymojo_rng_create.restype = ctypes.c_void_p
    lib.dynestymojo_rng_destroy.argtypes = [ctypes.c_void_p]
    lib.dynestymojo_rng_destroy.restype = None
    lib.dynestymojo_fit_ellipsoid.argtypes = [
        ctypes.c_int64,  # ndim
        ctypes.c_int64,  # n
        f64p,  # U[n*ndim] row-major
        ctypes.c_double,  # enlarge
        f64p,  # out_mean[ndim]
        f64p,  # out_chol[ndim*ndim]
        f64p,  # out_logvol[1]
    ]
    lib.dynestymojo_fit_ellipsoid.restype = ctypes.c_int32
    lib.dynestymojo_propose_batch.argtypes = [
        ctypes.c_void_p,  # rng handle
        ctypes.c_int64,  # ndim
        f64p,  # mean[ndim]
        f64p,  # chol[ndim*ndim]
        ctypes.c_int32,  # mode
        ctypes.c_int64,  # max_out
        f64p,  # out[max_out*ndim]
    ]
    lib.dynestymojo_propose_batch.restype = ctypes.c_int64


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
                    abi = int(lib.dynestymojo_abi_version())
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
    """True if the native kernel can propose right now. Never raises."""
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
    info["abi_version_native"] = int(lib.dynestymojo_abi_version())
    return info


def fit_ellipsoid(u_points: np.ndarray, enlarge: float) -> EllipsoidFit | None:
    """Fit the enlarged bounding ellipsoid of the live points (native kernel).

    Returns None when the kernel reports a numerically singular covariance
    (the caller then samples from the whole unit cube for that iteration).
    """
    lib = _load()  # raises NativeUnavailable
    u = np.ascontiguousarray(u_points, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"u_points must be 2-D (n, ndim); got shape {u.shape}")
    n, ndim = u.shape
    f64p = ctypes.POINTER(ctypes.c_double)
    mean = np.empty(ndim, dtype=np.float64)
    chol = np.empty(ndim * ndim, dtype=np.float64)
    logvol = np.empty(1, dtype=np.float64)
    rc = lib.dynestymojo_fit_ellipsoid(
        ctypes.c_int64(ndim),
        ctypes.c_int64(n),
        u.ctypes.data_as(f64p),
        ctypes.c_double(enlarge),
        mean.ctypes.data_as(f64p),
        chol.ctypes.data_as(f64p),
        logvol.ctypes.data_as(f64p),
    )
    if rc == 2:
        return None
    if rc != 0:
        raise NativeUnavailable(f"native ellipsoid fit failed with status {rc}")
    return EllipsoidFit(mean=mean, chol=chol.reshape(ndim, ndim), logvol=float(logvol[0]))


class NativeRng:
    """Owned handle to a native seeded RNG state. Not thread-safe to close twice."""

    def __init__(self, seed: int) -> None:
        lib = _load()  # raises NativeUnavailable
        handle = lib.dynestymojo_rng_create(ctypes.c_uint64(seed))
        if not handle:
            raise NativeUnavailable("native kernel rejected the RNG seed")
        self._lib = lib
        self._handle = handle

    def propose_batch(
        self,
        ndim: int,
        fit: EllipsoidFit | None,
        mode: int,
        max_out: int,
    ) -> np.ndarray:
        """Up to max_out candidate unit-cube points, shape (k, ndim), 0 <= k."""
        if self._handle is None:
            raise NativeUnavailable("native RNG is closed")
        if mode == MODE_CUBE or fit is None:
            mean = np.zeros(ndim, dtype=np.float64)
            chol = np.zeros(ndim * ndim, dtype=np.float64)
            mode = MODE_CUBE
        else:
            mean = np.ascontiguousarray(fit.mean, dtype=np.float64)
            chol = np.ascontiguousarray(fit.chol, dtype=np.float64)
        out = np.empty(max_out * ndim, dtype=np.float64)
        f64p = ctypes.POINTER(ctypes.c_double)
        k = self._lib.dynestymojo_propose_batch(
            ctypes.c_void_p(self._handle),
            ctypes.c_int64(ndim),
            mean.ctypes.data_as(f64p),
            chol.ctypes.data_as(f64p),
            ctypes.c_int32(mode),
            ctypes.c_int64(max_out),
            out.ctypes.data_as(f64p),
        )
        return out[: k * ndim].reshape(int(k), ndim).copy()

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.dynestymojo_rng_destroy(ctypes.c_void_p(handle))

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass


def create_rng(seed: int) -> NativeRng:
    """Create a native RNG handle. Raises NativeUnavailable if unloaded."""
    return NativeRng(seed)
