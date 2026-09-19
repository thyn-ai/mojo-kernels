"""ctypes loader for the metpycapemojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$METPY_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``metpy_mojo/_native/``
    3. the repository development build output ``kernels/metpy-cape/build/``

If the library cannot be found, fails to load, or reports an ABI version
this package does not understand, :class:`NativeUnavailable` is raised and
the caller falls back to the vendored pure-Python reference implementation.
Set ``METPY_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t metpycapemojo_abi_version(void)
    int32_t metpycapemojo_cape_cin(int64_t ncols, int64_t nlev,
                                   const double* p, const double* t,
                                   const double* td, int32_t which,
                                   double mu_depth_hpa,
                                   double* out_cape, double* out_cin,
                                   double* out_lclp, double* out_lfcp,
                                   double* out_elp)

    which: 0 = surface parcel, 1 = most-unstable parcel. Column c occupies
    p/t/td[c*nlev, (c+1)*nlev) with pressure strictly decreasing (hPa),
    temperatures in K. Return 0 on success; nonzero on invalid input.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/metpy-cape/src/metpycapemojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

WHICH_SURFACE = 0
WHICH_MOST_UNSTABLE = 1

_ENV_LIB = "METPY_MOJO_NATIVE_LIB"
_ENV_DISABLE = "METPY_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native metpycapemojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libmetpycapemojo.dylib"
    if sys.platform.startswith("linux"):
        return "libmetpycapemojo.so"
    if sys.platform.startswith("win"):
        return "metpycapemojo.dll"  # no Mojo toolchain builds this today
    return "libmetpycapemojo.so"


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
                    here, "..", "..", "..", "kernels", "metpy-cape", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.metpycapemojo_abi_version.argtypes = []
    lib.metpycapemojo_abi_version.restype = ctypes.c_int32
    lib.metpycapemojo_cape_cin.argtypes = [
        ctypes.c_int64,  # ncols
        ctypes.c_int64,  # nlev
        f64p,  # p[ncols*nlev]
        f64p,  # t[ncols*nlev]
        f64p,  # td[ncols*nlev]
        ctypes.c_int32,  # which
        ctypes.c_double,  # mu_depth_hpa
        f64p,  # out_cape[ncols]
        f64p,  # out_cin[ncols]
        f64p,  # out_lclp[ncols]
        f64p,  # out_lfcp[ncols]
        f64p,  # out_elp[ncols]
    ]
    lib.metpycapemojo_cape_cin.restype = ctypes.c_int32


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
                    abi = int(lib.metpycapemojo_abi_version())
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
    info["abi_version_native"] = int(lib.metpycapemojo_abi_version())
    return info


def cape_cin_batched(
    p: np.ndarray,
    t: np.ndarray,
    td: np.ndarray,
    which: int,
    mu_depth_hpa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the native kernel on a validated (ncols, nlev) column batch.

    Inputs must be C-contiguous float64 and already validated (strictly
    decreasing pressure, finite, es < p). Raises NativeUnavailable if the
    kernel cannot be loaded or reports an error.
    """
    lib = _load()  # raises NativeUnavailable
    f64p = ctypes.POINTER(ctypes.c_double)
    p = np.ascontiguousarray(p, dtype=np.float64)
    t = np.ascontiguousarray(t, dtype=np.float64)
    td = np.ascontiguousarray(td, dtype=np.float64)
    ncols, nlev = p.shape
    cape = np.empty(ncols, dtype=np.float64)
    cin = np.empty(ncols, dtype=np.float64)
    lclp = np.empty(ncols, dtype=np.float64)
    lfcp = np.empty(ncols, dtype=np.float64)
    elp = np.empty(ncols, dtype=np.float64)
    rc = lib.metpycapemojo_cape_cin(
        np.int64(ncols),
        np.int64(nlev),
        p.ctypes.data_as(f64p),
        t.ctypes.data_as(f64p),
        td.ctypes.data_as(f64p),
        ctypes.c_int32(which),
        ctypes.c_double(mu_depth_hpa),
        cape.ctypes.data_as(f64p),
        cin.ctypes.data_as(f64p),
        lclp.ctypes.data_as(f64p),
        lfcp.ctypes.data_as(f64p),
        elp.ctypes.data_as(f64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native kernel rejected the batch (status {rc})")
    return cape, cin, lclp, lfcp, elp
