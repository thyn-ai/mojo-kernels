"""ctypes loader for the rupturesmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$RUPTURES_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``ruptures_mojo/_native/``
    3. the repository development build output ``kernels/ruptures/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python implementation. Set
``RUPTURES_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t rupturesmojo_abi_version(void)
    int32_t rupturesmojo_detect(const double* signal, int64_t n,
                                int32_t method, int32_t model,
                                int64_t min_size, int64_t jump,
                                int64_t n_bkps, double pen,
                                int64_t* out_bkps, int64_t out_cap)

``rupturesmojo_detect`` returns the number of breakpoints written (>= 0) or a
negative status: -1 impossible segmentation, -2 output capacity, -3 invalid
arguments, -4 empty candidate set, -5 too large for the native path, -6
scratch allocation failure.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/ruptures/src/rupturesmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

# Detection methods, mirrored from the kernel.
METHOD_DYNP = 0
METHOD_PELT = 1
METHOD_BINSEG = 2

# Cost models, mirrored from the kernel.
MODEL_L2 = 0
MODEL_L1 = 1

# Negative status codes returned by the kernel.
STATUS_BAD_SEGMENTATION = -1
STATUS_OUTPUT_CAPACITY = -2
STATUS_INVALID_ARGUMENT = -3
STATUS_EMPTY_CANDIDATES = -4
STATUS_TOO_LARGE = -5
STATUS_ALLOC_FAILED = -6

_ENV_LIB = "RUPTURES_MOJO_NATIVE_LIB"
_ENV_DISABLE = "RUPTURES_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native rupturesmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "librupturesmojo.dylib"
    if sys.platform.startswith("linux"):
        return "librupturesmojo.so"
    if sys.platform.startswith("win"):
        return "rupturesmojo.dll"  # no Mojo toolchain builds this today
    return "librupturesmojo.so"


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
                    here, "..", "..", "..", "kernels", "ruptures", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.rupturesmojo_abi_version.argtypes = []
    lib.rupturesmojo_abi_version.restype = ctypes.c_int32
    lib.rupturesmojo_detect.argtypes = [
        f64p,  # signal[n]
        ctypes.c_int64,  # n
        ctypes.c_int32,  # method
        ctypes.c_int32,  # model
        ctypes.c_int64,  # min_size
        ctypes.c_int64,  # jump
        ctypes.c_int64,  # n_bkps
        ctypes.c_double,  # pen
        i64p,  # out_bkps[out_cap]
        ctypes.c_int64,  # out_cap
    ]
    lib.rupturesmojo_detect.restype = ctypes.c_int32


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
                    abi = int(lib.rupturesmojo_abi_version())
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
    """True if the native kernel can run right now. Never raises."""
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
    info["abi_version_native"] = int(lib.rupturesmojo_abi_version())
    return info


def native_detect(
    signal: np.ndarray,
    method: int,
    model: int,
    min_size: int,
    jump: int,
    n_bkps: int,
    pen: float,
) -> int | list[int]:
    """Run one detection on the native kernel.

    `signal` must be a contiguous 1-D float64 array. Returns the sorted list
    of breakpoints (always ending with n), or a negative status code (-1
    impossible segmentation, -4 empty candidate set, -5 too large) that the
    caller translates into the matching Python behaviour.
    """
    lib = _load()  # raises NativeUnavailable
    x = np.ascontiguousarray(signal, dtype=np.float64)
    n = x.shape[0]
    # Output capacity: breakpoints are a subset of the jump grid plus n, so
    # n//jump + 2 always suffices; PELT's count is data-dependent.
    cap = n // jump + 2
    out = np.empty(cap, dtype=np.int64)
    rc = lib.rupturesmojo_detect(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        ctypes.c_int32(method),
        ctypes.c_int32(model),
        ctypes.c_int64(min_size),
        ctypes.c_int64(jump),
        ctypes.c_int64(n_bkps),
        ctypes.c_double(pen),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(cap),
    )
    if rc < 0:
        return int(rc)
    return out[:rc].tolist()
