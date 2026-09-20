"""ctypes loader for the spatialmathmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$SPATIALMATH_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``spatialmath_mojo/_native/``
    3. the repository development build output ``kernels/spatialmath/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``SPATIALMATH_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t spatialmathmojo_abi_version(void)
    int32_t spatialmathmojo_se3_compose(const double* a, int64_t a_stride,
                                        const double* b, int64_t b_stride,
                                        int64_t n, double* out)
    int32_t spatialmathmojo_so3_compose(const double* a, int64_t a_stride,
                                        const double* b, int64_t b_stride,
                                        int64_t n, double* out)
    int32_t spatialmathmojo_se3_inverse(const double* t, int64_t n, double* out)
    int32_t spatialmathmojo_so3_inverse(const double* r, int64_t n, double* out)
    int32_t spatialmathmojo_se3_transform(const double* t, int64_t n,
                                          const double* points, int64_t m,
                                          double* out)
    int32_t spatialmathmojo_so3_transform(const double* r, int64_t n,
                                          const double* points, int64_t m,
                                          double* out)

Poses are C-order row-major float64: 16 slots per SE(3) pose, 9 per SO(3)
pose. A compose stride of 0 broadcasts that operand's singleton pose across
the batch; otherwise the stride is the per-pose element count. Points are
m C-order row-major float64 triples; transform outputs are n*m C-order
row-major triples (out[i*m + j] = pose_i applied to point_j).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/spatialmath/src/spatialmathmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "SPATIALMATH_MOJO_NATIVE_LIB"
_ENV_DISABLE = "SPATIALMATH_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native spatialmathmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libspatialmathmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libspatialmathmojo.so"
    if sys.platform.startswith("win"):
        return "spatialmathmojo.dll"  # no Mojo toolchain builds this today
    return "libspatialmathmojo.so"


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
                    here, "..", "..", "..", "kernels", "spatialmath", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.spatialmathmojo_abi_version.argtypes = []
    lib.spatialmathmojo_abi_version.restype = ctypes.c_int32
    lib.spatialmathmojo_se3_compose.argtypes = [
        f64p,  # a[n*16 or 16]
        ctypes.c_int64,  # a_stride (16 or 0)
        f64p,  # b[n*16 or 16]
        ctypes.c_int64,  # b_stride (16 or 0)
        ctypes.c_int64,  # n
        f64p,  # out[n*16]
    ]
    lib.spatialmathmojo_se3_compose.restype = ctypes.c_int32
    lib.spatialmathmojo_so3_compose.argtypes = [
        f64p,  # a[n*9 or 9]
        ctypes.c_int64,  # a_stride (9 or 0)
        f64p,  # b[n*9 or 9]
        ctypes.c_int64,  # b_stride (9 or 0)
        ctypes.c_int64,  # n
        f64p,  # out[n*9]
    ]
    lib.spatialmathmojo_so3_compose.restype = ctypes.c_int32
    lib.spatialmathmojo_se3_inverse.argtypes = [
        f64p,  # t[n*16]
        ctypes.c_int64,  # n
        f64p,  # out[n*16]
    ]
    lib.spatialmathmojo_se3_inverse.restype = ctypes.c_int32
    lib.spatialmathmojo_so3_inverse.argtypes = [
        f64p,  # r[n*9]
        ctypes.c_int64,  # n
        f64p,  # out[n*9]
    ]
    lib.spatialmathmojo_so3_inverse.restype = ctypes.c_int32
    lib.spatialmathmojo_se3_transform.argtypes = [
        f64p,  # t[n*16]
        ctypes.c_int64,  # n
        f64p,  # points[m*3]
        ctypes.c_int64,  # m
        f64p,  # out[n*m*3]
    ]
    lib.spatialmathmojo_se3_transform.restype = ctypes.c_int32
    lib.spatialmathmojo_so3_transform.argtypes = [
        f64p,  # r[n*9]
        ctypes.c_int64,  # n
        f64p,  # points[m*3]
        ctypes.c_int64,  # m
        f64p,  # out[n*m*3]
    ]
    lib.spatialmathmojo_so3_transform.restype = ctypes.c_int32


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
                    abi = int(lib.spatialmathmojo_abi_version())
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
    """True if the native kernel can evaluate right now. Never raises."""
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
    info["abi_version_native"] = int(lib.spatialmathmojo_abi_version())
    return info


def _f64p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def compose(a: np.ndarray, b: np.ndarray, n: int, so3: bool) -> np.ndarray:
    """Batched pose composition on the native kernel; raises NativeUnavailable.

    ``a``/``b`` are contiguous float64 pose batches of shape (1, d, d) or
    (n, d, d) with d = 4 (SE3) or 3 (SO3); at least one of them must have
    n poses. Returns a fresh (n, d, d) array. Input buffers are only read.
    """
    lib = _load()  # raises NativeUnavailable
    per = 9 if so3 else 16
    if a.shape[0] not in (1, n) or b.shape[0] not in (1, n):
        raise NativeUnavailable("compose operands must have 1 or n poses")
    if n == 0:
        return np.empty((0, a.shape[1], a.shape[2]), dtype=np.float64)
    out = np.empty((n, a.shape[1], a.shape[2]), dtype=np.float64)
    fn = lib.spatialmathmojo_so3_compose if so3 else lib.spatialmathmojo_se3_compose
    rc = fn(
        _f64p(a),
        ctypes.c_int64(0 if a.shape[0] == 1 else per),
        _f64p(b),
        ctypes.c_int64(0 if b.shape[0] == 1 else per),
        ctypes.c_int64(n),
        _f64p(out),
    )
    if rc != 0:
        raise NativeUnavailable(f"native compose failed with status {rc}")
    return out


def inverse(t: np.ndarray, so3: bool) -> np.ndarray:
    """Batched pose inverse on the native kernel; raises NativeUnavailable.

    ``t`` is a contiguous float64 (n, 4, 4) or (n, 3, 3) batch; returns a
    fresh array of the same shape. The input buffer is only read.
    """
    lib = _load()  # raises NativeUnavailable
    n = t.shape[0]
    if n == 0:
        return np.empty(t.shape, dtype=np.float64)
    out = np.empty(t.shape, dtype=np.float64)
    fn = lib.spatialmathmojo_so3_inverse if so3 else lib.spatialmathmojo_se3_inverse
    rc = fn(_f64p(t), ctypes.c_int64(n), _f64p(out))
    if rc != 0:
        raise NativeUnavailable(f"native inverse failed with status {rc}")
    return out


def transform(t: np.ndarray, points: np.ndarray, so3: bool) -> np.ndarray:
    """Batched point transform on the native kernel; raises NativeUnavailable.

    ``t`` is a contiguous float64 (n, 4, 4) or (n, 3, 3) batch, ``points`` a
    contiguous float64 (m, 3) array; returns a fresh (n, m, 3) array,
    out[i, j] = t[i] applied to points[j]. Input buffers are only read.
    """
    lib = _load()  # raises NativeUnavailable
    n = t.shape[0]
    m = points.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m, 3), dtype=np.float64)
    out = np.empty((n, m, 3), dtype=np.float64)
    fn = lib.spatialmathmojo_so3_transform if so3 else lib.spatialmathmojo_se3_transform
    rc = fn(_f64p(t), ctypes.c_int64(n), _f64p(points), ctypes.c_int64(m), _f64p(out))
    if rc != 0:
        raise NativeUnavailable(f"native transform failed with status {rc}")
    return out
