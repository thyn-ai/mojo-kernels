"""ctypes loader for the nuscenesevalmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$NUSCENES_EVAL_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``nuscenes_eval_mojo/_native/``
    3. the repository development build output ``kernels/nuscenes-eval/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``NUSCENES_EVAL_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  nuscenesevalmojo_abi_version(void)
    int32_t  nuscenesevalmojo_match(int64_t n_classes, int64_t n_samples,
                                    int64_t n_gt_total, int64_t n_pred_total,
                                    const int64_t* gt_class_offsets,
                                    const int64_t* gt_sample_offsets,
                                    const double* gt_vals,
                                    const int32_t* gt_attr,
                                    const int64_t* pred_class_offsets,
                                    const int32_t* pred_sample,
                                    const double* pred_vals,
                                    const int32_t* pred_attr,
                                    const double* periods,
                                    const double* dist_ths, int64_t n_ths,
                                    double* out)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/nuscenes-eval/src/nuscenesevalmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library disagree;
# fall back.
ABI_VERSION = 1

_ENV_LIB = "NUSCENES_EVAL_MOJO_NATIVE_LIB"
_ENV_DISABLE = "NUSCENES_EVAL_MOJO_DISABLE_NATIVE"

# Float64 fields per box record and per (threshold, prediction) output row;
# mirrored from the kernel.
BOX_STRIDE = 8
OUT_STRIDE = 6


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native nuscenesevalmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libnuscenesevalmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libnuscenesevalmojo.so"
    if sys.platform.startswith("win"):
        return "nuscenesevalmojo.dll"  # no Mojo toolchain builds this today
    return "libnuscenesevalmojo.so"


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
                    here, "..", "..", "..", "kernels", "nuscenes-eval", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.nuscenesevalmojo_abi_version.argtypes = []
    lib.nuscenesevalmojo_abi_version.restype = ctypes.c_int32
    lib.nuscenesevalmojo_match.argtypes = [
        ctypes.c_int64,  # n_classes
        ctypes.c_int64,  # n_samples
        ctypes.c_int64,  # n_gt_total
        ctypes.c_int64,  # n_pred_total
        i64p,  # gt_class_offsets[n_classes+1]
        i64p,  # gt_sample_offsets[n_classes*(n_samples+1)]
        f64p,  # gt_vals[n_gt_total*8]
        i32p,  # gt_attr[n_gt_total]
        i64p,  # pred_class_offsets[n_classes+1]
        i32p,  # pred_sample[n_pred_total]
        f64p,  # pred_vals[n_pred_total*8]
        i32p,  # pred_attr[n_pred_total]
        f64p,  # periods[n_classes]
        f64p,  # dist_ths[n_ths]
        ctypes.c_int64,  # n_ths
        f64p,  # out[n_ths*n_pred_total*6]
    ]
    lib.nuscenesevalmojo_match.restype = ctypes.c_int32


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
                    abi = int(lib.nuscenesevalmojo_abi_version())
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
    """True if the native kernel can match right now. Never raises."""
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
    info["abi_version_native"] = int(lib.nuscenesevalmojo_abi_version())
    return info


def native_match(
    *,
    n_classes: int,
    n_samples: int,
    gt_class_offsets: np.ndarray,
    gt_sample_offsets: np.ndarray,
    gt_vals: np.ndarray,
    gt_attr: np.ndarray,
    pred_class_offsets: np.ndarray,
    pred_sample: np.ndarray,
    pred_vals: np.ndarray,
    pred_attr: np.ndarray,
    periods: np.ndarray,
    dist_ths: np.ndarray,
) -> np.ndarray:
    """Run every (class, threshold) greedy-matching pass on the native kernel.

    Returns a float64 array of shape (n_ths, n_pred_total, 6): tp flag and
    trans/vel/scale/orient/attr errors per prediction (NaN when unmatched).
    Raises NativeUnavailable when the kernel cannot be used.
    """
    lib = _load()  # raises NativeUnavailable
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)

    gt_class_offsets = np.ascontiguousarray(gt_class_offsets, dtype=np.int64)
    gt_sample_offsets = np.ascontiguousarray(gt_sample_offsets, dtype=np.int64)
    gt_vals = np.ascontiguousarray(gt_vals, dtype=np.float64)
    gt_attr = np.ascontiguousarray(gt_attr, dtype=np.int32)
    pred_class_offsets = np.ascontiguousarray(pred_class_offsets, dtype=np.int64)
    pred_sample = np.ascontiguousarray(pred_sample, dtype=np.int32)
    pred_vals = np.ascontiguousarray(pred_vals, dtype=np.float64)
    pred_attr = np.ascontiguousarray(pred_attr, dtype=np.int32)
    periods = np.ascontiguousarray(periods, dtype=np.float64)
    dist_ths = np.ascontiguousarray(dist_ths, dtype=np.float64)

    n_gt_total = np.int64(gt_vals.shape[0] // BOX_STRIDE)
    n_pred_total = np.int64(pred_vals.shape[0] // BOX_STRIDE)
    n_ths = np.int64(dist_ths.shape[0])
    out = np.empty(int(n_ths) * int(n_pred_total) * OUT_STRIDE, dtype=np.float64)

    rc = lib.nuscenesevalmojo_match(
        ctypes.c_int64(n_classes),
        ctypes.c_int64(n_samples),
        n_gt_total,
        n_pred_total,
        gt_class_offsets.ctypes.data_as(i64p),
        gt_sample_offsets.ctypes.data_as(i64p),
        gt_vals.ctypes.data_as(f64p),
        gt_attr.ctypes.data_as(i32p),
        pred_class_offsets.ctypes.data_as(i64p),
        pred_sample.ctypes.data_as(i32p),
        pred_vals.ctypes.data_as(f64p),
        pred_attr.ctypes.data_as(i32p),
        periods.ctypes.data_as(f64p),
        dist_ths.ctypes.data_as(f64p),
        n_ths,
        out.ctypes.data_as(f64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native matching failed with status {rc}")
    return out.reshape(int(n_ths), int(n_pred_total), OUT_STRIDE)
