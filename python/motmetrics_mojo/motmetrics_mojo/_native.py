"""ctypes loader for the motmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$MOTMETRICS_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``motmetrics_mojo/_native/``
    3. the repository development build output ``kernels/motmetrics/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``MOTMETRICS_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  motmojo_abi_version(void)
    void*    motmojo_create(double max_switch_time)
    int32_t  motmojo_update(void* handle, const int64_t* oids, int64_t n_oids,
                            const int64_t* hids, int64_t n_hids,
                            const double* dists, int64_t frameid)
    int32_t  motmojo_finalize(void* handle, double* out, int64_t capacity)
    void     motmojo_destroy(void* handle)

`dists` is the row-major n_oids x n_hids distance matrix (NaN/+/-inf entries
signal do-not-pair constellations). The kernel keeps all state; `finalize`
writes the COUNTS_LEN float64 counters documented in `core.py`.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/motmetrics/src/motmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

# Number of float64 counters written by motmojo_finalize.
COUNTS_LEN = 21

_ENV_LIB = "MOTMETRICS_MOJO_NATIVE_LIB"
_ENV_DISABLE = "MOTMETRICS_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native motmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libmotmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libmotmojo.so"
    if sys.platform.startswith("win"):
        return "motmojo.dll"  # no Mojo toolchain builds this today
    return "libmotmojo.so"


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
                    here, "..", "..", "..", "kernels", "motmetrics", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.motmojo_abi_version.argtypes = []
    lib.motmojo_abi_version.restype = ctypes.c_int32
    lib.motmojo_create.argtypes = [ctypes.c_double]
    lib.motmojo_create.restype = ctypes.c_void_p
    lib.motmojo_update.argtypes = [
        ctypes.c_void_p,
        i64p,
        ctypes.c_int64,
        i64p,
        ctypes.c_int64,
        f64p,
        ctypes.c_int64,
    ]
    lib.motmojo_update.restype = ctypes.c_int32
    lib.motmojo_finalize.argtypes = [ctypes.c_void_p, f64p, ctypes.c_int64]
    lib.motmojo_finalize.restype = ctypes.c_int32
    lib.motmojo_destroy.argtypes = [ctypes.c_void_p]
    lib.motmojo_destroy.restype = None


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
                    abi = int(lib.motmojo_abi_version())
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
    """True if the native kernel can accumulate right now. Never raises."""
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
    info["abi_version_native"] = int(lib.motmojo_abi_version())
    return info


# Counter slots written by motmojo_finalize (keep in sync with the kernel and
# with the keys `core._metrics_from_counts` consumes).
_FINALIZE_KEYS = (
    "num_frames",
    "num_objects",
    "num_predictions",
    "num_matches",
    "num_switches",
    "num_misses",
    "num_false_positives",
    "num_transfer",
    "num_ascend",
    "num_migrate",
    "dist_sum",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_fragmentations",
    "idtp",
    "idfp",
    "idfn",
    "ocs_sum",
    "hcs_sum",
)


class NativeAccumulator:
    """Owned handle to a native accumulator. Not thread-safe to close twice."""

    def __init__(self, max_switch_time: float) -> None:
        lib = _load()  # raises NativeUnavailable
        handle = lib.motmojo_create(ctypes.c_double(max_switch_time))
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the configuration "
                f"(max_switch_time={max_switch_time!r}); "
                "falling back to the pure-Python reference"
            )
        self._lib = lib
        self._handle = handle

    def update(
        self,
        oids: np.ndarray,
        hids: np.ndarray,
        dists: np.ndarray,
        frameid: int,
    ) -> None:
        """Accumulate one frame of int64 ids + a C-contiguous float64 matrix."""
        if self._handle is None:
            raise NativeUnavailable("native accumulator is closed")
        i64p = ctypes.POINTER(ctypes.c_int64)
        f64p = ctypes.POINTER(ctypes.c_double)
        rc = self._lib.motmojo_update(
            self._handle,
            oids.ctypes.data_as(i64p),
            ctypes.c_int64(oids.shape[0]),
            hids.ctypes.data_as(i64p),
            ctypes.c_int64(hids.shape[0]),
            dists.ctypes.data_as(f64p),
            ctypes.c_int64(frameid),
        )
        if rc != 0:
            raise NativeUnavailable(f"native update failed with status {rc}")

    def counts(self) -> dict[str, float]:
        """Read the accumulated counters out of the kernel."""
        if self._handle is None:
            raise NativeUnavailable("native accumulator is closed")
        out = np.empty(COUNTS_LEN, dtype=np.float64)
        rc = self._lib.motmojo_finalize(
            self._handle,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(COUNTS_LEN),
        )
        if rc != 0:
            raise NativeUnavailable(f"native finalize failed with status {rc}")
        result = {}
        for name, value in zip(_FINALIZE_KEYS, out):
            # Counts are exact integers below 2^53; only dist_sum is fractional.
            result[name] = float(value) if name == "dist_sum" else int(value)
        return result

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.motmojo_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
