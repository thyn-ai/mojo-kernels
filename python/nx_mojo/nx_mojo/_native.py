"""ctypes loader for the nxgraphmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$NX_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``nx_mojo/_native/``
    3. the repository development build output ``kernels/networkx-graph/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``NX_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  nxgraphmojo_abi_version(void)
    void*    nxgraphmojo_graph_create(int64_t n, int32_t directed,
                                      const int64_t* row_ptr,
                                      const int32_t* col,
                                      const double* weight,
                                      int32_t has_weight)
    void     nxgraphmojo_graph_destroy(void* handle)
    int32_t  nxgraphmojo_betweenness(void* handle, int32_t use_weights,
                                     double* out_bc)
    int64_t  nxgraphmojo_dijkstra(void* handle, int64_t source,
                                  int32_t has_target, int64_t target,
                                  int32_t has_cutoff, double cutoff,
                                  double* out_dist, int64_t* out_parent)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/networkx-graph/src/nxgraphmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

# betweenness() status codes, mirrored from the kernel.
_BC_OK = 0
_BC_NO_WEIGHTS = 2

# dijkstra() negative status codes, mirrored from the kernel.
DIJKSTRA_BAD_NODE = -2
DIJKSTRA_CONTRADICTORY_PATHS = -3
DIJKSTRA_NO_WEIGHTS = -4

_ENV_LIB = "NX_MOJO_NATIVE_LIB"
_ENV_DISABLE = "NX_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native nxgraphmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libnxgraphmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libnxgraphmojo.so"
    if sys.platform.startswith("win"):
        return "nxgraphmojo.dll"  # no Mojo toolchain builds this today
    return "libnxgraphmojo.so"


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
                    here, "..", "..", "..",
                    "kernels", "networkx-graph", "build", _lib_basename(),
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.nxgraphmojo_abi_version.argtypes = []
    lib.nxgraphmojo_abi_version.restype = ctypes.c_int32
    lib.nxgraphmojo_graph_create.argtypes = [
        ctypes.c_int64,  # n
        ctypes.c_int32,  # directed
        i64p,  # row_ptr[n+1]
        i32p,  # col[nnz]
        f64p,  # weight[nnz] (>=1 slot even when unused)
        ctypes.c_int32,  # has_weight
    ]
    lib.nxgraphmojo_graph_create.restype = ctypes.c_void_p
    lib.nxgraphmojo_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.nxgraphmojo_graph_destroy.restype = None
    lib.nxgraphmojo_betweenness.argtypes = [ctypes.c_void_p, ctypes.c_int32, f64p]
    lib.nxgraphmojo_betweenness.restype = ctypes.c_int32
    lib.nxgraphmojo_dijkstra.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,  # source
        ctypes.c_int32,  # has_target
        ctypes.c_int64,  # target
        ctypes.c_int32,  # has_cutoff
        ctypes.c_double,  # cutoff
        f64p,  # out_dist[n]
        i64p,  # out_parent[n]
    ]
    lib.nxgraphmojo_dijkstra.restype = ctypes.c_int64


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
                    abi = int(lib.nxgraphmojo_abi_version())
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
    info["abi_version_native"] = int(lib.nxgraphmojo_abi_version())
    return info


def _as_csr_arrays(row_ptr, col, weight, weighted: bool):
    """Contiguous, correctly-typed copies of the CSR buffers.

    When `weighted` is False the kernel is handed a 1-slot dummy weight
    buffer that it never reads.
    """
    row_ptr = np.ascontiguousarray(row_ptr, dtype=np.int64)
    col = np.ascontiguousarray(col, dtype=np.int32)
    if weighted:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        if weight.shape[0] != col.shape[0]:
            raise ValueError("weight buffer must have one value per CSR arc")
    else:
        weight = np.zeros(1, dtype=np.float64)
    return row_ptr, col, weight


class NativeGraph:
    """Owned handle to a native CSR graph. Not thread-safe to close twice."""

    def __init__(self, n: int, directed: bool, row_ptr, col, weight, weighted: bool) -> None:
        lib = _load()  # raises NativeUnavailable
        f64p = ctypes.POINTER(ctypes.c_double)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        row_ptr, col, weight = _as_csr_arrays(row_ptr, col, weight, weighted)
        handle = lib.nxgraphmojo_graph_create(
            ctypes.c_int64(n),
            ctypes.c_int32(1 if directed else 0),
            row_ptr.ctypes.data_as(i64p),
            col.ctypes.data_as(i32p),
            weight.ctypes.data_as(f64p),
            ctypes.c_int32(1 if weighted else 0),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the graph (invalid CSR buffers); "
                "falling back to the pure-Python reference"
            )
        # The kernel copies every buffer; these arrays may be freed by the GC.
        self._lib = lib
        self._handle = handle
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    def betweenness(self, weighted: bool) -> np.ndarray:
        """Raw Brandes sums (unscaled) as float64[n]. Raises NativeUnavailable."""
        if self._handle is None:
            raise NativeUnavailable("native graph is closed")
        out = np.empty(self._n, dtype=np.float64)
        rc = self._lib.nxgraphmojo_betweenness(
            self._handle,
            ctypes.c_int32(1 if weighted else 0),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc == _BC_NO_WEIGHTS:
            raise NativeUnavailable("native graph was created without weights")
        if rc != _BC_OK:
            raise NativeUnavailable(f"native betweenness failed with status {rc}")
        return out

    def dijkstra(
        self,
        source: int,
        target: int | None,
        cutoff: float | None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Single-source Dijkstra.

        Returns (n_finalized, dist, parent) with dist as float64[n] (-1.0 for
        unfinalized nodes) and parent as int64[n] (-1 for source/unreached).
        The negative status codes DIJKSTRA_* are returned as the first
        element instead of raising, so the caller can mirror oracle errors.
        """
        if self._handle is None:
            raise NativeUnavailable("native graph is closed")
        dist = np.empty(self._n, dtype=np.float64)
        parent = np.empty(self._n, dtype=np.int64)
        rc = self._lib.nxgraphmojo_dijkstra(
            self._handle,
            ctypes.c_int64(source),
            ctypes.c_int32(1 if target is not None else 0),
            ctypes.c_int64(target if target is not None else -1),
            ctypes.c_int32(1 if cutoff is not None else 0),
            ctypes.c_double(float(cutoff) if cutoff is not None else 0.0),
            dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            parent.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        return int(rc), dist, parent

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.nxgraphmojo_graph_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
