"""ctypes loader for the aseneighborlistmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$ASE_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``ase_mojo/_native/``
    3. the repository development build output ``kernels/ase_neighborlist/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``ASE_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  aseneighborlistmojo_abi_version(void)
    int32_t  aseneighborlistmojo_build(int64_t n, const double* pos,
                                       const double* cell, const int32_t* pbc,
                                       int32_t cutoff_mode,
                                       const int32_t* atom_types,
                                       int64_t n_types,
                                       const double* cut_matrix,
                                       const double* radii,
                                       int32_t self_interaction,
                                       int32_t bothways, int64_t max_nbins,
                                       int64_t* out_i, int64_t* out_j,
                                       int64_t* out_S, int64_t capacity,
                                       int64_t* out_count)

    cutoff_mode 0: rc(A,B) = cut_matrix[type[A]*n_types + type[B]]
    cutoff_mode 1: rc(A,B) = radii[A] + radii[B]
    returns 0 on success, 1 when capacity is too small (out_count = needed),
    2 on invalid geometry, 3 on invalid arguments.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in
# kernels/ase_neighborlist/src/aseneighborlistmojo.mojo. A mismatch means the
# installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

MODE_MATRIX = 0
MODE_RADII = 1

_ENV_LIB = "ASE_MOJO_NATIVE_LIB"
_ENV_DISABLE = "ASE_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native aseneighborlistmojo kernel could not be found/loaded/verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libaseneighborlistmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libaseneighborlistmojo.so"
    if sys.platform.startswith("win"):
        return "aseneighborlistmojo.dll"  # no Mojo toolchain builds this today
    return "libaseneighborlistmojo.so"


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
                    here,
                    "..",
                    "..",
                    "..",
                    "kernels",
                    "ase_neighborlist",
                    "build",
                    _lib_basename(),
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.aseneighborlistmojo_abi_version.argtypes = []
    lib.aseneighborlistmojo_abi_version.restype = ctypes.c_int32
    lib.aseneighborlistmojo_build.argtypes = [
        ctypes.c_int64,  # n
        f64p,  # pos[3n]
        f64p,  # cell[9]
        i32p,  # pbc[3]
        ctypes.c_int32,  # cutoff_mode
        i32p,  # atom_types[n]
        ctypes.c_int64,  # n_types
        f64p,  # cut_matrix[n_types*n_types]
        f64p,  # radii[n]
        ctypes.c_int32,  # self_interaction
        ctypes.c_int32,  # bothways
        ctypes.c_int64,  # max_nbins
        i64p,  # out_i[capacity]
        i64p,  # out_j[capacity]
        i64p,  # out_S[3*capacity]
        ctypes.c_int64,  # capacity
        i64p,  # out_count[1]
    ]
    lib.aseneighborlistmojo_build.restype = ctypes.c_int32


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
                    abi = int(lib.aseneighborlistmojo_abi_version())
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
    """True if the native kernel can build neighbor lists right now. Never raises."""
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
    info["abi_version_native"] = int(lib.aseneighborlistmojo_abi_version())
    return info


def build_ijs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff_mode: int,
    atom_types: np.ndarray,
    n_types: int,
    cut_matrix: np.ndarray,
    radii: np.ndarray,
    self_interaction: bool,
    bothways: bool,
    max_nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (i, j, S) on the native kernel; raises NativeUnavailable on failure.

    All inputs must already be contiguous typed arrays (the wrapper's
    validation layer guarantees this). Returns int64 arrays i, j (npairs,)
    and S (npairs, 3) in the kernel's deterministic emission order.
    """
    if cutoff_mode not in (MODE_MATRIX, MODE_RADII):
        raise NativeUnavailable(f"unknown cutoff mode {cutoff_mode}")
    lib = _load()  # raises NativeUnavailable
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)

    n = int(positions.shape[0])
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )

    # Heuristic initial capacity; the kernel reports the exact required size
    # on overflow, so at most one retry is ever needed.
    capacity = max(64, n * 32)
    for _attempt in range(8):
        out_i = np.empty(capacity, dtype=np.int64)
        out_j = np.empty(capacity, dtype=np.int64)
        out_s = np.empty(3 * capacity, dtype=np.int64)
        out_count = np.empty(1, dtype=np.int64)
        rc = lib.aseneighborlistmojo_build(
            ctypes.c_int64(n),
            positions.ctypes.data_as(f64p),
            cell.ctypes.data_as(f64p),
            pbc.ctypes.data_as(i32p),
            ctypes.c_int32(cutoff_mode),
            atom_types.ctypes.data_as(i32p),
            ctypes.c_int64(n_types),
            cut_matrix.ctypes.data_as(f64p),
            radii.ctypes.data_as(f64p),
            ctypes.c_int32(1 if self_interaction else 0),
            ctypes.c_int32(1 if bothways else 0),
            ctypes.c_int64(max_nbins),
            out_i.ctypes.data_as(i64p),
            out_j.ctypes.data_as(i64p),
            out_s.ctypes.data_as(i64p),
            ctypes.c_int64(capacity),
            out_count.ctypes.data_as(i64p),
        )
        if rc == 0:
            npairs = int(out_count[0])
            return out_i[:npairs], out_j[:npairs], out_s[: 3 * npairs].reshape(npairs, 3)
        if rc == 1:
            needed = int(out_count[0])
            if needed <= capacity:  # defensive: kernel must report growth
                raise NativeUnavailable(
                    "native kernel reported capacity overflow without growth"
                )
            capacity = needed
            continue
        if rc == 2:
            raise ValueError(
                "invalid geometry: the cell is singular on a periodic axis, "
                "or the bin grid exceeds the safety ceiling"
            )
        raise NativeUnavailable(f"native neighbor-list build failed with status {rc}")
    raise NativeUnavailable("native neighbor-list build exceeded retry budget")
