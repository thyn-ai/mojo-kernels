"""ctypes loader for the gaussgridmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$CCLIB_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``cclib_mojo/_native/``
    3. the repository development build output ``kernels/gaussgrid/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``CCLIB_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  gaussgridmojo_abi_version(void)
    int32_t  gaussgridmojo_eval(int64_t n_bf, const int64_t* bf_offsets,
                                const int32_t* bf_l, const int32_t* bf_m,
                                const int32_t* bf_n,
                                const double* bf_cx, const double* bf_cy,
                                const double* bf_cz, const double* bf_norm,
                                const double* prim_alpha, const double* prim_w,
                                const double* ax, const double* ay,
                                const double* az,
                                int64_t nx, int64_t ny, int64_t nz,
                                int64_t n_mo, const double* mo_coeff,
                                int32_t mode, double* out)

    mode 0 (wavefunction): out = sum_bf mo_coeff[0,bf] * bf(r)   (n_mo == 1)
    mode 1 (density):      out = sum_mo (sum_bf coeff*bf(r))^2
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/gaussgrid/src/gaussgridmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

MODE_WAVEFUNCTION = 0
MODE_DENSITY = 1

_ENV_LIB = "CCLIB_MOJO_NATIVE_LIB"
_ENV_DISABLE = "CCLIB_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native gaussgridmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libgaussgridmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libgaussgridmojo.so"
    if sys.platform.startswith("win"):
        return "gaussgridmojo.dll"  # no Mojo toolchain builds this today
    return "libgaussgridmojo.so"


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
                    here, "..", "..", "..", "kernels", "gaussgrid", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.gaussgridmojo_abi_version.argtypes = []
    lib.gaussgridmojo_abi_version.restype = ctypes.c_int32
    lib.gaussgridmojo_eval.argtypes = [
        ctypes.c_int64,  # n_bf
        i64p,  # bf_offsets[n_bf+1]
        i32p,  # bf_l[n_bf]
        i32p,  # bf_m[n_bf]
        i32p,  # bf_n[n_bf]
        f64p,  # bf_cx[n_bf]
        f64p,  # bf_cy[n_bf]
        f64p,  # bf_cz[n_bf]
        f64p,  # bf_norm[n_bf]
        f64p,  # prim_alpha[n_prims]
        f64p,  # prim_w[n_prims]
        f64p,  # ax[nx]
        f64p,  # ay[ny]
        f64p,  # az[nz]
        ctypes.c_int64,  # nx
        ctypes.c_int64,  # ny
        ctypes.c_int64,  # nz
        ctypes.c_int64,  # n_mo
        f64p,  # mo_coeff[n_mo * n_bf]
        ctypes.c_int32,  # mode
        f64p,  # out[nx*ny*nz]
    ]
    lib.gaussgridmojo_eval.restype = ctypes.c_int32


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
                    abi = int(lib.gaussgridmojo_abi_version())
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
    info["abi_version_native"] = int(lib.gaussgridmojo_abi_version())
    return info


def eval_grid(
    basis,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    coeff: np.ndarray,
    mode: int,
) -> np.ndarray:
    """Evaluate on the native kernel; raises NativeUnavailable on any failure.

    ``basis`` is a ``_basis.BasisArrays`` (already contiguous typed arrays),
    ``ax``/``ay``/``az`` are the grid axes in bohr, ``coeff`` is the
    (n_mo, n_bf) coefficient matrix, and the return is the C-order raveled
    grid, float64[nx*ny*nz]. The kernel writes into a fresh buffer per call;
    input buffers are only read.
    """
    if mode == MODE_WAVEFUNCTION and coeff.shape[0] != 1:
        raise NativeUnavailable("wavefunction mode requires exactly one MO row")
    if mode not in (MODE_WAVEFUNCTION, MODE_DENSITY):
        raise NativeUnavailable(f"unknown mode {mode}")
    lib = _load()  # raises NativeUnavailable
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    ax = np.ascontiguousarray(ax, dtype=np.float64)
    ay = np.ascontiguousarray(ay, dtype=np.float64)
    az = np.ascontiguousarray(az, dtype=np.float64)
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    nx, ny, nz = ax.shape[0], ay.shape[0], az.shape[0]
    out = np.empty(nx * ny * nz, dtype=np.float64)
    rc = lib.gaussgridmojo_eval(
        ctypes.c_int64(basis.n_bf),
        basis.offsets.ctypes.data_as(i64p),
        basis.powers_l.ctypes.data_as(i32p),
        basis.powers_m.ctypes.data_as(i32p),
        basis.powers_n.ctypes.data_as(i32p),
        basis.center_x.ctypes.data_as(f64p),
        basis.center_y.ctypes.data_as(f64p),
        basis.center_z.ctypes.data_as(f64p),
        basis.bf_norm.ctypes.data_as(f64p),
        basis.prim_alpha.ctypes.data_as(f64p),
        basis.prim_w.ctypes.data_as(f64p),
        ax.ctypes.data_as(f64p),
        ay.ctypes.data_as(f64p),
        az.ctypes.data_as(f64p),
        ctypes.c_int64(nx),
        ctypes.c_int64(ny),
        ctypes.c_int64(nz),
        ctypes.c_int64(coeff.shape[0]),
        coeff.ctypes.data_as(f64p),
        ctypes.c_int32(mode),
        out.ctypes.data_as(f64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native evaluation failed with status {rc}")
    return out
