"""ctypes loader for the pykalmanmojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$PYKALMAN_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``pykalman_mojo/_native/``
    3. the repository development build output ``kernels/pykalman/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``PYKALMAN_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  pykalmanmojo_abi_version(void)
    void*    pykalmanmojo_model_create(int64_t n_dim_state, int64_t n_dim_obs,
                                       const double* A, const double* b,
                                       const double* Q, const double* C,
                                       const double* d, const double* R,
                                       const double* x0, const double* P0)
    int32_t  pykalmanmojo_filter(void* handle, int64_t n_timesteps,
                                 const double* obs, const int32_t* mask,
                                 double* out_filt_mean, double* out_filt_cov)
    int32_t  pykalmanmojo_smooth(void* handle, int64_t n_timesteps,
                                 const double* obs, const int32_t* mask,
                                 double* out_smooth_mean, double* out_smooth_cov)
    void     pykalmanmojo_model_destroy(void* handle)

Kernel return codes: 0 success, 1 NULL handle, 2 invalid n_timesteps,
3 singular matrix (the kernel inverts exactly; the caller should serve that
call from the reference implementation, whose pseudo-inverse tolerates
singularity).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/pykalman/src/pykalmanmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "PYKALMAN_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PYKALMAN_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native pykalmanmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libpykalmanmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libpykalmanmojo.so"
    if sys.platform.startswith("win"):
        return "pykalmanmojo.dll"  # no Mojo toolchain builds this today
    return "libpykalmanmojo.so"


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
                    here, "..", "..", "..", "kernels", "pykalman", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.pykalmanmojo_abi_version.argtypes = []
    lib.pykalmanmojo_abi_version.restype = ctypes.c_int32
    lib.pykalmanmojo_model_create.argtypes = [
        ctypes.c_int64,  # n_dim_state
        ctypes.c_int64,  # n_dim_obs
        f64p,  # A[n_s*n_s]
        f64p,  # b[n_s]
        f64p,  # Q[n_s*n_s]
        f64p,  # C[n_o*n_s]
        f64p,  # d[n_o]
        f64p,  # R[n_o*n_o]
        f64p,  # x0[n_s]
        f64p,  # P0[n_s*n_s]
    ]
    lib.pykalmanmojo_model_create.restype = ctypes.c_void_p
    for name in ("pykalmanmojo_filter", "pykalmanmojo_smooth"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, f64p, i32p, f64p, f64p]
        fn.restype = ctypes.c_int32
    lib.pykalmanmojo_model_destroy.argtypes = [ctypes.c_void_p]
    lib.pykalmanmojo_model_destroy.restype = None


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
                    abi = int(lib.pykalmanmojo_abi_version())
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
    """True if the native kernel can filter right now. Never raises."""
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
    info["abi_version_native"] = int(lib.pykalmanmojo_abi_version())
    return info


def _as_f64(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


class NativeModel:
    """Owned handle to a native system model. Not thread-safe to close twice."""

    def __init__(self, params) -> None:
        lib = _load()  # raises NativeUnavailable
        f64p = ctypes.POINTER(ctypes.c_double)
        n_s = int(params.n_dim_state)
        n_o = int(params.n_dim_obs)
        # The kernel copies every buffer; the temporaries may be freed after.
        bufs = [_as_f64(getattr(params, name)) for name in ("A", "b", "Q", "C", "d", "R", "x0", "P0")]
        handle = lib.pykalmanmojo_model_create(
            ctypes.c_int64(n_s),
            ctypes.c_int64(n_o),
            *(b.ctypes.data_as(f64p) for b in bufs),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the model (invalid dimensions); "
                "falling back to the pure-Python reference"
            )
        self._lib = lib
        self._handle = handle
        self._n_s = n_s
        self._n_o = n_o

    @property
    def n_dim_state(self) -> int:
        return self._n_s

    def _run(self, fn_name: str, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._handle is None:
            raise NativeUnavailable("native model is closed")
        f64p = ctypes.POINTER(ctypes.c_double)
        i32p = ctypes.POINTER(ctypes.c_int32)
        obs = _as_f64(obs)
        mask = np.ascontiguousarray(mask, dtype=np.int32)
        n_timesteps = obs.shape[0]
        if obs.ndim != 2 or obs.shape[1] != self._n_o:
            raise NativeUnavailable(
                f"observation width {obs.shape[1] if obs.ndim == 2 else '?'} "
                f"!= model n_dim_obs {self._n_o}"
            )
        out_mean = np.empty((n_timesteps, self._n_s), dtype=np.float64)
        out_cov = np.empty((n_timesteps, self._n_s, self._n_s), dtype=np.float64)
        rc = getattr(self._lib, fn_name)(
            self._handle,
            ctypes.c_int64(n_timesteps),
            obs.ctypes.data_as(f64p),
            mask.ctypes.data_as(i32p),
            out_mean.ctypes.data_as(f64p),
            out_cov.ctypes.data_as(f64p),
        )
        if rc != 0:
            raise NativeUnavailable(f"native {fn_name} failed with status {rc}")
        return out_mean, out_cov

    def filter(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filtered state means/covariances. Returns float64 (T, n_s), (T, n_s, n_s)."""
        return self._run("pykalmanmojo_filter", obs, mask)

    def smooth(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed state means/covariances. Returns float64 (T, n_s), (T, n_s, n_s)."""
        return self._run("pykalmanmojo_smooth", obs, mask)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.pykalmanmojo_model_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
