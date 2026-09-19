"""ctypes loader for the elephantsurrogatesmojo native kernel, ABI handshake.

Resolution order:

    1. ``$ELEPHANT_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``elephant_mojo/_native/``
    3. the repository development build output
       ``kernels/elephant-surrogates/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``ELEPHANT_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  elephantsurrogatesmojo_abi_version(void)

    int32_t  elephantsurrogatesmojo_dither(
                 int64_t n_trains, const int64_t* spike_counts,
                 const double* spike_times,
                 double t_start, double t_stop, double bin_size,
                 int64_t n_bins, double dither, double refractory_period,
                 int32_t method, int32_t edges_mode,
                 int64_t n_surrogates, uint64_t seed,
                 uint8_t* out)   # [n_surrogates * n_trains * n_bins]

    int64_t  elephantsurrogatesmojo_pvalue_spec_count(
                 int64_t n_rows, int64_t n_sizes, int64_t winlen,
                 const double* max_occs, int64_t min_occ)

    int32_t  elephantsurrogatesmojo_pvalue_spec_fill(
                 int64_t n_rows, int64_t n_sizes, int64_t winlen,
                 const double* max_occs, int64_t min_spikes, int64_t min_occ,
                 int64_t n_surr,
                 int32_t* out_size, int32_t* out_occ, int32_t* out_dur,
                 double* out_p)

    (max_occs has n_rows rows; n_surr is only the p-value denominator, as in
    the reference, which divides by the given surrogate count)

    method 0 (plain dither):   spike + U(-dither, +dither); edges_mode 0 drops
                               spikes outside (t_start, t_stop), edges_mode 1
                               clamps them to the range ends
    method 1 (refractory):     dither range shrunk by the refractory period
                               min(given, smallest ISI), spikes perturbed in a
                               uniform random order
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in
# kernels/elephant-surrogates/src/elephantsurrogatesmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back.
ABI_VERSION = 1

METHOD_PLAIN = 0
METHOD_REFRACTORY = 1

EDGES_DROP = 0
EDGES_CLAMP = 1

_ENV_LIB = "ELEPHANT_MOJO_NATIVE_LIB"
_ENV_DISABLE = "ELEPHANT_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native elephantsurrogatesmojo kernel could not be used."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libelephantsurrogatesmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libelephantsurrogatesmojo.so"
    if sys.platform.startswith("win"):
        return "elephantsurrogatesmojo.dll"  # no Mojo toolchain builds this today
    return "libelephantsurrogatesmojo.so"


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
                    "elephant-surrogates",
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
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.elephantsurrogatesmojo_abi_version.argtypes = []
    lib.elephantsurrogatesmojo_abi_version.restype = ctypes.c_int32
    lib.elephantsurrogatesmojo_dither.argtypes = [
        ctypes.c_int64,  # n_trains
        i64p,  # spike_counts[n_trains]
        f64p,  # spike_times[total_spikes]
        ctypes.c_double,  # t_start
        ctypes.c_double,  # t_stop
        ctypes.c_double,  # bin_size
        ctypes.c_int64,  # n_bins
        ctypes.c_double,  # dither
        ctypes.c_double,  # refractory_period
        ctypes.c_int32,  # method
        ctypes.c_int32,  # edges_mode
        ctypes.c_int64,  # n_surrogates
        ctypes.c_uint64,  # seed
        u8p,  # out[n_surrogates * n_trains * n_bins]
    ]
    lib.elephantsurrogatesmojo_dither.restype = ctypes.c_int32
    lib.elephantsurrogatesmojo_pvalue_spec_count.argtypes = [
        ctypes.c_int64,  # n_rows
        ctypes.c_int64,  # n_sizes
        ctypes.c_int64,  # winlen
        f64p,  # max_occs[n_surr * n_sizes * winlen]
        ctypes.c_int64,  # min_occ
    ]
    lib.elephantsurrogatesmojo_pvalue_spec_count.restype = ctypes.c_int64
    lib.elephantsurrogatesmojo_pvalue_spec_fill.argtypes = [
        ctypes.c_int64,  # n_rows
        ctypes.c_int64,  # n_sizes
        ctypes.c_int64,  # winlen
        f64p,  # max_occs
        ctypes.c_int64,  # min_spikes
        ctypes.c_int64,  # min_occ
        ctypes.c_int64,  # n_surr (p-value denominator)
        i32p,  # out_size[count]
        i32p,  # out_occ[count]
        i32p,  # out_dur[count]
        f64p,  # out_p[count]
    ]
    lib.elephantsurrogatesmojo_pvalue_spec_fill.restype = ctypes.c_int32


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
                    abi = int(lib.elephantsurrogatesmojo_abi_version())
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
    info["abi_version_native"] = int(lib.elephantsurrogatesmojo_abi_version())
    return info


def dither_native(
    spike_counts: np.ndarray,
    spike_times: np.ndarray,
    t_start: float,
    t_stop: float,
    bin_size: float,
    n_bins: int,
    dither: float,
    refractory_period: float,
    method: int,
    edges_mode: int,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    """Dither on the native kernel; raises NativeUnavailable on any failure.

    ``spike_counts``/``spike_times`` are the flattened per-train arrays
    (already contiguous, int64/float64). Returns the C-order uint8 buffer of
    shape (n_surrogates, n_trains, n_bins); occupied bins are 1.
    """
    lib = _load()  # raises NativeUnavailable
    i64p = ctypes.POINTER(ctypes.c_int64)
    f64p = ctypes.POINTER(ctypes.c_double)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    n_trains = int(spike_counts.shape[0])
    out = np.empty((n_surrogates, n_trains, n_bins), dtype=np.uint8)
    rc = lib.elephantsurrogatesmojo_dither(
        ctypes.c_int64(n_trains),
        spike_counts.ctypes.data_as(i64p),
        spike_times.ctypes.data_as(f64p),
        ctypes.c_double(t_start),
        ctypes.c_double(t_stop),
        ctypes.c_double(bin_size),
        ctypes.c_int64(n_bins),
        ctypes.c_double(dither),
        ctypes.c_double(refractory_period),
        ctypes.c_int32(method),
        ctypes.c_int32(edges_mode),
        ctypes.c_int64(n_surrogates),
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        out.ctypes.data_as(u8p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native dither failed with status {rc}")
    return out


def pvalue_spec_native(
    max_occs: np.ndarray,
    min_spikes: int,
    min_occ: int,
    n_surr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """P-value spectrum on the native kernel; raises NativeUnavailable.

    ``max_occs`` is the contiguous float64 C-order cube
    (n_rows, n_sizes, winlen) — for the 2-D spectrum '#' pass winlen == 1.
    ``n_surr`` is the p-value denominator (the reference divides by the
    given surrogate count, which may exceed the row count). Returns
    (size, occ, dur, p) int32/int32/int32/float64 arrays in the reference
    emission order (size, then duration, then occurrence).
    """
    lib = _load()  # raises NativeUnavailable
    i32p = ctypes.POINTER(ctypes.c_int32)
    f64p = ctypes.POINTER(ctypes.c_double)
    n_rows = int(max_occs.shape[0])
    n_sizes = int(max_occs.shape[1])
    winlen = int(max_occs.shape[2])
    count = lib.elephantsurrogatesmojo_pvalue_spec_count(
        ctypes.c_int64(n_rows),
        ctypes.c_int64(n_sizes),
        ctypes.c_int64(winlen),
        max_occs.ctypes.data_as(f64p),
        ctypes.c_int64(min_occ),
    )
    if count < 0:
        raise NativeUnavailable(f"native pvalue_spec count failed with status {count}")
    out_size = np.empty(count, dtype=np.int32)
    out_occ = np.empty(count, dtype=np.int32)
    out_dur = np.empty(count, dtype=np.int32)
    out_p = np.empty(count, dtype=np.float64)
    rc = lib.elephantsurrogatesmojo_pvalue_spec_fill(
        ctypes.c_int64(n_rows),
        ctypes.c_int64(n_sizes),
        ctypes.c_int64(winlen),
        max_occs.ctypes.data_as(f64p),
        ctypes.c_int64(min_spikes),
        ctypes.c_int64(min_occ),
        ctypes.c_int64(n_surr),
        out_size.ctypes.data_as(i32p),
        out_occ.ctypes.data_as(i32p),
        out_dur.ctypes.data_as(i32p),
        out_p.ctypes.data_as(f64p),
    )
    if rc != 0:
        raise NativeUnavailable(f"native pvalue_spec fill failed with status {rc}")
    return out_size, out_occ, out_dur, out_p
