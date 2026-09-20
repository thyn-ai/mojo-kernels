"""ctypes loader for the sacremojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$SACREBLEU_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``sacrebleu_mojo/_native/``
    3. the repository development build output ``kernels/sacrebleu/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``SACREBLEU_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  sacremojo_abi_version(void)
    int32_t  sacremojo_bleu_stats(const uint32_t* hyp_ids, const int64_t* hyp_off,
                                  const uint32_t* ref_ids, const int64_t* ref_off,
                                  const int64_t* seg_index, int64_t i0, int64_t i1,
                                  int64_t* stats /*[10]*/)
    int32_t  sacremojo_chrf_stats(const uint32_t* hypc, const int64_t* hypc_off,
                                  const uint32_t* refc, const int64_t* refc_off,
                                  const int64_t* seg_index,
                                  const uint32_t* hypw, const int64_t* hypw_off,
                                  const uint32_t* refw, const int64_t* refw_off,
                                  int64_t i0, int64_t i1,
                                  int64_t char_order, int64_t word_order,
                                  double beta,
                                  int64_t* out_m, int64_t* out_h, int64_t* out_r)

Both statistics functions accumulate into caller-owned output buffers for the
pair range [i0, i1) and return 0 on success. A positive return value is
(pair_index + 1): that pair exceeds the kernel's per-pair packing limits and
must be processed by the caller (the pure-Python reference path), after which
the kernel is called again with i0 = pair_index + 1.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# Must equal ABI_VERSION in kernels/sacrebleu/src/sacremojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back to the vendored pure-Python reference.
ABI_VERSION = 1

# Kernel parameter ceilings mirrored from the kernel: chrF parameter
# combinations outside these bounds are served by the fallback path.
MAX_CHAR_ORDER = 6
MAX_WORD_ORDER = 4

_ENV_LIB = "SACREBLEU_MOJO_NATIVE_LIB"
_ENV_DISABLE = "SACREBLEU_MOJO_DISABLE_NATIVE"

_U32P = ctypes.POINTER(ctypes.c_uint32)
_I64P = ctypes.POINTER(ctypes.c_int64)


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native sacremojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libsacremojo.dylib"
    if sys.platform.startswith("linux"):
        return "libsacremojo.so"
    if sys.platform.startswith("win"):
        return "sacremojo.dll"  # no Mojo toolchain builds this today
    return "libsacremojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "sacrebleu", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.sacremojo_abi_version.argtypes = []
    lib.sacremojo_abi_version.restype = ctypes.c_int32
    lib.sacremojo_bleu_stats.argtypes = [
        _U32P,
        _I64P,
        _U32P,
        _I64P,
        _I64P,
        ctypes.c_int64,
        ctypes.c_int64,
        _I64P,
    ]
    lib.sacremojo_bleu_stats.restype = ctypes.c_int32
    lib.sacremojo_chrf_stats.argtypes = [
        _U32P,
        _I64P,
        _U32P,
        _I64P,
        _I64P,
        _U32P,
        _I64P,
        _U32P,
        _I64P,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        _I64P,
        _I64P,
        _I64P,
    ]
    lib.sacremojo_chrf_stats.restype = ctypes.c_int32


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
                    abi = int(lib.sacremojo_abi_version())
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
    """True if the native kernel can compute statistics right now. Never raises."""
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
    info["abi_version_native"] = int(lib.sacremojo_abi_version())
    return info


def _as_u32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint32)


def _as_i64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def bleu_stats(
    hyp_ids: np.ndarray,
    hyp_off: np.ndarray,
    ref_ids: np.ndarray,
    ref_off: np.ndarray,
    seg_index: np.ndarray,
    i0: int,
    i1: int,
    stats: np.ndarray,
    reference_pair,
) -> None:
    """Accumulate BLEU statistics for pairs [i0, i1) into ``stats`` (int64[10]).

    Pairs that exceed the kernel's per-pair packing limit are computed by
    ``reference_pair`` (the vendored pure-Python path) and the kernel call is
    resumed after them, so results are exact for every corpus.
    """
    lib = _load()  # raises NativeUnavailable
    hyp_ids = _as_u32(hyp_ids)
    hyp_off = _as_i64(hyp_off)
    ref_ids = _as_u32(ref_ids)
    ref_off = _as_i64(ref_off)
    seg_index = _as_i64(seg_index)
    i = i0
    while i < i1:
        rc = lib.sacremojo_bleu_stats(
            hyp_ids.ctypes.data_as(_U32P),
            hyp_off.ctypes.data_as(_I64P),
            ref_ids.ctypes.data_as(_U32P),
            ref_off.ctypes.data_as(_I64P),
            seg_index.ctypes.data_as(_I64P),
            ctypes.c_int64(i),
            ctypes.c_int64(i1),
            stats.ctypes.data_as(_I64P),
        )
        if rc == 0:
            return
        if rc < 0 or rc > i1:
            raise NativeUnavailable(f"native BLEU stats failed with status {rc}")
        reference_pair(rc - 1)  # overflow pair: pure-Python single pair
        i = rc


def chrf_stats(
    hypc: np.ndarray,
    hypc_off: np.ndarray,
    refc: np.ndarray,
    refc_off: np.ndarray,
    seg_index: np.ndarray,
    hypw: np.ndarray,
    hypw_off: np.ndarray,
    refw: np.ndarray,
    refw_off: np.ndarray,
    i0: int,
    i1: int,
    char_order: int,
    word_order: int,
    beta: float,
    out_m: np.ndarray,
    out_h: np.ndarray,
    out_r: np.ndarray,
    reference_pair,
) -> None:
    """Accumulate chrF statistics for pairs [i0, i1) into the output arrays.

    Same overflow-pair protocol as :func:`bleu_stats`.
    """
    lib = _load()  # raises NativeUnavailable
    hypc = _as_u32(hypc)
    hypc_off = _as_i64(hypc_off)
    refc = _as_u32(refc)
    refc_off = _as_i64(refc_off)
    seg_index = _as_i64(seg_index)
    hypw = _as_u32(hypw)
    hypw_off = _as_i64(hypw_off)
    refw = _as_u32(refw)
    refw_off = _as_i64(refw_off)
    i = i0
    while i < i1:
        rc = lib.sacremojo_chrf_stats(
            hypc.ctypes.data_as(_U32P),
            hypc_off.ctypes.data_as(_I64P),
            refc.ctypes.data_as(_U32P),
            refc_off.ctypes.data_as(_I64P),
            seg_index.ctypes.data_as(_I64P),
            hypw.ctypes.data_as(_U32P),
            hypw_off.ctypes.data_as(_I64P),
            refw.ctypes.data_as(_U32P),
            refw_off.ctypes.data_as(_I64P),
            ctypes.c_int64(i),
            ctypes.c_int64(i1),
            ctypes.c_int64(char_order),
            ctypes.c_int64(word_order),
            ctypes.c_double(beta),
            out_m.ctypes.data_as(_I64P),
            out_h.ctypes.data_as(_I64P),
            out_r.ctypes.data_as(_I64P),
        )
        if rc == 0:
            return
        if rc == 2:
            raise ValueError(
                f"chrF parameters out of kernel range: char_order={char_order}, "
                f"word_order={word_order}"
            )
        if rc < 0 or rc > i1:
            raise NativeUnavailable(f"native chrF stats failed with status {rc}")
        reference_pair(rc - 1)  # overflow pair: pure-Python single pair
        i = rc
