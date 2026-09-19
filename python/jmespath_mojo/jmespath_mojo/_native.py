"""ctypes loader for the jmespathmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$JMESPATH_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``jmespath_mojo/_native/``
    3. the repository development build output ``kernels/jmespath/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``JMESPATH_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  jmespathmojo_abi_version(void)
    int32_t  jmespathmojo_search(const char* expr, int64_t expr_len,
                                 const char* data, int64_t data_len,
                                 char** out_buf, int64_t* out_len)
    void     jmespathmojo_free(char* buf, int64_t len)

``data`` is the document serialized with ``marshal.dumps(data, 4)`` (a
binary format preserving exact Python types); the result is a
kernel-allocated buffer holding the result as JSON, freed by the caller via
``jmespathmojo_free``. Status 0 means the result is ready; any other status
means the kernel cannot answer with reference-identical semantics and the
wrapper recomputes with the vendored implementation.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/jmespath/src/jmespathmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree.
ABI_VERSION = 1

# Kernel call statuses (non-zero => recompute with the vendored fallback).
STATUS_OK = 0
STATUS_EXPR = 1  # expression rejected by the kernel parser
STATUS_EVAL = 2  # evaluation-time type/arity/domain error
STATUS_UNSUPPORTED = 3  # in-spec construct outside the kernel subset
STATUS_DATA = 4  # data outside the kernel's JSON subset

_ENV_LIB = "JMESPATH_MOJO_NATIVE_LIB"
_ENV_DISABLE = "JMESPATH_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native jmespathmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libjmespathmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libjmespathmojo.so"
    if sys.platform.startswith("win"):
        return "jmespathmojo.dll"  # no Mojo toolchain builds this today
    return "libjmespathmojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "jmespath", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.jmespathmojo_abi_version.argtypes = []
    lib.jmespathmojo_abi_version.restype = ctypes.c_int32
    lib.jmespathmojo_search.argtypes = [
        ctypes.c_char_p,  # expression (UTF-8)
        ctypes.c_int64,  # expression length in bytes
        ctypes.c_char_p,  # data (JSON document)
        ctypes.c_int64,  # data length in bytes
        ctypes.POINTER(ctypes.c_char_p),  # out: kernel-allocated result buffer
        ctypes.POINTER(ctypes.c_int64),  # out: result length in bytes
    ]
    lib.jmespathmojo_search.restype = ctypes.c_int32
    lib.jmespathmojo_free.argtypes = [ctypes.c_char_p, ctypes.c_int64]
    lib.jmespathmojo_free.restype = None


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None

# Diagnostics counters (also surfaced through backend_info()).
_STATS = {"native_ok": 0, "fallback_engaged": 0}


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
                    abi = int(lib.jmespathmojo_abi_version())
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
        "stats": dict(_STATS),
    }
    try:
        lib = _load()
    except NativeUnavailable as exc:
        info["error"] = str(exc)
        return info
    info["native_available"] = True
    info["native_source"] = _LIB_SOURCE
    info["abi_version_native"] = int(lib.jmespathmojo_abi_version())
    return info


def search_native(expression: str, payload: bytes) -> tuple[int, bytes | None]:
    """Run one search on the native kernel.

    ``payload`` is the JSON document as UTF-8 bytes. Returns
    ``(STATUS_OK, result_json_bytes)`` on success, or ``(status, None)`` for
    any status the kernel cannot answer with reference-identical semantics —
    the caller must then recompute with the vendored implementation.
    """
    lib = _load()  # raises NativeUnavailable
    expr = expression.encode("utf-8")
    out = ctypes.c_char_p()
    out_len = ctypes.c_int64(0)
    status = lib.jmespathmojo_search(
        expr, len(expr), payload, len(payload), ctypes.byref(out), ctypes.byref(out_len)
    )
    if status != STATUS_OK:
        _STATS["fallback_engaged"] += 1
        return status, None
    try:
        result = ctypes.string_at(out.value, out_len.value)
    finally:
        lib.jmespathmojo_free(out, out_len)
    _STATS["native_ok"] += 1
    return STATUS_OK, result
