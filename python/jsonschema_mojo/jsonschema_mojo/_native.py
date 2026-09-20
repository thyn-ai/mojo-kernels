"""ctypes loader for the jsonschemamojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$JSONSCHEMA_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``jsonschema_mojo/_native/``
    3. the repository development build output ``kernels/jsonschema/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``JSONSCHEMA_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  jsonschemamojo_abi_version(void)
    void*    jsonschemamojo_schema_compile(const char* text, int64_t len)
    int32_t  jsonschemamojo_schema_flags(void* handle)
    void     jsonschemamojo_schema_destroy(void* handle)
    int32_t  jsonschemamojo_validate(void* handle, const char* inst,
                                     int64_t inst_len, uint8_t* out,
                                     int64_t out_cap, int32_t* meta)

``validate`` appends error/deferral records to ``out`` (see
``core._decode_records`` for the wire format) and reports counts, overflow,
and instance-feature flags through ``meta``.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/jsonschema/src/jsonschemamojo.mojo. A
# mismatch means the installed wheel and the resolved shared library disagree;
# fall back.
ABI_VERSION = 1

# schema_flags() / meta[4] bits, mirrored from the kernel.
FLAG_BIG_NUMBER = 1 << 0

_ENV_LIB = "JSONSCHEMA_MOJO_NATIVE_LIB"
_ENV_DISABLE = "JSONSCHEMA_MOJO_DISABLE_NATIVE"

# meta[] slots written by jsonschemamojo_validate.
_META_STATUS = 0
_META_N_ERR = 1
_META_N_DEFER = 2
_META_OVERFLOW = 3
_META_BIG_NUMBER = 4
_META_OUT_LEN = 5

_INITIAL_OUT_CAP = 64 * 1024
_MAX_OUT_GROWTH_ROUNDS = 12


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native jsonschemamojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libjsonschemamojo.dylib"
    if sys.platform.startswith("linux"):
        return "libjsonschemamojo.so"
    if sys.platform.startswith("win"):
        return "jsonschemamojo.dll"  # no Mojo toolchain builds this today
    return "libjsonschemamojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "jsonschema", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.jsonschemamojo_abi_version.argtypes = []
    lib.jsonschemamojo_abi_version.restype = ctypes.c_int32
    lib.jsonschemamojo_schema_compile.argtypes = [ctypes.c_char_p, ctypes.c_int64]
    lib.jsonschemamojo_schema_compile.restype = ctypes.c_void_p
    lib.jsonschemamojo_schema_flags.argtypes = [ctypes.c_void_p]
    lib.jsonschemamojo_schema_flags.restype = ctypes.c_int32
    lib.jsonschemamojo_schema_destroy.argtypes = [ctypes.c_void_p]
    lib.jsonschemamojo_schema_destroy.restype = None
    lib.jsonschemamojo_validate.argtypes = [
        ctypes.c_void_p,  # schema handle
        ctypes.c_char_p,  # instance JSON text (NUL-free: dumps escapes it)
        ctypes.c_int64,  # instance text length
        u8p,  # output buffer
        ctypes.c_int64,  # output buffer capacity
        i32p,  # meta[8]
    ]
    lib.jsonschemamojo_validate.restype = ctypes.c_int32


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_SOURCE: str | None = None
_LOAD_ERROR: str | None = None

# Per-thread scratch for validation records: allocating and zeroing a fresh
# 64 KiB buffer per call would dominate small-document latency. Buffers are
# thread-local (ctypes releases the GIL during the native call) and grow to
# the largest capacity ever needed.
_TLS = threading.local()


def _scratch_buffer(cap: int):
    buf = getattr(_TLS, "record_buffer", None)
    if buf is None or len(buf) < cap:
        buf = (ctypes.c_uint8 * cap)()
        _TLS.record_buffer = buf
    return buf


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
                    abi = int(lib.jsonschemamojo_abi_version())
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
    """True if the native kernel can validate right now. Never raises."""
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
    info["abi_version_native"] = int(lib.jsonschemamojo_abi_version())
    return info


class NativeSchema:
    """Owned handle to a natively compiled schema. Not thread-safe to close twice."""

    def __init__(self, schema_text: bytes) -> None:
        lib = _load()  # raises NativeUnavailable
        handle = lib.jsonschemamojo_schema_compile(schema_text, len(schema_text))
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the schema JSON; "
                "falling back to the pure-Python reference"
            )
        self._lib = lib
        self._handle = handle

    @property
    def flags(self) -> int:
        if self._handle is None:
            raise NativeUnavailable("native schema is closed")
        return int(self._lib.jsonschemamojo_schema_flags(self._handle))

    def validate_records(self, instance_text: bytes) -> tuple[bytes, int, int, bool]:
        """Validate one serialized instance.

        Returns ``(record_payload, n_errors, n_defers, big_number)``. Raises
        NativeUnavailable if the native walk itself failed (the caller then
        re-validates with the pure-Python reference).
        """
        if self._handle is None:
            raise NativeUnavailable("native schema is closed")
        cap = _INITIAL_OUT_CAP
        for _ in range(_MAX_OUT_GROWTH_ROUNDS):
            buf = _scratch_buffer(cap)
            meta = (ctypes.c_int32 * 8)()
            rc = self._lib.jsonschemamojo_validate(
                self._handle, instance_text, len(instance_text), buf, cap, meta
            )
            if rc != 0:
                raise NativeUnavailable(f"native validation failed with status {rc}")
            if meta[_META_STATUS] != 0:
                raise NativeUnavailable(
                    f"native instance parse failed with status {meta[_META_STATUS]}"
                )
            if meta[_META_OVERFLOW]:
                cap *= 4  # records are complete only up to overflow; retry bigger
                continue
            out_len = int(meta[_META_OUT_LEN])
            payload = bytes(bytearray(buf[:out_len]))
            return (
                payload,
                int(meta[_META_N_ERR]),
                int(meta[_META_N_DEFER]),
                bool(meta[_META_BIG_NUMBER]),
            )
        raise NativeUnavailable("native error-record buffer did not converge")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.jsonschemamojo_schema_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
