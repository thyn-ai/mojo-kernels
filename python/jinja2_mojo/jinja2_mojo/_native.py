"""ctypes loader for the jinja2mojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$JINJA2_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``jinja2_mojo/_native/``
    3. the repository development build output ``kernels/jinja2/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to stock jinja2's own lexer. Set
``JINJA2_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t jinja2mojo_abi_version(void)
    int32_t jinja2mojo_scan(const uint8_t* src, int64_t src_len,
                            uint8_t* out_tokens, int64_t out_cap,
                            int64_t* out_count)

Each output token is a 24-byte little-endian record; see
``kernels/jinja2/src/jinja2mojo.mojo`` for the layout. Scan statuses:
0 = tokens written, 1 = outside the kernel's subset (caller must run the
stock lexer), 2 = ``out_cap`` too small (``out_count`` = records needed).
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading

# Must equal ABI_VERSION in kernels/jinja2/src/jinja2mojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; the
# caller falls back to the stock lexer.
ABI_VERSION = 1

_ENV_LIB = "JINJA2_MOJO_NATIVE_LIB"
_ENV_DISABLE = "JINJA2_MOJO_DISABLE_NATIVE"

RECORD_SIZE = 24
_RECORD_FMT = "<BBBBIII8x"  # kind, aux, aux2, pad, lineno, offset, length, pad

STATUS_OK = 0
STATUS_FALLBACK = 1
STATUS_OVERFLOW = 2


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native jinja2mojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libjinja2mojo.dylib"
    if sys.platform.startswith("linux"):
        return "libjinja2mojo.so"
    if sys.platform.startswith("win"):
        return "jinja2mojo.dll"  # no Mojo toolchain builds this today
    return "libjinja2mojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "jinja2", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.jinja2mojo_abi_version.argtypes = []
    lib.jinja2mojo_abi_version.restype = ctypes.c_int32
    # c_char_p pins the caller's bytes object for the call (no copy; the
    # explicit length makes embedded NULs safe).
    lib.jinja2mojo_scan.argtypes = [ctypes.c_char_p, ctypes.c_int64, u8p, ctypes.c_int64, i64p]
    lib.jinja2mojo_scan.restype = ctypes.c_int32


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
                    abi = int(lib.jinja2mojo_abi_version())
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
    """True if the native kernel can scan right now. Never raises."""
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
    info["abi_version_native"] = int(lib.jinja2mojo_abi_version())
    return info


class _ScanBuffers(threading.local):
    """Grow-only per-thread output buffers (avoids per-call allocation)."""

    buf: ctypes.Array | None = None
    cap: int = 0


_TLS = _ScanBuffers()


def scan(src: bytes) -> list[tuple] | None:
    """Scan one UTF-8 template into token records.

    Returns a list of ``(kind, aux, aux2, lineno, offset, length)`` tuples
    (byte offsets into ``src``), or ``None`` when the kernel declines the
    template (the caller then runs the stock lexer).
    """
    lib = _load()  # raises NativeUnavailable
    n = len(src)
    if n == 0:
        return []
    cap = max(64, n // 2 + 16)
    count = ctypes.c_int64(0)
    for _attempt in range(3):
        if _TLS.cap < cap:
            _TLS.buf = (ctypes.c_uint8 * (cap * RECORD_SIZE))()
            _TLS.cap = cap
        rc = lib.jinja2mojo_scan(src, n, _TLS.buf, cap, ctypes.byref(count))  # type: ignore[arg-type]
        if rc == STATUS_OK:
            total = int(count.value)
            raw = memoryview(_TLS.buf)[: total * RECORD_SIZE]  # type: ignore[arg-type]
            return [t for t in struct.iter_unpack(_RECORD_FMT, raw)]
        if rc == STATUS_OVERFLOW and int(count.value) > cap:
            cap = int(count.value)
            continue
        return None  # STATUS_FALLBACK or an inconsistent overflow: stock lexer
    return None
