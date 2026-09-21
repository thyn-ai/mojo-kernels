"""ctypes loader for the mistunemojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$MISTUNE_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``mistune_mojo/_native/``
    3. the repository development build output ``kernels/mistune/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python engine. Set
``MISTUNE_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1, single-shot: render a whole document per call)::

    int32_t  mistunemojo_abi_version(void)
    void*    mistunemojo_render(const uint8_t* data, int64_t length)
    int32_t  mistunemojo_result_status(void* handle)  -> 0 ok / 2 unsupported
    int64_t  mistunemojo_result_size(void* handle)    -> html byte length
    uint8_t* mistunemojo_result_data(void* handle)    -> html bytes
    void     mistunemojo_result_destroy(void* handle)

Status 2 ("unsupported construct") is not an error: the wrapper transparently
re-renders the document with the pure-Python engine, which produces the same
bytes. End users never see a difference between the two backends.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/mistune/src/mistunemojo.mojo.
ABI_VERSION = 1

_ENV_LIB = "MISTUNE_MOJO_NATIVE_LIB"
_ENV_DISABLE = "MISTUNE_MOJO_DISABLE_NATIVE"

# render status codes (mirror the kernel)
_STATUS_OK = 0
_STATUS_UNSUPPORTED = 2


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native mistunemojo kernel could not be found, loaded, or verified."""


class UnsupportedConstruct(RuntimeError):  # noqa: N818
    """The native kernel declined this document; use the fallback engine."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libmistunemojo.dylib"
    if sys.platform.startswith("linux"):
        return "libmistunemojo.so"
    if sys.platform.startswith("win"):
        return "mistunemojo.dll"  # no Mojo toolchain builds this today
    return "libmistunemojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "mistune", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.mistunemojo_abi_version.argtypes = []
    lib.mistunemojo_abi_version.restype = ctypes.c_int32
    lib.mistunemojo_render.argtypes = [u8p, ctypes.c_int64]
    lib.mistunemojo_render.restype = ctypes.c_void_p
    lib.mistunemojo_result_status.argtypes = [ctypes.c_void_p]
    lib.mistunemojo_result_status.restype = ctypes.c_int32
    lib.mistunemojo_result_size.argtypes = [ctypes.c_void_p]
    lib.mistunemojo_result_size.restype = ctypes.c_int64
    lib.mistunemojo_result_data.argtypes = [ctypes.c_void_p]
    lib.mistunemojo_result_data.restype = u8p
    lib.mistunemojo_result_destroy.argtypes = [ctypes.c_void_p]
    lib.mistunemojo_result_destroy.restype = None


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
                    abi = int(lib.mistunemojo_abi_version())
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
    """True if the native kernel can render right now. Never raises."""
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
    info["abi_version_native"] = int(lib.mistunemojo_abi_version())
    return info


def render_native(text: str) -> str:
    """Render one document with the native kernel.

    Raises NativeUnavailable if the kernel cannot be loaded and
    UnsupportedConstruct if the kernel declines the document (the caller then
    uses the pure-Python engine, which is byte-identical).
    """
    lib = _load()  # raises NativeUnavailable
    data = text.encode("utf-8")
    u8p = ctypes.POINTER(ctypes.c_uint8)
    handle = lib.mistunemojo_render(
        (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None,
        len(data),
    )
    if not handle:
        raise NativeUnavailable("native kernel returned a null result handle")
    try:
        status = int(lib.mistunemojo_result_status(handle))
        if status == _STATUS_UNSUPPORTED:
            raise UnsupportedConstruct("native kernel declined this document")
        if status != _STATUS_OK:
            raise NativeUnavailable(f"native kernel failed with status {status}")
        size = int(lib.mistunemojo_result_size(handle))
        buf = lib.mistunemojo_result_data(handle)
        return ctypes.string_at(buf, size).decode("utf-8")
    finally:
        lib.mistunemojo_result_destroy(handle)
