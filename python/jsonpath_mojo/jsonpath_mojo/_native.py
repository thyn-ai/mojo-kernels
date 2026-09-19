"""ctypes loader for the jsonpathmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$JSONPATH_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``jsonpath_mojo/_native/``
    3. the repository development build output ``kernels/jsonpath/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python implementation. Set
``JSONPATH_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  jsonpathmojo_abi_version(void)
    void*    jsonpathmojo_find(const uint8_t* expr, int64_t expr_len,
                               const uint8_t* json, int64_t json_len,
                               int64_t* status_out /* [0]=status [1]=aux */)
    int64_t  jsonpathmojo_count(void* handle, int32_t which)
    void     jsonpathmojo_copy(void* handle, uint8_t* kinds, int64_t* vals,
                               int64_t* path_off, int64_t* str_off,
                               uint8_t* str_buf)
    void     jsonpathmojo_destroy(void* handle)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/jsonpath/src/jsonpathmojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "JSONPATH_MOJO_NATIVE_LIB"
_ENV_DISABLE = "JSONPATH_MOJO_DISABLE_NATIVE"

# find() status codes, mirrored from the kernel.
ST_OK = 0
ST_PARSE = 1
ST_TE_INT = 2  # TypeError from int() coercion; aux = value tag
ST_TE_ORD = 3  # TypeError from ordering comparison; aux = op*64 + ltag*8 + rtag
ST_INDEXERROR = 4  # aux = 0 list, 1 string
ST_KEYERROR = 5  # aux = index
ST_VE_SLICE = 6
ST_NOTIMPL = 7
ST_FALLBACK = 8  # out of the native domain; the wrapper must use the fallback
ST_TE_LEN = 9  # "object of type 'X' has no len()"; aux = tag

# Output step kinds, mirrored from the kernel.
OUT_FIELD = 0
OUT_INDEX = 1
OUT_DICTPOS = 2
OUT_SELFIDX = 3

# Value tags for error message composition (mirror the kernel's TAG_*).
_TAG_NAMES = {
    0: "NoneType",
    1: "bool",
    2: "bool",
    3: "int",
    4: "float",
    5: "str",
    6: "list",
    7: "dict",
}
_OP_CHARS = {0: "==", 1: "!=", 2: "<", 3: "<=", 4: ">", 5: ">="}


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native jsonpathmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libjsonpathmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libjsonpathmojo.so"
    if sys.platform.startswith("win"):
        return "jsonpathmojo.dll"  # no Mojo toolchain builds this today
    return "libjsonpathmojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "jsonpath", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.jsonpathmojo_abi_version.argtypes = []
    lib.jsonpathmojo_abi_version.restype = ctypes.c_int32
    lib.jsonpathmojo_find.argtypes = [u8p, ctypes.c_int64, u8p, ctypes.c_int64, i64p]
    lib.jsonpathmojo_find.restype = ctypes.c_void_p
    lib.jsonpathmojo_count.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.jsonpathmojo_count.restype = ctypes.c_int64
    lib.jsonpathmojo_copy.argtypes = [ctypes.c_void_p, u8p, i64p, i64p, i64p, u8p]
    lib.jsonpathmojo_copy.restype = None
    lib.jsonpathmojo_destroy.argtypes = [ctypes.c_void_p]
    lib.jsonpathmojo_destroy.restype = None


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
                    abi = int(lib.jsonpathmojo_abi_version())
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
    info["abi_version_native"] = int(lib.jsonpathmojo_abi_version())
    return info


def find_native(
    expr_bytes: bytes, json_bytes: bytes
) -> tuple[int, int, list[int], list[int], list[int], list[str]]:
    """Run one native query.

    Returns ``(status, aux, kinds, vals, path_off, strings)``; on status != 0
    the last four are empty. The caller interprets status (raising the
    matching Python exception or falling back).
    """
    lib = _load()  # raises NativeUnavailable
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)

    def _u8buf(data: bytes) -> ctypes.Array:
        if not data:
            return (ctypes.c_uint8 * 1)()
        return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)

    expr_buf = _u8buf(expr_bytes)
    json_buf = _u8buf(json_bytes)
    status_out = (ctypes.c_int64 * 2)()
    handle = lib.jsonpathmojo_find(
        expr_buf, len(expr_bytes), json_buf, len(json_bytes), status_out
    )
    if not handle:
        raise NativeUnavailable("native kernel returned no result handle")
    try:
        status, aux = int(status_out[0]), int(status_out[1])
        if status != ST_OK:
            return status, aux, [], [], [], []
        n_matches = lib.jsonpathmojo_count(handle, 0)
        n_steps = lib.jsonpathmojo_count(handle, 1)
        n_strings = lib.jsonpathmojo_count(handle, 2)
        n_bytes = lib.jsonpathmojo_count(handle, 3)
        kinds = (ctypes.c_uint8 * max(n_steps, 1))()
        vals = (ctypes.c_int64 * max(n_steps, 1))()
        path_off = (ctypes.c_int64 * (n_matches + 1))()
        str_off = (ctypes.c_int64 * (n_strings + 1))()
        str_buf = (ctypes.c_uint8 * max(n_bytes, 1))()
        lib.jsonpathmojo_copy(handle, kinds, vals, path_off, str_off, str_buf)
        raw = bytes(str_buf[:n_bytes])
        strings = [
            raw[str_off[i] : str_off[i + 1]].decode("utf-8", "surrogatepass")
            for i in range(n_strings)
        ]
        return (
            status,
            aux,
            list(kinds[:n_steps]),
            list(vals[:n_steps]),
            list(path_off[: n_matches + 1]),
            strings,
        )
    finally:
        lib.jsonpathmojo_destroy(handle)


def ordering_type_error(aux: int) -> TypeError:
    """Rebuild the CPython 3.12 message for a mixed-type ordering comparison."""
    op = _OP_CHARS[(aux // 64) % 8]
    lname = _TAG_NAMES[(aux // 8) % 8]
    rname = _TAG_NAMES[aux % 8]
    return TypeError(f"'{op}' not supported between instances of '{lname}' and '{rname}'")


def int_coercion_type_error(aux: int) -> TypeError:
    name = _TAG_NAMES[aux % 8]
    return TypeError(
        "int() argument must be a string, a bytes-like object or a real number, "
        f"not '{name}'"
    )


def len_type_error(aux: int) -> TypeError:
    name = _TAG_NAMES[aux % 8]
    return TypeError(f"object of type '{name}' has no len()")
