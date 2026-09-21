"""ctypes loader for the slugifymojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$SLUGIFY_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``slugify_mojo/_native/``
    3. the repository development build output ``kernels/slugify/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python implementation.
Set ``SLUGIFY_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  slugifymojo_abi_version(void)
    void*    slugifymojo_table_create(const uint8_t* blob, int64_t blob_len)
    int32_t  slugifymojo_transform(void* handle, const uint8_t* in,
                                   int64_t in_len, uint8_t* out,
                                   int64_t out_cap, int64_t* out_len,
                                   uint32_t flags)
    int32_t  slugifymojo_transform_batch(void* handle, const uint8_t* in,
                                         const int64_t* in_offsets,
                                         int64_t n_items, uint8_t* out,
                                         int64_t* out_offsets,
                                         int64_t out_cap, uint32_t flags)
    void     slugifymojo_table_destroy(void* handle)

``slugifymojo_transform`` returns 0 on success, 1 when ``out_cap`` is too
small (``out_len`` then holds the required size), 2 on invalid arguments.
The kernel copies the blob during ``create``; the Python side may release it
afterwards.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/slugify/src/slugifymojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall
# back.
ABI_VERSION = 1

# transform() flags, mirrored from the kernel. Stages always run in the
# kernel's fixed order: quote folding, transliterate, named entities,
# decimal entities, hex entities, ASCII filter.
F_TRANS = 1
F_ENT_NAMED = 2
F_ENT_DECIMAL = 4
F_ENT_HEX = 8
F_NUMERIC_LEGACY = 16
F_FILTER = 32
F_QUOTE_DASH = 64
F_LOWER = 128

_ENV_LIB = "SLUGIFY_MOJO_NATIVE_LIB"
_ENV_DISABLE = "SLUGIFY_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native slugifymojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libslugifymojo.dylib"
    if sys.platform.startswith("linux"):
        return "libslugifymojo.so"
    if sys.platform.startswith("win"):
        return "slugifymojo.dll"  # no Mojo toolchain builds this today
    return "libslugifymojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "slugify", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.slugifymojo_abi_version.argtypes = []
    lib.slugifymojo_abi_version.restype = ctypes.c_int32
    lib.slugifymojo_table_create.argtypes = [u8p, ctypes.c_int64]
    lib.slugifymojo_table_create.restype = ctypes.c_void_p
    lib.slugifymojo_transform.argtypes = [
        ctypes.c_void_p,  # handle
        u8p,  # in bytes
        ctypes.c_int64,  # in_len
        u8p,  # out bytes
        ctypes.c_int64,  # out_cap
        i64p,  # out_len
        ctypes.c_uint32,  # flags
    ]
    lib.slugifymojo_transform.restype = ctypes.c_int32
    lib.slugifymojo_transform_batch.argtypes = [
        ctypes.c_void_p,  # handle
        u8p,  # in bytes (packed)
        i64p,  # in_offsets[n+1]
        ctypes.c_int64,  # n_items
        u8p,  # out bytes (packed)
        i64p,  # out_offsets[n+1]
        ctypes.c_int64,  # out_cap
        ctypes.POINTER(ctypes.c_uint32),  # item_flags[n]
    ]
    lib.slugifymojo_transform_batch.restype = ctypes.c_int32
    lib.slugifymojo_table_destroy.argtypes = [ctypes.c_void_p]
    lib.slugifymojo_table_destroy.restype = None


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
                    abi = int(lib.slugifymojo_abi_version())
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
    """True if the native kernel can transform right now. Never raises."""
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
    info["abi_version_native"] = int(lib.slugifymojo_abi_version())
    return info


def _as_u8_buffer(data: bytes) -> tuple[ctypes.POINTER(ctypes.c_uint8), object]:
    """Pin a bytes object behind a uint8 pointer for the duration of a call."""
    view = ctypes.c_char_p(data)
    return ctypes.cast(view, ctypes.POINTER(ctypes.c_uint8)), view


class NativeTable:
    """Owned handle to the native table store. Not thread-safe to close twice."""

    def __init__(self, blob: bytes) -> None:
        lib = _load()  # raises NativeUnavailable
        ptr, _pin = _as_u8_buffer(blob)
        handle = lib.slugifymojo_table_create(ptr, ctypes.c_int64(len(blob)))
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the table blob (malformed or version "
                "mismatch); falling back to the pure-Python reference"
            )
        # The kernel copied every byte of the blob; the pin may be released.
        self._lib = lib
        self._handle = handle
        # Hot-path caches: CDLL attribute lookup rebuilds a ForeignFunction
        # on every access, so bind the callables once.
        self._transform_fn = lib.slugifymojo_transform
        self._batch_fn = lib.slugifymojo_transform_batch

    def transform(self, data: bytes, flags: int) -> bytes:
        """Run the kernel stages selected by `flags` over one byte string."""
        if self._handle is None:
            raise NativeUnavailable("native table is closed")
        in_ptr, _pin = _as_u8_buffer(data)
        cap = max(64, len(data) * 8 + 64)
        for _ in range(2):  # one exact-size retry after a capacity miss
            out = (ctypes.c_uint8 * cap)()
            out_len = ctypes.c_int64(0)
            rc = self._transform_fn(
                self._handle,
                in_ptr,
                ctypes.c_int64(len(data)),
                out,
                ctypes.c_int64(cap),
                ctypes.byref(out_len),
                ctypes.c_uint32(flags),
            )
            if rc == 0:
                return ctypes.string_at(out, out_len.value)
            if rc == 1:
                cap = int(out_len.value)
                continue
            raise NativeUnavailable(f"native transform failed with status {rc}")
        raise NativeUnavailable("native transform did not converge on a buffer size")

    def transform_batch(self, items: list[bytes], item_flags: list[int]) -> list[bytes]:
        """Run per-item flag-selected stages over a batch in one FFI call."""
        if self._handle is None:
            raise NativeUnavailable("native table is closed")
        n = len(items)
        if n == 0:
            return []
        if len(item_flags) != n:
            raise ValueError("item_flags must have one entry per item")
        offsets = [0]
        for item in items:
            offsets.append(offsets[-1] + len(item))
        packed = b"".join(items)
        in_ptr, _pin = _as_u8_buffer(packed)
        i64 = ctypes.c_int64 * (n + 1)
        in_off = i64(*offsets)
        flags_arr = (ctypes.c_uint32 * n)(*item_flags)
        total_in = offsets[-1]
        cap = max(64, total_in * 8 + 64)
        for _ in range(2):
            out = (ctypes.c_uint8 * cap)()
            out_off = i64()
            rc = self._batch_fn(
                self._handle,
                in_ptr,
                in_off,
                ctypes.c_int64(n),
                out,
                out_off,
                ctypes.c_int64(cap),
                flags_arr,
            )
            if rc == 0:
                return [
                    ctypes.string_at(
                        ctypes.byref(out, out_off[i]), out_off[i + 1] - out_off[i]
                    )
                    for i in range(n)
                ]
            if rc == 1:
                cap = int(out_off[n])
                continue
            raise NativeUnavailable(f"native batch transform failed with status {rc}")
        raise NativeUnavailable("native batch transform did not converge on a buffer size")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.slugifymojo_table_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
