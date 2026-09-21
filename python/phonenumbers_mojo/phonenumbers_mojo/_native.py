"""ctypes loader for the phonenumbersmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$PHONENUMBERS_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``phonenumbers_mojo/_native/``
    3. the repository development build output ``kernels/phonenumbers/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the pure-Python engine. Set
``PHONENUMBERS_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  phonenumbersmojo_abi_version(void)
    void*    phonenumbersmojo_metadata_create(const uint8_t* blob, int64_t blob_len)
    int32_t  phonenumbersmojo_parse(void* handle, const uint8_t* text, int64_t text_len,
                                    int32_t region_idx, PNMOut* out)
    int32_t  phonenumbersmojo_validate(void* handle, int32_t cc,
                                       const uint8_t* nsn, int64_t nsn_len,
                                       int32_t* out_flags)   // bit0 valid, bit1 possible
    int32_t  phonenumbersmojo_format(void* handle, int32_t cc,
                                     const uint8_t* nsn, int64_t nsn_len,
                                     const uint8_t* ext, int64_t ext_len,
                                     int32_t fmt, uint8_t* out, int64_t out_cap)
    int32_t  phonenumbersmojo_validate_batch(void* handle, const uint8_t* packed,
                                             const int64_t* offsets, const int32_t* region_idx,
                                             int64_t n, uint8_t* out_valid)
    void     phonenumbersmojo_metadata_destroy(void* handle)

PNMOut is a fixed-size POD (see _PNMOut below): parse status, error_type,
country code, leading-zero count, valid/possible flags and inline string
buffers for nsn/ext. Formatting goes through phonenumbersmojo_format, which
returns the byte length written (or the required length when the buffer is
too small, reported via the PNM_FMT_TRUNC status and retried with a bigger
buffer by the caller).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from phonenumbers_mojo import _fallback as fb

# Must equal ABI_VERSION in kernels/phonenumbers/src/phonenumbersmojo.mojo.
ABI_VERSION = 1

_ENV_LIB = "PHONENUMBERS_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PHONENUMBERS_MOJO_DISABLE_NATIVE"

NSN_CAP = 24
EXT_CAP = 24


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native phonenumbersmojo kernel could not be found, loaded, or verified."""


class ParseFailure(Exception):
    """The kernel rejected the input with a parse error (error_type, message)."""

    def __init__(self, error_type: int, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class _PNMOut(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),  # 0 ok, 1 parse error, 2 kernel overflow
        ("error_type", ctypes.c_int32),
        ("cc", ctypes.c_int32),
        ("ccs", ctypes.c_int32),
        ("leading_zeros", ctypes.c_int32),
        ("valid", ctypes.c_int32),
        ("possible", ctypes.c_int32),
        ("nsn_len", ctypes.c_int32),
        ("ext_len", ctypes.c_int32),
        ("nsn", ctypes.c_char * NSN_CAP),
        ("ext", ctypes.c_char * EXT_CAP),
    ]


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libphonenumbersmojo.dylib"
    if sys.platform.startswith("linux"):
        return "libphonenumbersmojo.so"
    if sys.platform.startswith("win"):
        return "phonenumbersmojo.dll"  # no Mojo toolchain builds this today
    return "libphonenumbersmojo.so"


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
                    here, "..", "..", "..", "kernels", "phonenumbers", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.phonenumbersmojo_abi_version.argtypes = []
    lib.phonenumbersmojo_abi_version.restype = ctypes.c_int32
    lib.phonenumbersmojo_metadata_create.argtypes = [u8p, ctypes.c_int64]
    lib.phonenumbersmojo_metadata_create.restype = ctypes.c_void_p
    lib.phonenumbersmojo_parse.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(_PNMOut),
    ]
    lib.phonenumbersmojo_parse.restype = ctypes.c_int32
    lib.phonenumbersmojo_validate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int64,
        i32p,
    ]
    lib.phonenumbersmojo_validate.restype = ctypes.c_int32
    lib.phonenumbersmojo_format.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int32,
        u8p,
        ctypes.c_int64,
    ]
    lib.phonenumbersmojo_format.restype = ctypes.c_int32
    lib.phonenumbersmojo_validate_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        i64p,
        i32p,
        ctypes.c_int64,
        u8p,
        i32p,
    ]
    lib.phonenumbersmojo_validate_batch.restype = ctypes.c_int32
    lib.phonenumbersmojo_metadata_destroy.argtypes = [ctypes.c_void_p]
    lib.phonenumbersmojo_metadata_destroy.restype = None


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
                    abi = int(lib.phonenumbersmojo_abi_version())
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
    """True if the native kernel can answer right now. Never raises."""
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
    info["abi_version_native"] = int(lib.phonenumbersmojo_abi_version())
    return info


def _as_u8_buffer(data: bytes) -> tuple[ctypes.POINTER(ctypes.c_uint8), object]:
    """Pin a bytes object behind a uint8 pointer for the duration of a call."""
    view = ctypes.c_char_p(data)
    return ctypes.cast(view, ctypes.POINTER(ctypes.c_uint8)), view


def _as_u8(data: bytes) -> tuple[ctypes.c_char_p, object]:
    """Cheaper pin: c_char_p over the bytes' internal buffer (zero-copy)."""
    return ctypes.c_char_p(data), data


# Parse error messages must match the oracle; the kernel returns the
# error_type and a message-variant selector in the `ccs` detail field.
_ERROR_MESSAGES = {
    (0, 1): "Could not interpret numbers after plus-sign.",
    (0, 5): "Country calling code supplied was not recognised.",
    (0, -1): "Missing or invalid default region.",
    (1, 0): "The string supplied did not seem to be a phone number.",
    (1, -3): "The phone-context value is invalid",
    (1, -4): "The phone number supplied was None.",
    (2, 0): "Phone number had an IDD, but after this was not long enough to be a viable phone number.",
    (3, 0): "The string supplied is too short to be a phone number.",
    (4, 0): "The string supplied is too long to be a phone number.",
    (4, -2): "The string supplied was too long to parse.",
}


class NativeStore:
    """Owned handle to the native metadata tables. Not thread-safe to close twice."""

    def __init__(self, blob: bytes) -> None:
        lib = _load()  # raises NativeUnavailable
        ptr, _pin = _as_u8_buffer(blob)
        handle = lib.phonenumbersmojo_metadata_create(ptr, ctypes.c_int64(len(blob)))
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the metadata blob (malformed or "
                "version mismatch); falling back to the pure-Python engine"
            )
        # The kernel copied every byte of the blob; the pin may be released.
        self._lib = lib
        self._handle = handle

    def parse(self, text: str, region_idx: int) -> fb.Parsed | None:
        """Parse via the kernel; None when the input is outside the kernel's
        ASCII scope and must be handled by the pure-Python engine."""
        if self._handle is None:
            raise NativeUnavailable("native store is closed")
        data = text.encode("utf-8")
        ptr, _pin = _as_u8(data)
        out = _PNMOut()
        rc = self._lib.phonenumbersmojo_parse(
            self._handle, ptr, ctypes.c_int64(len(data)), ctypes.c_int32(region_idx),
            ctypes.byref(out),
        )
        if rc != 0 or out.status == 2:
            raise NativeUnavailable(f"native parse failed with status {rc or out.status}")
        if out.status == 3:
            return None  # non-ASCII input: caller routes to the fallback engine
        if out.status == 1:
            msg = _ERROR_MESSAGES.get((out.error_type, out.ccs))
            if msg is None:
                msg = _ERROR_MESSAGES[(out.error_type, 0)]
            raise ParseFailure(out.error_type, msg)
        return fb.Parsed(
            cc=out.cc,
            nsn=out.nsn[: out.nsn_len].decode("ascii"),
            ext=out.ext[: out.ext_len].decode("ascii") if out.ext_len > 0 else None,
            ccs=out.ccs,
        )

    def validate(self, cc: int, nsn: str) -> tuple[bool, bool]:
        """Returns (valid, possible)."""
        if self._handle is None:
            raise NativeUnavailable("native store is closed")
        data = nsn.encode("ascii")
        ptr, _pin = _as_u8(data)
        flags = ctypes.c_int32(0)
        rc = self._lib.phonenumbersmojo_validate(
            self._handle, ctypes.c_int32(cc), ptr, ctypes.c_int64(len(data)),
            ctypes.byref(flags),
        )
        if rc != 0:
            raise NativeUnavailable(f"native validate failed with status {rc}")
        return bool(flags.value & 1), bool(flags.value & 2)

    def format(self, cc: int, nsn: str, ext: str | None, fmt: int) -> str:
        if self._handle is None:
            raise NativeUnavailable("native store is closed")
        ndata = nsn.encode("ascii")
        edata = (ext or "").encode("ascii")
        nptr, _npin = _as_u8(ndata)
        eptr, _epin = _as_u8(edata)
        cap = 128
        for _ in range(3):
            buf = (ctypes.c_uint8 * cap)()
            rc = self._lib.phonenumbersmojo_format(
                self._handle, ctypes.c_int32(cc), nptr, ctypes.c_int64(len(ndata)),
                eptr, ctypes.c_int64(len(edata)), ctypes.c_int32(fmt),
                buf, ctypes.c_int64(cap),
            )
            if rc >= 0:
                return bytes(buf[:rc]).decode("ascii")
            if rc == -2:  # buffer too small: kernel reports nothing; grow and retry
                cap *= 4
                continue
            raise NativeUnavailable(f"native format failed with status {rc}")
        raise NativeUnavailable("native format buffer keeps growing")

    def validate_column(self, values: list[str], region_idx: int) -> list[bool | None]:
        """One FFI call for the whole column. Rows the kernel cannot serve
        (non-ASCII input) come back as None and are routed by the caller to
        the pure-Python engine."""
        if self._handle is None:
            raise NativeUnavailable("native store is closed")
        n = len(values)
        if n == 0:
            return []
        encoded = [v.encode("utf-8") for v in values]
        packed = b"".join(encoded)
        offsets = [0]
        for e in encoded:
            offsets.append(offsets[-1] + len(e))
        off_arr = (ctypes.c_int64 * (n + 1))(*offsets)
        reg_arr = (ctypes.c_int32 * n)(*([region_idx] * n))
        out_arr = (ctypes.c_uint8 * n)()
        status_arr = (ctypes.c_int32 * n)()
        pptr, _pin = _as_u8(packed)
        rc = self._lib.phonenumbersmojo_validate_batch(
            self._handle, pptr, off_arr, reg_arr, ctypes.c_int64(n), out_arr, status_arr
        )
        if rc != 0:
            raise NativeUnavailable(f"native batch validate failed with status {rc}")
        out: list[bool | None] = []
        for i in range(n):
            if status_arr[i] == 0:
                out.append(bool(out_arr[i]))
            elif status_arr[i] == 1:
                out.append(False)
            else:  # 3: non-ASCII, out of kernel scope
                out.append(None)
        return out

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.phonenumbersmojo_metadata_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
