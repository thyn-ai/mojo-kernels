"""ctypes loader for the rlemojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$PYDICOM_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``pydicom_mojo/_native/``
    3. the repository development build output ``kernels/pydicom-rle/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``PYDICOM_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  rlemojo_abi_version(void)
    int64_t  rlemojo_decode_segment(const uint8_t* data, int64_t data_len,
                                    uint8_t* out_buf, int64_t out_cap)
    int64_t  rlemojo_encode_segment(const uint8_t* data, int64_t data_len,
                                    int64_t columns, uint8_t* out_buf,
                                    int64_t out_cap)

Both segment functions write into a caller-provided buffer and return the
number of bytes written. Negative return codes from the kernel: -1 invalid
parameters, -3 output buffer too small. The decoded size of a segment is at
most 64 * data_len; the encoded size is at most 2 * data_len + 2.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from pydicom_mojo._reference import RleCodecError

# Must equal ABI_VERSION in kernels/pydicom-rle/src/rlemojo.mojo. A mismatch
# means the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "PYDICOM_MOJO_NATIVE_LIB"
_ENV_DISABLE = "PYDICOM_MOJO_DISABLE_NATIVE"

# Hard bounds on segment codec expansion, proven from the packet geometry:
# a 2-byte replicate packet expands to at most 128 bytes (64x); every other
# packet shrinks or stays even. Encoding costs at most one header byte per
# input byte plus the one-byte even-length pad.
_DECODE_BOUND = 64
_ENCODE_BOUND = 2


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native rlemojo kernel could not be found, loaded, or verified."""


class _OutputTooSmall(Exception):
    """Internal: the caller-provided output buffer was too small (kernel -3)."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "librlemojo.dylib"
    if sys.platform.startswith("linux"):
        return "librlemojo.so"
    if sys.platform.startswith("win"):
        return "rlemojo.dll"  # no Mojo toolchain builds this today
    return "librlemojo.so"


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
                    here, "..", "..", "..", "kernels", "pydicom-rle", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    lib.rlemojo_abi_version.argtypes = []
    lib.rlemojo_abi_version.restype = ctypes.c_int32
    lib.rlemojo_decode_segment.argtypes = [
        ctypes.c_char_p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_void_p,  # out_buf
        ctypes.c_int64,  # out_cap
    ]
    lib.rlemojo_decode_segment.restype = ctypes.c_int64
    lib.rlemojo_encode_segment.argtypes = [
        ctypes.c_char_p,  # data
        ctypes.c_int64,  # data_len
        ctypes.c_int64,  # columns
        ctypes.c_void_p,  # out_buf
        ctypes.c_int64,  # out_cap
    ]
    lib.rlemojo_encode_segment.restype = ctypes.c_int64


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
                    abi = int(lib.rlemojo_abi_version())
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
    """True if the native kernel can code segments right now. Never raises."""
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
    info["abi_version_native"] = int(lib.rlemojo_abi_version())
    return info


def decode_segment_native(data: bytes, out_cap: int) -> bytes:
    """Decode one RLE segment via the native kernel into an `out_cap` buffer.

    Raises _OutputTooSmall when the decoded bytes exceed `out_cap`; the
    caller retries with the hard bound (64 * len(data)) when it must
    reproduce the oracle's over-long-segment behaviour exactly.
    """
    lib = _load()  # raises NativeUnavailable
    out = ctypes.create_string_buffer(max(out_cap, 1))
    rc = lib.rlemojo_decode_segment(data, len(data), ctypes.cast(out, ctypes.c_void_p), out_cap)
    if rc == -3:
        raise _OutputTooSmall
    if rc < 0:
        raise RleCodecError(f"native kernel rejected the segment (status {rc})")
    return out.raw[:rc]


def decode_segment_bounded_native(data: bytes) -> bytes:
    """Decode one RLE segment of unknown decoded size via the native kernel.

    Uses the proven hard bound 64 * len(data), so no retry is ever needed.
    """
    return decode_segment_native(data, _DECODE_BOUND * len(data))


def encode_segment_native(data: bytes, columns: int) -> bytes:
    """Encode one byte plane into an RLE segment via the native kernel.

    `columns` must be >= 1 (the wrapper enforces the oracle's `columns`
    edge semantics before calling). The output buffer uses the proven hard
    bound 2 * len(data) + 2, so the kernel never reports -3 here.
    """
    lib = _load()  # raises NativeUnavailable
    cap = _ENCODE_BOUND * len(data) + 2
    out = ctypes.create_string_buffer(max(cap, 1))
    rc = lib.rlemojo_encode_segment(
        data, len(data), columns, ctypes.cast(out, ctypes.c_void_p), cap
    )
    if rc < 0:
        raise RleCodecError(f"native kernel rejected the segment (status {rc})")
    return out.raw[:rc]
