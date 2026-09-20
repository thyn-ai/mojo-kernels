"""ctypes loader for the langdetectmojo native kernel, with an ABI handshake.

Resolution order:

    1. ``$LANGDETECT_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``langdetect_mojo/_native/``
    3. the repository development build output ``kernels/langdetect/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python implementation.
Set ``LANGDETECT_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  langdetectmojo_abi_version(void)
    void*    langdetectmojo_profiles_create(const uint8_t* blob, int64_t blob_len)
    int32_t  langdetectmojo_detect(void* handle, const uint8_t* text,
                                   int64_t text_len, uint64_t seed,
                                   double* out_probs, int64_t out_cap,
                                   uint32_t flags)
    void     langdetectmojo_profiles_destroy(void* handle)

``langdetectmojo_detect`` returns 0 on success, 1 when the text has no usable
features (the reference raises 'No features in text.'), 2 on invalid
arguments. The kernel copies the blob during ``create``; the Python side may
release it afterwards.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/langdetect/src/langdetectmojo.mojo. A
# mismatch means the installed wheel and the resolved shared library
# disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "LANGDETECT_MOJO_NATIVE_LIB"
_ENV_DISABLE = "LANGDETECT_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native langdetectmojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "liblangdetectmojo.dylib"
    if sys.platform.startswith("linux"):
        return "liblangdetectmojo.so"
    if sys.platform.startswith("win"):
        return "langdetectmojo.dll"  # no Mojo toolchain builds this today
    return "liblangdetectmojo.so"


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
                    here, "..", "..", "..", "kernels", "langdetect", "build", _lib_basename()
                )
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.langdetectmojo_abi_version.argtypes = []
    lib.langdetectmojo_abi_version.restype = ctypes.c_int32
    lib.langdetectmojo_profiles_create.argtypes = [u8p, ctypes.c_int64]
    lib.langdetectmojo_profiles_create.restype = ctypes.c_void_p
    lib.langdetectmojo_detect.argtypes = [
        ctypes.c_void_p,  # handle
        u8p,  # text bytes
        ctypes.c_int64,  # text_len
        ctypes.c_uint64,  # seed
        f64p,  # out_probs
        ctypes.c_int64,  # out_cap
        ctypes.c_uint32,  # flags
    ]
    lib.langdetectmojo_detect.restype = ctypes.c_int32
    lib.langdetectmojo_profiles_destroy.argtypes = [ctypes.c_void_p]
    lib.langdetectmojo_profiles_destroy.restype = None


# detect() flag bit 0: summation flavor for probability normalization.
# CPython 3.12 changed sum() for floats to Neumaier-compensated summation;
# older interpreters sum left-to-right. Match the running interpreter.
_SUM_FLAG = 0 if sys.version_info >= (3, 12) else 1

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
                    abi = int(lib.langdetectmojo_abi_version())
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
    """True if the native kernel can detect right now. Never raises."""
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
    info["abi_version_native"] = int(lib.langdetectmojo_abi_version())
    return info


def _as_u8_buffer(data: bytes) -> tuple[ctypes.POINTER(ctypes.c_uint8), object]:
    """Pin a bytes object behind a uint8 pointer for the duration of a call."""
    view = ctypes.c_char_p(data)
    return ctypes.cast(view, ctypes.POINTER(ctypes.c_uint8)), view


class NativeModel:
    """Owned handle to the native profile store. Not thread-safe to close twice."""

    def __init__(self, blob: bytes, n_langs: int) -> None:
        lib = _load()  # raises NativeUnavailable
        ptr, _pin = _as_u8_buffer(blob)
        handle = lib.langdetectmojo_profiles_create(ptr, ctypes.c_int64(len(blob)))
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the profile blob (malformed or "
                "version mismatch); falling back to the pure-Python reference"
            )
        # The kernel copied every byte of the blob; the pin may be released.
        self._lib = lib
        self._handle = handle
        self._n_langs = int(n_langs)

    @property
    def n_langs(self) -> int:
        return self._n_langs

    def detect_block(self, text: str, seed: int) -> tuple[int, list[float] | None]:
        """Run the full native pipeline. Returns (status, langprob|None):
        status 0 = success, 1 = no features in text, 2 = invalid argument."""
        if self._handle is None:
            raise NativeUnavailable("native model is closed")
        data = text.encode("utf-8", "surrogatepass")
        ptr, _pin = _as_u8_buffer(data)
        out = (ctypes.c_double * self._n_langs)()
        rc = self._lib.langdetectmojo_detect(
            self._handle,
            ptr,
            ctypes.c_int64(len(data)),
            ctypes.c_uint64(seed),
            out,
            ctypes.c_int64(self._n_langs),
            ctypes.c_uint32(_SUM_FLAG),
        )
        if rc != 0:
            return int(rc), None
        return 0, [out[i] for i in range(self._n_langs)]

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.langdetectmojo_profiles_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
