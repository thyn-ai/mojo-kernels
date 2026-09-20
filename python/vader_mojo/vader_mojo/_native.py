"""ctypes loader for the vadermojo native kernel, with an ABI-version handshake.

Resolution order:

    1. ``$VADER_MOJO_NATIVE_LIB`` (explicit path override, for development)
    2. the shared library bundled in this wheel under ``vader_mojo/_native/``
    3. the repository development build output ``kernels/vader/build/``

If the library cannot be found, fails to load, or reports an ABI version this
package does not understand, :class:`NativeUnavailable` is raised and the
caller falls back to the vendored pure-Python reference implementation.
Set ``VADER_MOJO_DISABLE_NATIVE=1`` to force that fallback (used by the
differential test suite).

Stable C ABI (v1)::

    int32_t  vadermojo_abi_version(void)
    void*    vadermojo_analyzer_create(const uint8_t* words_blob,
                                       const int64_t* word_offsets,
                                       const double* values, int64_t n_words,
                                       const uint32_t* emoji_keys,
                                       const uint8_t* desc_blob,
                                       const int64_t* desc_offsets,
                                       int64_t n_emojis)
    int32_t  vadermojo_polarity(void* handle, const uint8_t* text,
                                int64_t text_len, double* out4)
    void     vadermojo_analyzer_destroy(void* handle)

The kernel returns four raw (unrounded) float64 scores in reference order —
neg, neu, pos, compound. Rounding to the public 3/4-digit precision happens in
``core.py`` with Python's ``round()`` so the result is byte-identical to the
reference package on every platform.

Lexicon reachability filter: the reference implementation looks up
``token.lower()`` in its lexicon, so any key that is not equal to its own
``.lower()`` (the two non-ASCII emoticon keys ``:-Þ``/``:Þ``) can never match
and is dropped from the native tables; likewise emoji keys longer than one
code point (ZWJ sequences, skin-tone variants, flags) can never match the
reference's per-character lookup. Dropping both classes is behavior-identical,
and the differential suite asserts exact parity on corpora containing them.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

# Must equal ABI_VERSION in kernels/vader/src/vadermojo.mojo. A mismatch means
# the installed wheel and the resolved shared library disagree; fall back.
ABI_VERSION = 1

_ENV_LIB = "VADER_MOJO_NATIVE_LIB"
_ENV_DISABLE = "VADER_MOJO_DISABLE_NATIVE"


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native vadermojo kernel could not be found, loaded, or verified."""


def _lib_basename() -> str:
    if sys.platform == "darwin":
        return "libvadermojo.dylib"
    if sys.platform.startswith("linux"):
        return "libvadermojo.so"
    if sys.platform.startswith("win"):
        return "vadermojo.dll"  # no Mojo toolchain builds this today
    return "libvadermojo.so"


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
                os.path.join(here, "..", "..", "..", "kernels", "vader", "build", _lib_basename())
            ),
        )
    )
    return out


def _bind_abi(lib: ctypes.CDLL) -> None:
    i64p = ctypes.POINTER(ctypes.c_int64)
    f64p = ctypes.POINTER(ctypes.c_double)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.vadermojo_abi_version.argtypes = []
    lib.vadermojo_abi_version.restype = ctypes.c_int32
    lib.vadermojo_analyzer_create.argtypes = [
        ctypes.c_char_p,  # words_blob
        i64p,  # word_offsets[n_words+1]
        f64p,  # values[n_words]
        ctypes.c_int64,  # n_words
        u32p,  # emoji_keys[n_emojis]
        ctypes.c_char_p,  # desc_blob
        i64p,  # desc_offsets[n_emojis+1]
        ctypes.c_int64,  # n_emojis
    ]
    lib.vadermojo_analyzer_create.restype = ctypes.c_void_p
    lib.vadermojo_polarity.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, f64p]
    lib.vadermojo_polarity.restype = ctypes.c_int32
    lib.vadermojo_analyzer_destroy.argtypes = [ctypes.c_void_p]
    lib.vadermojo_analyzer_destroy.restype = None


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
                    abi = int(lib.vadermojo_abi_version())
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
    """True if the native kernel can score right now. Never raises."""
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
    info["abi_version_native"] = int(lib.vadermojo_abi_version())
    return info


def _serialize_lexicon(lexicon: dict[str, float]):
    """(blob_bytes, int64 offsets, float64 values) for reachable lexicon keys.

    Keys that can never match the reference's ``token.lower()`` lookup are
    dropped (see module docstring); the caller's dict order is preserved.
    """
    words = [w for w in lexicon if w == w.lower()]
    blob = "".join(words).encode("utf-8")
    offsets = (ctypes.c_int64 * (len(words) + 1))()
    values = (ctypes.c_double * len(words))()
    pos = 0
    for i, word in enumerate(words):
        offsets[i] = pos
        pos += len(word.encode("utf-8"))
        values[i] = lexicon[word]
    offsets[len(words)] = pos
    return blob, offsets, values, len(words)


def _serialize_emojis(emojis: dict[str, str]):
    """(uint32 keys, blob_bytes, int64 offsets) for single-code-point emojis.

    Multi-code-point keys (ZWJ sequences, skin tones, flags) are unreachable
    in the reference's per-character lookup and are dropped; this is
    behavior-identical (see module docstring).
    """
    singles = [(ord(k), v) for k, v in emojis.items() if len(k) == 1]
    blob = "".join(desc for _, desc in singles).encode("utf-8")
    keys = (ctypes.c_uint32 * len(singles))()
    offsets = (ctypes.c_int64 * (len(singles) + 1))()
    pos = 0
    for i, (cp, desc) in enumerate(singles):
        keys[i] = cp
        offsets[i] = pos
        pos += len(desc.encode("utf-8"))
    offsets[len(singles)] = pos
    return keys, blob, offsets, len(singles)


class NativeAnalyzer:
    """Owned handle to a native VADER analyzer. Not thread-safe to close twice.

    Scoring is stateless per call, so concurrent ``polarity`` calls on one
    analyzer are safe.
    """

    def __init__(self, lexicon: dict[str, float], emojis: dict[str, str]) -> None:
        lib = _load()  # raises NativeUnavailable
        words_blob, word_offsets, values, n_words = _serialize_lexicon(lexicon)
        emoji_keys, desc_blob, desc_offsets, n_emojis = _serialize_emojis(emojis)
        handle = lib.vadermojo_analyzer_create(
            words_blob,
            word_offsets,
            values,
            ctypes.c_int64(n_words),
            emoji_keys,
            desc_blob,
            desc_offsets,
            ctypes.c_int64(n_emojis),
        )
        if not handle:
            raise NativeUnavailable(
                "native kernel rejected the analyzer tables; "
                "falling back to the pure-Python reference"
            )
        # The kernel copies every buffer; the ctypes arrays above may be freed.
        self._lib = lib
        self._handle = handle

    def polarity(self, text: str) -> tuple[float, float, float, float]:
        """Raw (unrounded) neg, neu, pos, compound for one text."""
        if self._handle is None:
            raise NativeUnavailable("native analyzer is closed")
        data = text.encode("utf-8", "surrogatepass")
        out = (ctypes.c_double * 4)()
        rc = self._lib.vadermojo_polarity(self._handle, data, len(data), out)
        if rc != 0:
            raise NativeUnavailable(f"native scoring failed with status {rc}")
        return out[0], out[1], out[2], out[3]

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle and self._lib is not None:
            self._lib.vadermojo_analyzer_destroy(handle)

    def __del__(self) -> None:  # best-effort; never raise during GC
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
