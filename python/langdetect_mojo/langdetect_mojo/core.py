"""Drop-in language detection, API-compatible with the `langdetect` package.

`detect(text)` and `detect_langs(text)` return the same strings as their
`langdetect` counterparts for a given seed: detection runs the reference
algorithm (55-language profiles loaded at runtime from the installed
`langdetect` package) on the native Mojo kernel where available, and on the
vendored pure-Python fallback everywhere else. Both backends run the full
pipeline themselves; the differential suite asserts identical outputs on both.

Determinism: the reference is only deterministic once
`langdetect.detector_factory.DetectorFactory.seed` is set (its default is a
non-deterministic stream). This package is deterministic by default: the
module seed defaults to 0 and every call re-seeds, exactly like the reference
does for each fresh Detector. Use `set_seed(n)` to change it.
"""

from __future__ import annotations

import os
import threading

from langdetect_mojo import _data, _fallback
from langdetect_mojo._native import NativeModel, NativeUnavailable

__all__ = [
    "Language",
    "LangDetectError",
    "detect",
    "detect_langs",
    "set_seed",
    "get_seed",
    "backend",
]

PROB_THRESHOLD = 0.1
UNKNOWN_LANG = "unknown"
CANT_DETECT_ERROR = 5  # reference ErrorCode.CantDetectError

_MAX_SEED = (1 << 64) - 1


class LangDetectError(Exception):
    """Mirror of langdetect.LangDetectException: str(exc) is the message,
    .code the numeric error code (5 = CantDetectError)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code

    def get_code(self) -> int:
        return self.code


class Language:
    """Mirror of langdetect.language.Language: repr is 'lang:prob' and
    ordering compares probabilities, so sorted/str output matches."""

    __slots__ = ("lang", "prob")

    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob

    def __repr__(self) -> str:
        if self.lang is None:
            return ""
        return "%s:%s" % (self.lang, self.prob)

    def __lt__(self, other) -> bool:
        return self.prob < other.prob


_seed = 0  # deterministic by default (see module docstring)


def set_seed(seed: int) -> None:
    """Set the module detection seed (mirrors DetectorFactory.seed = n).

    Only integer seeds in [-(2**64-1), 2**64-1] are supported; a negative
    seed behaves like its absolute value, as CPython's random.Random does.
    """
    _normalize_seed(seed)  # validate first; never leave a bad seed behind
    globals()["_seed"] = abs(int(seed))


def get_seed() -> int:
    return _seed


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, not {type(seed).__name__}")
    value = abs(seed)
    if value > _MAX_SEED:
        raise ValueError(f"seed magnitude exceeds 2**64-1: {seed}")
    return value


_LOCK = threading.Lock()
_STATE: tuple[_data.Profiles, NativeModel | None] | None = None


def _get_state() -> tuple[_data.Profiles, NativeModel | None]:
    """Load profiles once; build the native model when the kernel is up."""
    global _STATE
    if _STATE is not None:
        return _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE
        profiles = _data.get_profiles()
        model: NativeModel | None = None
        try:
            model = NativeModel(profiles.blob(), profiles.n_langs)
        except NativeUnavailable:
            model = None
        _STATE = (profiles, model)
        return _STATE


def _backend() -> tuple[_data.Profiles, NativeModel | None]:
    profiles, model = _get_state()
    # Forcing the fallback must work even after a native model was cached
    # (the differential suite runs the whole suite both ways).
    if os.environ.get("LANGDETECT_MOJO_DISABLE_NATIVE") == "1":
        return profiles, None
    return profiles, model


def backend() -> str:
    """Which engine serves detection right now: "native" or "fallback"."""
    _, model = _backend()
    return "native" if model is not None else "fallback"


def _detect_block(text: str) -> tuple[_data.Profiles, list[float]]:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")
    profiles, model = _backend()
    seed = _normalize_seed(_seed)
    if model is not None:
        rc, probs = model.detect_block(text, seed)
    else:
        rc, probs = _fallback.detect_block(profiles, text, seed)
    if rc != 0 or probs is None:
        raise LangDetectError(CANT_DETECT_ERROR, "No features in text.")
    return profiles, probs


def _sorted_languages(profiles: _data.Profiles, probs: list[float]) -> list[Language]:
    # Reference _sort_probability: keep p > 0.1, sort by probability
    # descending; CPython's sort is stable, so ties keep langlist order.
    items = [
        (lang, p) for lang, p in zip(profiles.langlist, probs) if p > PROB_THRESHOLD
    ]
    items.sort(key=lambda kv: -kv[1])
    return [Language(lang, p) for lang, p in items]


def detect(text: str) -> str:
    """Most likely language code for `text` (same string as langdetect.detect)."""
    profiles, probs = _detect_block(text)
    langs = _sorted_languages(profiles, probs)
    return langs[0].lang if langs else UNKNOWN_LANG


def detect_langs(text: str) -> list[Language]:
    """Ranked candidate languages (same string form as langdetect.detect_langs)."""
    profiles, probs = _detect_block(text)
    return _sorted_languages(profiles, probs)
