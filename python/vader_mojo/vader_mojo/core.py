"""Drop-in VADER sentiment scoring, API-compatible with the `vaderSentiment` package.

Scoring runs on the native Mojo kernel when its shared library is available
(macOS arm64 / Linux x86_64 wheels) and transparently falls back to the
vendored pure-Python implementation otherwise. Both backends consume the same
vendored lexicon tables and return raw float64 scores in reference order;
this module applies the reference's public rounding (``neg``/``neu``/``pos``
to 3 decimals, ``compound`` to 4) so the returned dicts are identical to the
reference package on every platform. The differential test suite asserts exact
dict parity with `vaderSentiment` on both backends.
"""

from __future__ import annotations

import os
import threading

from vader_mojo import _reference
from vader_mojo._native import NativeAnalyzer, NativeUnavailable

__all__ = ["SentimentIntensityAnalyzer", "polarity_scores"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_lexicon() -> dict[str, float]:
    """Parse the vendored VADER lexicon exactly as the reference does."""
    lex_dict: dict[str, float] = {}
    with open(os.path.join(_DATA_DIR, "vader_lexicon.txt"), encoding="utf-8") as f:
        lexicon_text = f.read()
    for line in lexicon_text.rstrip("\n").split("\n"):
        if not line:
            continue
        word, measure = line.strip().split("\t")[0:2]
        lex_dict[word] = float(measure)
    return lex_dict


def _load_emojis() -> dict[str, str]:
    """Parse the vendored emoji lexicon exactly as the reference does."""
    emoji_dict: dict[str, str] = {}
    with open(os.path.join(_DATA_DIR, "emoji_utf8_lexicon.txt"), encoding="utf-8") as f:
        emoji_text = f.read()
    for line in emoji_text.rstrip("\n").split("\n"):
        emoji, description = line.strip().split("\t")[0:2]
        emoji_dict[emoji] = description
    return emoji_dict


def _round_scores(raw: tuple[float, float, float, float]) -> dict[str, float]:
    """The reference's public rounding: neg/neu/pos to 3, compound to 4."""
    neg, neu, pos, compound = raw
    return {
        "neg": round(neg, 3),
        "neu": round(neu, 3),
        "pos": round(pos, 3),
        "compound": round(compound, 4),
    }


class SentimentIntensityAnalyzer:
    """Give a sentiment intensity score to sentences (drop-in replacement).

    Same constructor shape and same ``polarity_scores(text) -> dict`` contract
    as ``vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer``. The
    ``backend`` attribute reports which engine is active: ``"native"`` (Mojo
    kernel) or ``"fallback"`` (vendored pure-Python).
    """

    def __init__(self) -> None:
        self.lexicon = _load_lexicon()
        self.emojis = _load_emojis()
        try:
            self._native: NativeAnalyzer | None = NativeAnalyzer(self.lexicon, self.emojis)
        except NativeUnavailable:
            self._native = None
        self._fallback = (
            None if self._native is not None else _reference.PurePythonAnalyzer(self.lexicon, self.emojis)
        )
        self.backend = "native" if self._native is not None else "fallback"

    def polarity_scores(self, text: str) -> dict[str, float]:
        """Return a dict of sentiment scores: neg, neu, pos (proportions
        rounded to 3 decimals) and compound (normalized, rounded to 4)."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if self._native is not None:
            return _round_scores(self._native.polarity(text))
        assert self._fallback is not None
        return _round_scores(self._fallback.polarity(text))

    def close(self) -> None:
        """Release the native analyzer handle, if one is held."""
        if self._native is not None:
            self._native.close()


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_ANALYZER: SentimentIntensityAnalyzer | None = None


def _global_analyzer() -> SentimentIntensityAnalyzer:
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER is not None:
        return _GLOBAL_ANALYZER
    with _GLOBAL_LOCK:
        if _GLOBAL_ANALYZER is None:
            _GLOBAL_ANALYZER = SentimentIntensityAnalyzer()
        return _GLOBAL_ANALYZER


def polarity_scores(text: str) -> dict[str, float]:
    """Module-level convenience: score one text with a shared analyzer.

        >>> import vader_mojo
        >>> vader_mojo.polarity_scores("VADER is smart, handsome, and funny!")
        {'neg': 0.0, 'neu': 0.248, 'pos': 0.752, 'compound': 0.8439}
    """
    return _global_analyzer().polarity_scores(text)
