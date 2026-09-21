"""Drop-in Markdown renderer, API-compatible with `mistune.markdown`.

`markdown(text)` renders CommonMark Markdown to HTML, byte-identical to
`mistune.markdown()` from the published mistune package (3.3.4) for the
supported scope (see the package README). Rendering runs on the native Mojo
kernel when its shared library is available (macOS arm64 / Linux x86_64
wheels) and transparently falls back to the vendored pure-Python engine
otherwise. Both engines implement the same rules and agree byte-for-byte on
every input; the differential suite asserts agreement with `mistune` on both
paths across the full CommonMark 0.31.2 spec corpus.
"""

from __future__ import annotations

from mistune_mojo import _reference
from mistune_mojo._native import NativeUnavailable, UnsupportedConstruct, render_native

__all__ = ["markdown"]


def markdown(text: str) -> str:
    """Render Markdown text to HTML, matching mistune.markdown() byte-for-byte.

    Mirrors mistune's own call contract: None and "" render as ""; any other
    non-string input fails the same way mistune fails (the engines perform
    the same string operations in the same order).
    """
    if text is None or text == "":
        return ""
    if not isinstance(text, str):
        return _reference.markdown(text)
    try:
        return render_native(text)
    except (NativeUnavailable, UnsupportedConstruct):
        return _reference.markdown(text)
