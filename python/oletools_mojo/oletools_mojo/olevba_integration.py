"""olevba integration: route oletools' own VBA extraction through oletools_mojo.

`oletools.olevba.VBA_Parser` decompresses every macro module stream and the
project `dir` stream through the module-level `decompress_stream` function in
`oletools.olevba`. Because those call sites resolve the function as a module
global at call time, monkeypatching that one attribute accelerates all of
oletools' VBA source extraction (olevba, olevba3, mraptor, VBA_Parser
embeddings) without forking it::

    import oletools_mojo.olevba_integration as integration
    integration.install()          # oletools.olevba now uses oletools_mojo

    from oletools.olevba import VBA_Parser
    with VBA_Parser("sample.xlsm") as parser:
        for (_, _, filename, code) in parser.extract_macros():
            print(filename)

    integration.uninstall()        # restore the stock oletools behaviour

`install()` imports oletools lazily; oletools is an optional, user-side
dependency — it is never imported (or required) by `oletools_mojo` itself.
The patched-in replacement is `oletools_mojo.decompress_stream`, which is
bit-exact with the stock function (including exception types), so behaviour
is unchanged apart from speed. `install()` is idempotent; `uninstall()`
restores whatever attribute was in place before the first `install()`.
"""

from __future__ import annotations

import oletools_mojo

__all__ = ["install", "uninstall", "is_installed"]

_ORIGINAL_ATTR: str = "decompress_stream"
_saved = None  # (module, original attribute value) of the first install()


def install() -> None:
    """Monkeypatch `oletools.olevba.decompress_stream` to oletools_mojo's.

    Raises ImportError if oletools is not installed in the environment.
    """
    global _saved
    from oletools import olevba  # user-side optional dependency

    if _saved is None:
        _saved = (olevba, getattr(olevba, _ORIGINAL_ATTR))
    setattr(olevba, _ORIGINAL_ATTR, oletools_mojo.decompress_stream)


def uninstall() -> None:
    """Restore the attribute saved by the first `install()` (if any)."""
    global _saved
    if _saved is None:
        return
    module, original = _saved
    setattr(module, _ORIGINAL_ATTR, original)
    _saved = None


def is_installed() -> bool:
    """True while an `install()` patch is active."""
    return _saved is not None
