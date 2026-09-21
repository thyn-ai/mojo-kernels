"""jinja2-mojo: faster jinja2 template *compilation*, powered by a Mojo lexer.

Same parser, same compiler, same render runtime as stock jinja2 — the
package swaps only the compile-time lexer for a native Mojo kernel
(macOS arm64 / Linux x86_64) and falls back to stock jinja2's lexer
everywhere else (including Windows and any template the kernel declines).
Render output is byte-identical to ``jinja2.Environment.from_string``
for the same context; the differential suite asserts this across the
supported corpus on both backends against the pip jinja2 oracle.

    import jinja2_mojo

    t = jinja2_mojo.compile_template("Hello {{ name }}!")
    t.render({"name": "World"})          # 'Hello World!'
    t(name="World")                      # same

    ts = jinja2_mojo.compile_batch(["{{ a }}", "{% if b %}x{% endif %}"])

Scope: compilation only. Rendering speed is unchanged (templates execute
as Python bytecode; autoescape goes through MarkupSafe's C extension).

Set JINJA2_MOJO_DISABLE_NATIVE=1 to force the stock-lexer path.
"""

from jinja2_mojo._native import backend_info, native_available
from jinja2_mojo.core import (
    CompiledTemplate,
    NativeUnavailable,
    compile_batch,
    compile_template,
)

__version__ = "0.1.2"  # x-release-please-version
__all__ = [
    "CompiledTemplate",
    "compile_template",
    "compile_batch",
    "backend_info",
    "native_available",
    "NativeUnavailable",
    "__version__",
]
