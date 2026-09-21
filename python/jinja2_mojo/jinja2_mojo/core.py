"""Public API: compile jinja2 templates with a native lexer fast path.

`compile_template` mirrors `jinja2.Environment.from_string` for the compile
path: the returned object renders byte-identical output to stock jinja2 for
the same context, because everything downstream of tokenisation — parser,
compiler, and the render runtime — is stock jinja2. The native Mojo kernel
only replaces the lexer's per-character scan (the compile hot spot); for
templates outside its probed subset it declines and the stock lexer runs
instead, so every template is handled, natively or not, with identical
results. Render performance is unchanged by design (templates execute as
Python bytecode and autoescape goes through MarkupSafe's C extension); this
package accelerates *compilation* only.

Supported natively: default delimiters, variables, blocks (if/for/set/
include/import/from/macros/filter/call/do/...), filters, tests, comments,
string/number literals (Python-literal escapes, hex/octal/binary/underscored
ints), whitespace control (``-``/``+`` markers, ``trim_blocks``,
``lstrip_blocks``, ``keep_trailing_newline``), and arbitrary nesting.
Not native (transparent stock-lexer fallback): ``{% raw %}``, custom
delimiters, line statements/comments, and any construct the kernel cannot
tokenise with oracle-identical results.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import jinja2
from jinja2.lexer import TokenStream

from . import _native, _tokens
from ._native import NativeUnavailable, backend_info, native_available

__all__ = [
    "CompiledTemplate",
    "compile_template",
    "compile_batch",
    "native_available",
    "backend_info",
    "NativeUnavailable",
]

_DEFAULT_DELIMITERS = (
    ("block_start_string", "{%"),
    ("block_end_string", "%}"),
    ("variable_start_string", "{{"),
    ("variable_end_string", "}}"),
    ("comment_start_string", "{#"),
    ("comment_end_string", "#}"),
)

_TLS = threading.local()


def _set_last_backend(value: str) -> None:
    _TLS.last_backend = value


def _last_backend() -> str:
    return getattr(_TLS, "last_backend", "stock")


def _env_supports_native(env: jinja2.Environment) -> bool:
    """Native lexing only covers default delimiters, no line statements."""
    for attr, default in _DEFAULT_DELIMITERS:
        if getattr(env, attr) != default:
            return False
    if getattr(env, "line_statement_prefix", None) is not None:
        return False
    if getattr(env, "line_comment_prefix", None) is not None:
        return False
    return True


class _NativeEnvironment(jinja2.Environment):
    """A stock Environment whose lexer is swapped for the native kernel.

    Only ``_tokenize`` is overridden; parsing, compilation, optimization,
    rendering, sandboxing and every extension hook are stock jinja2, which
    is what makes render output byte-identical to the oracle. Any template
    the kernel declines falls through to the stock lexer transparently.
    """

    def _tokenize(self, source, name, filename=None, state=None):  # noqa: D102
        if _env_supports_native(self):
            try:
                tokens = _tokens.build_tokens(
                    source,
                    trim_blocks=self.trim_blocks,
                    lstrip_blocks=self.lstrip_blocks,
                    keep_trailing_newline=self.keep_trailing_newline,
                )
            except NativeUnavailable:
                tokens = None
            if tokens is not None:
                _set_last_backend("native")
                return TokenStream(iter(tokens), name, filename)
        _set_last_backend("stock")
        return super()._tokenize(source, name, filename, state)


def _freeze(value):
    """Hashable form of an environment option value; None if not hashable."""
    if isinstance(value, (str, int, float, bool, type(None), type)):
        return value
    if isinstance(value, (list, tuple)):
        parts = tuple(_freeze(v) for v in value)
        return parts if all(p is not None or v is None for p, v in zip(parts, value)) else None
    if isinstance(value, dict):
        try:
            return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
        except TypeError:
            return None
    try:
        hash(value)
    except TypeError:
        return None
    return value


class CompiledTemplate:
    """A compiled jinja2 template plus compile-path metadata.

    Wraps a stock ``jinja2.Template``; rendering goes through the stock
    runtime untouched, so output is byte-identical to
    ``jinja2.Environment.from_string(...).render(...)`` for the same context.
    """

    __slots__ = ("_template", "backend", "source")

    def __init__(self, template: jinja2.Template, *, backend: str, source: str) -> None:
        self._template = template
        #: "native" when the native lexer tokenised this template, else "stock".
        self.backend = backend
        #: The template source this object was compiled from.
        self.source = source

    @property
    def template(self) -> jinja2.Template:
        """The underlying stock ``jinja2.Template``."""
        return self._template

    @property
    def environment(self) -> jinja2.Environment:
        return self._template.environment

    def render(self, *args, **kwargs) -> str:
        """Render like ``jinja2.Template.render`` (dict or keyword context)."""
        return self._template.render(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Stream-render like ``jinja2.Template.generate``."""
        return self._template.generate(*args, **kwargs)

    def __call__(self, context: dict | None = None, **kwargs) -> str:
        if context is None:
            context = {}
        if kwargs:
            context = {**context, **kwargs}
        return self._template.render(context)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CompiledTemplate backend={self.backend!r} len(source)={len(self.source)}>"


_ENV_LOCK = threading.Lock()
_ENV_CACHE_MAX = 64
_ENV_CACHE: OrderedDict[tuple, jinja2.Environment] = OrderedDict()

_TPL_LOCK = threading.Lock()
_TPL_CACHE_MAX = 4096
_TPL_CACHE: OrderedDict[tuple, CompiledTemplate] = OrderedDict()


def _env_key(loader, env_options: dict) -> tuple | None:
    frozen = tuple(sorted((k, _freeze(v)) for k, v in env_options.items()))
    if any(v is None and env_options[k] is not None for k, v in frozen):
        return None
    loader_key = id(loader) if loader is not None else None
    return (loader_key, frozen)


def _get_env(loader, env_options: dict) -> jinja2.Environment:
    key = _env_key(loader, env_options)
    if key is None:
        return _NativeEnvironment(loader=loader, **env_options)
    with _ENV_LOCK:
        env = _ENV_CACHE.get(key)
        if env is not None:
            _ENV_CACHE.move_to_end(key)
            return env
        env = _NativeEnvironment(loader=loader, **env_options)
        if len(_ENV_CACHE) >= _ENV_CACHE_MAX:
            _ENV_CACHE.clear()
        _ENV_CACHE[key] = env
        return env


def compile_template(
    source: str,
    *,
    loader: jinja2.BaseLoader | None = None,
    globals: dict | None = None,
    cache: bool = True,
    **env_options,
) -> CompiledTemplate:
    """Compile ``source`` like ``Environment.from_string`` with a fast lexer.

    Parameters mirror ``jinja2.Environment`` options (``trim_blocks``,
    ``lstrip_blocks``, ``keep_trailing_newline``, ``autoescape``,
    ``extensions``, ``undefined``, ...); ``loader`` makes
    ``{% include %}``/``{% extends %}``/``{% import %}`` resolvable at
    render time (e.g. ``jinja2.DictLoader``). ``cache`` (default on) memoises
    compiled templates by ``(source, loader, options)`` — compilation is
    deterministic, so a repeated compile returns the same object; pass
    ``cache=False`` to force recompilation. ``globals`` is passed through to
    ``from_string`` and disables caching (a mutable mapping cannot key a
    deterministic cache).
    """
    if not isinstance(source, str):
        raise TypeError(f"template source must be str, got {type(source).__name__}")
    env_key = _env_key(loader, env_options)
    use_cache = cache and globals is None and env_key is not None
    key = (source, env_key) if use_cache else None
    if key is not None:
        with _TPL_LOCK:
            hit = _TPL_CACHE.get(key)
            if hit is not None:
                _TPL_CACHE.move_to_end(key)
                return hit
    env = _get_env(loader, env_options)
    template = env.from_string(source, globals=globals)
    compiled = CompiledTemplate(template, backend=_last_backend(), source=source)
    if key is not None:
        with _TPL_LOCK:
            if len(_TPL_CACHE) >= _TPL_CACHE_MAX:
                _TPL_CACHE.popitem(last=False)
            _TPL_CACHE[key] = compiled
    return compiled


def compile_batch(
    sources,
    *,
    loader: jinja2.BaseLoader | None = None,
    globals: dict | None = None,
    cache: bool = True,
    workers: int = 0,
    **env_options,
) -> list[CompiledTemplate]:
    """Compile many templates; returns one CompiledTemplate per source.

    ``workers > 1`` compiles on a thread pool (the native scan releases the
    GIL inside the ctypes call); results stay in input order. With
    ``workers=0`` (default) compilation is sequential.
    """
    sources = list(sources)
    if workers and workers > 1 and len(sources) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda s: compile_template(
                        s, loader=loader, globals=globals, cache=cache, **env_options
                    ),
                    sources,
                )
            )
    return [
        compile_template(s, loader=loader, globals=globals, cache=cache, **env_options)
        for s in sources
    ]
