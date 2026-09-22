"""jsonpath-mojo: a fast JSONPath query engine, API-compatible with the
`jsonpath_ng.ext` dialect of the jsonpath-ng package.

    from jsonpath_mojo import find

    matches = find("$.store.book[?(@.price < 10)].title", data)
    # -> [(value, path), ...] exactly like
    #    [(m.value, str(m.full_path)) for m in jsonpath_ng.ext.parse(q).find(data)]

Evaluation runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back to
the vendored pure-Python engine everywhere else. Both backends return
identical results; the differential test suite asserts agreement with
jsonpath_ng on both paths.

Set JSONPATH_MOJO_DISABLE_NATIVE=1 to force the pure-Python engine; inspect
the active backend with jsonpath_mojo.backend_info().
"""

from jsonpath_mojo._native import backend_info, native_available
from jsonpath_mojo.core import JsonPathError, find, find_parsed, parse

__version__ = "0.1.4"  # x-release-please-version
__all__ = [
    "JsonPathError",
    "backend_info",
    "find",
    "find_parsed",
    "native_available",
    "parse",
    "__version__",
]
