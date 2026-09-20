"""jmespath_mojo.search: native-first dispatch with a vendored fallback.

Every call takes one of two paths:

- **native**: the document is serialized once with ``marshal.dumps`` (a
  C-speed binary walk that preserves exact Python types — tuples stay
  distinct from lists, non-string keys stay visible) and evaluated by the
  Mojo kernel. The kernel answers only when it can guarantee
  reference-identical semantics; the result comes back as JSON and is
  decoded with ``json.loads`` (bit-identical floats).
- **fallback**: anything outside that contract — non-string expressions,
  documents marshal cannot serialize, expressions or types outside the
  kernel subset, and every error case — is evaluated by the vendored
  pure-Python implementation (:mod:`jmespath_mojo._reference`), which
  mirrors the reference package exactly, including the error classes it
  raises.
"""

from __future__ import annotations

import json
import marshal

from jmespath_mojo import _native, _reference
from jmespath_mojo._native import NativeUnavailable

__all__ = ["search", "backend_info", "native_available"]


def search(expression, data):
    """Search `data` with the JMESPath `expression` — same results and same
    errors as ``jmespath.search``, powered by the native Mojo kernel where
    possible (macOS arm64 / Linux x86_64), with a vendored pure-Python
    fallback everywhere else.
    """
    if isinstance(expression, str):
        try:
            payload = marshal.dumps(data, 4)
        except (TypeError, ValueError, RecursionError, OverflowError):
            payload = None
        if payload is not None:
            try:
                status, out = _native.search_native(expression, payload)
            except NativeUnavailable:
                status, out = -1, None
            if status == _native.STATUS_OK and out is not None:
                try:
                    return json.loads(out)
                except ValueError:  # pragma: no cover - kernel output is valid JSON
                    pass
    return _reference.search(expression, data)


backend_info = _native.backend_info
native_available = _native.native_available
