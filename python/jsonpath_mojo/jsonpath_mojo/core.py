"""Public query API: one-shot JSONPath find with native + fallback backends.

``find(expr, data)`` returns a list of ``(value, path)`` pairs identical —
in values, order, and path rendering — to
``[(m.value, str(m.full_path)) for m in jsonpath_ng.ext.parse(expr).find(data)]``.

Backend selection, per call:

1. If the native Mojo kernel is available and ``data`` is JSON-representable
   (``json.dumps(data, allow_nan=False)`` succeeds), the query runs natively:
   the kernel parses the expression and the JSON text and returns the matched
   paths; values are then resolved against the original Python objects, so
   they are the same objects the oracle would return.
2. Otherwise (no kernel, non-JSON data, or any kernel/wrapper disagreement
   such as >63-bit integers or duplicate-after-serialization keys), the
   vendored pure-Python engine evaluates the query with identical results.

Set ``JSONPATH_MOJO_DISABLE_NATIVE=1`` to force the pure-Python engine.
"""

from __future__ import annotations

import json
from itertools import islice

from jsonpath_mojo import _reference
from jsonpath_mojo._native import (
    ST_INDEXERROR,
    ST_KEYERROR,
    ST_NOTIMPL,
    ST_OK,
    ST_PARSE,
    ST_FALLBACK,
    ST_TE_INT,
    ST_TE_LEN,
    ST_TE_ORD,
    ST_VE_SLICE,
    NativeUnavailable,
    backend_info,
    find_native,
    int_coercion_type_error,
    len_type_error,
    native_available,
    ordering_type_error,
)
from jsonpath_mojo._reference import JsonPathError

__all__ = [
    "JsonPathError",
    "backend_info",
    "find",
    "find_parsed",
    "native_available",
    "parse",
]


class _NativeMismatch(Exception):
    """Kernel-result paths could not be resolved against the original data
    (only possible for data outside the JSON domain, e.g. non-string dict
    keys). The caller re-runs the query on the pure-Python engine."""


def _serialize(data) -> bytes | None:
    """JSON-representable check + serialization; None when out of domain."""
    try:
        return json.dumps(data, allow_nan=False).encode("ascii")
    except (TypeError, ValueError):
        return None


def _resolve(data, kinds, vals, path_off, strings):
    """Resolve kernel path steps against the original Python objects and
    render oracle-style path strings."""
def _resolve(data, kinds, vals, path_off, strings):
    """Resolve kernel path steps against the original Python objects and
    render oracle-style path strings.

    Matches arrive in document order, so consecutive paths share long
    prefixes; a chain of already-resolved (step, node, rendered-prefix)
    entries is kept and only the diverging tail is walked per match. This
    keeps big result sets (e.g. `$..*`) fast.
    """
    out = []
    # rendered field fragments, memoized per output string id
    field_frags: dict[int, str] = {}
    # index fragments, memoized per index value
    idx_frags: dict[int, str] = {}
    # resolution chain for the previous match: steps, nodes, rendered prefixes
    chain_steps: list = []
    chain_nodes: list = [data]
    chain_render: list = [""]
    n_matches = len(path_off) - 1
    for m in range(n_matches):
        start, end = path_off[m], path_off[m + 1]
        # longest common prefix with the previous match's chain
        depth = end - start
        common = 0
        limit = depth if depth < len(chain_steps) else len(chain_steps)
        while common < limit:
            s = start + common
            if chain_steps[common] != (kinds[s], vals[s]):
                break
            common += 1
        del chain_steps[common:]
        del chain_nodes[common + 1 :]
        del chain_render[common + 1 :]
        node = chain_nodes[-1]
        rendered = chain_render[-1]
        for s in range(start + common, end):
            kind = kinds[s]
            val = vals[s]
            if kind == 0:  # object field
                key = strings[val]
                try:
                    node = node[key]
                except (KeyError, TypeError) as exc:
                    raise _NativeMismatch from exc
                frag = field_frags.get(val)
                if frag is None:
                    frag = field_frags[val] = _reference._render_field(key)
            elif kind == 1:  # sequence index (list/tuple/str; may be negative)
                try:
                    node = node[val]
                except (IndexError, KeyError, TypeError) as exc:
                    raise _NativeMismatch from exc
                frag = idx_frags.get(val)
                if frag is None:
                    frag = idx_frags[val] = f"[{val}]"
            elif kind == 4 or kind == 5:  # list-only steps (filter/descend):
                # the oracle applies these only to lists, so tuple data must
                # reroute to the pure-Python engine
                if not isinstance(node, list):
                    raise _NativeMismatch
                node = node[val]
                frag = idx_frags.get(val)
                if frag is None:
                    frag = idx_frags[val] = f"[{val}]"
            elif kind == 2:  # dict-position index (filter on dicts)
                values = getattr(node, "values", None)
                if values is None:
                    raise _NativeMismatch
                view = values()
                if val >= len(view):
                    raise _NativeMismatch
                node = next(islice(view, val, None))
                frag = idx_frags.get(val)
                if frag is None:
                    frag = idx_frags[val] = f"[{val}]"
            else:  # render-only [val]: the node itself ([*] on dict/str/scalar)
                frag = idx_frags.get(val)
                if frag is None:
                    frag = idx_frags[val] = f"[{val}]"
            rendered = frag if not rendered else rendered + "." + frag
            chain_steps.append((kind, val))
            chain_nodes.append(node)
            chain_render.append(rendered)
        out.append((node, rendered if chain_steps else "$"))
    return out


def _raise_for_status(status: int, aux: int):
    """Translate a native error status into the oracle's exception."""
    if status == ST_PARSE:
        raise JsonPathError(f"Parse error at 1:{aux}")
    if status == ST_TE_INT:
        raise int_coercion_type_error(aux)
    if status == ST_TE_ORD:
        raise ordering_type_error(aux)
    if status == ST_TE_LEN:
        raise len_type_error(aux)
    if status == ST_INDEXERROR:
        raise IndexError("list index out of range" if aux == 0 else "string index out of range")
    if status == ST_KEYERROR:
        raise KeyError(aux)
    if status == ST_VE_SLICE:
        raise ValueError("slice step cannot be zero")
    if status == ST_NOTIMPL:
        raise NotImplementedError()
    raise NativeUnavailable(f"unexpected native status {status}")


def find(expr, data):
    """Evaluate a JSONPath expression; return oracle-equal (value, path) pairs."""
    if not isinstance(expr, str):
        raise TypeError(f"expression must be a str, not {type(expr).__name__}")
    json_bytes = _serialize(data)
    if json_bytes is not None:
        try:
            status, aux, kinds, vals, path_off, strings = find_native(
                expr.encode("utf-8", "surrogatepass"), json_bytes
            )
        except NativeUnavailable:
            status = ST_FALLBACK
        if status == ST_OK:
            try:
                return _resolve(data, kinds, vals, path_off, strings)
            except _NativeMismatch:
                pass  # non-JSON-domain data: fall through to pure Python
        elif status == ST_FALLBACK:
            pass  # kernel flagged out-of-domain input: pure Python takes over
        else:
            _raise_for_status(status, aux)
    return _reference.find(_reference.parse(expr), data)


def parse(expr):
    """Compile a JSONPath expression for repeated use with find_parsed."""
    return _reference.parse(expr)


def find_parsed(plan, data):
    """Evaluate a plan from parse(); always uses the pure-Python engine."""
    return _reference.find(plan, data)
