# jsonpath-mojo

A fast JSONPath query engine, API-compatible with the
[`jsonpath-ng`](https://pypi.org/project/jsonpath-ng/) package (the
`jsonpath_ng.ext` extended dialect), powered by a clean-room Mojo kernel —
with a vendored pure-Python engine for platforms without a native build
(including Windows).

```python
from jsonpath_mojo import find

data = {"store": {"book": [{"title": "Sayings", "price": 8.95},
                           {"title": "Sword", "price": 12.99}]}}

find("$.store.book[*].title", data)        # [('Sayings', 'store.book.[0].title'),
                                           #  ('Sword',  'store.book.[1].title')]
find("$.store.book[?(@.price < 10)]", data)  # filter scripts
find("$..price", data)                     # recursive descent
```

`find(expr, data)` returns the same `(value, str(full_path))` sequence as
`[(m.value, str(m.full_path)) for m in jsonpath_ng.ext.parse(expr).find(data)]`
— same values, same order, same path rendering, verified element-for-element
by the differential test suite on both backends.

- **Same results**: the suite runs every query against the published
  `jsonpath_ng==1.7.0` oracle and asserts exact sequence parity — plus
  exception-type parity for the error contract (parse errors raise
  `jsonpath_mojo.JsonPathError`; runtime errors raise the same builtin
  `TypeError`/`KeyError`/`IndexError`/`ValueError`/`NotImplementedError` the
  oracle raises).
- **Much faster**: a native expression parser + evaluator instead of a PLY
  parser rebuilt per call — 49x faster at 100-book documents, 3.6x at 2k,
  1.7x at 20k (Apple M4 Max; full method and numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python engine
  (which is itself ~2-28x faster than the oracle on these datasets).
- Force the fallback with `JSONPATH_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `jsonpath_mojo.backend_info()`.

## Supported syntax

- child steps: `$.a`, `$['a']`, `$.a.b`, dot-number children `$.a.0`
- field unions: `$['a', 'b']` (expression order, duplicates repeated); a
  quoted `'*'` — alone or in a union — is a wildcard
- wildcards: `$.*` (object values), `$[*]` (sequence elements; see README in
  the repo for the exact non-list truth table)
- recursive descent: `$..a`, `$..*`, `$..[0]`, `$..['a']` (DFS pre-order)
- index: `$.a[0]`, `$.a[-1]`
- slices: `$.a[1:5:2]`, `$.a[::-1]` (Python slice semantics)
- filters: `$.a[?(@.x > 2)]` with `== != < <= > >=`, `&` conjunctions,
  existence tests `?(@.x)`, arith (`+ - *`) on the left side, literals
  (numbers, quoted strings, `true`/`false`, bare words) on the right, and
  `@` paths with fields/indexes/slices/wildcards

Expressions outside this grammar (or data outside the JSON domain — non-string
dict keys, NaN/Infinity, tuples, custom objects) are handled with identical
results: grammar gaps raise `JsonPathError`, non-JSON data transparently
reroutes to the pure-Python engine. Currently unsupported oracle features:
`?(...)` filters nested inside a filter path (e.g. `?(@.a[?(@ > 1)] == 2)`),
and sorting/`~` regex extensions.

## Benchmarks

Method (see `benchmarks/bench_jsonpath.py` in the repository): store-shaped
JSON documents (100 / 2,000 / 20,000 book records) and a 14-query battery
covering the supported scope. Each cell runs `jsonpath_ng.ext.parse(expr).
find(data)` per query against `jsonpath_mojo.find(expr, data)`. COLD = first
battery call (includes one-time parser setup); WARM = median of 5 battery
runs. Correctness is asserted before timing.

Measured on Apple M4 Max, macOS 26.6.2, Python 3.12.14, jsonpath_ng 1.7.0,
Mojo 1.1.0, 2026-09-19:

| dataset | backend | cold (s) | warm (s) | speedup (warm) |
|---|---|---|---|---|
| 100 books | oracle jsonpath_ng | 0.1568 | 0.1993 | 1.0x |
| | jsonpath-mojo native | 0.0050 | 0.0041 | **49.2x** |
| | jsonpath-mojo fallback | 0.0078 | 0.0071 | 27.9x |
| 2,000 books | oracle jsonpath_ng | 0.3209 | 0.2750 | 1.0x |
| | jsonpath-mojo native | 0.0728 | 0.0766 | **3.6x** |
| | jsonpath-mojo fallback | 0.1411 | 0.1353 | 2.0x |
| 20,000 books | oracle jsonpath_ng | 1.7522 | 1.4909 | 1.0x |
| | jsonpath-mojo native | 0.7877 | 0.8738 | **1.7x** |
| | jsonpath-mojo fallback | 1.4734 | 1.2992 | 1.1x |

The oracle rebuilds its PLY parser on every `parse()` call (~7 ms), so small
and medium documents are dominated by expression parsing — where the native
kernel and even the pure-Python engine win overwhelmingly. On very large
documents the evaluation itself dominates; the native pipeline
(`json.dumps` → kernel parse+eval → path resolution) still leads, and result
correctness is identical either way.

Source, tests, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
