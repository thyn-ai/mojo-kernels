# jsonschema-mojo

Drop-in faster replacement for the [`jsonschema`](https://pypi.org/project/jsonschema/)
package (draft 2020-12 subset), powered by a Mojo kernel.

Same call shapes, same pass/fail outcomes, same failing instance paths —
validated by a differential test suite against the published `jsonschema`
package on both backends. The native Mojo kernel runs on macOS arm64 and
Linux x86_64; everywhere else (Windows, missing/unloadable kernel, or values
the native path cannot represent exactly) the package transparently uses a
vendored pure-Python fallback with identical semantics. Zero dependencies.

```python
import jsonschema_mojo

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name"],
    "additionalProperties": False,
}

jsonschema_mojo.validate({"name": "Ada", "age": 36}, schema)   # None (valid)
jsonschema_mojo.validate({"age": -1}, schema)                  # raises ValidationError

validator = jsonschema_mojo.Validator(schema)    # compiles the schema once
validator.is_valid({"name": "Ada"})              # True
for error in validator.iter_errors({"age": -1}):
    print(error.json_path, error.message)
# $.age -1 is less than the minimum of 0
# $ 'name' is a required property
```

`ValidationError` mirrors the reference's attributes: `message`,
`absolute_path`, `json_path`, `absolute_schema_path`, `validator`,
`validator_value`, `instance`, `schema`, `context`.
`jsonschema_mojo.Draft202012Validator` is an alias of `Validator`.

## Supported schema subset (draft 2020-12)

`type`, `properties`, `required`, `items`, `additionalProperties`, `enum`,
`const`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
`minLength`, `maxLength`, `pattern`, `minItems`, `maxItems`, `uniqueItems`,
`minProperties`, `maxProperties`, `multipleOf` — plus boolean subschemas
(`true`/`false`, including `additionalProperties: false` and `items: false`).

Annotation/inert keywords are ignored exactly like the reference does by
default: `$schema`, `$id`, `$defs`/`definitions`, `$anchor`, `$dynamicAnchor`,
`$comment`, `$vocabulary`, `title`, `description`, `default`, `deprecated`,
`readOnly`/`writeOnly`, `examples`, `format` (annotation-only without a
format checker), `contentEncoding`/`contentMediaType`/`contentSchema`.

Schemas using any *other* validation-affecting keyword — `$ref`,
`$dynamicRef`, `allOf`/`anyOf`/`oneOf`/`not`, `if`/`then`/`else`,
`patternProperties`, `prefixItems`, `contains`, `minContains`/`maxContains`,
`propertyNames`, `dependentRequired`/`dependentSchemas`/`dependencies`,
`unevaluatedItems`/`unevaluatedProperties` — raise
`jsonschema_mojo.UnsupportedSchemaError` at `Validator` construction (and in
`validate()`), instead of silently validating differently from the
reference. Invalid keyword value shapes (e.g. `{"type": "frob"}`,
`{"multipleOf": 0}`) also raise `UnsupportedSchemaError`; the reference's
`validate()` raises `SchemaError` for these via metaschema checking.

## Parity contract with `jsonschema` (measured, not aspirational)

Asserted on both backends by `tests/test_jsonschema_differential.py`
(229 tests per backend: curated per-keyword cases, edge semantics, and a
seeded randomized schema/instance sweep) against `jsonschema` 4.26.0:

- **pass/fail outcome — exact** (100% agreement);
- **failing instance JSON paths + keywords — exact** (multiset comparison,
  100% agreement);
- **error message text — exact** for every supported keyword (the
  reference's templates are replicated verbatim; `str(error)` differs: the
  reference appends schema/instance excerpts, this package's `str()` is the
  message itself);
- **error ordering — not part of the contract** (documented divergence):
  the reference orders errors by schema-dict insertion order, the native
  backend uses a fixed canonical keyword order. `Validator.validate()`
  raises its own first error (identity best-effort; the module-level
  `jsonschema_mojo.validate()` mirrors the reference's `best_match`
  deepest-path heuristic).

Edge semantics replicated from the reference: `1.0` counts as `"integer"`;
booleans are never numbers and `True`/`False` never equal `1`/`0` for
`const`/`enum`/`uniqueItems` (but `1 == 1.0`); `"array"` is `list`,
`"object"` is `dict`; `multipleOf` uses floored-modulo semantics for integer
divisors and the `int(quotient) != quotient` test for float divisors with an
exact `Fraction` fallback; `pattern` is evaluated with Python's `re` on the
original string (the kernel defers it), so regex behavior is identical
—including `re.error` on invalid patterns.

## Native path boundaries (auto-degrade to the exact fallback)

The native kernel validates the JSON-document projection of the instance
(what `json.dumps` produces). It detects and silently routes to the
vendored fallback (which mirrors the reference on arbitrary Python objects)
when a value is not exactly representable:

- integers beyond ±2**53 (kernel big-number flag, in schema or instance);
- non-finite floats (`NaN`, `±Infinity`) and non-JSON-serializable values
  (`Decimal`, bytes, sets, arbitrary objects) — `json.dumps` failure.

Instances containing **tuples** or **non-string dict keys** are not JSON
documents; the reference does not treat tuples as arrays and does not
stringify keys. The native path validates their JSON projection (tuple →
array, `1` → `"1"`), which can differ from the reference for such inputs;
use `JSONSCHEMA_MOJO_DISABLE_NATIVE=1` (or the oracle) if you rely on
Python-object semantics there. JSON documents loaded via `json.load` are
unaffected and always parity-exact.

## How it works

`Validator(schema)` checks the schema against the subset gate, serializes it
once (`json.dumps`), and hands it to the Mojo kernel, which parses it into a
flat node arena and compiles a keyword tree. Per `validate`/`iter_errors`
call the wrapper serializes the instance (C-speed `json.dumps`) and the
kernel parses it and walks the compiled schema tree in one pass, streaming
`(keyword, instance path, schema path)` records back. Error objects and
messages are then materialized in Python from the original objects; regex
`pattern` checks and `multipleOf` edge cases are deferred by the kernel to
the wrapper's exact Python path. Schema and instance never round-trip
through Python object reconstruction — the walk is fully native.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14, Mojo
1.1.0, oracle jsonschema 4.26.0), seed 20260919, median of 5 runs,
correctness-gated before timing. Reproduce with
`PYTHONPATH=python/jsonschema_mojo python benchmarks/bench_jsonschema.py`
from the repository root. Scenarios: **A** = 10,000 order records against a
shared schema (incl. two `pattern` checks per record, resolved via Python
`re`); **B** = one ~3 MB nested document (22,500 leaf objects) per call;
**C** = 2,000 invalid records producing ~4 errors each. Cold = fresh
`Validator` + first call (gate + native compile + first walk); warm =
steady-state per document.

| scenario | cold jsonschema (ms) | cold jsonschema_mojo (ms) | warm jsonschema (ms/doc) | warm jsonschema_mojo (ms/doc) | warm speedup |
|---|---:|---:|---:|---:|---:|
| A: 10k order records | 0.050 | 0.047 | 0.0589 | 0.0189 | 3.1x |
| B: big nested document | 222.803 | 9.470 | 213.1359 | 9.0478 | 23.6x |
| C: 2k invalid records | 0.080 | 0.063 | 0.0627 | 0.0292 | 2.1x |

Warm throughput: A — 16,976 vs 52,984 docs/s; B — 5 vs 111 docs/s; C —
15,947 vs 34,244 docs/s. The vendored pure-Python fallback (what Windows
gets) measured 0.0186 / 66.7 / 0.0396 ms/doc warm on A/B/C — itself ~3x
faster than the oracle on small documents, and on tiny documents roughly
equal to the native path (fixed serialization/FFI cost dominates); the
kernel's advantage grows with document size. Numbers on a shared machine
varied ±20% run to run; speedup ratios were stable (2.1–3.2x small docs,
21–28x large).

## Backend selection & diagnostics

```python
import jsonschema_mojo
jsonschema_mojo.backend_info()       # native source, ABI version, errors
jsonschema_mojo.native_available()   # bool
jsonschema_mojo.Validator(schema).backend  # "native" or "fallback"
```

- `JSONSCHEMA_MOJO_DISABLE_NATIVE=1` forces the fallback (used by the
  differential suite).
- `JSONSCHEMA_MOJO_NATIVE_LIB=/path/to/libjsonschemamojo.{dylib,so}`
  overrides kernel resolution (development).

The wrapper verifies an ABI-version handshake with the kernel before use;
any mismatch, load failure, or missing library falls back silently.

## Tests

From the repository root (oracle required: `pip install jsonschema`):

```bash
bash kernels/jsonschema/build.sh
bash scripts/test_all_jsonschema.sh   # native backend, then forced fallback
```

229 differential tests + 7 loader tests per backend run (native and forced
fallback).

## License

Apache-2.0. Copyright Algenta.
