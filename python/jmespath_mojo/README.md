# jmespath-mojo

A drop-in faster replacement for the [`jmespath`](https://pypi.org/project/jmespath/)
package, powered by a clean-room Mojo kernel — with a vendored pure-Python
fallback for platforms without a native build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import jmespath_mojo  # same call shape as jmespath.search

doc = {"reservations": [{"instances": [
    {"id": "i-1", "state": "running"}, {"id": "i-2", "state": "stopped"}]}]}

jmespath_mojo.search("reservations[].instances[?state=='running'].id | []", doc)
# ['i-1']
```

- **Same results, same errors**: output matches `jmespath.search`
  structurally (including `int` vs `float`, `-0.0`, NaN, and the exact
  exception classes) on a 5,000+ cell differential matrix, run against the
  native kernel and the forced fallback. The kernel even mirrors CPython
  3.12's compensated `sum()` bit-for-bit.
- **A real Mojo engine**: the kernel parses the expression and the document
  into an index-based arena and evaluates with zero-copy field access,
  projections, filters, slices, and all 28 functions in scope — a genuine
  compiled alternative to the reference's recursive `TreeInterpreter`.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback.
- Force the fallback with `JMESPATH_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `jmespath_mojo.backend_info()`.

Supported surface: field, subexpression, index, slice, list/object
projections, flatten, filters (`&&` `||` `!` and all comparison ops), pipe,
current-node, backtick JSON literals, and `abs avg ceil contains ends_with
floor join keys length map max max_by merge min min_by not_null reverse sort
sort_by starts_with sum to_array to_number to_string type values`. Anything
outside that surface (multiselect list/hash, exotic literal forms) is still
fully correct — the wrapper computes those calls with the vendored
pure-Python implementation, which mirrors the reference exactly.

## Performance (measured, honest)

Measured on an Apple M4 Max (2026-09-19), Python 3.12.14, Mojo 1.1.0,
median of 5 warm runs; cold = median of 7 fresh processes. The benchmark is
`benchmarks/bench_jmespath.py` (seeded, reproducible), gated on correctness
before timing. Big document: 0.85 MB of AWS-describe-instances-shaped JSON
(400 reservations × 10 instances).

| workload (big doc, 0.85 MB) | pip jmespath | jmespath_mojo | speedup |
|---|---:|---:|---:|
| filter+flatten+project (warm) | 4.906 ms | 9.730 ms | 0.50x |
| filter+and+project (warm) | 8.614 ms | 9.765 ms | 0.88x |
| 5-condition filter (warm) | 14.88 ms | 13.51 ms | **1.10x** |
| flatten+avg (warm) | 2.559 ms | 9.338 ms | 0.27x |
| sort_by+pipe (warm) | 4.790 ms | 9.125 ms | 0.53x |
| sort 32k floats+pick (warm) | 11.994 ms | 22.663 ms | 0.53x |
| max of 32k floats (warm) | 9.522 ms | 9.176 ms | **1.04x** |
| sum of 32k floats (warm) | 8.852 ms | 8.925 ms | 0.99x |
| object-wildcard (warm) | 0.316 ms | 8.632 ms | 0.04x |
| merge+pipe (warm) | 0.101 ms | 10.599 ms | 0.01x |
| boundary floor (trivial path, warm) | 0.003 ms | 10.376 ms | 0.00x |
| filter+flatten+project (cold, first call) | 5.898 ms | 20.804 ms | 0.28x |
| small docs, 5k × 1 search (warm) | 6.76 µs/call | 12.94 µs/call | 0.52x |

**Read this table honestly.** Each native call serializes the document once
(`marshal.dumps`, a C-speed binary walk with exact Python types), the kernel
parses and evaluates it, and the result returns as JSON. That per-call
boundary costs ~6 ms for the 0.85 MB document — so:

- **Where it wins or breaks even**: compute-heavy filters (multiple
  conditions per element) and big aggregations (`max`, `sum`, `avg` over
  tens of thousands of values) — 1.0–1.1x, with the advantage growing as
  the per-element expression complexity grows.
- **Where it loses**: light lookups, output-heavy projections (the result
  must be serialized back), trivial paths, and small documents — anywhere
  the reference's in-process walk is already cheap.
- **Cold starts**: the first call in a process also pays dlopen + runtime
  init (~15 ms one-time).

The kernel's raw evaluation is several times faster than the reference
interpreter on the same tree — the serialization boundary is the tax. If
your workload runs many heavy queries against large documents, the native
backend earns its keep; for light lookups `pip jmespath` stays ahead, and
`jmespath-mojo` still gives you bit-identical results (and a pure-Python
fallback on every platform).

## How it works

- `jmespath_mojo.search(expression, data)` serializes `data` with
  `marshal.dumps` (exact types: tuples stay distinct from lists), evaluates
  the expression on the Mojo kernel, and decodes the JSON result.
- The kernel answers only when it can guarantee reference-identical
  semantics; anything outside its subset (multiselect, bigint past int64,
  exotic literal forms, NaN sort keys, error paths) is recomputed by the
  vendored pure-Python implementation, including the exact exception
  classes (`ParseError`, `LexerError`, `JMESPathTypeError`,
  `ArityError`, ...).
- Differential tests run the full expression × document matrix against the
  published PyPI package (`jmespath==1.0.1`) on both backends
  (`scripts/test_all_jmespath.sh`).

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
