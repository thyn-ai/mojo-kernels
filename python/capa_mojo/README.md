# capa-mojo

An accelerated **rule-evaluation engine** for [capa](https://github.com/mandiant/capa)-style
rule sets, powered by a clean-room Mojo kernel — with a pure-Python fallback
for platforms without a native build (including Windows).

Given a rule set (capa's documented YAML format) and a feature map per scope,
`capa_mojo.match(rules, features)` produces match results identical to the
reference engine's `capa.engine.match` for the supported rule subset — rule
for rule, statement for statement, location for location.

```python
import capa_mojo

rules = capa_mojo.load_rules([open("create-two-anonymous-pipes.yml").read()])

# one scope (e.g. one function): a feature map {(name, value): {locations}}
result = rules.match({
    ("api", "CreatePipe"): {0x401010, 0x401020},   # count(api(CreatePipe)): 2 -> match
    ("mnemonic", "push"): {0x401030},
})
assert "create two anonymous pipes" in result

# many scopes at once (the batch fast path): a list of (addr, feature_map)
results = rules.match([(0x401000, function_0_features), (0x402000, function_1_features)])
for scope in results:
    print(hex(scope.addr), scope.rule_names)
```

- **Same results**: the differential suite asserts identical matched rule
  sets AND identical per-statement detail trees (success flags + feature
  locations) against `capa.engine.match` from the published `flare-capa`
  9.2.0 package — on both the native and the fallback backend, with zero
  tolerance (match results are booleans and integer locations; there is
  nothing to approximate).
- **Much faster matching**: the rule set is compiled once into flat,
  postfix-ordered node arrays; the Mojo kernel then evaluates **every rule
  against 64 scopes per pass**, packing one scope per bit of a 64-bit word so
  each boolean statement is one bitwise instruction (measured numbers below).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python fallback
  with identical results.
- Force the fallback with `CAPA_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `capa_mojo.backend_info()`.

## What this is (and is not)

capa's pipeline is *feature extraction* (disassembly via vivisect & friends —
the larger cost) followed by *rule evaluation* (`capa.engine.match`: every
rule's boolean tree against every scope's feature map). **This package
accelerates rule evaluation only.** Extraction is entirely untouched and out
of scope: callers bring feature maps in the documented
`{(name, value): {locations}}` form (integer locations; a key may be present
with an *empty* location set, which counts as present for leaf matching and
as count 0 for `count(...)` — the reference semantics).

To use it with capa's own extractors, run extraction as usual and convert the
resulting `capa.engine.FeatureSet` keys with the documented mapping below.

### Supported rule subset

Statements: `and`, `or`, `not`, `optional` (alias for `0 or more`),
`N or more`, and `count(...)` with an exact count, `N or more`, `N or fewer`,
or an inclusive `(min, max)` range with either bound open.

Features (rule keys → canonical feature-map keys): `api` (DLL prefixes are
trimmed like the reference: `kernel32.CreateFileW` → `CreateFileW`),
`string` (plain, or `/regex/` with an optional `i` flag), `substring`,
`bytes` (prefix match), `number`, `offset`, `mnemonic`, `characteristic`,
`section`, `export`, `import`, `function-name`, `os`, `arch`, `format`,
`class`, `namespace`, `property`, `property/<access>`, `match` (rule names
and namespaces, with namespace prefixes published on match, in topological
rule order), and `count(basic blocks)` (canonical key `("basicblock", 0)`).
Inline ` = descriptions` and `description:` entries are parsed and ignored,
exactly as they do not affect matching.

### Deliberately unsupported (raises `UnsupportedRuleError`)

- **Subscope statements**: `instruction:`, `basic block:`, `function:`,
  `process:`, `thread:`, `span of calls:`, `call:`. The kernel evaluates one
  scope's feature map; nested-scope rules are the caller's responsibility to
  split.
- **`com/...` features** and **`operand[N].number` / `operand[N].offset`**.
- **`count(...)` over a `match` feature.**
- Anything outside the documented capa rule format subset above.

Notable behavioural notes:

- A `match:` reference to an unknown rule or namespace is a load-time error
  (the reference engine fails the same way); dependency cycles are rejected
  with a clear message instead of the reference's `RecursionError`. Very deep
  match chains (>~900 rules) exceed Python's recursion limit in *both*
  engines.
- Scope filtering (`rule.meta.scopes`) is parsed but not interpreted: the
  caller matches each scope against the rules for that scope, mirroring how
  `capa.engine.match` takes a pre-filtered, topologically ordered rule list.
- capa's own optimized matcher (`RuleSet.match`) documents edge cases where
  it diverges from `capa.engine.match`; this package tracks
  **`capa.engine.match`**, the canonical reference.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2, Python 3.12.14,
Mojo 1.1.0, flare-capa 9.2.0) on 2026-09-19 with
`benchmarks/bench_capa.py`: 1000 seeded rules (capa's YAML grammar mix:
and/or/not/optional/N-or-more/count, exact + substring/regex/bytes leaves,
`match:` dependencies) × 2000 function scopes (~37k feature entries).
Correctness gate (identical matched-name sets per scope) passed before
timing; warm = median of 5 full passes. Reproduce:

```
PYTHONPATH="python/capa_mojo:.oracle-capa" PYTHONNOUSERSITE=1 pixi run python benchmarks/bench_capa.py
```

| phase | capa `engine.match` | capa_mojo (native) | speedup |
|---|---:|---:|---:|
| rule parse/compile (cold) | 0.104s | 0.0769s | **1.4x** |
| first match pass (cold) | 37.080s | 4.7716s | **7.8x** |
| warm match pass (median) | 38.757s | 6.1246s | **6.3x** |

warm: 19.378 ms/scope -> 3.0623 ms/scope (52 vs 327 scopes/sec)

Cold vs warm: the cold rows are one-time costs (rule parsing/compilation,
then the first full pass including native-library load and buffer
allocation); the warm row is the steady-state cost of matching the whole rule
set against 2000 scopes. The native kernel evaluates the compiled trees at
64 scopes per bitwise pass; the remaining warm cost is the Python-side
conversion of input feature maps into the kernel's flat count/presence
buffers and the per-match detail reconstruction. (Numbers above are one
measured run on the machine described; absolute times vary with system load,
the ratios were stable across runs at ~6-8x for match passes.)

## Development

- Kernel: `kernels/capa/` (clean-room Mojo; `bash kernels/capa/build.sh`).
- Wrapper: `python/capa_mojo/` (ctypes loader with an ABI handshake, pure
  fallback evaluator, wheel build via `build_wheel.sh` + delocate/auditwheel).
- Tests: `scripts/test_all_capa.sh` runs the differential suite twice — once
  native, once with `CAPA_MOJO_DISABLE_NATIVE=1` — against the pinned
  flare-capa oracle (provisioned with
  `pixi run python -m pip install --target .oracle-capa "flare-capa==9.2.0"`).

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
