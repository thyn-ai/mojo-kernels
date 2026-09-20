# toml-mojo

A drop-in faster replacement for the stdlib [`tomllib`](https://docs.python.org/3/library/tomllib.html)
parser (and a much faster alternative to [`tomlkit`](https://pypi.org/project/tomlkit/),
Poetry's parser), powered by a clean-room Mojo kernel — with the stdlib
`tomllib` itself as the transparent fallback on platforms without a native
build (including Windows — tested there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

```python
import toml_mojo

doc = toml_mojo.loads("[tool.example]\nversion = 1\n")
assert doc == {"tool": {"example": {"version": 1}}}

with open("pyproject.toml", "rb") as f:
    doc = toml_mojo.load(f)
```

- **Same results**: `toml_mojo.loads(text)` returns exactly what
  `tomllib.loads(text)` returns — the same typed objects
  (`dict`/`list`/`str`/`int`/`float`/`bool`/`datetime`). Integers and floats
  are converted from their source literals with Python's own `int()`/
  `float()`, so they agree with the reference bit-for-bit (including bignum
  integers outside int64 and every representable double; `nan` included).
  The differential suite asserts strict structural equality on both the
  native and fallback backends: handwritten valid/invalid corpora, 250
  seeded generated documents, the repo's own TOML files, and 1,500+
  mutation-fuzz cases checking accept/reject parity with `tomllib` (23.8k
  additional fuzz cases showed zero divergence). Invalid input raises
  `toml_mojo.TOMLDecodeError`, a `ValueError` like
  `tomllib.TOMLDecodeError`.
- **tomllib-shaped data, not tomlkit documents**: like `tomllib` (and unlike
  `tomlkit`), comments and formatting trivia are discarded; documents cannot
  be round-tripped back to text. `tomlkit` (0.15.1) is used as a secondary
  acceptance oracle in the test suite; every generated document also parses
  under it. One known parser-family divergence: `tomllib` (and therefore
  this package) accepts dotted keys that extend an implicitly-created
  super-table (`[t.a.b]` then `[t]` then `a.c = 1`); `tomlkit` rejects that
  as a redefinition. Our oracle is `tomllib`.
- **Faster parsing**: the clean-room Mojo kernel (TOML 1.0, written fresh
  from the public spec) parses in a single pass into a compact binary record
  stream that a small assembler turns into Python objects. ~1.8x faster than
  `tomllib` on 64 KB-1 MB documents, ~8x faster than `tomlkit` (measured
  below; full method and numbers in the repository README/benchmarks).
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently parses with stdlib `tomllib`
  (requires Python >= 3.11), so it is correct by construction on every
  platform.
- Force the fallback with `TOML_MOJO_DISABLE_NATIVE=1`; inspect the active
  backend with `toml_mojo.backend_info()`.

## Benchmarks

Measured on this machine (Apple M4 Max, macOS 26.6.2 arm64, Python 3.12.14,
Mojo 1.1.0, tomlkit 0.15.1, 2026-09-19). Documents generated locally from
fixed seeds; correctness gate (strict equality vs `tomllib`, floats
bit-exact) passes before timing. Warm steady-state: per-call latency, median
of 5 runs. Cold first-call: the first `loads()` in a fresh Python process
(includes the one-time native library load + ABI handshake), median of 5
processes. Method: `benchmarks/bench_toml.py`.

Warm steady-state:

| document | backend | ms/call | MB/s | vs tomllib |
|---|---:|---:|---:|---:|
| small (~1 KB config) | tomllib | 0.0817 | 9.2 | 1.00x |
| small (~1 KB config) | tomlkit | 0.6769 | 1.1 | 0.12x |
| small (~1 KB config) | toml_mojo (native) | 0.0772 | 9.7 | 1.06x |
| small (~1 KB config) | toml_mojo (fallback) | 0.1042 | 7.2 | 0.78x |
| medium (~64 KB) | tomllib | 8.0711 | 8.1 | 1.00x |
| medium (~64 KB) | tomlkit | 62.3814 | 1.1 | 0.13x |
| medium (~64 KB) | toml_mojo (native) | 4.5118 | 14.6 | 1.79x |
| medium (~64 KB) | toml_mojo (fallback) | 8.4801 | 7.7 | 0.95x |
| large (~1 MB) | tomllib | 121.6210 | 8.6 | 1.00x |
| large (~1 MB) | tomlkit | 1038.3020 | 1.0 | 0.12x |
| large (~1 MB) | toml_mojo (native) | 67.4869 | 15.5 | 1.80x |
| large (~1 MB) | toml_mojo (fallback) | 114.0140 | 9.2 | 1.07x |

Cold first-call (fresh process):

| document | tomllib | tomlkit | toml_mojo (native) |
|---|---:|---:|---:|
| small (~1 KB config) | 0.164 ms | 0.983 ms | 5.329 ms |
| medium (~64 KB) | 6.719 ms | 49.468 ms | 8.865 ms |
| large (~1 MB) | 105.047 ms | 804.412 ms | 71.549 ms |

Notes: on tiny documents the one-time native library load (~5 ms) dominates
the cold call, and warm per-call fixed costs leave little room to beat
`tomllib`; the native advantage grows with document size (already ahead cold
at ~64 KB). The fallback rows measure stdlib `tomllib` through the wrapper's
resolver, i.e. the worst case on an unsupported platform.

## Unsupported scope / limits

- Value nesting deeper than 2000 levels (arrays/inline tables) is rejected
  with `TOMLDecodeError` (kernel guard against native stack overflow).
  `tomllib` itself raises `RecursionError` beyond ~450 array levels, so no
  document `tomllib` can parse is affected.
- Error *messages* are not byte-identical to `tomllib`'s (only the
  exception type and the accept/reject verdict match).
- Windows: no Mojo toolchain exists there; the package transparently uses
  the `tomllib` fallback (correct, just not accelerated) — tested on Windows
  in CI ([`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
