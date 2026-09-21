# jinja2-mojo

Faster [jinja2](https://pypi.org/project/Jinja/) template **compilation**, powered by a
Mojo lexer kernel — with stock jinja2's own lexer as the transparent fallback.

`jinja2-mojo` accelerates the compile path only. Everything downstream of tokenisation —
the parser, the compiler, and the entire render runtime — **is stock jinja2**, running
in-process. Render output is byte-identical to `jinja2.Environment.from_string` for the
same context, and the differential suite asserts exactly that (plus byte-identical
*generated Python source*) across the supported corpus on both backends, against the
pip jinja2 3.1.6 oracle.

```python
import jinja2_mojo

t = jinja2_mojo.compile_template("Hello {{ name }}!")
t.render({"name": "World"})     # 'Hello World!'
t(name="World")                 # same

# batch + loaders
ts = jinja2_mojo.compile_batch(["{{ a }}", "{% include 'row' %}"],
                               loader=jinja2.DictLoader({"row": "<td>{{ v }}</td>"}))
```

## Why

`Environment.from_string` recompiles on every call, and the lexer's per-character scan
is pure Python. The Mojo kernel performs that scan natively; measured on an Apple M4 Max
(2026-09-20, Python 3.12.5, jinja2 3.1.6, mojo 1.1.x) the kernel's per-character scan is
**18.7–34.1×** faster than the stock lexer's, the full tokenize stage is **2.2–2.3×**
faster, and steady-state cold compile is **1.12–1.22×** faster. A deterministic compile
cache (on by default) makes repeated compiles of an already-seen source ~0.6 µs.

Render speed is unchanged by design: compiled templates execute as Python bytecode and
autoescaping goes through MarkupSafe's C extension — there is nothing to accelerate in
this package's scope.

## Benchmarks

Measured with `benchmarks/bench_jinja2.py` (correctness gate first: render output
byte-identical to stock jinja2 on every template before any timing). Templates are
generated locally from a fixed seed. "stock" = `jinja2.Environment().from_string`;
"jinja2_mojo" = `jinja2_mojo.compile_template(cache=False)` unless noted.

Cold compile, steady state (30 unique sources, median of 5 runs):

| template | stock | native | forced fallback |
|---|---|---|---|
| small (213 B) | 622.7 µs | 512.3 µs (**1.22×**) | 574.9 µs (1.08×) |
| medium (6 026 B) | 11 193.5 µs | 9 803.7 µs (**1.14×**) | 11 392.5 µs (0.98×) |
| large (29 207 B) | 58 396.6 µs | 52 266.3 µs (**1.12×**) | 56 833.1 µs (1.03×) |

Tokenize stage only (`env._tokenize` vs native scan + token build; kernel scan alone
shown for reference):

| template | stock tokenize | native scan+build | kernel scan only |
|---|---|---|---|
| small (213 B) | 91.7 µs | 41.0 µs (**2.24×**) | 4.9 µs (18.65×) |
| medium (6 026 B) | 2 212.3 µs | 972.4 µs (**2.28×**) | 65.0 µs (34.06×) |
| large (29 207 B) | 11 220.9 µs | 5 114.8 µs (**2.19×**) | 572.6 µs (19.60×) |

Warm compile (repeat compile of one already-compiled source; stock always recompiles):

| template | stock | jinja2_mojo (cache hit) |
|---|---|---|
| small (213 B, 2000 iters) | 678.7 µs | 0.7 µs (**~1018×**) |
| medium (6 026 B, 200 iters) | 11 865.1 µs | 0.5 µs (**~21 696×**) |
| large (29 207 B, 30 iters) | 58 634.6 µs | 0.6 µs (**~100 516×**) |

Batch compile (300 unique 6 KB sources):

| path | time | speedup |
|---|---|---|
| stock `from_string` loop | 3 840.6 ms | — |
| `compile_batch` (sequential) | 3 526.9 ms | 1.09× |
| `compile_batch` (workers=4) | 3 197.4 ms | 1.20× |

Cold compile, fresh process (one compile in a fresh interpreter, median of 7 — includes
import, dlopen and the ABI handshake; the honest first-request number):

| template | stock | jinja2_mojo |
|---|---|---|
| small (213 B) | 1.71 ms | 4.91 ms (**0.35× — slower**) |
| medium (6 026 B) | 12.93 ms | 14.42 ms (0.90×) |
| large (29 207 B) | 58.06 ms | 54.65 ms (1.06×) |

**Honest read:** loading the native runtime costs ~3 ms one time, so a process that
compiles exactly one tiny template is slower with this package. It wins when a process
compiles many templates (static-site generators, template precompilation, test suites)
or recompiles the same templates repeatedly (the cache). The forced-fallback path is
within ±8 % of stock everywhere — the wrapper adds no meaningful overhead when the
kernel is absent.

## Install

```bash
pip install jinja2-mojo-<version>-py3-none-<platform>.whl
```

Quickstart after install (prints rendered samples and a checksum):

```bash
python -m jinja2_mojo
```

The wheel is self-contained: the Mojo kernel and its runtime libraries are vendored in
(repaired with `delocate` on macOS / `auditwheel` on Linux), so no Mojo toolchain is
needed at install or run time. If the native library cannot load on your platform, the
package silently uses stock jinja2's lexer — results are identical either way.

## API

- `compile_template(source, *, loader=None, globals=None, cache=True, **env_options)`
  → `CompiledTemplate`. `env_options` are `jinja2.Environment` options
  (`trim_blocks`, `lstrip_blocks`, `keep_trailing_newline`, `autoescape`, `extensions`,
  `undefined`, ...). `loader` (e.g. `jinja2.DictLoader`) resolves
  `{% include %}`/`{% extends %}`/`{% import %}` at render time. `cache` memoises the
  compiled template by `(source, loader, options)`; pass `cache=False` to force
  recompilation.
- `compile_batch(sources, *, workers=0, **same_options)` → list in input order;
  `workers>1` uses a thread pool (the native scan releases the GIL).
- `CompiledTemplate`: `.render(...)` and `.__call__(context)` like
  `jinja2.Template.render`, `.generate()`, `.template` (the underlying stock
  `jinja2.Template`), `.environment`, `.source`, `.backend` (`"native"` or `"stock"`).
- `native_available()`, `backend_info()` — backend diagnostics.

Set `JINJA2_MOJO_DISABLE_NATIVE=1` to force the stock-lexer path (used by the
differential suite).

## Supported scope (native lexer)

Default delimiters; variables; blocks (`if`/`elif`/`else`, `for`/`else`, `set`,
`include`, `import`/`from`, `extends`/`block`, `macro`/`call`, `filter`, `do`
extension, `with`, ...); filters and tests; comments; whitespace control (`-`/`+`
markers, `trim_blocks`, `lstrip_blocks`, `keep_trailing_newline`); string literals with
Python-style escapes (including `\xHH`, `\uHHHH`, `\UHHHHHHHH`, `\N{...}`, octal, and
backslash-newline continuation); numbers (decimal/hex/octal/binary integers with
underscores, floats with fraction/exponent, Python-literal leading-zero rules); unicode
names and content; `\r\n`/`\r` normalisation; arbitrary bracket nesting.

**Not native** — the kernel declines and the stock lexer takes over transparently
(identical results, stock speed for that template): `{% raw %}` blocks, custom
delimiters, `line_statement_prefix`/`line_comment_prefix`, and anything the kernel
cannot tokenise with oracle-identical results (it never guesses). Lex-time errors
(unterminated strings/comments, bad escapes, unbalanced brackets, unknown characters)
always come from the stock lexer, so error types and messages match the oracle exactly.

**Out of scope by design:** rendering (templates execute as Python bytecode; MarkupSafe
escaping is C) — this package changes compile time only.

## Parity

Oracle: pip `jinja2==3.1.6`, used strictly black-box (token streams and render outputs
observed on probe inputs; no oracle source read or adapted). The differential suite
(`tests/test_jinja2_differential.py`) asserts, on both the native backend and the
forced fallback:

- render output **byte-identical** to the oracle across a 50-template corpus
  (variables, control flow, includes/extends/imports via `DictLoader`, macros, filters,
  comments, whitespace control, escapes, numbers, unicode) × contexts;
- generated Python source (`Environment.compile(raw=True)`) **byte-identical**;
- token stream identical (type, value, line number; block/variable end tokens may carry
  the bare marker where the oracle folds trimmed whitespace into the value — the parser
  never reads end-token values);
- error behavior identical (same `TemplateSyntaxError` and message, always produced by
  the stock lexer);
- 300 seeded fuzz templates, plus a 4 000-template token-level fuzz sweep with zero
  divergences.

Comparison tolerance: none — outputs must be byte-identical (`==`), and they are.

## Development

```bash
# build the native kernel (needs the repo pixi environment with Mojo 1.1.x)
bash kernels/jinja2/build.sh

# differential tests, native then forced fallback (provisions the oracle
# jinja2==3.1.6 into .oracle-jinja2/ on first run)
pixi run bash scripts/test_all_jinja2.sh

# benchmark (prints the tables above)
PYTHONPATH=python/jinja2_mojo pixi run python benchmarks/bench_jinja2.py

# self-contained platform wheel (build + delocate/auditwheel repair)
pixi run bash python/jinja2_mojo/build_wheel.sh
```

Attribution: Algenta. License: Apache-2.0.
