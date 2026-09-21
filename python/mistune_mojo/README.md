# mistune-mojo

A drop-in faster replacement for [`mistune`](https://pypi.org/project/mistune/)'s
`mistune.markdown()`, powered by a clean-room Mojo kernel — with a vendored
pure-Python engine for platforms without a native build (including Windows —
the pure-Python fallback runs there in CI: [`windows-fallback`](https://github.com/thyn-ai/mojo-kernels/actions/workflows/windows-fallback.yml)).
Zero runtime dependencies.

```python
import mistune_mojo

mistune_mojo.markdown("# Hello *world*\n\n- a\n- b with `code`\n")
# '<h1>Hello <em>world</em></h1>\n<ul>\n<li>a</li>\n<li>b with <code>code</code></li>\n</ul>\n'
# byte-identical to mistune.markdown()
```

- **Byte-identical output**: the renderer is byte-identical to
  `mistune.markdown()` from the published `mistune` package (3.3.4) for the
  supported scope — the differential suite compares the full CommonMark
  0.31.2 spec corpus (652 cases) plus a seeded generated-document corpus,
  on both the native and fallback backends. Measured agreement: **652/652
  spec cases on both backends** (byte-for-byte, zero tolerance).
- **Not plain CommonMark**: `mistune.markdown()`'s default configuration
  deliberately deviates from the CommonMark spec, and both engines mirror
  those deviations exactly: no raw inline HTML and no HTML blocks are
  emitted (tags are HTML-escaped), named/numeric entities are *not* decoded
  in text (they *are* decoded in link destinations and titles), fenced code
  keeps its trailing newline while indented code does not, `<li>` is never
  followed by a newline, link/image/autolink URLs are scheme-checked against
  an allowlist (`http`, `https`, `ftp`, `ftps`, `irc`, `ircs`, `mailto`,
  `tel`; anything else renders as `#harmful-link`), and mistune's
  quote/list lazy-continuation rules apply. Both engines were built
  clean-room from the CommonMark spec plus black-box observation of the
  published package (no mistune source was read or adapted).
- **Much faster**: single-pass block+inline parse in the Mojo kernel —
  see the benchmark below.
- **No toolchain needed**: per-platform wheels ship the compiled kernel.
  Everywhere else the package transparently uses its pure-Python engine,
  which returns identical bytes.
- Force the fallback with `MISTUNE_MOJO_DISABLE_NATIVE=1`; inspect the
  active backend with `mistune_mojo.backend_info()`.

## Supported scope

Everything `mistune.markdown()` (default configuration, no plugins) handles:
paragraphs, ATX/setext headings, thematic breaks, indented and fenced code
blocks, block quotes (including lazy continuation), bullet/ordered lists
(nested, tight/loose, custom start numbers), link reference definitions
(forward references, multi-line titles), inline code spans, emphasis and
strong emphasis (full delimiter-run algorithm with the rule-of-3), links
(inline/reference/collapsed/shortcut), images, autolinks (URI and email,
with the scheme allowlist above), hard/soft line breaks, backslash escapes,
HTML blocks and inline raw HTML (rendered escaped, as mistune does), and
entity/numeric references in link destinations and titles (HTML5 named table
and numeric replacement rules, including the C1 remapping and noncharacter
drop).

**Not supported** (out of scope, by design):

- **mistune plugins** (`strikethrough`, `table`, `footnotes`, `task_lists`,
  `mark`, `url`, `ruby`, `spoiler`, etc.) — this package mirrors the
  *default* `mistune.markdown()` only. Inputs meant for a plugin parse as
  they would in default mistune (e.g. `~~x~~` renders literally, pipe-table
  syntax renders as a paragraph).
- **mistune's renderer/hook API** (`HTMLRenderer`, `create_markdown`,
  custom renderers, AST/token access) — only the `markdown(text) -> str`
  entry point is provided.
- In the **native kernel only**: named HTML entities outside the kernel's
  built-in table when they appear in a link destination or title, and
  non-ASCII link-reference labels (Unicode casefolding). Such documents are
  transparently re-rendered by the pure-Python engine with identical bytes —
  end users see no difference.

## Benchmark

Measured on this machine (Apple M4 Max, macOS arm64, Python 3.12, Mojo
1.1.0, mistune 3.3.4), median of 5 runs. "Cold" = first render of each of 30
distinct seeded documents (no result caching anywhere — neither mistune nor
mistune-mojo caches results); "warm" = steady-state per-render cost over 50
repeats of the same document. Correctness (byte identity with mistune) is
asserted before any timing.

| workload | cold mistune | cold mistune-mojo (native) | speedup | warm mistune | warm mistune-mojo (native) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| small doc (~0.8 KB) | 449 us | 104 us | **4.32x** | 589 us | 117 us | **5.05x** |
| medium doc (~7.4 KB) | 4580 us | 938 us | **4.88x** | 5296 us | 999 us | **5.30x** |
| large doc (~75 KB) | 45547 us | 8407 us | **5.42x** | 49545 us | 9273 us | **5.34x** |

The pure-Python fallback returns identical bytes (its own speed is roughly
mistune-level; the native kernel is the speedup above). Full method and
repro script: `benchmarks/bench_mistune.py` in the repository.

## How it works

`mistune_mojo.markdown(text)` renders through the native Mojo kernel
(single-pass block+inline parser, `kernels/mistune`) when the shared library
is available (macOS arm64 / Linux x86_64 wheels) and through the vendored
pure-Python engine otherwise. Both engines implement the same rules and are
pinned to byte-identity with `mistune` and with each other by the
differential suite. If the kernel declines a document (the narrow
native-only scope above), the wrapper re-renders with the pure-Python
engine, so every supported input renders everywhere.

Source, benchmarks, and development: <https://github.com/thyn-ai/mojo-kernels>

License: Apache-2.0, © 2026 Algenta
