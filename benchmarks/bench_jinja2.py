#!/usr/bin/env python3
"""Reproducible benchmark: stock jinja2 compile vs jinja2_mojo.compile_template.

Templates are generated locally from fixed seeds (no network, no datasets):
a small variable page (~0.5 KB), a medium HTML page (~5 KB) with
loops/macros/comments/whitespace control, and a large generated page
(~100 KB). Correctness is asserted (render output byte-identical to stock
jinja2) before any timing happens, so the numbers below always come from a
verified-correct build.

Measured cells:

* cold (fresh process): one compile in a fresh interpreter, median of 7 —
  includes import, dlopen and the ABI handshake. This is the honest
  "first request" number.
* cold (steady, unique sources): per-template compile with the template
  cache bypassed (cache=False), median of 5 runs over 30 unique sources.
* tokenize stage only: stock env._tokenize vs the native scan + token
  build (same inputs) — the component the kernel actually accelerates.
* warm (cached): compile_template of an already-compiled source (cache
  hit) vs stock from_string (stock jinja2 always recompiles).
* batch: compile_batch over 300 unique sources, sequential and workers=4,
  vs sequential stock from_string.

Run from the repository root:

    PYTHONPATH=python/jinja2_mojo pixi run python benchmarks/bench_jinja2.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

SEED = 20260920
N_RUNS = 5
N_COLD_SOURCES = 30
N_BATCH = 300
N_WARM = 2000

SMALL = """<h1>{{ title|upper }}</h1>
{% if user %}<p>{{ user.name }} ({{ user['age'] }})</p>{% endif %}
<ul>{% for x in items %}<li>{{ loop.index }}. {{ x * 1.5 }}</li>{% endfor %}</ul>
{# footer #}<p>{{ 'bye' ~ '!' }}</p>
"""


def _medium(rng) -> str:
    rows = "\n".join(
        f'  <tr class="{{{{ loop.cycle(\'a\', \'b\') }}}}"><td>{{{{ r{i} }}}}</td>'
        f"<td>{{{{ r{i}|default('n/a') }}}}</td></tr>"
        for i in range(60)
    )
    return (
        "{# generated #}\n{% macro cell(v) %}<td>{{ v }}</td>{% endmacro %}\n"
        "<table>\n{% for row in rows -%}\n" + rows + "\n{%- endfor %}\n</table>\n"
        "{% set total = rows|length %}{{ total }} rows\n"
    )


def _large(rng) -> str:
    parts = []
    for i in range(900):
        kind = i % 6
        if kind == 0:
            parts.append(f"<div id='d{i}'>{{{{ v{i % 40} }}}}</div>")
        elif kind == 1:
            parts.append(f"{{% if v{i % 40} %}}yes{i}{{% else %}}no{i}{{% endif %}}")
        elif kind == 2:
            parts.append(f"{{# comment {i} #}}text {i} ")
        elif kind == 3:
            parts.append(f"{{% set s{i} = v{i % 40} * 2 %}}{{{{ s{i} }}}}")
        elif kind == 4:
            parts.append(f"{{{{ '{i}x\\t' ~ v{i % 40} }}}}")
        else:
            parts.append(f"{{% for x in v{i % 40} %}}{{{{ x }}}}{{% endfor %}}")
    return "\n".join(parts)


def _make_unique(src: str, i: int) -> str:
    # A trailing comment changes the source (and its cache key) without
    # changing semantics.
    return src + f"{{# unique {i} #}}"


def _median_us(fn, runs=N_RUNS) -> float:
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def _ctx() -> dict:
    return {
        "title": "t",
        "user": {"name": "n", "age": 3},
        "items": [1, 2, 3],
        "rows": [1],
        **{f"v{i}": [i, i + 1] for i in range(40)},
    }


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
        f"- python: {platform.python_version()}",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def _cold_fresh_process(src: str) -> tuple[float, float]:
    """Median wall time for one compile in a fresh interpreter (7 samples)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".tpl", delete=False) as f:
        f.write(src)
        path = f.name
    code_stock = (
        "import time,jinja2;"
        "t0=time.perf_counter();"
        f"jinja2.Environment().from_string(open({path!r}).read());"
        "print(f'{(time.perf_counter()-t0)*1e6:.1f}')"
    )
    code_ours = (
        "import time,jinja2_mojo;"
        "t0=time.perf_counter();"
        f"jinja2_mojo.compile_template(open({path!r}).read(), cache=False);"
        "print(f'{(time.perf_counter()-t0)*1e6:.1f}')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath("python/jinja2_mojo"), os.path.abspath(".oracle-jinja2")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    ours, stock = [], []
    for _ in range(7):
        stock.append(float(subprocess.run([sys.executable, "-c", code_stock],
                                          capture_output=True, text=True, env=env).stdout))
        ours.append(float(subprocess.run([sys.executable, "-c", code_ours],
                                         capture_output=True, text=True, env=env).stdout))
    os.unlink(path)
    return statistics.median(stock), statistics.median(ours)


def main() -> None:
    import random

    import jinja2

    import jinja2_mojo
    from jinja2_mojo import _native, _tokens

    rng = random.Random(SEED)
    mediums = {"small (~0.2 KB)": SMALL, "medium (~6 KB)": _medium(rng), "large (~30 KB)": _large(rng)}
    info = jinja2_mojo.backend_info()

    print("== environment ==")
    print(machine_info())
    print(f"- jinja2 (oracle): {jinja2.__version__}")
    print(f"- jinja2_mojo backend: {'native' if info['native_available'] else 'FALLBACK'}")
    print()

    # ---- correctness gate -------------------------------------------------
    # Autoescape stays off everywhere below: the benchmark must mirror stock
    # jinja2 defaults (autoescape is opt-in upstream) to keep parity meaningful.
    for label, src in mediums.items():
        expected = jinja2.Environment().from_string(src).render(_ctx())  # nosemgrep: missing-autoescape-disabled
        got = jinja2_mojo.compile_template(src).render(_ctx())
        assert got == expected, f"render mismatch on {label}"
    print("correctness gate: render output byte-identical to stock jinja2 on all sizes")
    print()

    sizes = {k: len(v) for k, v in mediums.items()}

    # ---- cold: fresh process ----------------------------------------------
    print("== cold compile, fresh process (import + dlopen + one compile) ==")
    for label, src in mediums.items():
        stock_us, ours_us = _cold_fresh_process(src)
        print(f"{label:>18} ({sizes[label]:>7} B): stock {stock_us/1e3:8.2f} ms | "
              f"jinja2_mojo {ours_us/1e3:8.2f} ms | ratio {stock_us/ours_us:5.2f}x")
    print()

    # ---- cold: steady state, unique sources --------------------------------
    print(f"== cold compile, steady state ({N_COLD_SOURCES} unique sources, cache off) ==")
    for label, src in mediums.items():
        sources = [_make_unique(src, i) for i in range(N_COLD_SOURCES)]
        env = jinja2.Environment()  # nosemgrep: missing-autoescape-disabled
        stock_us = _median_us(lambda: [env.from_string(s) for s in sources]) / len(sources)
        ours_us = _median_us(
            lambda: [jinja2_mojo.compile_template(s, cache=False) for s in sources]
        ) / len(sources)
        fb_us = None
        os.environ["JINJA2_MOJO_DISABLE_NATIVE"] = "1"
        try:
            fb_us = _median_us(
                lambda: [jinja2_mojo.compile_template(s, cache=False) for s in sources]
            ) / len(sources)
        finally:
            del os.environ["JINJA2_MOJO_DISABLE_NATIVE"]
        print(f"{label:>18} ({sizes[label]:>7} B): stock {stock_us:9.1f} us | "
              f"native {ours_us:9.1f} us ({stock_us/ours_us:5.2f}x) | "
              f"forced-fallback {fb_us:9.1f} us ({stock_us/fb_us:5.2f}x)")
    print()

    # ---- tokenize stage only ------------------------------------------------
    print("== tokenize stage only (stock _tokenize vs native scan+build) ==")
    for label, src in mediums.items():
        env = jinja2.Environment()  # nosemgrep: missing-autoescape-disabled
        src_bytes = src.encode("utf-8")
        stock_us = _median_us(lambda: list(env._tokenize(src, None, None, None)))
        ours_us = _median_us(lambda: _tokens.build_tokens(src))
        scan_us = _median_us(lambda: _native.scan(src_bytes))
        print(f"{label:>18} ({sizes[label]:>7} B): stock {stock_us:9.1f} us | "
              f"native scan+build {ours_us:9.1f} us ({stock_us/ours_us:6.2f}x) | "
              f"kernel scan only {scan_us:7.1f} us ({stock_us/scan_us:6.2f}x)")
    print()

    # ---- warm: cached compile ----------------------------------------------
    print("== warm compile (repeats of one already-compiled source) ==")
    warm_iters = {"small (~0.2 KB)": 2000, "medium (~6 KB)": 200, "large (~30 KB)": 30}
    for label, src in mediums.items():
        iters = warm_iters[label]
        jinja2_mojo.compile_template(src)  # prime the cache
        env = jinja2.Environment()  # nosemgrep: missing-autoescape-disabled
        stock_us = _median_us(lambda: [env.from_string(src) for _ in range(iters)]) / iters
        ours_us = _median_us(
            lambda: [jinja2_mojo.compile_template(src) for _ in range(iters)]
        ) / iters
        print(f"{label:>18} ({sizes[label]:>7} B, {iters:>5} iters): stock {stock_us:9.1f} us | "
              f"cached {ours_us:9.1f} us ({stock_us/ours_us:7.2f}x)")
    print()

    # ---- batch ----------------------------------------------------------------
    print(f"== batch compile ({N_BATCH} unique sources) ==")
    src = mediums["medium (~6 KB)"]
    sources = [_make_unique(src, i) for i in range(N_BATCH)]
    env = jinja2.Environment()  # nosemgrep: missing-autoescape-disabled
    stock_ms = _median_us(lambda: [env.from_string(s) for s in sources], runs=3) / 1e3
    seq_ms = _median_us(lambda: jinja2_mojo.compile_batch(sources, cache=False), runs=3) / 1e3
    par_ms = _median_us(lambda: jinja2_mojo.compile_batch(sources, cache=False, workers=4), runs=3) / 1e3
    print(f"{'stock from_string':>22}: {stock_ms:8.1f} ms")
    print(f"{'batch (sequential)':>22}: {seq_ms:8.1f} ms ({stock_ms/seq_ms:5.2f}x)")
    print(f"{'batch (workers=4)':>22}: {par_ms:8.1f} ms ({stock_ms/par_ms:5.2f}x)")


if __name__ == "__main__":
    main()
