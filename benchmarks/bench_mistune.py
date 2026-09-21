#!/usr/bin/env python3
"""Reproducible benchmark: mistune.markdown vs mistune_mojo (native kernel and
pure-Python engine).

Documents are generated locally from fixed seeds (no network, no datasets):
a small README-style page (~1 KB), a medium documentation page (~10 KB), and
a large reference document (~100 KB). Timings are the median of 5 runs.
"Cold" = first render of each of N distinct documents (steady per-render
cost including any first-touch page effects, no result caching anywhere —
neither mistune nor mistune-mojo caches results); "warm" = steady-state
per-render cost over R repeats of the same document. Correctness (byte
identity with mistune 3.3.4) is asserted before any timing happens, so the
numbers below always come from a verified-correct build.

Run from the repository root:

    PYTHONPATH=python/mistune_mojo python benchmarks/bench_mistune.py
"""

from __future__ import annotations

import platform
import random
import statistics
import subprocess
import sys
import time

import mistune  # oracle, 3.3.4

N_DOCS = 30  # distinct documents per size cell (cold)
N_RUNS = 5  # median over this many runs
WARM_REPS = 50  # repeats per warm cell
SEED = 20260920
SIZES = {"small (~1KB)": 12, "medium (~10KB)": 120, "large (~100KB)": 1200}

FRAGS = [
    "# Title\n\n",
    "## Section with *emphasis* and **strong** text\n\n",
    "Body text with `code spans`, [links](https://example.com/page?a=1&b=2), "
    "and _inline emphasis_ distributed through the prose.\n\n",
    "- list item one\n- list item two with `code`\n  - nested item\n  - second nested\n\n",
    "1. ordered first\n2. ordered second\n3. ordered third\n\n",
    "> a block quote with [a link](/url) and *emphasis*\n> spanning lines\n\n",
    "```python\ndef render(doc):\n    return parse(doc) & 0xFF\n```\n\n",
    "Reference-style [link one][r1] and [link two][r2] inline.\n\n"
    "[r1]: https://example.com/one \"First\"\n[r2]: https://example.com/two \"Second\"\n\n",
    "Hard line break here  \nand a soft break\nfollowed by more text.\n\n",
    "***\n\n",
    "Inline raw HTML <span class=\"x\">escaped</span> and entities &copy; &#169;.\n\n",
    "![image alt](/images/pic.png \"caption\")\n\n",
    "Setext heading level two\n------------------------\n\n",
    "Final paragraph with *nested **strong** emphasis* and <user@example.com>.\n\n",
]


def make_doc(seed: int, n_frags: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(FRAGS) for _ in range(n_frags))


def time_median(fn, docs, reps):
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for d in docs:
            for _ in range(reps):
                fn(d)
        samples.append((time.perf_counter() - t0) / (len(docs) * reps))
    return statistics.median(samples)


def main() -> None:
    if mistune.__version__ != "3.3.4":
        sys.exit(f"bench pins mistune==3.3.4, found {mistune.__version__}")
    import mistune_mojo
    from mistune_mojo import _reference

    info = mistune_mojo.backend_info()
    if not info["native_available"]:
        sys.exit(f"native kernel unavailable: {info['error']}")

    machine = f"{platform.processor() or platform.machine()}, {platform.system()}"
    py = platform.python_version()
    import subprocess as sp

    mojo_v = sp.run(
        ["mojo", "--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"# mistune-mojo benchmark")
    print(f"# {machine}, Python {py}, {mojo_v}, mistune {mistune.__version__}")
    print(f"# median of {N_RUNS} runs; correctness asserted before timing\n")

    rows = []
    for label, n_frags in SIZES.items():
        docs = [make_doc(SEED + i, n_frags) for i in range(N_DOCS)]
        # correctness gate on the whole cell
        for d in docs:
            want = mistune.markdown(d)
            assert mistune_mojo.markdown(d) == want, label
            assert _reference.markdown(d) == want, label
        size_kb = sum(len(d) for d in docs) / len(docs) / 1024

        # cold: render each distinct document once
        t_cold_ref = time_median(mistune.markdown, docs, 1)
        t_cold_mojo = time_median(mistune_mojo.markdown, docs, 1)
        t_cold_fb = time_median(_reference.markdown, docs, 1)
        # warm: repeat the same document
        doc = docs[0]
        one = [doc]
        t_warm_ref = time_median(mistune.markdown, one, WARM_REPS)
        t_warm_mojo = time_median(mistune_mojo.markdown, one, WARM_REPS)
        t_warm_fb = time_median(_reference.markdown, one, WARM_REPS)

        rows.append(
            (
                f"{label} ({size_kb:.1f} KB)",
                t_cold_ref,
                t_cold_mojo,
                t_cold_fb,
                t_warm_ref,
                t_warm_mojo,
                t_warm_fb,
            )
        )

    print("| workload | cold mistune | cold native | cold fallback | speedup | warm mistune | warm native | warm fallback | speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, cr, cm, cf, wr, wm, wf in rows:
        print(
            f"| {name} | {cr*1e6:.0f} us | {cm*1e6:.0f} us | {cf*1e6:.0f} us "
            f"| {cr/cm:.2f}x | {wr*1e6:.0f} us | {wm*1e6:.0f} us | {wf*1e6:.0f} us "
            f"| {wr/wm:.2f}x |"
        )


if __name__ == "__main__":
    main()
