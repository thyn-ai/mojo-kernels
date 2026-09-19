#!/usr/bin/env python3
"""Reproducible benchmark: vaderSentiment vs vader_mojo (native backend).

A 2000-sentence corpus is generated locally from a fixed seed (no network, no
datasets): social-media-length sentences mixing VADER lexicon words, boosters,
negators, emoticons, emojis, and punctuation floods. Timings are the median of
5 runs. Two regimes are measured for both the reference and vader_mojo:

- cold: import + analyzer construction + first polarity_scores call, each in a
  fresh Python process (median of 5);
- warm: steady-state scoring of the full corpus (median of 5 batch runs).

Correctness is asserted (exact dict parity vs vaderSentiment on a sample)
before any timing happens, so the numbers below always come from a
verified-correct build.

Run from the repository root (the script puts the package and, if needed, the
cached oracle on sys.path itself):

    pixi run python benchmarks/bench_vader.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_DIR = os.path.join(REPO_ROOT, "python", "vader_mojo")
ORACLE_DIR = os.environ.get(
    "VADER_ORACLE_DIR", os.path.expanduser("~/.cache/vader-oracle/pkg")
)
for path in (PKG_DIR, ORACLE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

N_SENTENCES = 2_000
N_RUNS = 5  # median over this many runs
N_COLD = 5  # fresh processes for the cold measurement
N_CORRECTNESS = 100
SEED = 20260919

_BOOSTERS = ["very", "really", "so", "uber", "friggin", "kind of", "almost", "slightly", "REALLY"]
_NEGATORS = ["not", "never", "no", "isn't", "can't", "won't", "without", "nor", "nothing", "hardly"]
_PUNCT = ["!", "!!", "!!!", "!!!!", "?", "??", "???", "?!", ".", "...", ","]
_EMOTICONS = [":)", ":(", ":D", ":-)", ":p", ";)", "lol", "xd"]
_EMOJIS = ["💘", "💋", "😁", "😢", "😡", "👍", "👎", "💔", "🎉", "😴"]
_IDIOMS = ["but", "least", "the shit", "badass", "yeah right", "to die for", "so", "this", "kind", "of"]
_FILLER = ["zzz", "qwerty", "lorem", "123", "can't-even"]


def make_corpus(seed: int) -> list[str]:
    from vader_mojo.core import _load_lexicon

    lex_words = list(_load_lexicon())
    rng = random.Random(seed)
    corpus = []
    for _ in range(N_SENTENCES):
        parts = []
        for _ in range(rng.randint(3, 30)):
            r = rng.random()
            if r < 0.50:
                parts.append(rng.choice(lex_words))
            elif r < 0.62:
                parts.append(rng.choice(_BOOSTERS))
            elif r < 0.72:
                parts.append(rng.choice(_NEGATORS))
            elif r < 0.80:
                parts.append(rng.choice(_IDIOMS))
            elif r < 0.87:
                parts.append(rng.choice(_EMOTICONS))
            elif r < 0.91:
                parts.append(rng.choice(_EMOJIS))
            else:
                parts.append(rng.choice(_FILLER))
        s = " ".join(parts)
        if rng.random() < 0.5:
            s += rng.choice(_PUNCT)
        if rng.random() < 0.2:
            s = s.upper()
        corpus.append(s)
    return corpus


_COLD_CODE = {
    "vaderSentiment": (
        "import time\n"
        "t0 = time.perf_counter()\n"
        "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\n"
        "a = SentimentIntensityAnalyzer()\n"
        "a.polarity_scores('VADER is smart, handsome, and funny!')\n"
        "print(f'{time.perf_counter() - t0:.6f}')\n"
    ),
    "vader_mojo": (
        "import time\n"
        "t0 = time.perf_counter()\n"
        "import vader_mojo\n"
        "vader_mojo.polarity_scores('VADER is smart, handsome, and funny!')\n"
        "print(f'{time.perf_counter() - t0:.6f}')\n"
    ),
}


def cold_seconds(code: str) -> float:
    """Median seconds for import + construction + first call, fresh processes."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (PKG_DIR, ORACLE_DIR, env.get("PYTHONPATH", "")) if p
    )
    samples = []
    for _ in range(N_COLD):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


def warm_seconds(score, corpus: list[str]) -> float:
    """Median seconds for one full pass over the corpus."""
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for s in corpus:
            score(s)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    lines.append(f"- python: {platform.python_version()}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def main() -> None:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as RefAnalyzer

    import vader_mojo
    from vader_mojo import SentimentIntensityAnalyzer

    info = vader_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- vader_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    import vaderSentiment

    print(f"- vaderSentiment: {getattr(vaderSentiment, '__version__', '3.3.2 (PyPI)')}")
    print(f"- corpus: {N_SENTENCES} seeded sentences (seed={SEED}); warm: median of {N_RUNS} "
          f"batch runs; cold: median of {N_COLD} fresh processes")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'vader_mojo'")

    corpus = make_corpus(SEED)

    print("\n== correctness gate (exact dict parity) ==")
    ref = RefAnalyzer()
    ours = SentimentIntensityAnalyzer()
    assert ours.backend == "native"
    for s in corpus[:N_CORRECTNESS]:
        if ours.polarity_scores(s) != ref.polarity_scores(s):
            sys.exit(f"correctness gate failed on {s!r}")
    print(f"  {N_CORRECTNESS}/{N_CORRECTNESS} sentences: exact dict parity  [OK]")

    print("\n== cold: import + analyzer + first call (median of fresh processes) ==")
    cold_ref = cold_seconds(_COLD_CODE["vaderSentiment"])
    cold_ours = cold_seconds(_COLD_CODE["vader_mojo"])
    print(f"{'':>12} | {'vaderSentiment':>14} | {'vader_mojo':>10} | {'speedup':>8}")
    print(f"{'-' * 12}-+-{'-' * 14}-+-{'-' * 10}-+-{'-' * 8}")
    print(f"{'cold (ms)':>12} | {cold_ref * 1e3:>14.1f} | {cold_ours * 1e3:>10.1f} "
          f"| {cold_ref / cold_ours:>7.2f}x")

    print("\n== warm: steady-state scoring, full corpus ==")
    t_ref = warm_seconds(ref.polarity_scores, corpus)
    t_ours = warm_seconds(ours.polarity_scores, corpus)
    ms_ref, ms_ours = 1e3 * t_ref / N_SENTENCES, 1e3 * t_ours / N_SENTENCES
    print(f"{'':>12} | {'vaderSentiment':>14} | {'vader_mojo':>10} | {'speedup':>8}")
    print(f"{'-' * 12}-+-{'-' * 14}-+-{'-' * 10}-+-{'-' * 8}")
    print(f"{'ms/sentence':>12} | {ms_ref:>14.3f} | {ms_ours:>10.3f} | {ms_ref / ms_ours:>7.1f}x")
    print(f"{'sentences/sec':>12} | {N_SENTENCES / t_ref:>14,.0f} | {N_SENTENCES / t_ours:>10,.0f} "
          f"| {(N_SENTENCES / t_ours) / (N_SENTENCES / t_ref):>7.1f}x")

    # For context only: the vendored pure-Python fallback (never shipped as the
    # headline number) on the same corpus.
    from vader_mojo import _reference
    from vader_mojo.core import _load_emojis, _load_lexicon, _round_scores

    fallback = _reference.PurePythonAnalyzer(_load_lexicon(), _load_emojis())
    t_fb = warm_seconds(lambda s: _round_scores(fallback.polarity(s)), corpus)
    print(f"{'fallback ms/sent':>12} | {'(context only)':>14} | {1e3 * t_fb / N_SENTENCES:>10.3f} "
          f"| {ms_ref / (1e3 * t_fb / N_SENTENCES):>7.2f}x")

    print("\n== README paste block ==")
    print("| regime | vaderSentiment | vader_mojo (native) | speedup |")
    print("|---|---:|---:|---:|")
    print(f"| cold: import + first call (ms, median of {N_COLD} fresh processes) "
          f"| {cold_ref * 1e3:.1f} | {cold_ours * 1e3:.1f} | {cold_ref / cold_ours:.2f}x |")
    print(f"| warm: ms/sentence (median of {N_RUNS} runs, {N_SENTENCES} sentences) "
          f"| {ms_ref:.3f} | {ms_ours:.3f} | {ms_ref / ms_ours:.1f}x |")
    print(f"| warm: sentences/sec | {N_SENTENCES / t_ref:,.0f} | {N_SENTENCES / t_ours:,.0f} "
          f"| {(N_SENTENCES / t_ours) / (N_SENTENCES / t_ref):.1f}x |")


if __name__ == "__main__":
    main()
