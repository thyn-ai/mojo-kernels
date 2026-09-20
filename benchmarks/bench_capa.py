#!/usr/bin/env python3
"""Reproducible benchmark: capa's engine.match vs capa_mojo (rule evaluation).

Workload: 1000 synthesized rules in capa's documented YAML format (seeded;
and/or/not/optional/N-or-more/count with exact/regex/substring/bytes leaves
and `match:` dependencies, mirroring the capa-rules grammar mix) matched
against 2000 function scopes of 25-130 features each — the shape of matching
one rule set against one program's functions. Feature extraction is NOT part
of this benchmark (capa_mojo does not do extraction); both sides receive the
same pre-built feature maps, so this measures rule evaluation exactly.

Timings:
  * cold: one-time costs — rule parse/compile, then the first full match pass;
  * warm: median of 5 repeat full match passes over the same 2000 scopes.

Correctness is asserted (identical matched rule-name sets per scope vs
flare-capa's engine.match) before any timing happens, so the numbers always
come from a verified-correct build.

Run from the repository root (oracle in .oracle-capa, wrapper in python/):

    PYTHONPATH="python/capa_mojo:.oracle-capa" PYTHONNOUSERSITE=1 pixi run python benchmarks/bench_capa.py
"""

from __future__ import annotations

import collections
import json
import platform
import random
import statistics
import subprocess
import sys
import time

N_RULES = 1000
N_SCOPES = 2000
N_RUNS = 5  # median over this many warm passes
SEED = 20260919

_APIS = ["CreateFileA", "kernel32.CreateFileW", "ReadFile", "VirtualAlloc",
        "WriteFile", "RegOpenKeyA", "ntdll.NtCreateFile", "GetProcAddress",
        "LoadLibraryA", "CreateMutexA", "WSAStartup", "InternetOpenA"]
_NUMBERS = [0x10, 0x20, 0x40, 0x7FFE02D4, -1, 0x100, 0x1000, 0xFF, 42, 0x80000000]
_MNEMONICS = ["push", "mov", "xor", "shr", "ror", "call", "cmp", "test", "lea", "jmp"]
_CHARACTERISTICS = ["nzxor", "peb access", "fs access", "gs access", "indirect call",
                    "call $+5", "cross section flow", "unmanaged call", "tight loop",
                    "stack string", "loop", "recursive call", "calls from", "calls to"]
_STRINGS = ["hello world", "debug", "SQLite format 3", "Install_Dir",
            "Software\\FileZilla", "sitemanager.xml", "user-agent: test/1.0",
            "Content-Type: application/x-www-form-urlencoded", "POST", "GET"]
_SUBSTRINGS = ["sitemanager", "FileZilla", "format", "gent:", "Content", "POST"]
_REGEXES = ["/str[0-9]+end/", "/mix[0-9]/i", "/^hello/", "/a.c/", "/[0-9]{1,3}\\./"]
_OFFSETS = [0x0, 0x8, 0x40, -0x10, 0x100, 0x2C]
_BYTES = ["00 11 22", "AA BB CC DD", "4D 5A", "FF D0", "55 8B EC", "48 89 5C 24"]


def gen_rules(rng: random.Random, n_rules: int) -> tuple[list[str], dict]:
    """Seeded rule-set generator in capa's documented YAML subset (function
    scope; the same grammar mix as the differential suite)."""
    universe: dict[tuple, None] = {}

    def leaf() -> str:
        choice = rng.random()
        if choice < 0.20:
            api = rng.choice(_APIS)
            universe[("api", api.split(".")[-1] if api.count(".") == 1 and "::" not in api and ".#" not in api else api)] = None
            return f"api: {api}"
        if choice < 0.34:
            n = rng.choice(_NUMBERS)
            universe[("number", n)] = None
            return f"number: {n}"
        if choice < 0.44:
            m = rng.choice(_MNEMONICS)
            universe[("mnemonic", m)] = None
            return f"mnemonic: {m}"
        if choice < 0.52:
            c = rng.choice(_CHARACTERISTICS)
            universe[("characteristic", c)] = None
            return f"characteristic: {c}"
        if choice < 0.62:
            s = rng.choice(_STRINGS)
            universe[("string", s)] = None
            return f"string: {json.dumps(s)}"
        if choice < 0.69:
            return f"substring: {json.dumps(rng.choice(_SUBSTRINGS))}"
        if choice < 0.76:
            return f"string: {json.dumps(rng.choice(_REGEXES))}"
        if choice < 0.83:
            return f"bytes: {rng.choice(_BYTES)}"
        if choice < 0.91:
            o = rng.choice(_OFFSETS)
            universe[("offset", o)] = None
            return f"offset: {o}"
        p = rng.choice([0x40, 0x44])
        universe[("property", p)] = None
        return f"property: {p}"

    def count_term() -> str:
        kind = rng.choice(["number", "mnemonic", "api", "characteristic"])
        if kind == "number":
            v = rng.choice([n for n in _NUMBERS if isinstance(n, int)])
            universe[("number", v)] = None
        elif kind == "mnemonic":
            v = rng.choice(_MNEMONICS)
            universe[("mnemonic", v)] = None
        elif kind == "api":
            v = rng.choice(_APIS)
            universe[("api", v.split(".")[-1] if v.count(".") == 1 and "::" not in v and ".#" not in v else v)] = None
        else:
            v = rng.choice(_CHARACTERISTICS)
            universe[("characteristic", v)] = None
        spec = rng.choice(["1", "2", "2 or more", "3 or fewer", "(1, 3)", "(0, 1)"])
        return f"count({kind}({v})): {spec}"

    def stmt(depth: int, match_pool: list[str]) -> str:
        if depth >= 3 or rng.random() < 0.36:
            if match_pool and rng.random() < 0.08:
                return f"match: {rng.choice(match_pool)}"
            return leaf()
        kind = rng.random()
        n = rng.randint(2, 4)
        pad = "  " * (depth + 3)
        if kind < 0.34:
            children = [stmt(depth + 1, match_pool) for _ in range(n)]
            return "and:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.60:
            children = [stmt(depth + 1, match_pool) for _ in range(n)]
            return "or:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.66:
            return f"not:\n{pad}- " + stmt(depth + 1, match_pool)
        if kind < 0.74:
            k = rng.randint(1, n)
            children = [stmt(depth + 1, match_pool) for _ in range(n)]
            return f"{k} or more:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.80:
            children = [stmt(depth + 1, match_pool) for _ in range(2)]
            return "optional:\n" + "\n".join(f"{pad}- {c}" for c in children)
        return count_term()

    rule_texts = []
    # match: targets are drawn only from dependency-free rules (and namespaces
    # containing only such rules), so the dependency graph stays shallow —
    # both topological sorters are recursive and very deep match chains would
    # exceed Python's recursion limit in EITHER engine.
    dep_free: list[str] = []
    dep_free_ns: list[str] = []
    for i in range(n_rules):
        ns = f"bench/ns{i % 23}" if rng.random() < 0.30 else None
        name = f"bench rule {i}"
        meta_ns = f"\n    namespace: {ns}" if ns else ""
        # a rule may never reference itself, including through its own
        # namespace — that is a dependency cycle the reference engine's
        # topological sorter cannot order either.
        body = stmt(0, dep_free + [n for n in dep_free_ns if n != ns])
        rule_texts.append(
            f"rule:\n  meta:\n    name: {name}{meta_ns}\n    authors: [bench]\n"
            f"    scopes:\n      static: function\n      dynamic: unsupported\n"
            f"  features:\n    - {body}\n"
        )
        if "match: " not in body:
            dep_free.append(name)
            if ns:
                dep_free_ns.append(ns)
    return rule_texts, universe


def gen_scopes(rng: random.Random, universe: dict, n_scopes: int) -> list[tuple[int, dict]]:
    keys = sorted(universe.keys(), key=repr)
    scopes = []
    for s in range(n_scopes):
        fmap: dict[tuple, set] = {}
        base = 0x600000 + s * 0x1000
        for key in keys:
            r = rng.random()
            if r < 0.10:
                fmap[key] = {base + 4 * k for k in range(rng.randint(1, 5))}
            elif r < 0.12:
                fmap[key] = set()
        # realistic noise: extracted strings/bytes the rules never reference
        for k in range(rng.randint(3, 12)):
            fmap[("string", f"noise-{s}-{k} str{k}end")] = {base + 0x800 + 4 * k}
        for k in range(rng.randint(0, 3)):
            fmap[("bytes", bytes([rng.randrange(256) for _ in range(8)]))] = {base + 0x900 + 4 * k}
        for k in range(rng.randint(0, 4)):
            fmap[("api", f"NoiseApi{k}")] = {base + 0xA00 + 4 * k}
        scopes.append((0x400000 + s * 0x1000, fmap))
    return scopes


def to_capa_feature(key):
    """Map one canonical (name, value) feature key to a capa Feature object."""
    import capa.features.basicblock
    import capa.features.common
    import capa.features.file
    import capa.features.insn

    name, value = key
    insn = capa.features.insn
    common = capa.features.common
    file_ = capa.features.file
    table = {
        "api": insn.API,
        "string": common.String,
        "substring": common.Substring,
        "regex": common.Regex,
        "bytes": common.Bytes,
        "number": insn.Number,
        "offset": insn.Offset,
        "mnemonic": insn.Mnemonic,
        "characteristic": common.Characteristic,
        "export": file_.Export,
        "import": file_.Import,
        "section": file_.Section,
        "function-name": file_.FunctionName,
        "os": common.OS,
        "arch": common.Arch,
        "format": common.Format,
        "class": common.Class,
        "namespace": common.Namespace,
        "property": insn.Property,
        "match": common.MatchedRule,
    }
    if name == "basicblock":
        return capa.features.basicblock.BasicBlock()
    if name.startswith("property/"):
        return insn.Property(value, access=name[len("property/") :])
    return table[name](value)


def AVA(addr: int):
    from capa.features.address import AbsoluteVirtualAddress

    return AbsoluteVirtualAddress(addr)


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
    import numpy

    lines.append(f"- python: {platform.python_version()}, numpy: {numpy.__version__}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def main() -> None:
    import capa.engine
    import capa.rules

    import capa_mojo

    info = capa_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(f"- capa_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
          f"({info.get('native_source') or info.get('error')})")
    try:
        from capa.version import __version__ as capa_version
    except Exception:
        import importlib.metadata

        capa_version = importlib.metadata.version("flare-capa")
    print(f"- flare-capa: {capa_version}")
    print(f"- workload: {N_RULES} rules x {N_SCOPES} scopes, seed {SEED}; warm: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'capa_mojo'")

    rng = random.Random(SEED)
    rule_texts, universe = gen_rules(rng, N_RULES)
    scopes = gen_scopes(rng, universe, N_SCOPES)
    n_features = sum(len(f) for _, f in scopes)
    print(f"- generated {len(rule_texts)} rules, {len(scopes)} scopes, {n_features} feature entries")

    # -- cold: parse / compile ------------------------------------------------
    t0 = time.perf_counter()
    capa_rules = [capa.rules.Rule.from_yaml(text) for text in rule_texts]
    ordered = capa.rules.topologically_order_rules(capa_rules)
    t_parse_oracle = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours = capa_mojo.load_rules(rule_texts)
    t_parse_ours = time.perf_counter() - t0

    capa_scopes = []
    for addr, fmap in scopes:
        features = collections.defaultdict(set)
        for key, locs in fmap.items():
            features[to_capa_feature(key)] = {AVA(loc) for loc in locs}
        capa_scopes.append((addr, features))

    # -- correctness gate: identical matched-name sets on every scope --------
    print("\n== correctness gate (matched rule-name sets vs capa.engine.match) ==")
    oracle_names = []
    for addr, features in capa_scopes:
        _, results = capa.engine.match(ordered, features, AVA(addr))
        oracle_names.append(set(results.keys()))
    actual = ours.match(scopes)
    mismatches = sum(
        1 for names, res in zip(oracle_names, actual) if names != set(res.matches.keys())
    )
    print(f"  {len(scopes)} scopes compared, {mismatches} mismatched")
    if mismatches:
        sys.exit("correctness gate failed")

    # -- cold: first full match pass ------------------------------------------
    t0 = time.perf_counter()
    for addr, features in capa_scopes:
        capa.engine.match(ordered, features, AVA(addr))
    t_first_oracle = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours.match(scopes)
    t_first_ours = time.perf_counter() - t0

    # -- warm: median of N_RUNS full passes ------------------------------------
    def oracle_pass():
        for addr, features in capa_scopes:
            capa.engine.match(ordered, features, AVA(addr))

    def ours_pass():
        ours.match(scopes)

    warm_oracle = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        oracle_pass()
        warm_oracle.append(time.perf_counter() - t0)
    warm_ours = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        ours_pass()
        warm_ours.append(time.perf_counter() - t0)
    med_oracle = statistics.median(warm_oracle)
    med_ours = statistics.median(warm_ours)

    print("\n== results ==")
    print(f"{'phase':>28} | {'capa engine.match':>18} | {'capa_mojo':>12} | {'speedup':>8}")
    print(f"{'-' * 28}-+-{'-' * 18}-+-{'-' * 12}-+-{'-' * 8}")
    rows = [
        ("rule parse/compile (cold)", t_parse_oracle, t_parse_ours),
        ("first match pass (cold)", t_first_oracle, t_first_ours),
        ("warm match pass (median)", med_oracle, med_ours),
    ]
    for label, a, b in rows:
        print(f"{label:>28} | {a:>17.3f}s | {b:>11.4f}s | {a / b:>7.1f}x")
    print(f"\nwarm pass: {N_SCOPES / med_oracle:,.0f} vs {N_SCOPES / med_ours:,.0f} scopes/sec; "
          f"{1e3 * med_oracle / N_SCOPES:.3f} vs {1e3 * med_ours / N_SCOPES:.4f} ms/scope")

    print("\n== README paste block ==")
    print("| phase | capa `engine.match` | capa_mojo (native) | speedup |")
    print("|---|---:|---:|---:|")
    for label, a, b in rows:
        print(f"| {label} | {a:.3f}s | {b:.4f}s | **{a / b:.1f}x** |")
    print(f"\nwarm: {1e3 * med_oracle / N_SCOPES:.3f} ms/scope -> {1e3 * med_ours / N_SCOPES:.4f} ms/scope "
          f"({N_SCOPES / med_ours:,.0f} scopes/sec)")


if __name__ == "__main__":
    main()
