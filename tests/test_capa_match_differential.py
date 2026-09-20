"""Differential tests: capa_mojo must match capa's engine.match exactly.

Run twice by `scripts/test_all_capa.sh`: once against the native Mojo kernel
and once with CAPA_MOJO_DISABLE_NATIVE=1 (forced pure-Python fallback). Both
backends must agree with the oracle — the published PyPI package
flare-capa==9.2.0, provisioned into .oracle-capa/ — on:

  * the set of matched rule names per scope, and
  * the full per-rule match detail tree (statement-by-statement success and
    feature locations), canonicalized to plain tuples.

Agreement is exact (zero tolerance): match results are booleans and integer
locations, so there is nothing to approximate.

The real rules below are verbatim copies from mandiant/capa-rules v9.2.0
(Apache-2.0, (c) Mandiant), limited to the subset this kernel supports; the
remaining rules are synthesized in the documented capa YAML format.
"""

from __future__ import annotations

import collections
import json
import os
import random

import pytest

import capa.engine
import capa.features.basicblock
import capa.features.common
import capa.features.file
import capa.features.insn
import capa.rules
from capa.features.address import AbsoluteVirtualAddress as AVA

import capa_mojo
from capa_mojo.features import UnsupportedRuleError, trim_dll_part

# ---------------------------------------------------------------------------
# oracle bridge
# ---------------------------------------------------------------------------

_COMMON = capa.features.common
# characteristics the reference rule parser accepts per scope used here
# (function/basic-block scopes inherit the instruction-scope set)
_FUNCTION_CHARACTERISTICS = [
    "calls from", "calls to", "loop", "recursive call",
    "tight loop", "stack string",
    "nzxor", "peb access", "fs access", "gs access", "indirect call",
    "call $+5", "cross section flow", "unmanaged call",
]
_FILE_CHARACTERISTICS = ["embedded pe", "mixed mode", "forwarded export"]


def to_capa_feature(key):
    """Map one canonical (name, value) feature key to a capa Feature."""
    name, value = key
    insn = capa.features.insn
    file_ = capa.features.file
    table = {
        "api": insn.API,
        "string": _COMMON.String,
        "substring": _COMMON.Substring,
        "regex": _COMMON.Regex,
        "bytes": _COMMON.Bytes,
        "number": insn.Number,
        "offset": insn.Offset,
        "mnemonic": insn.Mnemonic,
        "characteristic": _COMMON.Characteristic,
        "export": file_.Export,
        "import": file_.Import,
        "section": file_.Section,
        "function-name": file_.FunctionName,
        "os": _COMMON.OS,
        "arch": _COMMON.Arch,
        "format": _COMMON.Format,
        "class": _COMMON.Class,
        "namespace": _COMMON.Namespace,
        "property": insn.Property,
        "match": _COMMON.MatchedRule,
    }
    if name == "basicblock":
        return capa.features.basicblock.BasicBlock()
    if name.startswith("property/"):
        return insn.Property(value, access=name[len("property/") :])
    cls = table.get(name)
    if cls is None:
        raise AssertionError(f"test generator emitted unknown feature {key!r}")
    return cls(value)


def _capa_feature_key(feature):
    if isinstance(feature, capa.features.insn.Property):
        name = f"property/{feature.access}" if feature.access else "property"
        return (name, feature.value)
    if isinstance(feature, capa.features.basicblock.BasicBlock):
        return ("basicblock", 0)
    return (feature.name, feature.value)


def canon_capa(node) -> tuple:
    """Canonicalize a capa.engine.Result into the same tuple layout that
    capa_mojo._reference.eval_detail produces."""
    st = node.statement
    locs = tuple(sorted(int(a) for a in node.locations))
    success = bool(node)
    eng = capa.engine
    if isinstance(st, eng.And):
        return ("and", success, tuple(canon_capa(c) for c in node.children))
    if isinstance(st, eng.Or):
        return ("or", success, tuple(canon_capa(c) for c in node.children))
    if isinstance(st, eng.Not):
        return ("not", success, tuple(canon_capa(c) for c in node.children))
    if isinstance(st, eng.Some):
        return ("some", success, st.count, tuple(canon_capa(c) for c in node.children))
    if isinstance(st, eng.Range):
        return ("range", success, _capa_feature_key(st.child), st.min, st.max, locs)
    # feature leaves; isinstance order matters (Regex/Substring subclass String)
    if isinstance(st, (_COMMON.Regex, _COMMON.Substring)):
        kind = "regex" if isinstance(st, _COMMON.Regex) else "substring"
        matches = tuple(
            sorted((s, tuple(sorted(int(a) for a in ls))) for s, ls in st.matches.items())
        )
        return (kind, success, st.value, locs, matches)
    if isinstance(st, _COMMON.Bytes):
        return ("bytes", success, st.value, locs)
    if isinstance(st, _COMMON.MatchedRule):
        return ("match", success, st.value, locs)
    if isinstance(st, capa.features.insn.Property):
        name = f"property/{st.access}" if st.access else "property"
        return ("feature", success, name, st.value, locs)
    if isinstance(st, capa.features.basicblock.BasicBlock):
        return ("feature", success, "basicblock", 0, locs)
    return ("feature", success, st.name, st.value, locs)


def oracle_match(rule_texts, scopes):
    """Run capa's engine.match over the given scopes.

    `rule_texts`: list of rule YAML strings. `scopes`: list of
    (addr, feature_map) with canonical keys. Returns, per scope, a dict of
    rule name -> canonical detail tree.
    """
    rules = [capa.rules.Rule.from_yaml(text) for text in rule_texts]
    ordered = capa.rules.topologically_order_rules(rules)
    out = []
    for addr, fmap in scopes:
        features = collections.defaultdict(set)
        for key, locs in fmap.items():
            features[to_capa_feature(key)] = {AVA(loc) for loc in locs}
        _, results = capa.engine.match(ordered, features, AVA(addr))
        out.append({name: canon_capa(hits[0][1]) for name, hits in results.items()})
    return out


def assert_full_parity(rule_texts, scopes, backend_expected):
    """The one assertion every parity test funnels through: identical matched
    sets and identical detail trees, on the backend this run exercises."""
    ours = capa_mojo.load_rules(rule_texts)
    assert ours.backend == backend_expected
    oracle = oracle_match(rule_texts, scopes)
    actual = ours.match(scopes)
    assert len(actual) == len(oracle)
    for scope_idx, (scope_result, oracle_matches) in enumerate(zip(actual, oracle)):
        assert set(scope_result.matches.keys()) == set(oracle_matches.keys()), (
            f"scope {scope_idx}: matched rule sets differ: "
            f"ours={sorted(scope_result.matches.keys())} oracle={sorted(oracle_matches.keys())}"
        )
        for name, oracle_tree in oracle_matches.items():
            assert scope_result.matches[name] == oracle_tree, (
                f"scope {scope_idx}, rule {name!r}: detail trees differ:\n"
                f"ours:   {scope_result.matches[name]!r}\noracle: {oracle_tree!r}"
            )


@pytest.fixture()
def backend_expected() -> str:
    # scripts/test_all_capa.sh runs the suite once per backend.
    return "fallback" if os.environ.get("CAPA_MOJO_DISABLE_NATIVE") == "1" else "native"


def make_rule(name, body, namespace=None, scope="function"):
    ns = f"\n    namespace: {namespace}" if namespace else ""
    return (
        f"rule:\n  meta:\n    name: {name}{ns}\n    authors: [capa_mojo test]\n"
        f"    scopes:\n      static: {scope}\n      dynamic: unsupported\n"
        f"  features:\n{body}\n"
    )


# ---------------------------------------------------------------------------
# grammar matrix: every statement and feature form, driven to true and false
# ---------------------------------------------------------------------------

GRAMMAR_RULES = [
    make_rule(
        "and-or-not",
        """    - and:
      - or:
        - api: CreateFileA
        - api: kernel32.CreateFileW
      - not:
        - string: "debug\"""",
    ),
    make_rule(
        "optional-present-and-absent",
        """    - and:
      - mnemonic: push
      - optional:
        - characteristic: nzxor
        - offset: 0x40""",
    ),
    make_rule(
        "two-or-more",
        """    - and:
      - number: 0x10
      - 2 or more:
        - mnemonic: push
        - mnemonic: pop
        - mnemonic: mov
        - offset: 0x20""",
    ),
    make_rule(
        "count-exact",
        "    - and:\n      - count(api(CreatePipe)): 2",
    ),
    make_rule(
        "count-or-more",
        "    - and:\n      - count(api(kernel32.QueryPerformanceCounter)): 2 or more",
    ),
    make_rule(
        "count-or-fewer",
        "    - and:\n      - count(mnemonic(shr)): 3 or fewer",
    ),
    make_rule(
        "count-range-bounded",
        "    - and:\n      - count(number(0x10)): (2, 4)",
    ),
    make_rule(
        "count-range-open-min",
        "    - and:\n      - count(number(0x20)): (, 2)",
    ),
    make_rule(
        "count-range-open-max",
        "    - and:\n      - count(number(0x30)): (1, )",
    ),
    make_rule(
        "count-zero-or-more",
        "    - and:\n      - count(number(0x77)): 0 or more",
    ),
    make_rule(
        "count-basic-blocks",
        "    - and:\n      - count(basic blocks): 1 or more",
    ),
    make_rule(
        "count-bytes-exact",
        "    - and:\n      - count(bytes(00 11)): 1",
    ),
    make_rule(
        "count-substring-degenerate",
        "    - and:\n      - count(substring(zzz)): 0",
    ),
    make_rule(
        "substring-hit",
        """    - and:
      - substring: "sitemanager"
      - substring: "FileZilla\"""",
    ),
    make_rule(
        "regex-forms",
        """    - and:
      - string: /str[0-9]+end/
      - string: /MiXeD[0-9]/i""",
    ),
    make_rule(
        "bytes-prefix",
        """    - and:
      - bytes: 00 11 22 33
      - bytes: AA BB = trailing description""",
    ),
    make_rule(
        "number-formats",
        """    - and:
      - number: 0x7FFE02D4 = hex with description
      - number: -1
      - number: 0xFFFFFFFFFFFFFFFF
      - number: 3.5
      - number: 42""",
    ),
    make_rule(
        "property-forms",
        """    - and:
      - property: 0x40
      - property/read: 0x44""",
    ),
    make_rule(
        "global-features",
        """    - and:
      - os: windows
      - arch: amd64
      - format: pe""",
    ),
    make_rule(
        "file-scope-features",
        """    - and:
      - section: .text
      - export: main
      - import: CreateFileA
      - function-name: sub_401000
      - class: Widget
      - namespace: core/io""",
        scope="file",
    ),
    make_rule(
        "duplicate-leaves-dedup",
        """    - and:
      - number: 0x10
      - number: 0x10
      - or:
        - mnemonic: push
        - mnemonic: push
        - offset: 0x8""",
    ),
    make_rule(
        "nested-not",
        """    - or:
      - and:
        - mnemonic: push
        - not:
          - or:
            - number: 0x1
            - number: 0x2
      - string: "fallback string\"""",
    ),
]

GRAMMAR_SCOPES = [
    # scope 0: rich function scope hitting most leaves
    (
        0x401000,
        {
            ("api", "CreateFileA"): {0x401010},
            ("string", "debug"): set(),  # present with empty locations
            ("mnemonic", "push"): {0x401020, 0x401030},
            ("characteristic", "nzxor"): {0x401040},
            ("offset", 0x40): {0x401050},
            ("number", 0x10): {0x401060, 0x401064, 0x401068},
            ("mnemonic", "pop"): {0x401070},
            ("api", "CreatePipe"): {0x401080, 0x401090},
            ("api", "QueryPerformanceCounter"): {0x4010A0, 0x4010B0, 0x4010C0},
            ("mnemonic", "shr"): {0x4010D0, 0x4010D4},
            ("number", 0x20): {0x4010E0},
            ("number", 0x30): {0x4010F0, 0x4010F4},
            ("string", "xx sitemanager yy"): {0x402000},
            ("string", "FileZilla Client"): {0x402010},
            ("string", "my str42end here"): {0x402020},
            ("string", "mixed7"): {0x402030},
            ("bytes", bytes.fromhex("001122334455")): {0x402100},
            ("bytes", bytes.fromhex("AABBCC")): {0x402110},
            ("bytes", bytes.fromhex("0011")): {0x402120},  # exact for count(bytes(00 11))
            ("number", 0x7FFE02D4): {0x402200},
            ("number", -1): {0x402210},
            ("number", 0xFFFFFFFFFFFFFFFF): {0x402220},
            ("number", 3.5): {0x402230},
            ("number", 42): {0x402240},
            ("property", 0x40): {0x402300},
            ("property/read", 0x44): {0x402310},
            ("os", "windows"): set(),  # global features: present, no location
            ("arch", "amd64"): set(),
            ("format", "pe"): set(),
            ("number", 0x1): {0x402400},
            ("string", "fallback string"): {0x402410},
            ("basicblock", 0): {0x402500, 0x402504},
        },
    ),
    # scope 1: sparse scope driving the false branches
    (
        0x403000,
        {
            ("api", "CreateFileW"): {0x403010},  # dll-trimmed form in the map
            ("mnemonic", "mov"): {0x403020},
            ("number", 0x10): {0x403030},  # only 1 location: count ranges fail
            ("api", "CreatePipe"): {0x403040},  # 1 of 2 needed
            ("string", "nothing relevant"): {0x403050},
            ("bytes", bytes.fromhex("001199")): {0x403060},  # wrong prefix tail
            ("os", "linux"): set(),
        },
    ),
    # scope 2: file scope
    (
        0x1000,
        {
            ("section", ".text"): {0x1000},
            ("export", "main"): {0x1010},
            ("import", "CreateFileA"): {0x1020},
            ("function-name", "sub_401000"): {0x1030},
            ("class", "Widget"): set(),
            ("namespace", "core/io"): set(),
            ("string", "SQLite format 3"): {0x3000},
        },
    ),
]


def test_grammar_matrix_full_parity(backend_expected):
    assert_full_parity(GRAMMAR_RULES, GRAMMAR_SCOPES, backend_expected)


# ---------------------------------------------------------------------------
# real rules from mandiant/capa-rules v9.2.0 (verbatim, Apache-2.0)
# ---------------------------------------------------------------------------

REAL_RULES = [
    # communication/named-pipe/create/create-two-anonymous-pipes.yml
    """rule:
  meta:
    name: create two anonymous pipes
    namespace: communication/named-pipe/create
    authors:
      - matthew.williams@mandiant.com
    scopes:
      static: function
      dynamic: span of calls
  features:
    - and:
      - count(api(CreatePipe)): 2""",
    # host-interaction/file-system/write/clear-file-content.yml
    """rule:
  meta:
    name: clear file content
    namespace: host-interaction/file-system/write
    authors:
      - jakeperalta7
    scopes:
      static: function
      dynamic: span of calls
  features:
    - and:
      - api: kernel32.SetEndOfFile
      - not:
        - api: kernel32.SetFilePointer""",
    # anti-analysis/anti-debugging/debugger-detection/
    # check-for-time-delay-via-queryperformancecounter.yml
    """rule:
  meta:
    name: check for time delay via QueryPerformanceCounter
    namespace: anti-analysis/anti-debugging/debugger-detection
    authors:
      - michael.hunhoff@mandiant.com
    scopes:
      static: function
      dynamic: span of calls
  features:
    - and:
      - count(api(kernel32.QueryPerformanceCounter)): 2 or more""",
    # collection/file-managers/gather-filezilla-information.yml
    """rule:
  meta:
    name: gather filezilla information
    namespace: collection/file-managers
    authors:
      - "@_re_fox"
    scopes:
      static: function
      dynamic: span of calls
  features:
    - or:
      - and:
        - substring: "\\\\sitemanager.xml"
        - substring: "\\\\recentservers.xml"
        - substring: "\\\\filezilla.xml"
      - and:
        - substring: "Software\\\\FileZilla"
        - string: "Install_Dir"
        - substring: "Software\\\\FileZilla Client"
      - 3 or more:
        - string: "Server Type"
        - string: "Remote Dir"
        - string: "Server.Port"
        - string: "Server.Host"
        - string: "Server.User"
        - string: "Last Server Type"
        - string: "Last Server Port"
        - string: "Last Server User"
        - string: "Last Server Host"
        - string: "Last Server Pass\"""",
    # linking/static/sqlite3/linked-against-sqlite3.yml
    """rule:
  meta:
    name: linked against sqlite3
    namespace: linking/static/sqlite3
    authors:
      - still@teamt5.org
    scopes:
      static: file
      dynamic: file
  features:
    - or:
      - 3 or more:
        - string: "database corruption"
        - string: "SQLITE_OK"
        - string: "SQLite format 3"
        - string: "sqlite3_extension_init"
        - substring: "cannot INSERT into generated column"
        - substring: "UPSERT not implemented for virtual table"
        - substring: "sqlite3_get_table()"
        - substring: "qualified table names are not allowed on\"""",
]

REAL_SCOPES = [
    # scope 0: triggers pipes, clear-file, qpc, filezilla (branches 1 and 3)
    (
        0x401000,
        {
            ("api", "CreatePipe"): {0x401010, 0x401020},
            ("api", "SetEndOfFile"): {0x401030},
            ("api", "QueryPerformanceCounter"): {0x401040, 0x401050},
            ("string", "C:\\Users\\sitemanager.xml"): {0x402000},
            ("string", "C:\\Users\\recentservers.xml"): {0x402010},
            ("string", "C:\\Users\\filezilla.xml"): {0x402020},
            ("string", "Server Type"): {0x402030},
            ("string", "Remote Dir"): {0x402040},
            ("string", "Server.Port"): {0x402050},
        },
    ),
    # scope 1: clear-file fails via NOT, qpc short by one, filezilla branch 2,
    # sqlite3 via a substring hit
    (
        0x403000,
        {
            ("api", "CreatePipe"): {0x403010},
            ("api", "SetEndOfFile"): {0x403020},
            ("api", "SetFilePointer"): {0x403030},
            ("api", "QueryPerformanceCounter"): {0x403040},
            ("string", "Software\\FileZilla"): {0x404000},
            ("string", "Install_Dir"): {0x404010},
            ("string", "Software\\FileZilla Client"): {0x404020},
            ("string", "SQLITE_OK"): {0x405000},
            ("string", "SQLite format 3"): {0x405010},
            ("string", "cannot INSERT into generated column of t"): {0x405020},
        },
    ),
    # scope 2: near-empty
    (0x5000, {("mnemonic", "push"): {0x5000}}),
]


def test_real_capa_rules_full_parity(backend_expected):
    assert_full_parity(REAL_RULES, REAL_SCOPES, backend_expected)


# ---------------------------------------------------------------------------
# match: / namespace semantics
# ---------------------------------------------------------------------------

MATCH_RULES = [
    make_rule(
        "base api rule",
        "    - and:\n      - api: CreateFileA",
        namespace="file-ops",
    ),
    make_rule(
        "base registry rule",
        "    - and:\n      - api: RegOpenKeyA",
        namespace="registry/win",
    ),
    make_rule(
        "second registry rule",
        "    - and:\n      - api: RegCreateKeyA",
        namespace="registry/win",
    ),
    make_rule(
        "depends on rule name",
        """    - and:
      - match: base api rule
      - mnemonic: push""",
    ),
    make_rule(
        "depends on namespace",
        """    - or:
      - match: registry
      - number: 0x99""",
    ),
    make_rule(
        "depends on deep namespace",
        "    - and:\n      - match: registry/win",
    ),
]

MATCH_SCOPES = [
    (
        0x401000,
        {
            ("api", "CreateFileA"): {0x401010},
            ("mnemonic", "push"): {0x401020},
        },
    ),
    (
        0x402000,
        {
            ("api", "RegOpenKeyA"): {0x402010},
            ("number", 0x99): {0x402020},
        },
    ),
    (0x403000, {("mnemonic", "push"): {0x403010}}),
    # a scope whose INPUT map already carries match features
    (
        0x404000,
        {
            ("match", "base api rule"): {0x404010},
            ("mnemonic", "push"): {0x404020},
            ("match", "registry/win"): set(),
        },
    ),
]


def test_match_and_namespace_parity(backend_expected):
    assert_full_parity(MATCH_RULES, MATCH_SCOPES, backend_expected)


def test_match_forward_reference_reorders(backend_expected):
    # rule order violates dependency order; both engines reorder topologically.
    rules = [
        make_rule(
            "consumer first in file",
            "    - and:\n      - match: producer later in file",
        ),
        make_rule(
            "producer later in file",
            "    - and:\n      - api: VirtualAlloc",
        ),
    ]
    scopes = [
        (0x401000, {("api", "VirtualAlloc"): {0x401010}}),
        (0x402000, {("api", "VirtualFree"): {0x402010}}),
    ]
    assert_full_parity(rules, scopes, backend_expected)


# ---------------------------------------------------------------------------
# seeded randomized differential round (the workhorse)
# ---------------------------------------------------------------------------

_APIS = ["CreateFileA", "kernel32.CreateFileW", "ReadFile", "ws2_32.#1",
        "VirtualAlloc", "ntdll.NtCreateFile", "System.Convert::FromBase64String"]
_NUMBERS = [0x10, 0x20, 0x7FFE02D4, -1, 0xFFFFFFFFFFFFFFFF, 42, 3.5, 0x100]
_MNEMONICS = ["push", "mov", "xor", "shr", "ror", "call", "cmp", "test"]
_STRINGS = ["hello world", "debug", "SQLite format 3", "Install_Dir",
            "Software\\FileZilla", "sitemanager.xml", "user-agent: test/1.0"]
_SUBSTRINGS = ["sitemanager", "FileZilla", "format", "gent:"]
_REGEXES = ["/str[0-9]+end/", "/mix[0-9]/i", "/^hello/", "/a.c/"]
_OFFSETS = [0x0, 0x8, 0x40, -0x10, 0x100]
_SECTIONS = [".text", ".data", ".rsrc"]
_BYTES = ["00 11 22", "AA BB CC DD", "4D 5A", "FF D0"]
_EXPORTS = ["main", "DllMain", "Run"]
_IMPORTS = ["CreateFileA", "RegOpenKeyA", "WSAStartup"]
_CLASSNAMES = ["Widget", "COptimizer", "std::vector"]


def _gen_rules(rng: random.Random, n_rules: int) -> tuple[list[str], dict]:
    """Generate random rules in the documented YAML subset.

    Returns (rule_texts, universe) where universe holds the canonical feature
    keys the rules can reference, used to build matching feature maps.
    """
    universe: dict[tuple, None] = {}

    def leaf(scope: str) -> str:
        if scope == "file":
            return file_leaf()
        choice = rng.random()
        if choice < 0.18:
            api = rng.choice(_APIS)
            universe[("api", trim_dll_part(api))] = None
            return f"api: {api}"
        if choice < 0.32:
            n = rng.choice(_NUMBERS)
            universe[("number", n)] = None
            return f"number: {n}"
        if choice < 0.42:
            m = rng.choice(_MNEMONICS)
            universe[("mnemonic", m)] = None
            return f"mnemonic: {m}"
        if choice < 0.50:
            c = rng.choice(_FUNCTION_CHARACTERISTICS)
            universe[("characteristic", c)] = None
            return f"characteristic: {c}"
        if choice < 0.60:
            s = rng.choice(_STRINGS)
            universe[("string", s)] = None
            return f"string: {json.dumps(s)}"
        if choice < 0.67:
            return f"substring: {json.dumps(rng.choice(_SUBSTRINGS))}"
        if choice < 0.74:
            return f"string: {json.dumps(rng.choice(_REGEXES))}"
        if choice < 0.81:
            return f"bytes: {rng.choice(_BYTES)}"
        if choice < 0.88:
            o = rng.choice(_OFFSETS)
            universe[("offset", o)] = None
            return f"offset: {o}"
        if choice < 0.94:
            universe[("os", "windows")] = None
            return "os: windows"
        p = rng.choice([0x40, 0x44])
        universe[("property", p)] = None
        return f"property: {p}"

    def file_leaf() -> str:
        choice = rng.random()
        if choice < 0.22:
            s = rng.choice(_STRINGS)
            universe[("string", s)] = None
            return f"string: {json.dumps(s)}"
        if choice < 0.36:
            return f"substring: {json.dumps(rng.choice(_SUBSTRINGS))}"
        if choice < 0.44:
            return f"string: {json.dumps(rng.choice(_REGEXES))}"
        if choice < 0.54:
            c = rng.choice(_FILE_CHARACTERISTICS)
            universe[("characteristic", c)] = None
            return f"characteristic: {c}"
        if choice < 0.64:
            s = rng.choice(_SECTIONS)
            universe[("section", s)] = None
            return f"section: {s}"
        if choice < 0.74:
            e = rng.choice(_EXPORTS)
            universe[("export", e)] = None
            return f"export: {e}"
        if choice < 0.84:
            i = rng.choice(_IMPORTS)
            universe[("import", i)] = None
            return f"import: {i}"
        if choice < 0.92:
            c = rng.choice(_CLASSNAMES)
            universe[("class", c)] = None
            return f"class: {c}"
        universe[("os", "windows")] = None
        return "os: windows"

    def count_term(scope: str) -> str:
        if scope == "file":
            kind = rng.choice(["string", "characteristic"])
            if kind == "string":
                v = rng.choice([s for s in _STRINGS if "(" not in s and ")" not in s])
                universe[("string", v)] = None
            else:
                v = rng.choice(_FILE_CHARACTERISTICS)
                universe[("characteristic", v)] = None
        else:
            kind = rng.choice(["number", "mnemonic", "api", "characteristic"])
            if kind == "number":
                v = rng.choice([n for n in _NUMBERS if isinstance(n, int)])
                universe[("number", v)] = None
            elif kind == "mnemonic":
                v = rng.choice(_MNEMONICS)
                universe[("mnemonic", v)] = None
            elif kind == "api":
                v = rng.choice(_APIS)
                universe[("api", trim_dll_part(v))] = None
            else:
                v = rng.choice(_FUNCTION_CHARACTERISTICS)
                universe[("characteristic", v)] = None
        if kind == "string":
            term = f"string({v})"
        else:
            term = f"{kind}({v})"
        spec = rng.choice(["1", "2", "2 or more", "3 or fewer", "(1, 3)", "(0, 1)"])
        return f"count({term}): {spec}"

    def stmt(depth: int, scope: str, match_pool: list[str]) -> str:
        if depth >= 3 or rng.random() < 0.34:
            if match_pool and rng.random() < 0.15:
                return f"match: {rng.choice(match_pool)}"
            return leaf(scope)
        kind = rng.random()
        n = rng.randint(2, 3)
        # children of a statement at `depth` must out-indent its `- ` entry;
        # the root statement sits at `    - ` (4 spaces + dash) in make_rule.
        pad = "  " * (depth + 3)
        if kind < 0.28:
            children = [stmt(depth + 1, scope, match_pool) for _ in range(n)]
            return "and:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.52:
            children = [stmt(depth + 1, scope, match_pool) for _ in range(n)]
            return "or:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.62:
            return f"not:\n{pad}- " + stmt(depth + 1, scope, match_pool)
        if kind < 0.72:
            k = rng.randint(1, n)
            children = [stmt(depth + 1, scope, match_pool) for _ in range(n)]
            return f"{k} or more:\n" + "\n".join(f"{pad}- {c}" for c in children)
        if kind < 0.80:
            children = [stmt(depth + 1, scope, match_pool) for _ in range(2)]
            return "optional:\n" + "\n".join(f"{pad}- {c}" for c in children)
        return count_term(scope)

    rule_texts = []
    earlier: list[str] = []
    namespaces: list[str] = []
    for i in range(n_rules):
        scope = "file" if rng.random() < 0.15 else "function"
        ns = f"gen/ns{i % 7}" if rng.random() < 0.25 else None
        name = f"gen rule {i}"
        # a rule may never reference itself, including through a namespace it
        # belongs to — that is a dependency cycle the reference engine also
        # cannot order.
        match_pool = earlier + [n for n in namespaces if n != ns]
        body = f"    - {stmt(0, scope, match_pool)}"
        rule_texts.append(make_rule(name, body, namespace=ns, scope=scope))
        earlier.append(name)
        if ns:
            namespaces.append(ns)
    return rule_texts, universe


def _gen_scopes(rng: random.Random, universe: dict, n_scopes: int) -> list[tuple[int, dict]]:
    keys = sorted(universe.keys(), key=repr)
    scopes = []
    for s in range(n_scopes):
        fmap: dict[tuple, set] = {}
        for key in keys:
            r = rng.random()
            if r < 0.22:
                base = 0x600000 + s * 0x1000
                fmap[key] = {base + 4 * k for k in range(rng.randint(1, 4))}
            elif r < 0.26:
                fmap[key] = set()  # present with empty locations
        # noise features the rules never reference
        if rng.random() < 0.5:
            fmap[("api", f"NoiseApi{s}")] = {0x700000 + s}
        if rng.random() < 0.3:
            fmap[("string", f"noise string {s} with sitemanager tail")] = {0x700100 + s}
        if rng.random() < 0.3:
            fmap[("string", "str99end" if s % 2 else "MIX4")] = {0x700200 + s}
        if rng.random() < 0.3:
            fmap[("bytes", bytes([s & 0xFF, 0xAA, 0xBB, 0xCC, 0xDD]))] = {0x700300 + s}
        scopes.append((0x400000 + s * 0x1000, fmap))
    return scopes


def test_seeded_random_differential(backend_expected):
    rng = random.Random(20260919)
    rule_texts, universe = _gen_rules(rng, 240)
    scopes = _gen_scopes(rng, universe, 150)  # crosses two 64-scope blocks
    assert_full_parity(rule_texts, scopes, backend_expected)


def test_seeded_random_differential_second_seed(backend_expected):
    rng = random.Random(1337)
    rule_texts, universe = _gen_rules(rng, 120)
    scopes = _gen_scopes(rng, universe, 70)  # crosses one block boundary
    assert_full_parity(rule_texts, scopes, backend_expected)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_unknown_match_dependency_raises(backend_expected):
    with pytest.raises(capa_mojo.InvalidRuleError, match="unknown rule or namespace"):
        capa_mojo.load_rules([make_rule("lonely", "    - and:\n      - match: no such rule")])


def test_duplicate_rule_names_raise(backend_expected):
    rule = make_rule("same name", "    - and:\n      - number: 0x1")
    with pytest.raises(capa_mojo.InvalidRuleError, match="duplicate rule names"):
        capa_mojo.load_rules([rule, rule])


def test_cycle_detection(backend_expected):
    rules = [
        make_rule("cycle a", "    - and:\n      - match: cycle b"),
        make_rule("cycle b", "    - and:\n      - match: cycle a"),
    ]
    with pytest.raises(capa_mojo.InvalidRuleError, match="cycle"):
        capa_mojo.load_rules(rules)


@pytest.mark.parametrize(
    "body",
    [
        "    - instruction:\n      - mnemonic: push",
        "    - basic block:\n      - and:\n        - number: 0x1",
        "    - and:\n      - com/unknown: IAspNetUser",
        "    - and:\n      - operand[0].number: 0x10",
        "    - and:\n      - count(match(some rule)): 1",
    ],
    ids=["instruction-subscope", "basic-block-subscope", "com-feature", "operand-feature", "count-match"],
)
def test_unsupported_constructs_raise(body, backend_expected):
    with pytest.raises(UnsupportedRuleError):
        capa_mojo.load_rules([make_rule("unsupported", body)])


def test_single_scope_form_returns_single_result(backend_expected):
    rule = make_rule("single", "    - and:\n      - number: 0x10")
    result = capa_mojo.load_rules([rule]).match({("number", 0x10): {1, 2}})
    assert isinstance(result, capa_mojo.ScopeMatches)
    assert result.backend == backend_expected
    assert "single" in result
    oracle = oracle_match([rule], [(0x401000, {("number", 0x10): {1, 2}})])
    assert set(result.matches.keys()) == set(oracle[0].keys())


def test_invalid_feature_map_rejected(backend_expected):
    rule = make_rule("strict", "    - and:\n      - number: 0x10")
    rs = capa_mojo.load_rules([rule])
    with pytest.raises(ValueError, match="feature keys"):
        rs.match({"not-a-tuple": {1}})
    with pytest.raises(ValueError, match="locations must be integers"):
        rs.match({("number", 0x10): {"not-an-int"}})
