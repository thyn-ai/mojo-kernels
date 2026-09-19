"""Differential tests: toml_mojo must match stdlib tomllib exactly.

Run twice by `scripts/test_all_toml.sh`: once against the native Mojo kernel
and once with TOML_MOJO_DISABLE_NATIVE=1 (forced stdlib-tomllib fallback).
Both backends must accept/reject every document identically to tomllib and
produce strictly equal objects (same types, same values, floats bit-exact)
for every accepted document.

The oracle is CPython 3.12's stdlib `tomllib` (the pip `tomli` it vendored).
Edge-case verdicts were probed empirically against it; the handwritten
corpora below encode those verdicts. tomlkit (Poetry's parser, the packaging
target) is used as a secondary acceptance oracle when installed: every
document we accept must also parse under tomlkit.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible.
"""

from __future__ import annotations

import math
import os
import random
import struct
import tomllib
import zlib
from pathlib import Path

import pytest

import toml_mojo

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stable_id(s: str) -> str:
    return f"{zlib.crc32(s.encode('utf-8', 'surrogatepass')) % 99999:05d}"

# ---------------------------------------------------------------------------
# Strict structural equality (types must match; floats compared bit-for-bit)
# ---------------------------------------------------------------------------


def strict_eq(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(strict_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def assert_same_doc(src: str) -> None:
    """Both backends must return strictly-equal objects for a valid doc."""
    expected = tomllib.loads(src)  # oracle: must be valid
    got = toml_mojo.loads(src)
    assert strict_eq(got, expected), (
        f"mismatch for {src!r}\ntoml_mojo: {got!r}\ntomllib:   {expected!r}"
    )


def assert_both_reject(src: str) -> None:
    with pytest.raises(ValueError):
        tomllib.loads(src)  # oracle: must be invalid (TOMLDecodeError)
    with pytest.raises(ValueError):
        toml_mojo.loads(src)


def _expected_backend() -> str:
    # scripts/test_all_toml.sh runs the suite once per backend.
    return "fallback" if os.environ.get("TOML_MOJO_DISABLE_NATIVE") == "1" else "native"


# ---------------------------------------------------------------------------
# Handwritten corpora (verdicts probed against CPython 3.12 tomllib)
# ---------------------------------------------------------------------------

VALID_DOCS = [
    # scalars and basics
    "",
    "# nothing\n",
    "a = 1\n",
    "a=1\n",
    "a = 42",
    "a = -17\nb = +99\nc = 0\nd = -0\n",
    "a = 1_000\nb = 5_349_221\nc = 1_2_3\n",
    "a = 0xDEADBEEF\nb = 0xdead_beef\nc = 0o755\nd = 0b11010110\n",
    "a = 999999999999999999999999999999999\n",
    "a = 0xFFFFFFFFFFFFFFFFFFFFFFFFF\n",
    "a = 1.0\nb = 3.1415\nc = -0.01\nd = 5e+22\ne = 1e06\nf = -2E-2\ng = 6.626e-34\n",
    "a = 1_0.0_1e0_2\nb = 0.0\nc = -0.0\nd = 0e0\n",
    "a = inf\nb = +inf\nc = -inf\nd = nan\ne = +nan\nf = -nan\n",
    "a = true\nb = false\n",
    'a = ""\nb = "hello"\nc = "héllo wörld"\n',
    "a = ''\nb = 'literal'\nc = 'C:\\Users\\x'\n",
    'a = "\\b\\t\\n\\f\\r\\"\\\\"\n',
    'a = "\\u00e9\\U0001F600\\u0000\\u007f"\n',
    'a = "\\U0010FFFF"\n',
    "a = 'héllo'\n",
    # strings: multiline
    'a = """\nhello"""\n',
    'a = """he said ""hi"" x"""\n',
    'a = """x""""\nb = """y"""""\n',
    'a = """"""\n',
    'a = """one \\\n   two"""\n',
    'a = """one \\\r\n  two"""\n',
    'a = """\r\nhello\r\nworld"""\r\n',
    'a = """x\ny\r\nz"""\n',
    "a = '''\nno ''escapes'' \\ here'''\n",
    "a = '''x 'y' z'''\n",
    "a = '''x''''\n",
    "a = ''''''\n",
    'a = """tab\there"""\n',
    # keys
    "1234 = 1\n",
    "1979-05-27 = 1\n",
    "a_b-c = 1\n- = 2\n_ = 3\n",
    "true = 1\nfalse = 2\n",
    '"" = 1\n',
    '"a"."b".c = 1\n',
    '"a.b" = 1\n',
    "a . b . c = 1\n",
    '"a" . "b" = 1\n',
    'a."" = 1\n',
    '""."" = 1\n',
    '"\\u0041" = 1\n',
    '"\\u0000" = 1\n',
    # tables
    "[a]\nb = 1\n",
    "[a]\nb = 1\n[c]\nd = 2\n",
    "[a.b.c]\nd = 1\n",
    "[x.y]\n[x]\n",
    "[ a . b ]\nc = 1\n",
    "[t]\na.b = 1\n[t.u]\nc = 2\n",
    "a.b = 1\n[a.c]\n",
    "a.b = 1\n[[a.c]]\n",
    "a.b = 1\n[a.c.d]\n",
    "[a]\nb = 1\n[a.c]\nd = 2\n",
    "[t.a]\nx = 1\n[t]\n",
    "[t.a.b]\n[t]\na.c = 1\n",
    "a.b = 1\na.c = 2\n",
    '[""]\nx = 1\n',
    '["a.b"]\nx = 1\n',
    # arrays of tables
    "[[a]]\nb = 1\n",
    "[[a]]\nb = 1\n[[a]]\nb = 2\n",
    "[[a]]\n[a.b]\n[[a]]\n[a.b]\n",
    "[[a]]\n[[a.b]]\n[[a.b]]\n[[a]]\n[[a.b]]\n",
    "[[a.b]]\n",
    "[[a]]\n[[a.b]]\n[a.b.c]\n",
    "[[a]]\nb.c = 1\n[[a]]\nb.c = 2\n",
    # arrays
    "a = []\n",
    "a = [1]\n",
    "a = [1, 2,]\n",
    "a = [\n 1, # one\n 2,\n]\n",
    'a = [1, "x", 2.0, true, 1979-05-27]\n',
    "a = [[[]]]\n",
    "a = [[1, 2], [\"a\", \"b\"]]\n",
    "a = [ {x = 1}, {y = 2} ]\n",
    # inline tables
    "a = {}\n",
    "a = {b = 1}\n",
    "a = {b = 1, c = 2}\n",
    "x = {a.b = {c = 1}, a.d = 2}\n",
    "a = {b = {c = {d = 1}}}\n",
    "a = {b = [{c = 1}]}\n",
    "a = [{b = 1}, {c = 2}]\n",
    # dates and times
    "a = 1979-05-27\n",
    "a = 2000-02-29\n",
    "a = 07:32:00\n",
    "a = 00:00:00.5\n",
    "a = 00:00:00.000001\n",
    "a = 1979-05-27T07:32:00Z\n",
    "a = 1979-05-27T07:32:00.5\n",
    "a = 1979-05-27t07:32:00z\n",
    "a = 1979-05-27 07:32:00\n",
    "a = 1979-05-27T07:32:00-08:00\n",
    "a = 1979-05-27T07:32:00+00:00\n",
    "a = 1979-05-27T07:32:00-00:00\n",
    "a = 2000-01-01T20:00:00+20:00\n",
    "a = 2000-01-01T20:00:00+23:59\n",
    "a = 2000-01-01T00:00:00.123456789\n",
    "a = 2000-01-01T00:00:00.9999995\n",
    "a = [2000-01-01, 00:00:01, 2000-01-01T00:00:00Z]\n",
    # comments / whitespace
    "a = 1 # done",
    "[a] # ok\nb = 1\n",
    "# héllo\na = 1\n",
    "# hello\tworld\na = 1\n",
    "\n\n\ta = 1\n\n",
    "a = 1\r\nb = 2\r\n",
    "\ta = 1\t\n",
]

INVALID_DOCS = [
    # duplicate / redefinition rules
    "a = 1\na = 2\n",
    "a.b = 1\na.b = 2\n",
    "a.b = 1\n[a]\n",
    "[a]\n[a]\n",
    "[[a]]\n[a]\n",
    "[a]\n[[a]]\n",
    "[[a]]\n[a.b]\n[a.b]\n",
    "x = {a = 1, a = 2}\n",
    "x = {a = 1}\nx.b = 2\n",
    "x = {a = 1}\n[x.b]\n",
    "a = {}\n[a]\n",
    "a = {}\na.b = 1\n",
    "a.b = {}\na.b.c = 1\n",
    "a = [1]\n[[a]]\n",
    "a = 1\na.b = 2\n",
    "a.b = 1\n[a.b.c]\n",
    "[t]\na.b = 1\n[t.a]\n",
    "[t.a]\nx = 1\n[t]\na.y = 2\n",
    "[[t.a]]\n[t]\na.x = 1\n",
    "[t]\nb.c = 1\nb = 2\n",
    "a = [{b = 1, b = 2}]\n",
    "x = {a = {}, a.b = 1}\n",
    # keys
    "don't = 1\n",
    "héllo = 1\n",
    "= 1\n",
    "[]\na = 1\n",
    "[[]]\na = 1\n",
    "a 1\n",
    'a = "x" y = 2\n',
    "[a] [b]\n",
    "[a] b = 1\n",
    # values
    "a =\n",
    "a = ]\n",
    "a = 01\n",
    "a = +01\n",
    "a = 00\n",
    "a = -00\n",
    "a = 0_0\n",
    "a = 1__000\n",
    "a = 0x_dead\n",
    "a = 0xdead_\n",
    "a = 0X10\n",
    "a = 0x\n",
    "a = 0b\n",
    "a = 0b2\n",
    "a = 0o8\n",
    "a = 1.\n",
    "a = .5\n",
    "a = 1e\n",
    "a = 1.e5\n",
    "a = 1.0e\n",
    "a = 1e_5\n",
    "a = 1_.5\n",
    "a = 1._5\n",
    "a = 01.5\n",
    "a = INF\n",
    "a = NaN\n",
    "a = +\n",
    "a = -\n",
    "a = 1.5x\n",
    "a = 1979-05-27garbage\n",
    # strings
    'a = "x\n',
    'a = "\\q"\n',
    'a = "\\uD800"\n',
    'a = "\\U00110000"\n',
    "a = 'it''s'\n",
    'a = """x """"""\n',
    'a = """x """""""\n',
    'a = """x """ y"""\n',
    "a = '''x''''''\n",
    'a = """x \\ y"""\n',
    'a = """x \\',
    'a = "x\x7fy"\n',
    'a = "x\x08y"\n',
    'a = """x\ry"""\n',
    # dates and times
    "a = 0000-01-01\n",
    "a = 24:00:00\n",
    "a = 23:59:60\n",
    "a = 2000-01-01T00:00:00+24:00\n",
    "a = 2000-01-01T00:00:00+99:60\n",
    "a = 2000-01-01T07:32\n",
    "a = 2001-02-29\n",
    "a = 1900-02-29\n",
    "a = 2000-13-01\n",
    "a = 2000-01-00\n",
    "a = 07:32:00Z\n",
    "a = 2000-1-1\n",
    "a = 2000-01-01T00:00:00.\n",
    "a = 2000-01-01T00:00:00+07:00:00\n",
    "a = 2000-01-0100:00:00\n",
    "a = 2000-01-01T00:00:00.5x\n",
    "a = 2000-01-01T00:00:00+0700\n",
    "a = -2000-01-01\n",
    "a = +00:00:00\n",
    "a = 20000-01-01\n",
    # structure / whitespace
    "a = 1\rb = 2\r",
    "a = 1\rb = 2\n",
    "# hello\x01world\na = 1\n",
    "# hello\x7fworld\na = 1\n",
    "\ufeffa = 1\n",
    "a = 1\n\ufeffb = 2\n",
    "a =\x0b1\n",
    "a =\x0c1\n",
    "a = 1\n",  # NBSP between = and 1
    "a = [1 2]\n",
    "a = [,]\n",
    "a = [1,,2]\n",
    "a = [1,\n",
    "x = {a = 1,}\n",
    "x = {a = 1,\n b = 2}\n",
    '"hello"\n',
]


@pytest.mark.parametrize("src", VALID_DOCS, ids=lambda s: f"valid-{_stable_id(s)}")
def test_valid_docs_match_tomllib(src):
    assert_same_doc(src)


@pytest.mark.parametrize("src", INVALID_DOCS, ids=lambda s: f"invalid-{_stable_id(s)}")
def test_invalid_docs_rejected_like_tomllib(src):
    assert_both_reject(src)


def test_backend_is_the_expected_one():
    info = toml_mojo.backend_info()
    if _expected_backend() == "native":
        assert info["native_available"], (
            "native run requires a built kernel (bash kernels/toml/build.sh)"
        )
        assert toml_mojo.native_available()
    else:
        assert not info["native_available"]


def test_native_actually_used_when_expected():
    # Guard against the native leg silently passing via the fallback.
    if _expected_backend() != "native":
        pytest.skip("fallback leg")
    assert toml_mojo._native.native_available()


# ---------------------------------------------------------------------------
# Real-world files from this repository as fixtures
# ---------------------------------------------------------------------------

REPO_TOML_FILES = [
    REPO_ROOT / "pixi.toml",
    REPO_ROOT / "python" / "bm25_mojo" / "pyproject.toml",
    REPO_ROOT / "python" / "toml_mojo" / "pyproject.toml",
    REPO_ROOT / "python" / "cclib_mojo" / "pyproject.toml",
]


@pytest.mark.parametrize("path", [p for p in REPO_TOML_FILES if p.exists()], ids=lambda p: p.name)
def test_repo_toml_files(path):
    assert_same_doc(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Seeded random valid-document generator
# ---------------------------------------------------------------------------

_BARE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
_SAFE_TEXT = (
    "abc XYZ 123 éüß 😀 !#$%&()*+,-./:;<=>?@[]^_`{|}~\t"
    "'' '' \"\" \\ "  # quote runs of at most two, a lone backslash
)
_ESCAPES = ["\\b", "\\t", "\\n", "\\f", "\\r", '\\"', "\\\\", "\\u00e9", "\\U0001F600"]


def _gen_key(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return "".join(rng.choice(_BARE_CHARS) for _ in range(rng.randint(1, 8)))
    # quoted key, possibly empty or containing dots/spaces/unicode
    pool = "ab .é😀\t-_%"
    inner = "".join(rng.choice(pool) for _ in range(rng.randint(0, 6)))
    inner = inner.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{inner}"'


def _gen_string(rng: random.Random) -> str:
    n = rng.randint(0, 12)
    style = rng.random()
    if style < 0.45:
        body = "".join(rng.choice(_SAFE_TEXT) for _ in range(n))
        body = body.replace("\\", "\\\\").replace('"', '\\"')
        body = body.replace("\t", "\\t") if rng.random() < 0.5 else body
        for esc in _ESCAPES:
            if rng.random() < 0.05:
                body += esc
        return f'"{body}"'
    if style < 0.7:
        pool = _SAFE_TEXT.replace("'", "").replace("\\", "")
        body = "".join(rng.choice(pool) for _ in range(n))
        return f"'{body}'"
    if style < 0.9:
        pool = _SAFE_TEXT.replace("\\", "") + "\n\n"
        body = "".join(rng.choice(pool) for _ in range(n))
        body = body.replace("\\", "\\\\").replace('"""', '""\\"')
        if rng.random() < 0.3:
            body = "\\\n  " + body  # line-ending backslash
        return f'"""{body}"""'
    pool = _SAFE_TEXT.replace("'", "").replace("\\", "") + "\n"
    body = "".join(rng.choice(pool) for _ in range(n))
    return f"'''{body}'''"


def _gen_number(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.3:
        v = rng.randint(-(10**rng.randint(1, 30)), 10**rng.randint(1, 30))
        s = str(v)
        if rng.random() < 0.3 and len(s.lstrip("-")) > 2:
            # sprinkle legal underscores between digits
            sign = "-" if s.startswith("-") else ""
            digs = s.lstrip("-")
            if rng.random() < 0.5:
                s = sign + digs[0] + "_" + digs[1:]
        return s
    if style < 0.45:
        return rng.choice(
            [
                "0x" + "%x" % rng.getrandbits(rng.randint(1, 64)),
                "0o" + "%o" % rng.randint(0, 511),
                "0b" + format(rng.randint(0, 255), "b"),
            ]
        )
    if style < 0.8:
        ip = rng.randint(0, 10**6)
        frac = rng.randint(0, 10**6)
        exp = rng.choice(["", f"e{rng.randint(-99, 99)}", f"e+{rng.randint(0, 20)}", f"E-{rng.randint(0, 20)}"])
        sign = rng.choice(["", "-", "+"])
        return f"{sign}{ip}.{frac:06d}{exp}"
    return rng.choice(["inf", "-inf", "+inf", "nan", "-nan", "+nan"])


def _gen_datetime(rng: random.Random) -> str:
    y = rng.randint(1, 9999)
    mo = rng.randint(1, 12)
    dim = [31, 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]
    d = rng.randint(1, dim)
    h, mi, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    frac = ""
    if rng.random() < 0.4:
        frac = "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 12)))
    kind = rng.random()
    if kind < 0.3:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    if kind < 0.45:
        return f"{h:02d}:{mi:02d}:{s:02d}{frac}"
    delim = rng.choice(["T", " "])
    off = rng.choice(["", "Z", "+07:30", "-08:00", "+00:00", "+14:00", "-00:00"])
    return f"{y:04d}-{mo:02d}-{d:02d}{delim}{h:02d}:{mi:02d}:{s:02d}{frac}{off}"


def _gen_scalar(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return _gen_string(rng)
    if r < 0.55:
        return _gen_number(rng)
    if r < 0.65:
        return rng.choice(["true", "false"])
    return _gen_datetime(rng)


def _gen_value(rng: random.Random, depth: int) -> str:
    if depth >= 3 or rng.random() < 0.55:
        return _gen_scalar(rng)
    if rng.random() < 0.5:
        n = rng.randint(0, 5)
        items = [_gen_value(rng, depth + 1) for _ in range(n)]
        trail = "," if items and rng.random() < 0.4 else ""
        sep = ",\n  " if rng.random() < 0.3 else ", "
        return "[" + sep.join(items) + trail + "]"
    n = rng.randint(0, 4)
    pairs = []
    used = set()
    for _ in range(n):
        k = _gen_key(rng)
        if k in used:
            continue
        used.add(k)
        pairs.append(f"{k} = {_gen_value(rng, depth + 1)}")
    return "{" + ", ".join(pairs) + "}"


def _gen_doc(rng: random.Random) -> str:
    """Build a random valid document via nested tables/AoTs/dotted keys.

    Validity is guaranteed by construction: within one table, plain keys,
    dotted-key first segments, and sub-table names come from three disjoint
    pools with unique names, and a sub-table is emitted either as [table]
    once or as [[aot]] one or more times — never both.
    """
    lines: list[str] = []

    def fresh_key(used: set[str]) -> str:
        for _ in range(20):
            k = _gen_key(rng)
            if k not in used:
                used.add(k)
                return k
        k = f"k{len(used)}"
        used.add(k)
        return k

    def emit_table(path: str | None, depth: int) -> None:
        if path is not None:
            lines.append(f"[{path}]")
        used: set[str] = set()
        n_kv = rng.randint(0, 5)
        for _ in range(n_kv):
            if rng.random() < 0.2 and depth < 3:
                # dotted key defining a nested table inline
                first = fresh_key(used)
                lines.append(f"{first}.{_gen_key(rng)} = {_gen_value(rng, depth + 1)}")
            else:
                lines.append(f"{fresh_key(used)} = {_gen_value(rng, depth)}")
            if rng.random() < 0.15:
                lines.append(f"# {rng.choice(['note', 'héllo', ''])}")
        if depth < 3:
            for _ in range(rng.randint(0, 3)):
                sub = fresh_key(used)
                full = f"{path}.{sub}" if path else sub
                if rng.random() < 0.35:
                    for _ in range(rng.randint(1, 3)):
                        lines.append(f"[[{full}]]")
                        aot_used: set[str] = set()
                        for _ in range(rng.randint(0, 3)):
                            lines.append(f"{fresh_key(aot_used)} = {_gen_value(rng, depth + 1)}")
                else:
                    emit_table(full, depth + 1)

    emit_table(None, 0)
    text = "\n".join(lines)
    if rng.random() < 0.7:
        text += "\n"
    return text


def test_generated_valid_docs_match_tomllib():
    rng = random.Random(20260919)
    for i in range(250):
        doc = _gen_doc(rng)
        assert_same_doc(doc)


def test_deep_nesting_parity_within_tomllib_limits():
    # tomllib itself hits RecursionError around ~450 array levels (its parser
    # is recursive in Python); within its working range we must agree exactly.
    for depth in (50, 200):
        for src in (
            "a = " + "[" * depth + "1" + "]" * depth + "\n",
            ".".join(["k"] * depth) + " = 1\n",
            "a = " + "{b = " * depth + "1" + "}" * depth + "\n",
        ):
            assert_same_doc(src)


def test_extreme_nesting_never_crashes():
    # Beyond the kernel's 2000-level value-nesting guard: a clean decode
    # error (never a native stack overflow). Between tomllib's own
    # RecursionError limit and the guard, either a correct result or a
    # Python-side RecursionError from the assembler is acceptable.
    src = "a = " + "[" * 3000 + "1" + "]" * 3000 + "\n"
    # Native leg: kernel guard raises TOMLDecodeError (a ValueError).
    # Fallback leg: tomllib hits its own RecursionError first.
    with pytest.raises((ValueError, RecursionError)):
        toml_mojo.loads(src)


def test_generated_docs_accepted_by_tomlkit_if_installed():
    """Secondary oracle: Poetry's parser must accept every document we and
    tomllib accept (we return tomllib-shaped data, not tomlkit's
    trivia-preserving document model)."""
    tomlkit = pytest.importorskip("tomlkit")
    rng = random.Random(11)
    for _ in range(60):
        doc = _gen_doc(rng)
        tomllib.loads(doc)  # sanity
        tomlkit.loads(doc)
    # Known parser divergence (tomllib is our primary oracle): tomllib
    # accepts dotted keys that extend an implicitly-created super-table
    # (`[t.a.b]` then `[t]` then `a.c = 1`); tomlkit rejects that as a
    # redefinition. The generated corpus above stays inside both grammars.
    TOMLKIT_STRICTER = {'[t.a.b]\n[t]\na.c = 1\n'}
    for src in VALID_DOCS:
        if src in TOMLKIT_STRICTER:
            continue
        tomlkit.loads(src)


# ---------------------------------------------------------------------------
# Mutation fuzz: accept/reject parity (and output parity when both accept)
# ---------------------------------------------------------------------------

_MUT_ALPHABET = list('[]{}=,."\'\\# \t\n\r+-_09aZzT:xoué')


def _mutations(rng: random.Random, text: str, n: int):
    out = []
    for _ in range(n):
        if not text:
            break
        s = text
        op = rng.random()
        i = rng.randrange(len(s))
        if op < 0.45:
            s = s[:i] + rng.choice(_MUT_ALPHABET) + s[i + 1 :]
        elif op < 0.8:
            s = s[:i] + rng.choice(_MUT_ALPHABET) + s[i:]
        else:
            s = s[:i] + s[i + 1 :]
        if s != text:
            out.append(s)
    return out


def test_mutation_fuzz_parity():
    rng = random.Random(777)
    bases = [
        'title = "x"\n[a]\nb = [1, 2.5, "z"]\nc = {d = 1}\n[[e]]\nf = 1979-05-27T07:32:00Z\n',
        "a.b.c = 1\n[a]\nx = '''ml\nstr'''\ny = 0x10\n",
        "[t]\na = {b.c = [true, {z = 00:00:01}], d = 1_000.5e-2}\n",
    ]
    checked = 0
    for base in bases:
        for mutant in _mutations(rng, base, 700):
            try:
                ref = tomllib.loads(mutant)
            except ValueError:
                ref = None
            try:
                got = toml_mojo.loads(mutant)
            except ValueError:
                got = None
            if ref is None:
                assert got is None, f"toml_mojo accepted what tomllib rejects: {mutant!r}\n{got!r}"
            else:
                assert got is not None, f"toml_mojo rejects what tomllib accepts: {mutant!r}"
                assert strict_eq(got, ref), (
                    f"mismatch on mutant {mutant!r}\ntoml_mojo: {got!r}\ntomllib:   {ref!r}"
                )
            checked += 1
    assert checked > 1500
