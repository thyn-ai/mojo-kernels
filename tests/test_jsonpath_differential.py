"""Differential tests: jsonpath_mojo must match jsonpath_ng (ext dialect).

Run twice by `scripts/test_all_jsonpath.sh`: once against the native Mojo
kernel and once with JSONPATH_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle everywhere: identical
(value, str(full_path)) sequences, and identical exception types for the
error contract (oracle parse/lex errors map to our JsonPathError).

The oracle is the published PyPI package (jsonpath_ng==1.7.0), installed to a
bootstrap directory by scripts/test_all_jsonpath.sh (it is not in pixi.toml;
this leg may not edit that file). Everything here is generated locally from
explicit seeds — no network, no randomness without a fixed seed.
"""

from __future__ import annotations

import os
import random

import pytest

from jsonpath_ng.exceptions import JSONPathError as OracleJSONPathError
from jsonpath_ng.ext import parse as oracle_parse

import jsonpath_mojo
from jsonpath_mojo import _reference
from jsonpath_mojo.core import JsonPathError

NATIVE_FORCED_OFF = os.environ.get("JSONPATH_MOJO_DISABLE_NATIVE") == "1"

STORE = {
    "store": {
        "book": [
            {"category": "reference", "author": "Nigel Rees", "title": "Sayings", "price": 8.95},
            {"category": "fiction", "author": "Evelyn Waugh", "title": "Sword", "price": 12.99},
            {"category": "fiction", "author": "Herman Melville", "title": "Moby Dick", "isbn": "0-553", "price": 8.99},
            {"category": "fiction", "author": "J. R. R. Tolkien", "title": "LOTR", "isbn": "0-395", "price": 22.99},
        ],
        "bicycle": {"color": "red", "price": 19.95},
    },
    "expensive": 10,
    "nums": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

EDGE_DOC = {
    "items": [
        {"v": 0, "s": "x"},
        {"v": 1},
        {"v": 2, "s": "y"},
        {"v": None, "s": ""},
        {"v": False},
        {"v": 5.9, "s": "5"},
        {"v": "7", "s": " 8 "},
    ],
    "mixed": [1, "a", 1.5, True, None, [1], {"k": 1}, False, 0, ""],
    "nested": {"deep": {"deeper": [{"w": 5}, {"w": 50}]}},
    "empty_list": [],
    "empty_dict": {},
    "empty_str": "",
    "zero": 0,
    "falsy": False,
    "nothing": None,
}

WEIRD_KEYS_DOC = {
    "a b": 1, "0": 2, "a.b": 3, "": 4, "it's": 5, "üñï": 6, "normal": 7, "*": 8,
    "a,b": 9, "a:b": 10, "a[b": 11, "a]b": 12, "a(b": 13, "a)b": 14, "a|b": 15,
    "a&b": 16, "a$b": 17, "a~b": 18, "a-b": 19, "a?b": 20, "a@b": 21, 'a"b': 22,
    "a'b": 23, "123": 24, "_": 25, "a#b": 26, "a;b": 27, "a=b": 28, "a<b": 29,
    "a\\b": 30, "a^b": 31, "a%b": 32, "a!b": 33, "a+b": 34, "a`b": 35, "a{b": 36,
    "a}b": 37, "a/b": 38, ".": 39, ",": 40, " ": 41, "and": 42, "null": 43,
    "tab\there": 44, "nl\nhere": 45,
}

# Queries valid for the oracle (and therefore for us). Each is checked for
# exact (value, path) sequence parity on every document it is paired with.
STORE_QUERIES = [
    "$", "$.expensive", "$.store.bicycle.color", "$['store']['bicycle']['price']",
    "$.store.book[0].title", "$.store.book[-1].title", "$.store.book[9]",
    "$.store.book[*].author", "$..author", "$..book[2].title", "$.store.*",
    "$.*", "$..*", "$.store.book[*]", "$.store.book[1:3].title",
    "$.store.book[::2].title", "$.store.book[::-1].title", "$.store.book[3:0:-2].title",
    "$.store.book[2:2].title", "$['store','expensive']", "$['store','store']",
    "$.store.book[?(@.price < 10)].title", "$.store.book[?(@.isbn)].title",
    "$.store.book[?(@.price < 10 & @.category == 'fiction')].title",
    '$.store.book[?(@.category == "fiction")].author',
    "$.store.book[?(@.price >= 8.95 & @.price <= 8.99)].price",
    "$.store.book[?(@ != 8.99)].price", "$.store.book[?(@.price > 100)]",
    "$.store.book[?(@.missing > 1)].title", "$.store.book[?(@.price)].title",
    "$.nums[1:8:3]", "$.nums[-3:]", "$.nums[::-1]", "$.nums[8:2:-2]", "$.nums[3:3]",
    "$.nums[?(@ > 2 & @ < 5)]", "$.nums[?(@ > 100)]",
    "$.nums[?(@ > 2 & @ < 8 & @ != 5)]", "$..price", "$..book[*].price",
    "$..book[?(@.price > 20)].title", "$.store.book[?(@.category == 'fiction')].price",
]

EDGE_QUERIES = [
    "$.items[?(@.v)]", "$.items[?(@.v == None)]",
    "$.items[?(@.v == null)]", "$.items[?(@.v == False)]",
    "$.items[?(@.v == 1.5)]",
    "$.items[?(@.s == 'x')]", "$.items[?(@.s < 'y')]", "$.items[?(@.s != 'x')]",
    "$.items[?(@.s == 5)]", "$.items[?(@.v + 1 > 1)]", "$.items[?(@.v * 2 == 2)]",
    "$.items[?(@.v - 1 == 1)]", "$.items[?(@.v + 1 * 2 == 5)]",
    "$.items[?((@.v + 1) * 2 == 4)]", "$.items[?(@.v + 1.5 == 2.5)]",
    "$.items[?(@.s * 2 == 'xx')]",
    "$.items[?(@.s + 'y' == 'xy')]", "$.items[?(@.s == ' 8 ')]",
    "$.items[?(@.s == 8)]", "$.items[?(@.v == 5.9)]",
    "$.mixed[?(@ == 1.5)]", "$.mixed[?(@ == True)]",
    "$.mixed[?(@ == null)]",
    "$.mixed[?(@ != null)]", "$.mixed[?(@ == 'a')]",
    "$.mixed[*]", "$.mixed[0]", "$.mixed[-1]", "$.mixed[2:5]", "$.nested.deep.deeper[*].w",
    "$..w", "$..deeper[?(@.w > 10)]", "$.nested.deep.deeper[?(@.w + 0 == 5)]",
    "$.empty_list[*]", "$.empty_dict[*]", "$.empty_str[*]", "$.zero[*]", "$.falsy[*]",
    "$.nothing[*]", "$.empty_list[0]", "$.zero[0]", "$.falsy[0]", "$.nothing[0]",
    "$.items[0].v",
]

RENDER_QUERIES = ["$.*", "$..*", "$['a b']", "$['0']", "$['a.b']", "$['']", "$['üñï']",
                  "$['it\\'s']", '$["it\'s"]', "$['*']", "$['a,b','a:b']",
                  "$['*','a b']", "$['a b','*']", "$..['*']"]


def oracle_find(expr, data):
    """(values, paths) or the exception the oracle raises."""
    return [(m.value, str(m.full_path)) for m in oracle_parse(expr).find(data)]


@pytest.fixture
def forbid_fallback(monkeypatch):
    """In the native run, in-domain cases must not touch the Python engine."""
    if NATIVE_FORCED_OFF:
        return
    monkeypatch.setattr(_reference, "find", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback engaged on native run")))


def _check_value_parity(expr, data):
    got = jsonpath_mojo.find(expr, data)
    want = oracle_find(expr, data)
    assert got == want, f"\nexpr: {expr}\nwant: {want}\ngot:  {got}"


@pytest.mark.parametrize("expr", STORE_QUERIES)
def test_store_queries(expr, forbid_fallback):
    _check_value_parity(expr, STORE)


@pytest.mark.parametrize("expr", EDGE_QUERIES)
def test_edge_queries(expr, forbid_fallback):
    _check_value_parity(expr, EDGE_DOC)


@pytest.mark.parametrize("expr", RENDER_QUERIES)
def test_path_rendering(expr, forbid_fallback):
    _check_value_parity(expr, WEIRD_KEYS_DOC)


def test_int_key_wildcard_rendering_error():
    # the oracle's path renderer dies on non-string keys
    with pytest.raises(TypeError):
        oracle_find("$.*", {0: "x"})
    with pytest.raises(TypeError, match="argument of type 'int' is not iterable"):
        jsonpath_mojo.find("$.*", {0: "x"})


def test_implicit_root():
    for q in ["store.book[0].title", "'expensive'", "*", "[0]", "[*]"]:
        data = STORE if not q.startswith("[") else [10, 20]
        _check_value_parity(q, data)


# ---------------------------------------------------------------------------
# Error contract: same exception types from both engines
# ---------------------------------------------------------------------------

PARSE_ERROR_QUERIES = [
    "$.store.book[0,2].title", "$.nums[?(@ > 2) | @ > 8)]", "$..[?(@.v > 1)]",
    "$.a[?(@.a == 1 and @.b == 2)]", "$.a[?(@.a == @.b)]", "$.a[?(@ > 1e2)]",
    "$.a[?(@.a == 1 + 1)]", "$.a[?(@.a == (1 + 1))]", "$.a[?(@.a == 'x' + 1)]",
    "$.a[?(@.a =< 2)]", "$.a[?(!@.b)]", "$.a[?(@.v % 2 == 0)]", "$.a[?(@ > $.x)]",
    "$.", "$..", "$.a[**]", "$.a[1.0]", "$['a',2]", "$.a[?(-@.a == -2)]",
    "$.a[?(@.a ==.5)]", "$.a[?(@.a == 5.)]", "$.café", "$.a[?(@.a > 'x' | @.a < 'y')]",
    "x y", "$.a[?(@ > 2]", "$['a'", "$.a[]", "$.a[?()]", "$.a[?(@.a == 1))]",
    "$.items[?(@.missing == 1 | @.v == 2)]", "$.items[?(@.v == 1 | @.v == 2)]",
    "$.items[?(@.v == 1 and @.s == 'x')]", "$..[?(@.v == 1)]",
]

# (query, data) pairs that must raise the same builtin exception type.
RUNTIME_ERROR_CASES = [
    ("$.a[-4]", {"a": [1, 2, 3]}, IndexError),
    ("$[0]", {"0": "zero"}, KeyError),
    ("$[-1]", {"a": 1}, KeyError),
    ("$[0]", 5, TypeError),
    ("$[3]", 5, TypeError),
    ("$[-1]", 5, TypeError),
    ("$[0]", True, TypeError),
    ("$[0]", 5.5, TypeError),
    ("$[*]", 5.5, TypeError),
    ("$..[0]", [10, 20], TypeError),
    ("$.a[::0]", {"a": [1, 2]}, ValueError),
    ("$.a[::0]", {"a": 5}, ValueError),
    ("$.a[1:2:0]", {"a": [1, 2]}, ValueError),
    ("$.items[?(@.v == 0)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v == false)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v > -1)]", EDGE_DOC, TypeError),
    ("$.mixed[?(@ == 1)]", EDGE_DOC, TypeError),
    ("$.nums[?(@ > 'a')]", {"nums": [1, 2]}, TypeError),
    ("$.nums[?(@ < 5.0)]", {"nums": [1, "x"]}, TypeError),
    ("$.nums[?(@ < 'x')]", {"nums": [5]}, TypeError),
    ("$.nums[?(@ < null)]", {"nums": [5]}, TypeError),
    ("$.nums[?(@ > foo)]", {"nums": [None]}, TypeError),
    ("$.items[?(@.a & @.b)]", {"items": [{"a": 1, "b": 1}]}, NotImplementedError),
    ("$.items[?(@.a & @.b == 1)]", {"items": [{"a": 1, "b": 1}]}, NotImplementedError),
    ("$.arr[?(@.a == 0 & @.b == 1)]", {"arr": [{"a": 0, "b": None}]}, TypeError),
    ("$.arr[?(@.* == 1)]", {"arr": [{"a": 1, "b": None}]}, TypeError),
    ("$.arr[?(@.a[5] == 1)]", {"arr": [{"a": 5}]}, TypeError),
    ("$.arr[?(@.a[-9] == 1)]", {"arr": [{"a": [1, 2]}]}, IndexError),
    ("$.arr[?(@.a[0] == 1)]", {"arr": [{"a": {"x": 1}}]}, KeyError),
    ("$.items[?(@.v == 0)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v == 5)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v > 1)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v > 0.5)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v == 8)]", EDGE_DOC, TypeError),
    ("$.mixed[?(@ == 1)]", EDGE_DOC, TypeError),
    ("$.mixed[?(@ == true)]", EDGE_DOC, TypeError),
    ("$.mixed[?(@ == false)]", EDGE_DOC, TypeError),
    ("$.mixed[?(@ > 1)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v == 1 & @.s)]", EDGE_DOC, TypeError),
    ("$.items[?(@.v & @.s == 'x')]", {"items": [{"v": 1, "s": "x"}]}, NotImplementedError),
    ("$.items[?(@.v > 1 & @.s == 'y' & @.v)]", EDGE_DOC, TypeError),
]


@pytest.mark.parametrize("expr", PARSE_ERROR_QUERIES)
def test_parse_errors(expr):
    with pytest.raises(OracleJSONPathError):
        oracle_parse(expr).find(STORE)
    with pytest.raises(JsonPathError):
        jsonpath_mojo.find(expr, STORE)


@pytest.mark.parametrize("expr,data,exc_type", RUNTIME_ERROR_CASES)
def test_runtime_errors(expr, data, exc_type):
    with pytest.raises(exc_type):
        oracle_find(expr, data)
    with pytest.raises(exc_type):
        jsonpath_mojo.find(expr, data)


# ---------------------------------------------------------------------------
# Out-of-native-domain data: silently correct via fallback on BOTH runs
# ---------------------------------------------------------------------------

REROUTE_CASES = [
    ("$[0]", ("a", "b")),
    ("$[*]", ("a", "b")),
    ("$[0:1]", ("a", "b")),
    ("$.*", ("a", "b")),
    ("$[?(@ == 'a')]", ("a", "b")),
    ("$..x", ({"x": 1},)),
    ("$.a[?(@ > 1)]", {"a": (1, 2)}),
    ("$..x", {"a": ({"x": 1},)}),
    ("$['0']", {0: "x"}),
    ("$[?(@ > 0)]", {0: 5}),
    ("$.a", {1.5: "float-key"}),
    ("$.a[0]", {True: [42]}),
    ("$.*", {"nan": float("nan")}),
    ("$.a", {"big": 2**70}),
    ("$.a[?(@ > 1)]", {"a": [1, 2**70]}),
    ("$..b", {"a": {"b": frozenset([1])}}),
]


@pytest.mark.parametrize("expr,data", REROUTE_CASES)
def test_out_of_domain_data(expr, data):
    got = jsonpath_mojo.find(expr, data)
    want = oracle_find(expr, data)
    assert got == want, f"\nexpr: {expr}\nwant: {want}\ngot:  {got}"


# ---------------------------------------------------------------------------
# Seeded fuzz: random JSON documents x query battery
# ---------------------------------------------------------------------------

_FUZZ_QUERIES = [
    "$", "$.*", "$..*", "$..a", "$..b", "$..z", "$.a", "$.b.c", "$.a.b.c",
    "$['a']", "$['a','b']", "$['zz','a']", "$.a[*]", "$.a[0]", "$.a[-1]",
    "$.a[1:3]", "$.a[::2]", "$.a[::-1]", "$.a[3:0:-1]", "$.a[5:]",
    "$.a[?(@ > 1)]", "$.a[?(@ == 1)]", "$.a[?(@ == 1.5)]", "$.a[?(@ == 'x')]",
    "$.a[?(@ != 'x')]", "$.a[?(@ > 1 & @ < 3)]", "$.a[?(@.b == 1)]",
    "$.a[?(@.b)]", "$.a[?(@.b.c == 2)]", "$[?(@ > 1)]", "$[?(@.a == 1)]",
    "$..[0]", "$..a[0]", "$..a[*]", "$.a.b[*].c", "$..c", "$..c.d",
]


def _rand_value(rng, depth):
    if depth <= 0:
        kind = rng.choice(["int", "float", "str", "bool", "null"])
    else:
        kind = rng.choice(["int", "float", "str", "bool", "null", "list", "dict", "list", "dict"])
    if kind == "int":
        return rng.choice([0, 1, -1, 2, 5, 7, 10, rng.randint(-100, 100)])
    if kind == "float":
        return rng.choice([0.0, 1.5, -2.25, 8.95, 1e10, rng.uniform(-50, 50)])
    if kind == "str":
        return rng.choice(["", "x", "y", "fiction", "a b", "üñï", "5", " 3 ", "null", "true"])
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "null":
        return None
    if kind == "list":
        return [_rand_value(rng, depth - 1) for _ in range(rng.randint(0, 5))]
    keys = rng.sample(["a", "b", "c", "d", "x", "y", "z", "0", "a b", "ü"], k=rng.randint(0, 4))
    return {k: _rand_value(rng, depth - 1) for k in keys}


@pytest.mark.parametrize("seed", range(12))
def test_seeded_fuzz(seed):
    rng = random.Random(10_000 + seed)
    docs = [_rand_value(rng, rng.randint(1, 4)) for _ in range(4)]
    for doc in docs:
        for expr in _FUZZ_QUERIES:
            try:
                want = oracle_find(expr, doc)
                want_exc = None
            except Exception as exc:  # noqa: BLE001 - mirroring whatever the oracle does
                want, want_exc = None, type(exc)
            try:
                got = jsonpath_mojo.find(expr, doc)
                got_exc = None
            except Exception as exc:  # noqa: BLE001
                got, got_exc = None, type(exc)
            if want_exc is not None:
                assert got_exc is not None and (
                    issubclass(got_exc, want_exc) or issubclass(want_exc, got_exc)
                ), f"seed {seed} expr {expr} doc {doc!r}: oracle raised {want_exc.__name__}, we raised {got_exc}"
            else:
                assert got_exc is None, f"seed {seed} expr {expr} doc {doc!r}: oracle ok, we raised {got_exc}"
                assert got == want, f"\nseed: {seed}\nexpr: {expr}\ndoc: {doc!r}\nwant: {want}\ngot:  {got}"


def test_deep_nesting():
    doc = cur = {}
    node = doc
    for i in range(60):
        node["lvl"] = i
        node["next"] = {}
        node = node["next"]
    node["lvl"] = 60
    for q in ["$..lvl", "$.next.next.next.lvl", "$..next..lvl"]:
        _check_value_parity(q, doc)
    # `$..[?...]` is an oracle parse error; `$..next[0]` hits KeyError on dicts
    import pytest as _pytest
    with _pytest.raises(JsonPathError):
        jsonpath_mojo.find("$..[?(@.lvl > 55)]", doc)
    with _pytest.raises(KeyError):
        jsonpath_mojo.find("$..next[0]", doc)


def test_big_flat_array():
    rng = random.Random(7)
    doc = {"vals": [rng.randint(-50, 50) for _ in range(300)]}
    for q in ["$.vals[?(@ > 40)]", "$.vals[100:200:7]", "$.vals[::-13]", "$.vals[*]",
              "$..vals[?(@ >= -1 & @ <= 1)]", "$.vals[?(@ * 2 > 60)]"]:
        _check_value_parity(q, doc)
