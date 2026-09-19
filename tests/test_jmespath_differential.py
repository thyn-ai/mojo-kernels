"""Differential tests: jmespath_mojo must match pip jmespath exactly.

Run twice by `scripts/test_all_jmespath.sh`: once against the native Mojo
kernel and once with JMESPATH_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the reference package (PyPI
jmespath 1.0.1) on every cell of a generated expression x document matrix:
structural equality with strict type checks (int vs float, -0.0 vs 0.0), and
equal exception class names when the expression raises.

Everything here is generated locally from explicit seeds — no network, no
randomness without a fixed seed — so the suite is bit-reproducible on any
machine.
"""

from __future__ import annotations

import math
import os
import random

import pytest
import jmespath  # the oracle (PyPI jmespath 1.0.1)

import jmespath_mojo
from jmespath_mojo import _native

EXPECTED_BACKEND = "fallback" if os.environ.get("JMESPATH_MOJO_DISABLE_NATIVE") == "1" else "native"


# ---------------------------------------------------------------------------
# Structural comparison: strict types (int vs float, bool vs int), sign of
# zero, NaN == NaN, exact float equality (kernels compute in reference order).
# ---------------------------------------------------------------------------


def assert_same(actual, expected, path="$"):
    ta, te = type(actual), type(expected)
    assert ta is te, f"{path}: type {ta.__name__} != {te.__name__} ({actual!r} vs {expected!r})"
    if te is float:
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: {actual!r} is not NaN"
        else:
            assert repr(actual) == repr(expected), f"{path}: {actual!r} != {expected!r}"
    elif te is list:
        assert len(actual) == len(expected), f"{path}: len {len(actual)} != {len(expected)}"
        for i, (x, y) in enumerate(zip(actual, expected)):
            assert_same(x, y, f"{path}[{i}]")
    elif te is dict:
        assert list(actual.keys()) == list(expected.keys()), (
            f"{path}: keys {list(actual.keys())} != {list(expected.keys())}"
        )
        for k in actual:
            assert_same(actual[k], expected[k], f"{path}.{k}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def run_cell(expression, data):
    """Returns (result, error_class_name) for ours and the reference."""
    try:
        ours = jmespath_mojo.search(expression, data)
        ours_err = None
    except Exception as e:  # noqa: BLE001 - comparing error classes is the point
        ours, ours_err = None, type(e).__name__
    try:
        ref = jmespath.search(expression, data)
        ref_err = None
    except Exception as e:  # noqa: BLE001
        ref, ref_err = None, type(e).__name__
    return (ours, ours_err), (ref, ref_err)


# ---------------------------------------------------------------------------
# Seeded document generator
# ---------------------------------------------------------------------------

WORDS = ["alpha", "beta", "gamma", "delta", "héllo", "wörld", "café", "𝄞note",
         "x", "yy", "", "0", "nested"]


def make_document(seed: int):
    rng = random.Random(seed)

    def scalar(depth):
        choice = rng.randrange(9)
        if choice == 0:
            return None
        if choice == 1:
            return rng.choice([True, False])
        if choice == 2:
            return rng.randint(-100, 100)
        if choice == 3:
            return rng.choice([0, 1, -1, 2**40, -2**40])
        if choice == 4:
            return round(rng.uniform(-100, 100), rng.randint(1, 6))
        if choice == 5:
            return rng.choice([0.0, -0.0, 1e-5, 1e16, 0.1, 2.5, -1.5e300 if depth < 2 else 3.14])
        if choice in (6, 7):
            return rng.choice(WORDS)
        return rng.choice(WORDS) * rng.randint(0, 3)

    def value(depth):
        if depth <= 0:
            return scalar(depth)
        choice = rng.randrange(10)
        if choice < 5:
            return scalar(depth)
        if choice < 8:
            return [value(depth - 1) for _ in range(rng.randint(0, 6))]
        keys = rng.sample(["name", "id", "age", "score", "tags", "state", "meta",
                           "x", "y", "z", "a b", "héllo"], k=rng.randint(0, 5))
        return {k: value(depth - 1) for k in keys}

    base = {
        "a": [0, 1, 2, 3, 4],
        "nums": [3, 1, 2, 1.5, -2, 0, 10],
        "strs": ["banana", "Apple", "cherry", "apple", "éclair", "zoo"],
        "empty": [],
        "eobj": {},
        "obj": {"k1": 1, "k2": None, "k3": [1, 2]},
        "people": [
            {"name": "ann", "age": 30, "tags": ["x", "y"], "state": {"live": True}},
            {"name": "bob", "age": 20, "tags": []},
            {"name": "cid", "age": 40, "tags": ["z"], "state": {"live": False}},
            {"name": "dot", "age": 30, "nick": "d"},
        ],
        "matrix": [[1, 2], [3, [4, 5]], [], [6]],
        "mixed": [1, "one", None, True, [1], {"k": 1}, 1.5, -0.0],
        "s": "héllo wörld",
        "n": None,
        "t": True,
        "f": False,
        "i": 42,
        "fl": 2.75,
        "weird keys": {"a b": 1, "": 2},
        "gen": value(rng.randint(1, 3)),
    }
    return base


@pytest.fixture(scope="module", params=list(range(12)))
def document(request):
    return make_document(request.param)


# ---------------------------------------------------------------------------
# Curated expression list (scope features, quirks, error cases)
# ---------------------------------------------------------------------------

EXPRESSIONS = [
    # fields / subexpressions
    "a", "nums", "missing", "missing.sub.deep", "obj.k1", "obj.k2", "obj.k3",
    "people[0].name", "people[3].nick", '"weird keys"."a b"', '"a b"',
    "people[0].state.live", "s", "n", "t", "f", "i", "fl",
    # index
    "a[0]", "a[-1]", "a[4]", "a[10]", "a[-10]", "n[0]", "s[0]", "obj[0]",
    "matrix[1][1][0]", "gen[0]",
    # slices
    "a[1:4]", "a[:2]", "a[2:]", "a[::2]", "a[::-1]", "a[3:1]", "a[3:1:-1]",
    "a[-3:]", "a[:-10:-1]", "a[1:100]", "a[10:20]", "a[-100:100]", "a[2:2]",
    "a[0:5:0]", "a[1:-1]", "mixed[2:5]", "people[0:2].name", "s[1:3]",
    # projections
    "a[*]", "mixed[*]", "people[*].name", "people[*].age", "people[].name",
    "people[].tags[]", "matrix[]", "matrix[][]", "obj.*", "obj.*.k1",
    "eobj.*", "empty[*]", "empty[].x", "n.*", "n[*]", "s[*]", "a.*",
    "people[*].state.live", "people[].tags", "*", "gen.*", "gen[*]",
    "people[*].missing", "matrix[*][*]", "matrix[0:2][]",
    # filters
    "people[?age > `25`].name", "people[?age >= `30`].name",
    "people[?age < `30`]", "people[?age <= `20`]",
    "people[?name == 'ann']", "people[?name != 'ann'].name",
    "people[?age > `20` && name == 'ann']", "people[?age > `35` || age < `25`].name",
    "people[?!(age > `20`)].name", "people[?tags].name", "people[?!tags].name",
    "people[?state.live].name", "people[?age == `30`].name",
    "people[?age > `20`].[name, age]",  # multiselect rhs (fallback)
    "a[? @ > `1`]", "a[? @ >= `2`]", "mixed[?@]", "mixed[? !@]",
    "people[?age > `20`][0]", "people[?age > `20`] | [0]",
    "people[?name > 'b'].name", "people[?name < 'c'].name",
    "people[?tags[0] == 'x'].name", "people[?length(tags) > `0`].name",
    "people[?age > `20`].age | [0]", "empty[?@]", "n[?@]", "s[?@]",
    # pipe
    "a | [0]", "a[*] | [0]", "obj.* | [0]", "obj.*.k1 | [0]",
    "people[*].name | [1]", "a | a[1]", "missing || `7`", "n || `8`",
    "empty || `9`", "eobj || `10`", "a | length(@)", "people | length(@)",
    "a | [1:3] | [0]", "a | [*]", "@ | a",
    # current node / identity / parens
    "@", "@.a", "(a)", "(missing || a)[1]", "(@)[0]", "people[].(@)",
    # literals
    "`1`", "`1.0`", "`1e2`", "`-0.0`", "`true`", "`false`", "`null`",
    "`\"str\"`", "`[1, 2]`", "`{\"a\": 1}`", "``", "`abc`", "`a b`", "`tru`",
    "`[1,2`", "` 1 `", "`1 `", "`01`", "`+1`", "`1_000`", "`TRUE`", "`NaN`",
    "`Infinity`", "`-Infinity`", "`null `", "`a\\u0041b`", "` a `", "`\"a\\nb\"`",
    "'raw'", "'a\\'b'", "'a\\\\'", "'a\\nb'", "'é'", "'0'",
    # comparisons
    "`1` == `1.0`", "`1` == `true`", "`1` == `\"1\"`", "`null` == `null`",
    "`[]` == `[]`", "`{}` == `{}`", "`[1,2]` == `[1,2]`", "`1` != `2`",
    "`1` < `2`", "`2` <= `2.0`", "`\"a\"` < `\"b\"`", "`\"10\"` < `\"2\"`",
    "`true` < `1`", "`null` < `1`", "`[]` < `1`", "`1` < `true`",
    "`{\"a\":1,\"b\":2}` == `{\"b\":2,\"a\":1}`", "`-0.0` == `0`",
    "s == `\"héllo wörld\"`", "s < `\"z\"`", "i < `\"z\"`", "i == `42`",
    "a == `1`", "obj.k1 == `1`", "missing == `null`", "missing != `null`",
    "`1e2` == `100`", "`9007199254740993` == `9007199254740992.0`",
    # and / or / not
    "`true` && `1`", "`false` && `1`", "`0` && `1`", "`\"\"` && `1`",
    "`[]` || `2`", "`{}` || `2`", "`\"0\"` || `2`", "`null` || `false` || `3`",
    "!`0`", "!`\"\"`", "!`[]`", "!`{}`", "!`null`", "!`true`", "!!`1`",
    "`true` || `false` && `false`", "`1` == `1` && `2` == `3`",
    "!`null` == `false`", "a[*] && `1`", "a[*] || `1`", "a[*] == `1`",
    "a[*].b && c", "a[?n] == `1`",
    # functions: the whole scoped set
    "abs(`-3`)", "abs(`-3.5`)", "abs(`3`)", "abs(i)", "abs(fl)", "abs('a')",
    "avg(nums)", "avg(`[]`)", "avg(`[1,2]`)", "avg(`[0.1,0.2]`)",
    "avg(`[9007199254740992, 9007199254740994]`)", "avg(strs)", "avg(a)",
    "ceil(`1.2`)", "ceil(`-1.2`)", "ceil(`3`)", "ceil(fl)", "floor(`1.8`)",
    "floor(`-1.8`)", "floor(i)",
    "contains(nums, `1`)", "contains(nums, `1.0`)", "contains(s, 'llo')",
    "contains(nums, 'a')", "contains(`[]`, `1`)", "contains(`[null]`, `null`)",
    "contains(`[{\"a\":1}]`, `{\"a\":1}`)", "contains('abc', `1`)",
    "contains(`1`, `1`)", "contains(`[\"abc\"]`, `\"b\"`)",
    "ends_with(s, 'rld')", "ends_with(s, 'xyz')", "starts_with(s, 'hél')",
    "starts_with(s, 'x')", "ends_with(`1`, 'a')",
    "join(`\",\"`, strs)", "join(`\"\"`, strs)", "join(`\",\"`, `[]`)",
    "join(`\",\"`, nums)", "join(`1`, strs)",
    "keys(obj)", "keys(`{}`)", "keys(`{\"b\":1,\"a\":2}`)", "keys(`[]`)",
    "values(obj)", "values(`{}`)", "values(`{\"b\":1,\"a\":2}`)",
    "length(s)", "length(`\"héllo\"`)", "length(`\"𝄞\"`)", "length(a)",
    "length(obj)", "length(`1`)", "length('')", "length(`\"\"`)",
    "map(&name, people)", "map(&age, people)", "map(&tags, people)",
    "map(&missing, people)", "map(&@, `[]`)", "map(&@, a)", "map(&name, `1`)",
    "map(&length(@), a)", "map(name, people)",
    "max(nums)", "max(`[]`)", "max(strs)", "max(mixed)", "max(`[1.5, 1]`)",
    "min(nums)", "min(`[]`)", "min(strs)", "max(`[true, 2]`)",
    "max_by(people, &age)", "max_by(people, &age).name", "min_by(people, &age)",
    "max_by(`[]`, &x)", "max_by(people, &name)", "max_by(people, &missing)",
    "min_by(people, &name)", "max_by(people, age)",
    "merge(obj, `{\"k1\": 9, \"k4\": 8}`)", "merge(`{}`)", "merge(`[]`, `{}`)",
    "merge()", "merge(obj, obj, `{\"k2\": 1}`)",
    "not_null(`null`, `null`)", "not_null(`null`, `1`, `2`)",
    "not_null(`0`, `1`)", "not_null(`\"\"`, `1`)", "not_null(`false`, `1`)",
    "not_null()", "not_null(missing, a)",
    "reverse(s)", "reverse(a)", "reverse(`\"héllo\"`)", "reverse(`[]`)",
    "reverse(`1`)",
    "sort(nums)", "sort(`[]`)", "sort(strs)", "sort(mixed)",
    "sort(`[2,1,1.0,0.5,-0.0]`)", "sort(`[\"b\",\"A\",\"a\",\"Z\",\"é\",\"z\"]`)",
    "sort(`[true, 1]`)",
    "sort_by(people, &age)", "sort_by(people, &age)[*].name",
    "sort_by(people, &name)", "sort_by(`[]`, &x)", "sort_by(people, &missing)",
    "sort_by(strs, &@)", "sort_by(nums, &@)", "sort_by(people, age)",
    "sort_by(`[{\"k\":1},{\"k\":\"a\"}]`, &k)", "sort_by(`[{\"k\":1},{\"k\":1.5}]`, &k)",
    "sum(nums)", "sum(`[]`)", "sum(`[1,2]`)", "sum(`[1.0,2]`)", "sum(a)",
    "sum(strs)", "sum(`[true, 1]`)", "sum(`[9007199254740992, 1]`)",
    "to_array(`1`)", "to_array(a)", "to_array(`[]`)",
    "to_number('3')", "to_number('3.5')", "to_number('-3')", "to_number(' 3 ')",
    "to_number('abc')", "to_number('1e3')", "to_number('03')", "to_number('3.')",
    "to_number('.5')", "to_number('1_0')", "to_number('inf')", "to_number('nan')",
    "to_number('-Infinity')", "to_number('0x10')", "to_number('')",
    "to_number(`true`)", "to_number(`null`)", "to_number(`[]`)",
    "to_number(to_string(`1.5`))", "to_number(`1`)", "to_number(`1.5`)",
    "to_string(`1`)", "to_string(`1.0`)", "to_string(`[1, 2]`)",
    "to_string(obj)", "to_string(`true`)", "to_string(`null`)",
    "to_string(s)", "to_string(`[\"café\"]`)", "to_string(`1e3`)",
    "to_string(`\"𝄞\"`)", "to_string(`[\"é\",\"𝄞\"]`)", "to_string(`1e16`)",
    "to_string(`1e15`)", "to_string(`1e-5`)", "to_string(`0.0001`)",
    "to_string(`-0.0`)", "to_string(`123456789012345678`)",
    "to_string(avg(`[0.1,0.2]`))", "to_string(avg(`[1,3]`))",
    "type(`1`)", "type(`1.0`)", "type('a')", "type(`true`)", "type(`null`)",
    "type(a)", "type(obj)", "type(&@)",
    "values(`{\"b\":1,\"a\":2}`)",
    # functions after dots
    "a.length(@)", "nums.sum(@)", "strs.sort(@)", "strs.reverse(@)",
    "obj.keys(@)", "obj.values(@)", "people.max_by(&age)", "s.reverse(@)",
    # expref edge
    "sort_by(people, &age || name)", "map(&a || `9`, `[{\"a\": 1}, {}]`)",
    "map(&a && `9`, `[{\"a\": 1}, {}]`)",
    # multiselect (kernel punts -> fallback must match)
    "a.{b: x}", "people[0].{n: name, a: age}", "{x: a, y: obj}",
    "people[0].[name, age]", "people[*].[name, age]", "n.{b: x}", "n.[x, y]",
    # CPython 3.12 compensated float summation (Neumaier), adversarial cases
    "sum(`[1e100, 1, -1e100]`)", "sum(`[0.1, 0.2, 0.3]`)", "sum(`[1e16, 1, 1]`)",
    "sum(`[1e100, 1, -1e100, 2]`)", "sum(`[1, 1e100, -1e100, 2]`)",
    "avg(`[1e100, 1, -1e100]`)", "avg(`[0.1, 0.2, 0.3]`)", "avg(`[1e16, 1, 1]`)",
    "sum(`[1e308, 1e308]`)", "sum(`[1e308, 1e308, -1e308]`)",
    "sum(`[9007199254740992.0, 1.0, -9007199254740992]`)",
    "sum(`[1e16, 1.5, -1.5]`)", "sum(`[0.1, 0.2]`)", "sum(`[0.1, 0.2, 0.7]`)",
    "avg(`[1e100, 1, -1e100, 1, -1, 1]`)",
    # invalid expressions (error-class parity)
    "", "foo(", "nosuchfunc(a)", "abs(a, a)", "abs()", "a[", ".a", "a..b",
    "sort_by(people, age)", "people[age > `1`]", "length()", "merge(a,)",
    "'a' 'b'", "foo\"bar\"", "1foo", "a[*].@", "a[].@", "=", "a ==",
    "not_null()", "people[?age > `20`].(a)", "@ foo", "`unclosed",
]

# Expressions whose results are non-deterministic in the reference itself
# (live object addresses) or that crash it; excluded from value comparison.
EXCLUDED = {"to_string(&@)"}


def test_matrix(document):
    stats_before = dict(_native._STATS)
    mismatches = []
    for expr in EXPRESSIONS:
        if expr in EXCLUDED:
            continue
        (ours, ours_err), (ref, ref_err) = run_cell(expr, document)
        if ours_err != ref_err:
            mismatches.append(
                f"{expr!r}: ours error={ours_err} ref error={ref_err}"
            )
            continue
        if ours_err is not None:
            continue  # same error class on both sides
        try:
            assert_same(ours, ref)
        except AssertionError as exc:
            mismatches.append(f"{expr!r}: {exc}")
    assert not mismatches, (
        f"{len(mismatches)} mismatches (backend={EXPECTED_BACKEND}):\n"
        + "\n".join(mismatches[:20])
    )
    # In the native run, most cells must have engaged the native kernel
    # (multiselect, exotic literals and error cells engage the fallback).
    stats_after = dict(_native._STATS)
    engaged = stats_after["fallback_engaged"] - stats_before["fallback_engaged"]
    native_ok = stats_after["native_ok"] - stats_before["native_ok"]
    if EXPECTED_BACKEND == "native":
        assert native_ok > 0, "native kernel was not used at all"
        # multiselect/error/unsupported cells engage the fallback by design;
        # they must stay a small minority of the matrix.
        total = len(EXPRESSIONS) - len(EXCLUDED)
        assert engaged < total // 3, (
            f"too many cells fell back to pure Python: {engaged}/{total}"
        )
    else:
        assert native_ok == 0, "fallback run must not touch the native kernel"


# ---------------------------------------------------------------------------
# Focused parity probes (documents chosen to exercise specific semantics)
# ---------------------------------------------------------------------------

FOCUSED = [
    # (expression, document)
    ("a[*]", {"a": [1, None, 2]}),
    ("a[]", {"a": [[1], None, [2], 3]}),
    ("o.*", {"o": {"x": 1, "y": None}}),
    ("a[0:9]", {"a": [1, None, 2]}),
    ("a[*][*]", {"a": [[1, None], [2]]}),
    ("people[?@]", {"people": [0, 1, None, "", "x", [], [1], {}, {"a": 1}, False, True, 0.0]}),
    ("map(&@, `[0, null, \"\", [], {}, false, true]`)", {}),
    ("contains(`[1]`, `1.0`)", {}),
    ("sum(`[0.1, 0.2, 0.3, 0.4]`)", {}),
    ("avg(`[1, 2, 3, 4]`)", {}),
    ("to_string(`9007199254740993`)", {}),
    ("abs(`-9223372036854775808`)", {}),
    ("sum(`[9223372036854775807, 1]`)", {}),
    ("`9223372036854775808`", {}),
    ("to_number('99999999999999999999')", {}),
    ("to_number('1_000')", {}),
    ("to_number('1__0')", {}),
    ("to_number('1_')", {}),
    ("to_number('_1')", {}),
    ("to_number('1.5_')", {}),
    ("to_number('1.5_0')", {}),
    ("to_number('1e1_0')", {}),
    ("to_number(' +3 ')", {}),
    ("to_number('INF')", {}),
    ("to_number('NaN')", {}),
    ("to_number('infinity')", {}),
    ("ceil(`1e17`)", {}),
    ("floor(`-1e17`)", {}),
    ("ceil(`1e100`)", {}),
    ("ceil(`0.5`)", {}),
    ("ceil(`-0.5`)", {}),
    ("floor(`0.5`)", {}),
    ("ceil(`-0.0`)", {}),
    ("to_string(ceil(`1.2`))", {}),
    ("max(`[1, 1.0]`)", {}),
    ("min(`[1.0, 1]`)", {}),
    ("max(`[1.0, 1]`)", {}),
    ("max(`[nan-safe]`)", {}),
    ("sort(`[3, 1, 2, 1.0, 1]`)", {}),
    ("sort_by(`[{\"k\":2.5,\"i\":0},{\"k\":2,\"i\":1},{\"k\":-1,\"i\":2}]`, &k)", {}),
    ("merge(`{\"a\":1,\"b\":2}`, `{\"b\":9}`, `{\"c\":3,\"a\":0}`)", {}),
    ("join(`\"--\"`, `[\"a\",\"é\",\"𝄞\"]`)", {}),
    ("reverse(`\"ab𝄞c\"`)", {}),
    ("length(`\"𝄞𝄞𝄞\"`)", {}),
    ("contains(`\"héllo\"`, `\"é\"`)", {}),
    ("starts_with(`\"𝄞abc\"`, `\"𝄞\"`)", {}),
    ("ends_with(`\"abc𝄞\"`, `\"𝄞\"`)", {}),
    ("keys(gen)", {"gen": {"b": 1, "a": 2, "é": 3}}),
    ("`\"\\ud834\\udd1e\"`", {}),
    ("`\\ud834\\udd1e`", {}),
    ("length(keys(`{\"𝄞\":1}`)[0])", {}),
    ("type(`[]`)", {}),
    ("not_null(not_null(`null`), `5`)", {}),
    ("deep.deeper[0].leaf", {"deep": {"deeper": [{"leaf": 1}]}}),
    ("deep.deeper[*].leaf", {"deep": {"deeper": [{"leaf": 1}, {"leaf": None}, {}]}}),
    ("to_array(@)", [1, 2]),
    ("@[0]", [10, 20]),
    ("@[?@ > `0`]", {"not": "an array"}),
    ("@ | [1]", ["a", "b", "c"]),
    ("@", None),
    ("@", 5),
    ("@", "just a string"),
    ("length(@)", "unicode é"),
    ("to_string(@)", {"a": [1, {"b": "é"}]}),
    ("@", {"a": 1}),
    ("@.missing || `\"fallback\"`", {}),
    ("`[1, [2, [3, [4]]]]`[1][1][1][0]", {}),
    ("people[?age > `20` && (name == 'ann' || name == 'cid')].name",
     {"people": [{"name": "ann", "age": 30}, {"name": "bob", "age": 20}, {"name": "cid", "age": 40}]}),
    ("people[? !(tags && age == `20`)].name",
     {"people": [{"name": "ann", "age": 30, "tags": ["x"]}, {"name": "bob", "age": 20, "tags": []}]}),
]


@pytest.mark.parametrize("expr,data", FOCUSED)
def test_focused(expr, data):
    (ours, ours_err), (ref, ref_err) = run_cell(expr, data)
    assert ours_err == ref_err, f"{expr!r}: ours error={ours_err} ref error={ref_err}"
    if ours_err is None:
        assert_same(ours, ref)


def test_native_stats_accounting():
    # In the native run, a plain in-scope expression must be native-served;
    # multiselect must engage the fallback (and still match the reference).
    before = dict(_native._STATS)
    jmespath_mojo.search("a[*]", {"a": [1, 2]})
    jmespath_mojo.search("a.{b: x}", {"a": {"x": 1}})
    after = dict(_native._STATS)
    if EXPECTED_BACKEND == "native":
        assert after["native_ok"] - before["native_ok"] == 1
        assert after["fallback_engaged"] - before["fallback_engaged"] == 1
    else:
        assert after["native_ok"] - before["native_ok"] == 0


def test_non_json_documents_use_fallback_but_match():
    # tuples / non-str keys: the reference leaks Python behaviour; we must too.
    docs = [
        ("a", {"a": (1, 2)}),
        ("a[0]", {"a": (1, 2)}),
        ("[0]", (10, 20)),
        ("length(a)", {"a": (1, 2, 3)}),
        ("to_string(a)", {"a": (1, 2)}),
        ("keys(@)", {1: "a", "b": "c"}),
        ("b", {1: "a", "b": "c"}),
        ("to_string(@)", {1: "a"}),
        ("type(a)", {"a": b"bytes"}),
        ("a", {"a": frozenset([1])}),
    ]
    for expr, data in docs:
        (ours, ours_err), (ref, ref_err) = run_cell(expr, data)
        assert ours_err == ref_err, f"{expr!r} on {data!r}: {ours_err} != {ref_err}"
        if ours_err is None:
            assert_same(ours, ref)


def test_summation_fuzz_seeded():
    """Seeded random float lists: sum/avg must match CPython 3.12's
    compensated summation bit-for-bit on both backends."""
    rng = random.Random(20260919)
    for trial in range(200):
        n = rng.randint(0, 8)
        xs = []
        for _ in range(n):
            kind = rng.randrange(4)
            if kind == 0:
                xs.append(rng.randint(-10**6, 10**6))
            elif kind == 1:
                xs.append(round(rng.uniform(-100, 100), rng.randint(1, 4)))
            elif kind == 2:
                xs.append(rng.uniform(-1e100, 1e100))
            else:
                xs.append(rng.choice([0.1, 0.2, 0.3, 1e16, -1e16, 1.5, -1.5, 1e-100]))
        lit = "`[" + ", ".join(repr(x) for x in xs) + "]`"
        for fn in ("sum", "avg"):
            expr = f"{fn}({lit})"
            (ours, ours_err), (ref, ref_err) = run_cell(expr, {})
            assert ours_err == ref_err, f"{expr}: {ours_err} != {ref_err}"
            if ours_err is None:
                assert_same(ours, ref, expr)


def test_expression_type_errors_match():
    for bad in [None, 123, ["a"], {"a": 1}, b"a"]:
        (ours, ours_err), (ref, ref_err) = run_cell(bad, {"a": 1})
        assert ours_err == ref_err, f"search({bad!r}): {ours_err} != {ref_err}"
        if ours_err is None:
            assert_same(ours, ref)
