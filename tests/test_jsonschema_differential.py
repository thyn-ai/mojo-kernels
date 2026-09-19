"""Differential tests: jsonschema_mojo must match the `jsonschema` package.

Run twice by `scripts/test_all_jsonschema.sh`: once against the native Mojo
kernel and once with JSONSCHEMA_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle everywhere:

* pass/fail outcome (is_valid) — exact;
* the multiset of failing instance JSON paths and keywords — exact;
* error message text — exact (the templates are replicated verbatim);
* error ordering is NOT part of the contract (documented): the oracle
  orders errors by schema-dict insertion order, the native backend uses a
  fixed canonical order. Comparisons here sort by (json_path, validator).

The oracle is the published PyPI package (jsonschema 4.26.0 at time of
writing; any 4.x with draft 2020-12 semantics works).

Instances stay inside the JSON-document domain (dicts with string keys,
lists, str, int within ±2**53, finite floats, bool, None). Python-specific
objects (tuples, non-string keys, Decimal, NaN/Infinity, huge ints) are
covered by test_auto_degrade_* — they are detected and validated by the
exact fallback, so parity holds there too; what the native path does with
them is documented in the package README.
"""

from __future__ import annotations

import math
import os
import random
import re

import pytest
from jsonschema import Draft202012Validator as OracleValidator

import jsonschema_mojo
from jsonschema_mojo import UnsupportedSchemaError, ValidationError, Validator

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _expected_backend() -> str:
    # scripts/test_all_jsonschema.sh runs the suite once per backend.
    return "fallback" if os.environ.get("JSONSCHEMA_MOJO_DISABLE_NATIVE") == "1" else "native"


def _keyed(errors):
    """Sorted multiset of (json_path, validator, message) for order-free parity."""
    return sorted((e.json_path, e.validator or "", e.message) for e in errors)


def _keyed_paths(errors):
    return sorted((e.json_path, e.validator or "") for e in errors)


def check_parity(schema, instance, *, check_messages: bool = True):
    """Full parity: pass/fail, failing path+keyword multiset, messages, and
    the raised-error identity of validate()."""
    ours = Validator(schema)
    assert ours.backend == _expected_backend()
    oracle = OracleValidator(schema)

    our_errors = list(ours.iter_errors(instance))
    ref_errors = list(oracle.iter_errors(instance))

    assert ours.is_valid(instance) == oracle.is_valid(instance)
    assert _keyed_paths(our_errors) == _keyed_paths(ref_errors)
    if check_messages:
        assert _keyed(our_errors) == _keyed(ref_errors)

    # Error object shape parity (on the sorted-first error, if any).
    for our_e, ref_e in zip(
        sorted(our_errors, key=lambda e: (e.json_path, e.validator or "")),
        sorted(ref_errors, key=lambda e: (e.json_path, e.validator or "")),
    ):
        assert list(our_e.absolute_path) == list(ref_e.absolute_path)
        assert [str(p) for p in our_e.absolute_schema_path] == [
            str(p) for p in ref_e.absolute_schema_path
        ]
        assert our_e.validator_value == ref_e.validator_value
        assert our_e.context == ref_e.context == []

    # validate(): same pass/fail; the raised error's identity is best-effort
    # (both sides raise their own first error; iteration orders differ), but
    # it must be one of the oracle's errors, and identical when there is
    # exactly one error.
    our_raised = None
    try:
        ours.validate(instance)
    except ValidationError as e:
        our_raised = e.message
    ref_raised = None
    try:
        oracle.validate(instance)
    except Exception as e:  # oracle ValidationError
        ref_raised = e.message
    assert (our_raised is None) == (ref_raised is None)
    if ref_raised is not None:
        assert our_raised in {e.message for e in ref_errors}
        if len(ref_errors) == 1:
            assert our_raised == ref_raised


# --------------------------------------------------------------------------
# Curated per-keyword cases
# --------------------------------------------------------------------------

CASES = [
    # --- type ------------------------------------------------------------
    pytest.param({"type": "string"}, "x", id="type-str-ok"),
    pytest.param({"type": "string"}, 1, id="type-str-fail"),
    pytest.param({"type": "integer"}, 1, id="type-int-ok"),
    pytest.param({"type": "integer"}, 1.0, id="type-int-float-integral"),
    pytest.param({"type": "integer"}, 1.5, id="type-int-float-fail"),
    pytest.param({"type": "integer"}, True, id="type-int-bool-fail"),
    pytest.param({"type": "number"}, True, id="type-num-bool-fail"),
    pytest.param({"type": "number"}, 2.5, id="type-num-ok"),
    pytest.param({"type": "boolean"}, False, id="type-bool-ok"),
    pytest.param({"type": "null"}, None, id="type-null-ok"),
    pytest.param({"type": "null"}, 0, id="type-null-fail"),
    pytest.param({"type": "array"}, [1, 2], id="type-array-ok"),
    pytest.param({"type": "object"}, {"a": 1}, id="type-object-ok"),
    pytest.param({"type": ["string", "integer"]}, 3, id="type-list-second"),
    pytest.param({"type": ["string", "integer"]}, 1.5, id="type-list-fail"),
    pytest.param({"type": ["null", "boolean"]}, None, id="type-list-null"),
    # --- enum / const ------------------------------------------------------
    pytest.param({"enum": [1, "a", None, [1, 2], {"k": True}]}, {"k": True}, id="enum-object-hit"),
    pytest.param({"enum": [1, "a", None, [1, 2], {"k": True}]}, {"k": False}, id="enum-object-miss"),
    pytest.param({"enum": [1, 2, 3]}, 1.0, id="enum-int-float-equal"),
    pytest.param({"enum": [1]}, True, id="enum-bool-not-int"),
    pytest.param({"enum": [True]}, 1, id="enum-int-not-bool"),
    pytest.param({"enum": [0.0]}, -0.0, id="enum-neg-zero"),
    pytest.param({"enum": ["1"]}, 1, id="enum-str-not-num"),
    pytest.param({"const": {"a": [1, {"b": 2}]}}, {"a": [1, {"b": 2}]}, id="const-deep-hit"),
    pytest.param({"const": {"a": [1, {"b": 2}]}}, {"a": [1, {"b": 2.5}]}, id="const-deep-miss"),
    pytest.param({"const": None}, None, id="const-null"),
    pytest.param({"const": False}, 0, id="const-false-not-zero"),
    pytest.param({"const": 2.5}, 2.5, id="const-float-hit"),
    # --- numeric bounds ----------------------------------------------------
    pytest.param({"minimum": 5}, 5, id="minimum-eq-ok"),
    pytest.param({"minimum": 5}, 4.999, id="minimum-fail"),
    pytest.param({"minimum": 2.5}, 2.5, id="minimum-float-eq"),
    pytest.param({"maximum": 5}, 5.001, id="maximum-fail"),
    pytest.param({"maximum": -3}, -2, id="maximum-neg-fail"),
    pytest.param({"exclusiveMinimum": 5}, 5, id="exclmin-eq-fail"),
    pytest.param({"exclusiveMinimum": 5}, 5.0000001, id="exclmin-ok"),
    pytest.param({"exclusiveMaximum": 5.5}, 5.5, id="exclmax-eq-fail"),
    pytest.param({"minimum": 0, "maximum": 10}, 11, id="minmax-max-fail"),
    pytest.param({"minimum": 1e-9}, 1e-10, id="minimum-tiny-fail"),
    pytest.param({"minimum": 0}, -0.0, id="minimum-neg-zero-ok"),
    # --- multipleOf ---------------------------------------------------------
    pytest.param({"multipleOf": 2}, 4, id="mult-int-ok"),
    pytest.param({"multipleOf": 2}, 3, id="mult-int-fail"),
    pytest.param({"multipleOf": 2}, 4.0, id="mult-int-float-ok"),
    pytest.param({"multipleOf": 2}, -4, id="mult-int-neg-ok"),
    pytest.param({"multipleOf": 2}, -3, id="mult-int-neg-fail"),
    pytest.param({"multipleOf": 3}, 1e18, id="mult-int-big-defer"),
    pytest.param({"multipleOf": 0.1}, 0.3, id="mult-float-03-fail"),
    pytest.param({"multipleOf": 0.5}, 2.5, id="mult-float-ok"),
    pytest.param({"multipleOf": 2.0}, 4.0, id="mult-float-int-ok"),
    pytest.param({"multipleOf": 1e-3}, 1e-9, id="mult-small-ok"),
    pytest.param({"multipleOf": 0.0001}, 1e300, id="mult-overflow-defer"),
    # --- strings -------------------------------------------------------------
    pytest.param({"minLength": 1}, "", id="minlen-empty-fail"),
    pytest.param({"minLength": 3}, "ab", id="minlen-fail"),
    pytest.param({"minLength": 2}, "héllo", id="minlen-unicode-ok"),
    pytest.param({"maxLength": 0}, "x", id="maxlen-zero-fail"),
    pytest.param({"maxLength": 2}, "héllo", id="maxlen-unicode-fail"),
    pytest.param({"maxLength": 5}, "héllo", id="maxlen-unicode-ok"),
    pytest.param({"minLength": 2}, "😀😀", id="minlen-surrogate-pair"),
    pytest.param({"maxLength": 1}, "😀😀", id="maxlen-surrogate-fail"),
    pytest.param({"minLength": 1}, "a\nb", id="minlen-escape"),
    # --- pattern (kernel-deferred to Python re) -----------------------------
    pytest.param({"pattern": "^a+$"}, "aaa", id="pattern-ok"),
    pytest.param({"pattern": "^a+$"}, "aaab", id="pattern-fail"),
    pytest.param({"pattern": "^[a-z_][a-z0-9_]*$"}, "valid_name_1", id="pattern-ident-ok"),
    pytest.param({"pattern": "^[a-z_][a-z0-9_]*$"}, "1bad", id="pattern-ident-fail"),
    pytest.param({"pattern": "needle"}, "hay needle stack", id="pattern-search-ok"),
    pytest.param({"pattern": "é+"}, "aéé", id="pattern-unicode-ok"),
    pytest.param({"pattern": "^\\d{3}-\\d{4}$"}, "555-0199", id="pattern-escaped-ok"),
    pytest.param({"pattern": "^\\d{3}-\\d{4}$"}, "555-01990", id="pattern-escaped-fail"),
    pytest.param({"pattern": "x"}, 5, id="pattern-nonstring-skip"),
    # --- arrays --------------------------------------------------------------
    pytest.param({"minItems": 1}, [], id="minitems-empty-fail"),
    pytest.param({"minItems": 2}, [1], id="minitems-fail"),
    pytest.param({"maxItems": 0}, [1], id="maxitems-zero-fail"),
    pytest.param({"maxItems": 2}, [1, 2, 3], id="maxitems-fail"),
    pytest.param({"uniqueItems": True}, [1, 2, 3], id="unique-ok"),
    pytest.param({"uniqueItems": True}, [1, 2, 1], id="unique-fail"),
    pytest.param({"uniqueItems": True}, [1, 1.0], id="unique-int-float-dup"),
    pytest.param({"uniqueItems": True}, [True, 1], id="unique-bool-int-not-dup"),
    pytest.param({"uniqueItems": True}, [0.0, -0.0], id="unique-neg-zero-dup"),
    pytest.param({"uniqueItems": True}, [[1, 2], [1, 2]], id="unique-array-dup"),
    pytest.param({"uniqueItems": True}, [{"a": 1, "b": 2}, {"b": 2, "a": 1}], id="unique-object-dup"),
    pytest.param({"uniqueItems": True}, ["a", "b", "a"], id="unique-str-dup"),
    pytest.param({"uniqueItems": True}, [None, None], id="unique-null-dup"),
    pytest.param({"uniqueItems": True}, list(range(500)), id="unique-big-ok"),
    pytest.param({"uniqueItems": True}, [divmod(i, 7)[0] for i in range(500)], id="unique-big-dup"),
    pytest.param({"uniqueItems": False}, [1, 1], id="unique-false-skip"),
    pytest.param({"items": {"type": "integer"}}, [1, 2, 3], id="items-schema-ok"),
    pytest.param({"items": {"type": "integer"}}, [1, "x", True], id="items-schema-fail"),
    pytest.param({"items": {"minimum": 2}}, [2, 1, 3], id="items-schema-nested-fail"),
    pytest.param({"items": False}, [], id="items-false-empty-ok"),
    pytest.param({"items": False}, [1, 2], id="items-false-fail"),
    pytest.param({"items": False}, [1], id="items-false-single"),
    pytest.param({"items": True}, [1, "x"], id="items-true-ok"),
    # --- objects -------------------------------------------------------------
    pytest.param({"required": ["a", "b"]}, {"a": 1}, id="required-one-missing"),
    pytest.param({"required": ["a", "b", "c"]}, {}, id="required-all-missing"),
    pytest.param({"required": ["a", "b", "c"]}, {"c": 0, "a": 1}, id="required-two-missing"),
    pytest.param({"required": []}, {}, id="required-empty"),
    pytest.param({"minProperties": 1}, {}, id="minprops-fail"),
    pytest.param({"minProperties": 2}, {"a": 1}, id="minprops-two-fail"),
    pytest.param({"maxProperties": 0}, {"a": 1}, id="maxprops-zero-fail"),
    pytest.param({"maxProperties": 1}, {"a": 1, "b": 2}, id="maxprops-fail"),
    pytest.param(
        {"properties": {"a": {"type": "integer"}}},
        {"a": "x", "b": "y"},
        id="properties-descend-fail",
    ),
    pytest.param(
        {"properties": {"a": {"type": "integer"}}},
        {"b": "y"},
        id="properties-absent-ok",
    ),
    pytest.param(
        {"properties": {"a": {"minLength": 2}}, "additionalProperties": False},
        {"a": "x", "b": 1, "c": 2},
        id="ap-false-multi-extra",
    ),
    pytest.param(
        {"additionalProperties": False},
        {"b": 1},
        id="ap-false-no-props",
    ),
    pytest.param(
        {"properties": {"a": {}}, "additionalProperties": False},
        {"a": 1, "b": 2},
        id="ap-false-one-extra",
    ),
    pytest.param(
        {"additionalProperties": {"type": "string"}},
        {"x": "ok", "y": 1},
        id="ap-schema-fail",
    ),
    pytest.param(
        {"properties": {"a": {}}, "additionalProperties": {"minimum": 2}},
        {"a": 0, "x": 1, "y": 3},
        id="ap-schema-one-fail",
    ),
    pytest.param({"additionalProperties": True}, {"x": 1}, id="ap-true-ok"),
    # --- boolean schemas / empty schema --------------------------------------
    pytest.param(True, 42, id="bool-true-accepts"),
    pytest.param(False, 42, id="bool-false-rejects"),
    pytest.param(False, {"a": [1]}, id="bool-false-rejects-object"),
    pytest.param({}, 42, id="empty-schema-accepts"),
    pytest.param({"properties": {"a": False}}, {"a": 1}, id="bool-false-subschema"),
    pytest.param({"properties": {"a": True}}, {"a": 1}, id="bool-true-subschema"),
    pytest.param({"items": {"items": {"items": False}}}, [[[1]]], id="nested-items-false"),
    pytest.param({"items": {"items": {"type": "string"}}}, [["a"], [1]], id="nested-items-type"),
    # --- paths / json_path quoting -------------------------------------------
    pytest.param(
        {"properties": {"a b": {"type": "integer"}, "0x": {"type": "integer"}, "it's": {"type": "integer"}, "héllo": {"type": "integer"}}},
        {"a b": "v", "0x": "v", "it's": "v", "héllo": "v"},
        id="jsonpath-quoting",
    ),
    pytest.param(
        {"properties": {"nested": {"properties": {"deep": {"const": 1}}, "required": ["deep"]}}},
        {"nested": {"deep": 2}},
        id="nested-const-fail",
    ),
    pytest.param(
        {"properties": {"nested": {"properties": {"deep": {"const": 1}}, "required": ["deep"]}}},
        {"nested": {}},
        id="nested-required-fail",
    ),
    # --- keyword combinations at one node ------------------------------------
    pytest.param(
        {"type": "integer", "minimum": 2, "maximum": 4, "multipleOf": 2, "enum": [2, 4]},
        3,
        id="combo-int-multi-fail",
    ),
    pytest.param(
        {"type": "string", "minLength": 3, "maxLength": 5, "pattern": "^[ab]+$"},
        "a",
        id="combo-str-multi-fail",
    ),
    pytest.param(
        {"type": "array", "minItems": 2, "maxItems": 3, "uniqueItems": True, "items": {"type": "integer"}},
        ["a"],
        id="combo-array-multi-fail",
    ),
    # --- annotations are inert ------------------------------------------------
    pytest.param(
        {"title": "t", "description": "d", "default": 5, "examples": [1], "$comment": "c",
         "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "https://x/y",
         "deprecated": True, "readOnly": False, "format": "email", "$defs": {"a": {"type": "string"}},
         "minimum": 3},
        2,
        id="annotations-inert",
    ),
]


@pytest.mark.parametrize("schema,instance", CASES)
def test_curated_parity(schema, instance):
    check_parity(schema, instance)


# --------------------------------------------------------------------------
# Auto-degrade: non-JSON-domain values still match the oracle exactly
# --------------------------------------------------------------------------


def test_auto_degrade_huge_int_instance():
    # Beyond ±2**53 the native path cannot represent the value exactly; the
    # wrapper must detect it (kernel big-number flag) and use the fallback.
    schema = {"minimum": 2**60, "maximum": 2**62}
    assert Validator(schema).is_valid(2**61) == OracleValidator(schema).is_valid(2**61)
    assert Validator(schema).is_valid(2**59) == OracleValidator(schema).is_valid(2**59)
    ours = [e.message for e in Validator(schema).iter_errors(2**59)]
    ref = [e.message for e in OracleValidator(schema).iter_errors(2**59)]
    assert ours == ref


def test_auto_degrade_huge_int_schema():
    schema = {"const": 2**60 + 1}
    assert Validator(schema).is_valid(2**60 + 1) == OracleValidator(schema).is_valid(2**60 + 1)
    assert Validator(schema).is_valid(2**60 + 2) == OracleValidator(schema).is_valid(2**60 + 2)


def test_auto_degrade_huge_int_enum_and_multipleof():
    schema = {"enum": [2**61], "multipleOf": 2**61}
    for inst in (2**61, 2**61 * 2, 2**61 + 1):
        ours = Validator(schema).is_valid(inst)
        ref = OracleValidator(schema).is_valid(inst)
        assert ours == ref


def test_auto_degrade_non_finite_float():
    schema = {"type": "number", "minimum": 0}
    for inst in (math.nan, math.inf, -math.inf):
        assert Validator(schema).is_valid(inst) == OracleValidator(schema).is_valid(inst)


def test_auto_degrade_non_serializable():
    schema = {"type": "null"}
    for inst in (object(), b"bytes", {1, 2}):
        assert Validator(schema).is_valid(inst) == OracleValidator(schema).is_valid(inst)
        ours = [e.message for e in Validator(schema).iter_errors(inst)]
        ref = [e.message for e in OracleValidator(schema).iter_errors(inst)]
        assert ours == ref


# --------------------------------------------------------------------------
# Subset gate
# --------------------------------------------------------------------------

UNSUPPORTED = [
    {"$ref": "#/definitions/x", "definitions": {"x": {}}},
    {"allOf": [{}]},
    {"anyOf": [{}]},
    {"oneOf": [{}]},
    {"not": {}},
    {"if": {}, "then": {}},
    {"else": {}},
    {"patternProperties": {"^a": {}}},
    {"prefixItems": [{}]},
    {"contains": {}},
    {"minContains": 1},
    {"maxContains": 2},
    {"propertyNames": {"minLength": 1}},
    {"dependentRequired": {"a": ["b"]}},
    {"dependentSchemas": {"a": {}}},
    {"dependencies": {"a": ["b"]}},
    {"unevaluatedItems": False},
    {"unevaluatedProperties": False},
    {"$dynamicRef": "#x"},
]


@pytest.mark.parametrize("schema", UNSUPPORTED)
def test_unsupported_keywords_raise(schema):
    with pytest.raises(UnsupportedSchemaError):
        Validator(schema)
    # nested positions must also be caught
    with pytest.raises(UnsupportedSchemaError):
        Validator({"properties": {"a": schema}})


BAD_VALUE_SHAPES = [
    {"type": "frobnicate"},
    {"type": []},
    {"type": ["string", "frob"]},
    {"type": 3},
    {"required": "name"},
    {"required": ["a", 1]},
    {"items": [True]},
    {"additionalProperties": 0},
    {"enum": "abc"},
    {"minimum": "5"},
    {"minimum": True},
    {"minLength": -1},
    {"minLength": 2.5},
    {"maxItems": True},
    {"pattern": 3},
    {"multipleOf": 0},
    {"multipleOf": "2"},
    {"uniqueItems": 1},
    {"properties": {"a": 5}},
]


@pytest.mark.parametrize("schema", BAD_VALUE_SHAPES)
def test_bad_keyword_values_raise(schema):
    with pytest.raises(UnsupportedSchemaError):
        Validator(schema)


# --------------------------------------------------------------------------
# Seeded randomized differential sweep
# --------------------------------------------------------------------------

_SAFE_PATTERNS = ["^a", "b$", "^[a-z]+$", "x{2}", "a|b", "^..$", r"\d+", "^$"]
_ATOMICS = [None, True, False, 0, 1, -2, 2.5, 0.1, -0.0, "", "a", "abc", "héllo", "😀"]


def _rand_value(rng: random.Random, depth: int):
    if depth <= 0 or rng.random() < 0.45:
        return rng.choice(_ATOMICS)
    if rng.random() < 0.5:
        return [_rand_value(rng, depth - 1) for _ in range(rng.randint(0, 5))]
    keys = rng.sample(["a", "b", "c", "d d", "0k", "é"], k=rng.randint(0, 4))
    return {k: _rand_value(rng, depth - 1) for k in keys}


def _rand_schema(rng: random.Random, depth: int) -> dict:
    schema: dict = {}
    n_kw = rng.randint(1, 4 if depth <= 0 else 3)
    choices = rng.sample(
        [
            "type",
            "enum",
            "const",
            "bounds",
            "mult",
            "strlen",
            "pattern",
            "arrlen",
            "unique",
            "proplen",
            "struct",
        ],
        k=n_kw,
    )
    for choice in choices:
        if choice == "type":
            names = rng.sample(
                ["null", "boolean", "object", "array", "number", "integer", "string"],
                k=rng.randint(1, 3),
            )
            schema["type"] = names[0] if len(names) == 1 else names
        elif choice == "enum":
            schema["enum"] = [_rand_value(rng, 1) for _ in range(rng.randint(1, 4))]
        elif choice == "const":
            schema["const"] = _rand_value(rng, 2)
        elif choice == "bounds":
            lo = rng.choice([-10, -2.5, 0, 1, 3])
            hi = rng.choice([5, 10, 2.5, 100])
            schema[rng.choice(["minimum", "exclusiveMinimum"])] = lo
            schema[rng.choice(["maximum", "exclusiveMaximum"])] = hi
        elif choice == "mult":
            schema["multipleOf"] = rng.choice([2, 3, 5, 0.5, 2.0, 0.25])
        elif choice == "strlen":
            schema["minLength"] = rng.randint(0, 3)
            schema["maxLength"] = rng.randint(3, 6)
        elif choice == "pattern":
            schema["pattern"] = rng.choice(_SAFE_PATTERNS)
        elif choice == "arrlen":
            schema["minItems"] = rng.randint(0, 2)
            schema["maxItems"] = rng.randint(2, 5)
        elif choice == "unique":
            schema["uniqueItems"] = rng.random() < 0.7
        elif choice == "proplen":
            schema["minProperties"] = rng.randint(0, 2)
            schema["maxProperties"] = rng.randint(2, 5)
        elif choice == "struct" and depth > 0:
            if rng.random() < 0.5:
                props = {
                    name: _rand_schema(rng, depth - 1)
                    for name in rng.sample(["a", "b", "c"], k=rng.randint(1, 3))
                }
                schema["properties"] = props
                if rng.random() < 0.6:
                    schema["required"] = rng.sample(
                        list(props), k=rng.randint(1, len(props))
                    )
                if rng.random() < 0.4:
                    schema["additionalProperties"] = (
                        False if rng.random() < 0.5 else _rand_schema(rng, depth - 1)
                    )
            else:
                schema["items"] = _rand_schema(rng, depth - 1)
    return schema


@pytest.mark.parametrize("seed", range(60))
def test_randomized_sweep(seed):
    rng = random.Random(20260919 + seed)
    schema = _rand_schema(rng, depth=3)
    for _ in range(6):
        instance = _rand_value(rng, depth=3)
        check_parity(schema, instance)


def test_randomized_deep_nesting():
    rng = random.Random(77)
    schema: dict = {"type": "array", "items": {}}
    node = schema
    for _ in range(8):
        node["items"] = {"type": "array", "minItems": 1, "items": {}}
        node = node["items"]
    node["items"] = {"type": "integer"}
    instance = [[[[[[[[[1, 2, "x"]]]]]]]]]
    check_parity(schema, instance)
    instance2 = _rand_value(rng, 0)
    check_parity(schema, instance2)


def test_pattern_deferral_keeps_record_order():
    # Errors around the deferred pattern check must all surface, and the
    # resolved pattern error must be among them.
    schema = {
        "properties": {
            "code": {"pattern": "^[0-9]+$", "minLength": 2},
            "qty": {"minimum": 1},
        },
        "required": ["code"],
    }
    instance = {"code": "12a", "qty": 0}
    check_parity(schema, instance)


def test_bad_regex_matches_oracle_behavior():
    # An invalid regex raises re.error from both the oracle and us (the
    # reference compiles lazily at search time).
    schema = {"pattern": "(["}
    with pytest.raises(re.error):
        list(OracleValidator(schema).iter_errors("x"))
    with pytest.raises(re.error):
        list(Validator(schema).iter_errors("x"))


def test_module_level_api():
    schema = {"type": "integer", "minimum": 3}
    jsonschema_mojo.validate(5, schema)
    with pytest.raises(ValidationError) as exc_info:
        jsonschema_mojo.validate(1, schema)
    oracle = None
    try:
        OracleValidator(schema).validate(1)
    except Exception as e:
        oracle = e.message
    assert exc_info.value.message == oracle
    errors = list(jsonschema_mojo.iter_errors(1, schema))
    assert len(errors) == 1 and errors[0].validator == "minimum"
    assert jsonschema_mojo.Draft202012Validator is jsonschema_mojo.Validator
