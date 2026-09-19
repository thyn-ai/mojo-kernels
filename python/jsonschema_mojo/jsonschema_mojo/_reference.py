"""Vendored pure-Python validator for the supported draft 2020-12 subset.

This is the portability fallback (Windows, missing/unloadable native kernel,
instances the native path cannot represent exactly) and the correctness
oracle of the package: it mirrors the published `jsonschema` package's
observable behaviour on Python objects, including its edge cases:

* ``1.0`` satisfies ``"integer"`` (floats with integral values count);
* booleans are never numbers, and ``True``/``False`` never equal ``1``/``0``
  for ``const``/``enum``/``uniqueItems`` (numbers do compare across
  int/float form, so ``1 == 1.0``);
* ``"array"`` means ``list`` (not tuple), ``"object"`` means ``dict``;
* ``multipleOf`` uses Python's floored ``%`` for integer divisors and the
  ``int(quotient) != quotient`` test for float divisors, with an exact
  Fraction fallback when the quotient overflows;
* errors are yielded in schema-dict insertion order, like the reference.

Yields ``(keyword, instance_path, schema_path)`` records — the same shape
the native backend decodes — so both backends build identical error objects
through ``core._render_error``.
"""

from __future__ import annotations

import numbers
import re
from fractions import Fraction

Record = tuple[str, tuple, tuple]


def is_type(instance, type_name: str) -> bool:
    if type_name == "null":
        return instance is None
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "integer":
        # bool inherits from int, so ensure bools aren't reported as ints;
        # floats with integral values count as integers (draft 2020-12).
        if isinstance(instance, bool):
            return False
        return isinstance(instance, int) or (
            isinstance(instance, float) and instance.is_integer()
        )
    if type_name == "number":
        return isinstance(instance, numbers.Number) and not isinstance(instance, bool)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "object":
        return isinstance(instance, dict)
    raise ValueError(f"unknown type name {type_name!r}")  # gated upstream


def _unbool(element, _true=object(), _false=object()):
    if element is True:
        return _true
    if element is False:
        return _false
    return element


def equal(one, two) -> bool:
    """JSON Schema equality: evades bool inheriting from int, recurses."""
    if one is two:
        return True
    if isinstance(one, str) or isinstance(two, str):
        return one == two
    if isinstance(one, list) and isinstance(two, list):
        return len(one) == len(two) and all(equal(i, j) for i, j in zip(one, two))
    if isinstance(one, dict) and isinstance(two, dict):
        return len(one) == len(two) and all(
            key in two and equal(value, two[key]) for key, value in one.items()
        )
    return _unbool(one) == _unbool(two)


def uniq(container) -> bool:
    """True when all elements are unique under `equal` (sort, else pairwise)."""
    try:
        sort = sorted(_unbool(i) for i in container)
        for i, j in zip(sort, sort[1:]):
            if equal(i, j):
                return False
    except TypeError:
        seen = []
        for e in container:
            e = _unbool(e)
            for i in seen:
                if equal(i, e):
                    return False
            seen.append(e)
    return True


def iter_errors(schema, instance) -> "list[Record]":
    return list(_walk(schema, instance, (), ()))


def _walk(schema, instance, ipath: tuple, spath: tuple):
    if schema is True:
        return
    if schema is False:
        yield ("false", ipath, spath)
        return
    # Iterate in schema-dict insertion order, like the reference validator.
    for keyword, value in schema.items():
        if keyword == "type":
            types = [value] if isinstance(value, str) else value
            if not any(is_type(instance, t) for t in types):
                yield ("type", ipath, spath + ("type",))
        elif keyword == "enum":
            if all(not equal(each, instance) for each in value):
                yield ("enum", ipath, spath + ("enum",))
        elif keyword == "const":
            if not equal(instance, value):
                yield ("const", ipath, spath + ("const",))
        elif keyword == "minimum":
            if is_type(instance, "number") and instance < value:
                yield ("minimum", ipath, spath + ("minimum",))
        elif keyword == "maximum":
            if is_type(instance, "number") and instance > value:
                yield ("maximum", ipath, spath + ("maximum",))
        elif keyword == "exclusiveMinimum":
            if is_type(instance, "number") and instance <= value:
                yield ("exclusiveMinimum", ipath, spath + ("exclusiveMinimum",))
        elif keyword == "exclusiveMaximum":
            if is_type(instance, "number") and instance >= value:
                yield ("exclusiveMaximum", ipath, spath + ("exclusiveMaximum",))
        elif keyword == "multipleOf":
            if not is_type(instance, "number"):
                continue
            if isinstance(value, float):
                quotient = instance / value
                try:
                    failed = int(quotient) != quotient
                except OverflowError:
                    # Exact path for quotients too large to truncate to int.
                    failed = (Fraction(instance) / Fraction(value)).denominator != 1
            else:
                failed = bool(instance % value)
            if failed:
                yield ("multipleOf", ipath, spath + ("multipleOf",))
        elif keyword == "minLength":
            if is_type(instance, "string") and len(instance) < value:
                yield ("minLength", ipath, spath + ("minLength",))
        elif keyword == "maxLength":
            if is_type(instance, "string") and len(instance) > value:
                yield ("maxLength", ipath, spath + ("maxLength",))
        elif keyword == "pattern":
            if is_type(instance, "string") and not re.search(value, instance):
                yield ("pattern", ipath, spath + ("pattern",))
        elif keyword == "minItems":
            if is_type(instance, "array") and len(instance) < value:
                yield ("minItems", ipath, spath + ("minItems",))
        elif keyword == "maxItems":
            if is_type(instance, "array") and len(instance) > value:
                yield ("maxItems", ipath, spath + ("maxItems",))
        elif keyword == "uniqueItems":
            if value and is_type(instance, "array") and not uniq(instance):
                yield ("uniqueItems", ipath, spath + ("uniqueItems",))
        elif keyword == "minProperties":
            if is_type(instance, "object") and len(instance) < value:
                yield ("minProperties", ipath, spath + ("minProperties",))
        elif keyword == "maxProperties":
            if is_type(instance, "object") and len(instance) > value:
                yield ("maxProperties", ipath, spath + ("maxProperties",))
        elif keyword == "required":
            if not is_type(instance, "object"):
                continue
            for prop in value:
                if prop not in instance:
                    yield ("required", ipath, spath + ("required",))
        elif keyword == "properties":
            if not is_type(instance, "object"):
                continue
            for prop, subschema in value.items():
                if prop in instance:
                    yield from _walk(
                        subschema,
                        instance[prop],
                        ipath + (prop,),
                        spath + ("properties", prop),
                    )
        elif keyword == "items":
            if not is_type(instance, "array"):
                continue
            if value is False:
                if len(instance) > 0:
                    yield ("items", ipath, spath + ("items",))
            else:
                for index, element in enumerate(instance):
                    yield from _walk(
                        value,
                        element,
                        ipath + (index,),
                        spath + ("items",),
                    )
        elif keyword == "additionalProperties":
            if not is_type(instance, "object"):
                continue
            properties = schema.get("properties", {})
            extras = [k for k in instance if k not in properties]
            if value is False:
                if extras:
                    yield ("additionalProperties", ipath, spath + ("additionalProperties",))
            elif value is not True:
                for extra in extras:
                    yield from _walk(
                        value,
                        instance[extra],
                        ipath + (extra,),
                        spath + ("additionalProperties",),
                    )
        # Anything else was rejected by the subset gate (or is an inert
        # annotation the reference also ignores).
