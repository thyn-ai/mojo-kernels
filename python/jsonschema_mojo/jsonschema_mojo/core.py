"""Drop-in JSON Schema validation (draft 2020-12 subset), Mojo-accelerated.

Public surface mirrors the `jsonschema` package: ``validate(instance,
schema)``, ``Validator(schema).iter_errors(instance)`` /
``.validate(instance)`` / ``.is_valid(instance)``, and ``ValidationError``
with the same attributes (``message``, ``absolute_path``, ``json_path``,
``validator``, ``validator_value``, ``instance``, ``schema``,
``absolute_schema_path``, ``context``).

Validation runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and the schema/instance pair
is exactly representable, and transparently falls back to the vendored
pure-Python reference implementation otherwise. Both backends must agree
with the `jsonschema` package; the differential suite asserts pass/fail and
failing-path-set parity on both paths.

Documented divergences from the `jsonschema` package (parity contract is
pass/fail outcome + the set of failing instance JSON paths):

* Error *ordering* is not part of the contract. The native backend emits
  errors in a fixed canonical keyword order; the reference orders errors by
  schema-dict insertion order (and set-iteration order for
  additionalProperties). ``validate()`` raises the best match by the same
  relevance heuristic (deepest path wins).
* Error message *text* parity is best-effort: messages for every supported
  keyword use the reference's templates and are asserted exact in the test
  suite; ``str(error)`` is the message only (the reference appends
  schema/instance excerpts).
* Schemas using keywords outside the supported subset raise
  ``UnsupportedSchemaError`` instead of being applied ($ref, allOf/anyOf/
  oneOf/not, if/then/else, patternProperties, prefixItems, contains,
  propertyNames, dependent*, unevaluated*, format assertion, ...).
  Annotation keywords ($schema, $id, $defs/definitions, title, description,
  default, examples, deprecated, readOnly/writeOnly, $comment, $anchor,
  $dynamicAnchor, $vocabulary, format, content*) are ignored, matching the
  reference's default (no format checker) behaviour.
* The native path validates the JSON-document projection of the instance
  (what ``json.dumps`` produces). Instances containing tuples, non-string
  dict keys, Decimals, non-finite floats, or integers beyond ±2**53 are not
  JSON documents; they are detected (serialization failure or a kernel
  big-number flag) and validated by the exact pure-Python fallback, which
  mirrors the reference on arbitrary Python objects.
"""

from __future__ import annotations

import json
import re
import struct
from collections import deque
from fractions import Fraction

from jsonschema_mojo import _reference
from jsonschema_mojo._native import (
    FLAG_BIG_NUMBER,
    NativeSchema,
    NativeUnavailable,
)

__all__ = [
    "Draft202012Validator",
    "UnsupportedSchemaError",
    "ValidationError",
    "Validator",
    "iter_errors",
    "validate",
]

_TYPE_NAMES = frozenset(
    {"null", "boolean", "object", "array", "number", "integer", "string"}
)

_SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "items",
        "additionalProperties",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "multipleOf",
    }
)

# Keywords with no validation effect under the reference's default
# configuration (annotations and inert structure): ignored, as the
# reference ignores them.
_INERT_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$vocabulary",
        "definitions",
        "title",
        "description",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        "format",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
    }
)

# Kernel error-record keyword codes (bit positions), mirrored from
# kernels/jsonschema/src/jsonschemamojo.mojo.
_KW_BY_CODE = (
    "type",  # 0
    "properties",  # 1
    "required",  # 2
    "items",  # 3
    "additionalProperties",  # 4
    "enum",  # 5
    "const",  # 6
    "minimum",  # 7
    "maximum",  # 8
    "exclusiveMinimum",  # 9
    "exclusiveMaximum",  # 10
    "minLength",  # 11
    "maxLength",  # 12
    "pattern",  # 13
    "minItems",  # 14
    "maxItems",  # 15
    "uniqueItems",  # 16
    "minProperties",  # 17
    "maxProperties",  # 18
    "multipleOf",  # 19
    "false",  # 20
)

_REC_ERROR = 0
_REC_DEFER = 1

_JSON_PATH_COMPATIBLE_PROPERTY = re.compile("^[a-zA-Z][a-zA-Z0-9_]*$")


class UnsupportedSchemaError(ValueError):  # noqa: N818
    """The schema uses a keyword or value shape outside the supported subset."""


class ValidationError(Exception):
    """An instance was invalid under a provided schema.

    Attribute-compatible with ``jsonschema.exceptions.ValidationError`` for
    the supported subset. ``__str__`` is the message (the reference class
    appends schema/instance excerpts; message text itself is identical).
    """

    def __init__(
        self,
        message: str,
        *,
        path,
        schema_path,
        validator,
        validator_value,
        instance,
        schema,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = deque(path)
        self.absolute_path = deque(path)
        self.schema_path = deque(schema_path)
        self.absolute_schema_path = deque(schema_path)
        self.validator = validator
        self.validator_value = validator_value
        self.instance = instance
        self.schema = schema
        self.context: list = []

    @property
    def json_path(self) -> str:
        path = "$"
        for elem in self.absolute_path:
            if isinstance(elem, int):
                path += "[" + str(elem) + "]"
            elif _JSON_PATH_COMPATIBLE_PROPERTY.match(elem):
                path += "." + elem
            else:
                escaped = elem.replace("\\", "\\\\").replace("'", "\\'")
                path += "['" + escaped + "']"
        return path

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"<ValidationError: {self.message!r}>"


# --------------------------------------------------------------------------
# Subset gate
# --------------------------------------------------------------------------


def _check_schema(schema, spath: str = "$") -> None:
    """Reject schemas outside the supported subset. Never silently ignores a
    validation-affecting keyword (the reference would apply it)."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise UnsupportedSchemaError(
            f"schema at {spath} must be an object or boolean, "
            f"got {type(schema).__name__}"
        )
    for key, value in schema.items():
        if not isinstance(key, str):
            raise UnsupportedSchemaError(
                f"schema at {spath} has a non-string key {key!r}"
            )
        loc = f"{spath}.{key}"
        if key in _INERT_KEYWORDS:
            continue
        if key not in _SUPPORTED_KEYWORDS:
            raise UnsupportedSchemaError(
                f"keyword {key!r} at {spath} is outside the supported "
                "draft 2020-12 subset (see jsonschema_mojo's README)"
            )
        if key == "type":
            names = [value] if isinstance(value, str) else value
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(t, str) or t not in _TYPE_NAMES for t in names)
            ):
                raise UnsupportedSchemaError(
                    f"'type' at {loc} must be one of {sorted(_TYPE_NAMES)} "
                    "or a non-empty list of those"
                )
        elif key == "properties":
            if not isinstance(value, dict):
                raise UnsupportedSchemaError(f"'properties' at {loc} must be an object")
            for prop_name, subschema in value.items():
                if not isinstance(prop_name, str):
                    raise UnsupportedSchemaError(
                        f"'properties' at {loc} has a non-string key {prop_name!r}"
                    )
                _check_schema(subschema, f"{loc}.{prop_name}")
        elif key == "required":
            if not isinstance(value, list) or any(
                not isinstance(p, str) for p in value
            ):
                raise UnsupportedSchemaError(
                    f"'required' at {loc} must be a list of strings"
                )
        elif key in ("items", "additionalProperties"):
            if not isinstance(value, (dict, bool)):
                raise UnsupportedSchemaError(
                    f"{key!r} at {loc} must be a schema (object or boolean)"
                )
            _check_schema(value, loc)
        elif key == "enum":
            if not isinstance(value, list):
                raise UnsupportedSchemaError(f"'enum' at {loc} must be a list")
        elif key == "const":
            pass  # any JSON value
        elif key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsupportedSchemaError(f"{key!r} at {loc} must be a number")
        elif key in (
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "minProperties",
            "maxProperties",
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UnsupportedSchemaError(
                    f"{key!r} at {loc} must be a non-negative integer"
                )
        elif key == "pattern":
            if not isinstance(value, str):
                raise UnsupportedSchemaError(f"'pattern' at {loc} must be a string")
        elif key == "multipleOf":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsupportedSchemaError(f"'multipleOf' at {loc} must be a number")
            if value == 0:
                raise UnsupportedSchemaError(
                    f"'multipleOf' at {loc} must be non-zero"
                )
        elif key == "uniqueItems":
            if not isinstance(value, bool):
                raise UnsupportedSchemaError(f"'uniqueItems' at {loc} must be a boolean")


# --------------------------------------------------------------------------
# Native record decoding
# --------------------------------------------------------------------------


def _dumps(obj) -> bytes:
    """Canonical JSON text for the kernel: ASCII-only, no NaN/Infinity."""
    return json.dumps(
        obj, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")


def _decode_segment(mv: memoryview, pos: int):
    tag = mv[pos]
    pos += 1
    if tag == 1:
        (index,) = struct.unpack_from("<q", mv, pos)
        return index, pos + 8
    (length,) = struct.unpack_from("<I", mv, pos)
    pos += 4
    key = bytes(mv[pos : pos + length]).decode("utf-8", errors="surrogatepass")
    return key, pos + length


def _decode_records(payload: bytes) -> list[tuple[int, str, tuple, tuple]]:
    """Decode the kernel's record stream: (kind, keyword, ipath, spath)."""
    mv = memoryview(payload)
    pos = 0
    out = []
    while pos < len(mv):
        kind = mv[pos]
        kw = _KW_BY_CODE[mv[pos + 1]]
        n_inst, n_schema = struct.unpack_from("<HH", mv, pos + 2)
        pos += 6
        ipath = []
        for _ in range(n_inst):
            seg, pos = _decode_segment(mv, pos)
            ipath.append(seg)
        spath = []
        for _ in range(n_schema):
            seg, pos = _decode_segment(mv, pos)
            spath.append(seg)
        out.append((kind, kw, tuple(ipath), tuple(spath)))
    return out


def _navigate(obj, path: tuple):
    for seg in path:
        obj = obj[seg]
    return obj


# --------------------------------------------------------------------------
# Error message rendering (reference templates)
# --------------------------------------------------------------------------


def _message(kw: str, inst_sub, subschema, missing_prop) -> str:
    if kw == "false":
        return f"False schema does not allow {inst_sub!r}"
    value = subschema[kw]
    if kw == "type":
        types = [value] if isinstance(value, str) else value
        reprs = ", ".join(repr(t) for t in types)
        return f"{inst_sub!r} is not of type {reprs}"
    if kw == "required":
        return f"{missing_prop!r} is a required property"
    if kw == "minimum":
        return f"{inst_sub!r} is less than the minimum of {value!r}"
    if kw == "maximum":
        return f"{inst_sub!r} is greater than the maximum of {value!r}"
    if kw == "exclusiveMinimum":
        return f"{inst_sub!r} is less than or equal to the minimum of {value!r}"
    if kw == "exclusiveMaximum":
        return f"{inst_sub!r} is greater than or equal to the maximum of {value!r}"
    if kw == "multipleOf":
        return f"{inst_sub!r} is not a multiple of {value}"
    if kw in ("minLength", "minItems"):
        word = "should be non-empty" if value == 1 else "is too short"
        return f"{inst_sub!r} {word}"
    if kw in ("maxLength", "maxItems"):
        word = "is expected to be empty" if value == 0 else "is too long"
        return f"{inst_sub!r} {word}"
    if kw == "pattern":
        return f"{inst_sub!r} does not match {value!r}"
    if kw == "uniqueItems":
        return f"{inst_sub!r} has non-unique elements"
    if kw == "minProperties":
        word = "should be non-empty" if value == 1 else "does not have enough properties"
        return f"{inst_sub!r} {word}"
    if kw == "maxProperties":
        word = "is expected to be empty" if value == 0 else "has too many properties"
        return f"{inst_sub!r} {word}"
    if kw == "enum":
        return f"{inst_sub!r} is not one of {value!r}"
    if kw == "const":
        return f"{value!r} was expected"
    if kw == "additionalProperties":
        properties = subschema.get("properties", {})
        extras = sorted((k for k in inst_sub if k not in properties), key=str)
        verb = "was" if len(extras) == 1 else "were"
        joined = ", ".join(repr(each) for each in extras)
        return f"Additional properties are not allowed ({joined} {verb} unexpected)"
    if kw == "items":
        # Only reachable when `items` is false: one aggregate error.
        count = len(inst_sub)
        rest = inst_sub if count != 1 else inst_sub[0]
        return f"Expected at most 0 items but found {count} extra: {rest!r}"
    raise ValueError(f"no message template for keyword {kw!r}")  # unreachable


def _best_match(errors: list[ValidationError]) -> ValidationError:
    """Mirror the reference's relevance heuristic for the supported subset:
    the deepest error wins (the weak/strong keyword classes and the
    type-match tiebreak are all constant within the subset); ties break by
    path (numeric segments numerically, like the reference)."""

    def key(error: ValidationError):
        return (
            -len(error.absolute_path),
            tuple(
                (0, part) if isinstance(part, int) else (1, part)
                for part in error.absolute_path
            ),
        )

    return max(errors, key=key)


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------


class Validator:
    """Draft 2020-12 (subset) validator. API-compatible with the reference's
    ``Draft202012Validator`` for the supported keywords."""

    def __init__(self, schema) -> None:
        _check_schema(schema)
        self.schema = schema
        self._native: NativeSchema | None = None
        self._backend = "fallback"
        try:
            schema_text = _dumps(schema)
        except (TypeError, ValueError, OverflowError):
            schema_text = None  # not JSON-representable: always fall back
        if schema_text is not None:
            native = None
            try:
                native = NativeSchema(schema_text)
            except NativeUnavailable:
                native = None
            if native is not None:
                if native.flags & FLAG_BIG_NUMBER:
                    # Schema numbers must be exact; the kernel flags >15-digit
                    # integer literals and we validate with the fallback.
                    native.close()
                else:
                    self._native = native
                    self._backend = "native"

    @property
    def backend(self) -> str:
        """Which engine serves this instance: "native" or "fallback"."""
        return self._backend

    # -- validation -------------------------------------------------------

    def _records(self, instance) -> list[tuple[str, tuple, tuple]]:
        """(keyword, instance_path, schema_path) records from the active backend."""
        if self._native is not None:
            try:
                instance_text = _dumps(instance)
            except (TypeError, ValueError, OverflowError):
                instance_text = None  # not a JSON document: exact fallback
            if instance_text is not None:
                try:
                    payload, _n_err, _n_defer, big = self._native.validate_records(
                        instance_text
                    )
                except NativeUnavailable:
                    payload = None
                if payload is not None and not big:
                    return self._resolve_defers(_decode_records(payload), instance)
        return _reference.iter_errors(self.schema, instance)

    def _resolve_defers(self, records, instance) -> list[tuple[str, tuple, tuple]]:
        """Evaluate kernel-deferred checks (regex patterns, exact multipleOf
        edge cases) with the original Python objects, in record order."""
        out = []
        for kind, kw, ipath, spath in records:
            if kind == _REC_ERROR:
                out.append((kw, ipath, spath))
                continue
            inst_sub = _navigate(instance, ipath)
            subschema = _navigate(self.schema, spath[:-1])
            if kw == "pattern":
                if not re.search(subschema["pattern"], inst_sub):
                    out.append((kw, ipath, spath))
            elif kw == "multipleOf":
                divisor = subschema["multipleOf"]
                if isinstance(divisor, float):
                    quotient = inst_sub / divisor
                    try:
                        failed = int(quotient) != quotient
                    except OverflowError:
                        failed = (
                            Fraction(inst_sub) / Fraction(divisor)
                        ).denominator != 1
                else:
                    failed = bool(inst_sub % divisor)
                if failed:
                    out.append((kw, ipath, spath))
        return out

    # -- public API (mirrors the reference Validator) ---------------------

    def iter_errors(self, instance):
        records = self._records(instance)
        # required errors: the record does not name the missing property;
        # assign missing properties (in `required` order) per occurrence.
        occurrence: dict[tuple, int] = {}
        errors = []
        for kw, ipath, spath in records:
            key = (kw, ipath, spath)
            n = occurrence.get(key, 0)
            occurrence[key] = n + 1
            missing_prop = None
            if kw == "required":
                inst_sub = _navigate(instance, ipath)
                subschema = _navigate(self.schema, spath[:-1])
                missing = [p for p in subschema["required"] if p not in inst_sub]
                missing_prop = missing[min(n, len(missing) - 1)]
            errors.append(
                self._render(kw, ipath, spath, instance, missing_prop)
            )
        return iter(errors)

    def _render(self, kw, ipath, spath, instance, missing_prop) -> ValidationError:
        inst_sub = _navigate(instance, ipath)
        if kw == "false":
            # Boolean-false subschema: the reference yields the error at the
            # *containing* keyword's level (its descend does not append the
            # final path segments), but the message/instance use the rejected
            # child value. Records carry full paths for navigation; the
            # displayed paths drop the last segment.
            subschema = _navigate(self.schema, spath)
            validator_value = None
            ipath = ipath[:-1]
            spath = spath[:-1]
        else:
            subschema = _navigate(self.schema, spath[:-1])
            validator_value = subschema[kw]
        return ValidationError(
            _message(kw, inst_sub, subschema, missing_prop),
            path=ipath,
            schema_path=spath,
            validator=None if kw == "false" else kw,
            validator_value=validator_value,
            instance=inst_sub,
            schema=subschema,
        )

    def validate(self, instance) -> None:
        # Mirrors the reference's Validator.validate: raise the first error
        # (in this backend's documented canonical order — see module docs).
        for error in self.iter_errors(instance):
            raise error

    def is_valid(self, instance) -> bool:
        return not self._records(instance)


def validate(instance, schema) -> None:
    """Validate ``instance`` under ``schema``, raising the best-matching
    :class:`ValidationError` if invalid. Mirrors ``jsonschema.validate``
    (which additionally metaschema-checks the schema; here the subset gate
    plays that role and raises :class:`UnsupportedSchemaError`)."""
    validator = Validator(schema)
    errors = list(validator.iter_errors(instance))
    if errors:
        raise _best_match(errors)


def iter_errors(instance, schema):
    """Lazily iterate all validation errors. Mirrors
    ``Draft202012Validator(schema).iter_errors(instance)``."""
    return Validator(schema).iter_errors(instance)


# Drop-in convenience alias: same subset, same call shape as the reference's
# draft 2020-12 validator class.
Draft202012Validator = Validator
