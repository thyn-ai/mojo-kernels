"""Clean-room feature model for capa-style rules.

A feature is identified by a ``(name, value)`` pair — exactly the identity a
capa ``Feature`` gives itself (``hash((self.name, self.value))``). Values are
plain Python ``str`` / ``int`` / ``float`` / ``bytes`` objects, so callers can
build feature maps without any dependency:

    {("api", "CreateFileA"): {0x401010},
     ("number", 0x10): {0x401020, 0x401030},
     ("string", "hello"): {0x402000}}

This module also holds the value-parsing helpers used by the rule parser:
inline descriptions (``number: 0x10 = CONST``), integer literals (``0x`` hex
or decimal, negatives), byte strings (``"00 11 22"``), and the DLL-prefix
trimming applied to ``api`` values. Written fresh against the documented rule
format; no capa code is used or adapted.
"""

from __future__ import annotations

import binascii

# Feature kinds whose evaluation scans the scope's features instead of doing a
# single exact lookup: substring/regex match against `string` values, bytes
# prefix-match against `bytes` values.
SPECIAL_SUBSTRING = "substring"
SPECIAL_REGEX = "regex"
SPECIAL_BYTES = "bytes"

#: the separator between a feature value and its inline description,
#: like ``number: 42 = ENUM_FAVORITE_NUMBER``.
DESCRIPTION_SEPARATOR = " = "

#: byte sequences in rules must be no larger than this (same limit as the
#: reference rule format documents).
MAX_BYTES_FEATURE_SIZE = 0x100

# Feature keys accepted in rules, mapped to the canonical feature-map name.
# `string` is listed for completeness; it becomes StringFactory semantics
# (plain string, or regex when wrapped in /.../).
KNOWN_FEATURES = frozenset(
    {
        "api",
        "string",
        "substring",
        "bytes",
        "number",
        "offset",
        "mnemonic",
        "basic blocks",
        "characteristic",
        "export",
        "import",
        "section",
        "match",
        "function-name",
        "os",
        "format",
        "arch",
        "class",
        "namespace",
        "property",
    }
)

# Canonical feature-map name for each rule key (most are identity). These
# follow the reference engine's feature identities: `basic blocks` is the
# "basicblock" feature, while `function-name` already is its identity.
_FEATURE_NAME_MAP = {
    "basic blocks": "basicblock",
}


class InvalidRuleError(ValueError):
    """A rule (or feature value) does not conform to the supported format."""


class UnsupportedRuleError(InvalidRuleError):
    """A rule uses a construct this kernel deliberately does not support."""


def canonical_feature_name(key: str) -> str:
    return _FEATURE_NAME_MAP.get(key, key)


def parse_int(s: str) -> int:
    s = s.strip()
    if s.startswith(("0x", "0X")):
        return int(s, 0x10)
    return int(s, 10)


def parse_range(spec: str) -> tuple[int | None, int | None]:
    """Parse a ``(min, max)`` count range; either side may be empty (unbound)."""
    if not spec.startswith("(") or not spec.endswith(")"):
        raise InvalidRuleError(f"invalid range: {spec}")
    body = spec[1:-1]
    min_spec, _, max_spec = body.partition(",")
    min_spec = min_spec.strip()
    max_spec = max_spec.strip()

    min_ = parse_int(min_spec) if min_spec else None
    max_ = parse_int(max_spec) if max_spec else None
    if min_ is not None and min_ < 0:
        raise InvalidRuleError("range min less than zero")
    if max_ is not None and max_ < 0:
        raise InvalidRuleError("range max less than zero")
    if min_ is not None and max_ is not None and max_ < min_:
        raise InvalidRuleError("range max less than min")
    return min_, max_


def parse_count_spec(count) -> tuple[int | None, int | None]:
    """Parse the RHS of ``count(...)``: ``2``, ``"2 or more"``,
    ``"2 or fewer"``, or ``"(1, 5)"`` into a (min, max) pair."""
    if isinstance(count, int) and not isinstance(count, bool):
        return count, count
    if not isinstance(count, str):
        raise InvalidRuleError(f"unexpected range: {count}")
    if count.endswith(" or more"):
        return parse_int(count[: -len(" or more")]), None
    if count.endswith(" or fewer"):
        return None, parse_int(count[: -len(" or fewer")])
    if count.startswith("("):
        return parse_range(count)
    raise InvalidRuleError(f"unexpected range: {count}")


def parse_bytes(s: str) -> bytes:
    try:
        b = bytes.fromhex(s.replace(" ", ""))
    except (ValueError, binascii.Error) as exc:
        raise InvalidRuleError(
            f'unexpected bytes value: must be a valid hex sequence: "{s}"'
        ) from exc
    if len(b) > MAX_BYTES_FEATURE_SIZE:
        raise InvalidRuleError(
            f"unexpected bytes value: byte sequences must be no larger "
            f"than {MAX_BYTES_FEATURE_SIZE} bytes"
        )
    return b


def trim_dll_part(api: str) -> str:
    """``kernel32.CreateFileW`` -> ``CreateFileW``; ordinal imports (``ws2_32.#1``)
    keep the DLL part; .NET-ish names containing ``::`` are left alone."""
    if ".#" in api:
        return api
    if api.count(".") == 1 and "::" not in api:
        api = api.split(".")[1]
    return api


def parse_description(s, value_type: str, description=None):
    """Split an inline description off a non-string feature value and cast the
    value to its proper type.

    String features cannot have inline descriptions: the whole RHS is the
    string (``string: foo = bar`` is the string "foo = bar").
    """
    if value_type == "string":
        return s, description
    if isinstance(s, str):
        if DESCRIPTION_SEPARATOR in s:
            if description:
                raise InvalidRuleError(
                    f'unexpected value: "{s}", only one description allowed '
                    f"(inline description with `{DESCRIPTION_SEPARATOR}`)"
                )
            value, _, description = s.partition(DESCRIPTION_SEPARATOR)
            if description == "":
                raise InvalidRuleError(
                    f'unexpected value: "{s}", description cannot be empty'
                )
        else:
            value = s
        if value_type == "bytes":
            value = parse_bytes(value)
        elif value_type in ("number", "offset"):
            try:
                value = parse_int(value)
            except ValueError as exc:
                raise InvalidRuleError(
                    f'unexpected value: "{value}", must begin with numerical value'
                ) from exc
        return value, description
    return s, description


def string_feature(value: str) -> tuple[str, str]:
    """StringFactory semantics: ``/.../`` (optionally ``/.../i``) is a regex
    feature, anything else is a plain string."""
    if value.startswith("/") and (value.endswith("/") or value.endswith("/i")):
        return ("regex", value)
    return ("string", value)
