"""Clean-room parser for the capa rule YAML subset this kernel evaluates.

Supported rule format (a subset of the documented capa rule format):

    rule:
      meta:
        name: <rule name>
        namespace: <optional/namespace>     # used by `match:` and ordering
        scopes: {static: ..., dynamic: ...} # parsed, not interpreted
      features:
        - and|or|not|optional|<N> or more:
          - <feature>: <value>
          - count(<feature>(<value>)): <N> | "<N> or more" | "<N> or fewer" | "(<lo>, <hi>)"
        - description: <ignored>

Supported statement nodes: ``and``, ``or``, ``not``, ``optional`` (an alias
for ``0 or more``), ``<N> or more``, and ``count(...)`` with an exact count,
``N or more``, ``N or fewer``, or an inclusive ``(min, max)`` range (either
bound may be empty).

Supported feature keys: ``api``, ``string`` (plain, or ``/regex/`` with an
optional ``i`` flag), ``substring``, ``bytes``, ``number``, ``offset``,
``mnemonic``, ``characteristic``, ``section``, ``export``, ``import``,
``function-name``, ``os``, ``arch``, ``format``, ``class``, ``namespace``,
``property``, ``property/<access>``, ``match``, and ``basic blocks`` (only as
``count(basic blocks)``). Inline `` = descriptions`` and ``description:``
entries are parsed and ignored, as they do not affect matching.

Deliberately unsupported (raise :class:`UnsupportedRuleError`): subscope
statements (``instruction:``, ``basic block:``, ``function:``, ``process:``,
``thread:``, ``span of calls:``, ``call:``), ``com/...`` features,
``operand[N].number`` / ``operand[N].offset``, and ``count(...)`` over a
``match`` feature. Feature *extraction* (vivisect & friends) is entirely out
of scope: this package consumes feature maps, it does not produce them.

No capa code is used or adapted; this is a fresh implementation of the
documented rule format.
"""

from __future__ import annotations

import re

import yaml

from capa_mojo.features import (
    KNOWN_FEATURES,
    InvalidRuleError,
    UnsupportedRuleError,
    canonical_feature_name,
    parse_count_spec,
    parse_description,
    string_feature,
    trim_dll_part,
)

#: capa's unbounded-range sentinel is ``1 << 64 - 1`` which, by Python
#: operator precedence, is ``1 << 63``; mirrored exactly.
RANGE_MAX_UNBOUNDED = 1 << 63

_SUBSCOPE_KEYS = frozenset(
    {"process", "thread", "span of calls", "call", "function", "basic block", "instruction"}
)


# --- node model -------------------------------------------------------------


class Node:
    """Base class for statement-tree nodes."""

    __slots__ = ()


class Statement(Node):
    __slots__ = ()


class And(Statement):
    __slots__ = ("children",)

    def __init__(self, children):
        self.children = list(children)


class Or(Statement):
    __slots__ = ("children",)

    def __init__(self, children):
        self.children = list(children)


class Not(Statement):
    __slots__ = ("child",)

    def __init__(self, child):
        self.child = child


class Some(Statement):
    """Match if at least ``count`` children match (``optional`` is count 0)."""

    __slots__ = ("count", "children")

    def __init__(self, count, children):
        self.count = count
        self.children = list(children)


class Range(Statement):
    """Match if the child feature's location count is within [min, max]."""

    __slots__ = ("feature", "min", "max")

    def __init__(self, feature, min_, max_):
        self.feature = feature  # exact feature key (name, value)
        self.min = 0 if min_ is None else min_
        self.max = RANGE_MAX_UNBOUNDED if max_ is None else max_


class Leaf(Node):
    """An exact-match feature leaf: present iff the key is in the feature map."""

    __slots__ = ("feature",)

    def __init__(self, feature):
        self.feature = feature  # (name, value)

    @property
    def identity(self):
        return ("leaf", self.feature)


class MatchLeaf(Node):
    """A ``match:`` leaf: satisfied by an earlier rule/namespace match."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    @property
    def identity(self):
        return ("match", self.name)


class SpecialLeaf(Node):
    """A scanning leaf: ``substring``/``regex`` against string values, or a
    ``bytes`` prefix match against bytes values."""

    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind  # "substring" | "regex" | "bytes"
        self.value = value

    @property
    def identity(self):
        return (self.kind, self.value)


# --- parsed rules -----------------------------------------------------------


class ParsedRule:
    __slots__ = ("name", "namespace", "statement", "scopes", "source")

    def __init__(self, name, namespace, statement, scopes, source=None):
        self.name = name
        self.namespace = namespace
        self.statement = statement
        self.scopes = scopes or {}
        self.source = source

    def __repr__(self):
        return f"ParsedRule(name={self.name!r})"


def _pop_statement_description(children):
    """Remove a single ``{"description": ...}`` child, mirroring the reference
    rule format (statement descriptions do not affect matching)."""
    if not isinstance(children, list):
        return
    descriptions = [
        c for c in children if isinstance(c, dict) and len(c) == 1 and "description" in c
    ]
    if len(descriptions) > 1:
        raise InvalidRuleError("statements can only have one description")
    if descriptions:
        children.remove(descriptions[0])


def _unique(nodes):
    """Deduplicate feature leaves within one statement's children.

    The reference format deduplicates identical *features* (equal name and
    value) while never deduplicating statements; mirrored here: leaves compare
    by identity key, statements are always kept.
    """
    seen = set()
    out = []
    for node in nodes:
        if isinstance(node, (Leaf, MatchLeaf, SpecialLeaf)):
            ident = node.identity
            if ident in seen:
                continue
            seen.add(ident)
        out.append(node)
    return out


def _count_feature_key(term: str, arg: str | None):
    """Build the exact feature key for a ``count(...)`` child."""
    if term not in KNOWN_FEATURES or term.startswith("property/"):
        raise InvalidRuleError(f"unexpected statement: {term}")
    if term == "match":
        raise UnsupportedRuleError("count() over a match feature is not supported")
    if arg is None or arg == "":
        if term == "basic blocks":
            return ("basicblock", 0)
        raise InvalidRuleError(f"count({term}) requires a value")
    if term == "string":
        # no inline descriptions for strings; /.../ becomes a regex feature
        return string_feature(arg)
    value, _ = parse_description(arg, term)
    if term == "api":
        value = trim_dll_part(value)
    return (canonical_feature_name(term), value)


def _build_feature(key: str, raw, description=None) -> Node:
    if key not in KNOWN_FEATURES and not key.startswith("property/"):
        raise InvalidRuleError(f"unexpected statement: {key}")
    if key == "string" and not isinstance(raw, str):
        raise InvalidRuleError(
            f"ambiguous string value {raw}, must be defined as explicit string"
        )
    if key == "string":
        name, value = string_feature(raw)
        if name == "regex":
            return SpecialLeaf("regex", value)
        return Leaf(("string", value))
    if key == "substring":
        return SpecialLeaf("substring", raw)
    if key == "match":
        value, _ = parse_description(raw, key, description)
        return MatchLeaf(value)
    if key == "bytes":
        value, _ = parse_description(raw, key, description)
        return SpecialLeaf("bytes", value)
    if key.startswith("property/"):
        access = key[len("property/") :]
        value, _ = parse_description(raw, key, description)
        return Leaf((f"property/{access}", value))
    value, _ = parse_description(raw, key, description)
    if key == "api":
        value = trim_dll_part(value)
    return Leaf((canonical_feature_name(key), value))


def _build(d) -> Node:
    if not isinstance(d, dict):
        raise InvalidRuleError(f"unexpected statement: {d!r}")
    if len(d.keys()) > 2:
        raise InvalidRuleError("too many statements")
    key = list(d.keys())[0]
    value = d[key]
    description = d.get("description")
    if isinstance(value, list):
        _pop_statement_description(value)

    if key == "and":
        return And(_unique(_build(c) for c in value))
    if key == "or":
        return Or(_unique(_build(c) for c in value))
    if key == "not":
        if not isinstance(value, list) or len(value) != 1:
            raise InvalidRuleError("not statement must have exactly one child statement")
        return Not(_build(value[0]))
    if key.endswith(" or more") and key[: -len(" or more")].strip().isdigit():
        count = int(key[: -len(" or more")])
        return Some(count, _unique(_build(c) for c in value))
    if key == "optional":
        return Some(0, _unique(_build(c) for c in value))
    if key in _SUBSCOPE_KEYS:
        raise UnsupportedRuleError(
            f"subscope statement {key!r} is not supported by capa_mojo "
            "(the kernel evaluates a single scope's feature map)"
        )
    if key.startswith("count(") and key.endswith(")"):
        term = key[len("count(") : -len(")")]
        term_name, _, arg = term.partition("(")
        if arg:
            arg = arg[: -len(")")]
        feature = _count_feature_key(term_name.strip(), arg if arg else None)
        min_, max_ = parse_count_spec(value)
        return Range(feature, min_, max_)
    if key.startswith("operand["):
        raise UnsupportedRuleError(
            f"operand features ({key!r}) are not supported by capa_mojo"
        )
    if key.startswith("com/"):
        raise UnsupportedRuleError(
            f"COM features ({key!r}) are not supported by capa_mojo"
        )
    return _build_feature(key, value, description)


def parse_rule_dict(doc: dict, source=None) -> ParsedRule:
    """Parse one already-YAML-loaded rule document."""
    try:
        rule = doc["rule"]
        meta = rule["meta"]
        features = rule["features"]
    except (KeyError, TypeError) as exc:
        raise InvalidRuleError(f"rule document must have rule.meta and rule.features: {exc}")
    name = meta.get("name")
    if not name:
        raise InvalidRuleError("rule meta must have a name")
    if not isinstance(features, list) or len(features) != 1:
        raise InvalidRuleError("rule features must be a list with exactly one root statement")
    statement = _build(features[0])
    return ParsedRule(
        name=name,
        namespace=meta.get("namespace"),
        statement=statement,
        scopes=meta.get("scopes") or {},
        source=source,
    )


def parse_rule_yaml(text: str, source=None) -> ParsedRule:
    """Parse one rule from its YAML text."""
    # libyaml (CSafeLoader) when available — same documents, ~8x faster;
    # this is the dominant one-time cost when loading ~1k rules.
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        doc = yaml.load(text, Loader=loader)
    except yaml.YAMLError as exc:
        raise InvalidRuleError(f"invalid YAML: {exc}") from exc
    return parse_rule_dict(doc, source=source)


def compile_regex(value: str) -> re.Pattern:
    """Compile a ``/pattern/`` or ``/pattern/i`` regex feature value."""
    flags = re.DOTALL
    if value.endswith("/i"):
        pat = value[len("/") : -len("/i")]
        flags |= re.IGNORECASE
    else:
        pat = value[len("/") : -len("/")]
    try:
        return re.compile(pat, flags)
    except re.error as exc:
        raise InvalidRuleError(
            f"invalid regular expression: {value} it should use Python syntax"
        ) from exc
