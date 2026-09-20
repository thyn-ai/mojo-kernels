"""Pure-Python evaluation of a compiled rule set against one scope.

This module is the semantic heart of the package: it mirrors, in plain
Python, the documented reference behaviour that the native kernel also
implements — boolean decision (used directly as the fallback backend) and
full detail-tree reconstruction (used by BOTH backends to describe matched
rules, since the native kernel reports only the matched/not-matched bit per
rule).

Semantics implemented here (documented reference behaviour):

  * exact feature leaf: present iff the feature *key* is in the scope map —
    even with an empty location set; its locations are the stored set;
  * ``count(...)`` (Range): the number of *locations* of the child feature
    within [min, max]; the child is looked up exactly (a substring/regex
    child therefore always counts 0 — the reference behaves this way because
    extracted feature maps never contain substring/regex keys);
  * ``and`` / ``or`` / ``not`` / ``N or more`` / ``optional``: plain boolean
    combination over child results;
  * ``substring(x)``: true iff any ``string`` value in the scope contains
    ``x``; locations are the union of the matching strings' locations;
  * ``string: /re/`` (regex): true iff the pattern searches successfully
    against any ``string`` value; locations as for substring;
  * ``bytes: B``: true iff any ``bytes`` value in the scope *starts with*
    ``B``; locations of the first such value in feature-map order;
  * ``match: name``: true iff a rule publishing ``name`` (rule name or
    namespace prefix) already matched in this scope, or a ``("match", name)``
    key is present in the input map; locations are the input locations plus
    the scope's own address when a published match exists.

Detail trees are emitted as plain nested tuples so callers can compare or
render them without depending on this package's classes:

    ("and"|"or", success, (child, ...))
    ("not", success, (child,))
    ("some", success, count, (child, ...))
    ("range", success, (name, value), min, max, locations)
    ("feature", success, name, value, locations)
    ("match", success, name, locations)
    ("substring"|"regex", success, value, locations, matches)
    ("bytes", success, value, locations)

Every tuple has the boolean success flag at index 1. ``locations`` is a
sorted tuple of ints; ``matches`` for substring/regex is a tuple of
``(string_value, locations)`` sorted by value.

No capa code is used or adapted.
"""

from __future__ import annotations

from capa_mojo.rules import (
    And,
    Leaf,
    MatchLeaf,
    Not,
    Or,
    Range,
    Some,
    SpecialLeaf,
)

INT32_MAX = (1 << 31) - 1


class ScopeView:
    """Normalized, index-ready view of one scope's feature map.

    `features` maps ``(name, value)`` feature keys to iterables of integer
    locations. Keys are validated once here; everything downstream relies on
    this normalization.
    """

    __slots__ = ("addr", "features", "string_items", "bytes_items", "_special_cache")

    def __init__(self, addr, features):
        self.addr = addr
        normalized = {}
        strings = []
        byte_items = []
        for key, locations in features.items():
            if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str)):
                raise ValueError(
                    f"feature keys must be (name, value) tuples, got {key!r}"
                )
            locs = frozenset(_as_location(loc) for loc in locations)
            normalized[key] = locs
            # the reference's substring/regex scan covers every String-family
            # feature key (string, but also pathological substring/regex keys)
            if key[0] in ("string", "substring", "regex") and isinstance(key[1], str):
                strings.append((key[1], locs))
            elif key[0] == "bytes" and isinstance(key[1], (bytes, bytearray)):
                byte_items.append((bytes(key[1]), locs))
        self.features = normalized
        self.string_items = strings
        self.bytes_items = byte_items
        self._special_cache: dict[tuple, tuple] = {}

    def count_of(self, key) -> int:
        locs = self.features.get(key)
        return 0 if locs is None else min(len(locs), INT32_MAX)

    def locations_of(self, key) -> tuple:
        locs = self.features.get(key)
        return () if locs is None else tuple(sorted(locs))

    def has(self, key) -> bool:
        return key in self.features

    def special(self, leaf: SpecialLeaf, regexes) -> tuple:
        """Evaluate a scanning leaf once per scope; returns the detail tuple
        minus the kind prefix."""
        ident = leaf.identity
        cached = self._special_cache.get(ident)
        if cached is not None:
            return cached
        if leaf.kind == "substring":
            needle = leaf.value
            matches = tuple(
                sorted(
                    ((s, tuple(sorted(locs))) for s, locs in self.string_items if needle in s),
                    key=lambda item: item[0],
                )
            )
        elif leaf.kind == "regex":
            pattern = regexes[id(leaf)] if isinstance(regexes, dict) else regexes
            matches = tuple(
                sorted(
                    (
                        (s, tuple(sorted(locs)))
                        for s, locs in self.string_items
                        if pattern.search(s)
                    ),
                    key=lambda item: item[0],
                )
            )
        elif leaf.kind == "bytes":
            prefix = leaf.value
            hits = tuple(
                (tuple(sorted(locs)))
                for value, locs in self.bytes_items
                if value.startswith(prefix)
            )[:1]  # the reference collects only the first prefix match
            result = (bool(hits), hits[0] if hits else (), None)
            self._special_cache[ident] = result
            return result
        else:  # pragma: no cover - defensive
            raise ValueError(f"unexpected special leaf kind {leaf.kind!r}")
        locations = tuple(sorted({loc for _, locs in matches for loc in locs}))
        result = (bool(matches), locations, matches)
        self._special_cache[ident] = result
        return result


def _as_location(loc) -> int:
    if isinstance(loc, bool) or not isinstance(loc, int):
        raise ValueError(f"locations must be integers, got {loc!r}")
    return loc


# --- boolean decision (fallback backend) ------------------------------------


def eval_bool(node, view: ScopeView, matched: set[str], regex_of) -> bool:
    """Short-circuit boolean evaluation of a rule subtree against one scope.

    `matched` holds every name published by rules that already matched in
    this scope (rule names and namespace prefixes), plus input match keys.
    """
    if isinstance(node, Leaf):
        return view.has(node.feature)
    if isinstance(node, MatchLeaf):
        return node.name in matched or view.has(("match", node.name))
    if isinstance(node, SpecialLeaf):
        regexes = regex_of(node) if node.kind == "regex" else None
        return view.special(node, regexes)[0]
    if isinstance(node, Range):
        count = len(view.features.get(node.feature, ()))
        return node.min <= count <= node.max
    if isinstance(node, Not):
        return not eval_bool(node.child, view, matched, regex_of)
    if isinstance(node, And):
        return all(eval_bool(c, view, matched, regex_of) for c in node.children)
    if isinstance(node, Or):
        return any(eval_bool(c, view, matched, regex_of) for c in node.children)
    if isinstance(node, Some):
        satisfied = 0
        for c in node.children:
            if eval_bool(c, view, matched, regex_of):
                satisfied += 1
            # checked after every child, true or not: this mirrors the
            # reference short-circuit exactly (Some(0, [x]) is always True,
            # Some(0, []) is always False).
            if satisfied >= node.count:
                return True
        return False
    raise ValueError(f"unexpected node {node!r}")  # pragma: no cover


# --- detail trees (both backends, matched rules only) -----------------------


def eval_detail(node, view: ScopeView, matched: set[str], regex_of) -> tuple:
    """Full (non-short-circuit) evaluation producing the canonical tree."""
    if isinstance(node, Leaf):
        name, value = node.feature
        locs = view.features.get(node.feature)
        locations = () if locs is None else tuple(sorted(locs))
        return ("feature", locs is not None, name, value, locations)
    if isinstance(node, MatchLeaf):
        input_locs = view.features.get(("match", node.name), frozenset())
        published = node.name in matched
        locations = set(input_locs)
        if published and view.addr is not None:
            locations.add(view.addr)
        return ("match", published or ("match", node.name) in view.features, node.name, tuple(sorted(locations)))
    if isinstance(node, SpecialLeaf):
        regexes = regex_of(node) if node.kind == "regex" else None
        success, locations, matches = view.special(node, regexes)
        if node.kind == "bytes":
            return ("bytes", success, node.value, locations)
        return (node.kind, success, node.value, locations, matches)
    if isinstance(node, Range):
        locs = view.features.get(node.feature, frozenset())
        count = len(locs)
        success = node.min <= count <= node.max
        locations = () if (node.min == 0 and count == 0) else tuple(sorted(locs))
        return ("range", success, node.feature, node.min, node.max, locations)
    if isinstance(node, Not):
        child = eval_detail(node.child, view, matched, regex_of)
        return ("not", not child[1], (child,))
    if isinstance(node, And):
        children = tuple(eval_detail(c, view, matched, regex_of) for c in node.children)
        return ("and", all(c[1] for c in children), children)
    if isinstance(node, Or):
        children = tuple(eval_detail(c, view, matched, regex_of) for c in node.children)
        return ("or", any(c[1] for c in children), children)
    if isinstance(node, Some):
        children = tuple(eval_detail(c, view, matched, regex_of) for c in node.children)
        return ("some", sum(1 for c in children if c[1]) >= node.count, node.count, children)
    raise ValueError(f"unexpected node {node!r}")  # pragma: no cover
