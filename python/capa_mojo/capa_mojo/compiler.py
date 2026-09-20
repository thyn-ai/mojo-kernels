"""Compile parsed rules into the flat arrays the native kernel evaluates.

The compiler

  * orders rules topologically, so that a rule whose tree contains a
    ``match:`` leaf is evaluated after every rule it can reference
    (namespace references expand to all rules under that namespace,
    prefixes included) — the same ordering the reference engine requires;
  * interns every exact feature key (leaf and ``count(...)`` child) into a
    vocabulary, and every scanning leaf (substring/regex/bytes) into a
    special-leaf list;
  * emits each rule's statement tree in postfix order (children before
    parents) into one concatenated node array.

No capa code is used or adapted.
"""

from __future__ import annotations

import numpy as np

from capa_mojo.features import InvalidRuleError
from capa_mojo.rules import (
    And,
    Leaf,
    MatchLeaf,
    Not,
    Or,
    ParsedRule,
    Range,
    Some,
    SpecialLeaf,
    compile_regex,
)

OP_AND = 1
OP_OR = 2
OP_NOT = 3
OP_SOME = 4
OP_RANGE = 5
OP_LEAF = 6


def _namespace_prefixes(namespace):
    """"c2/shell" -> ["c2/shell", "c2"] (the namespaces a match publishes)."""
    out = []
    while namespace:
        out.append(namespace)
        namespace, _, _ = namespace.rpartition("/")
    return out


def _iter_match_leaves(node):
    if isinstance(node, MatchLeaf):
        yield node.name
    elif isinstance(node, Not):
        yield from _iter_match_leaves(node.child)
    elif isinstance(node, (And, Or, Some)):
        for child in node.children:
            yield from _iter_match_leaves(child)


class CompiledRules:
    """A compiled, topologically ordered rule set ready for evaluation."""

    def __init__(self, rules: list[ParsedRule]):
        if not rules:
            raise InvalidRuleError("a rule set needs at least one rule")
        names = [r.name for r in rules]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise InvalidRuleError(f"duplicate rule names: {dupes}")

        # namespace -> member rule names, prefix-expanded (input order).
        namespaces: dict[str, list[str]] = {}
        for rule in rules:
            for prefix in _namespace_prefixes(rule.namespace):
                namespaces.setdefault(prefix, []).append(rule.name)

        rules_by_name = {r.name: r for r in rules}

        def dependencies(rule: ParsedRule) -> list[str]:
            deps = []
            for name in _iter_match_leaves(rule.statement):
                if name in namespaces:
                    # a namespace reference takes precedence over a rule name
                    deps.extend(namespaces[name])
                elif name in rules_by_name:
                    deps.append(name)
                else:
                    raise InvalidRuleError(
                        f"rule {rule.name!r} depends on unknown rule or namespace {name!r}"
                    )
            return deps

        # Topological order: depth-first over dependencies, rules in input
        # order — the same shape the reference ordering produces.
        ordered: list[ParsedRule] = []
        seen: set[str] = set()
        visiting: list[str] = []

        def visit(rule: ParsedRule) -> None:
            if rule.name in seen:
                return
            if rule.name in visiting:
                cycle = " -> ".join([*visiting[visiting.index(rule.name) :], rule.name])
                raise InvalidRuleError(f"rule dependencies form a cycle: {cycle}")
            visiting.append(rule.name)
            for dep in dependencies(rule):
                visit(rules_by_name[dep])
            visiting.pop()
            seen.add(rule.name)
            ordered.append(rule)

        for rule in rules:
            visit(rule)

        self.rules: list[ParsedRule] = ordered
        self.name_to_idx = {r.name: i for i, r in enumerate(ordered)}
        #: per rule, the match-feature names a match publishes (rule name +
        #: namespace prefixes) — used to reconstruct match-leaf detail.
        self.rule_publishes: list[tuple[str, ...]] = [
            (rule.name, *_namespace_prefixes(rule.namespace)) for rule in ordered
        ]

        self._compile()

    # -- flattening ---------------------------------------------------------

    def _compile(self) -> None:
        exact_index: dict[tuple, int] = {}
        special_index: dict[tuple, int] = {}
        self.special_leaves: list[SpecialLeaf] = []
        group_index: dict[str, int] = {}
        self.match_groups: list[list[int]] = []
        self.match_group_names: list[str] = []

        # namespace -> member rule indices, prefix-expanded, over the final
        # (topological) ordering — resolved once up front.
        ordered_namespaces: dict[str, list[int]] = {}
        for i, rule in enumerate(self.rules):
            for prefix in _namespace_prefixes(rule.namespace):
                ordered_namespaces.setdefault(prefix, []).append(i)

        def exact_slot(key) -> int:
            if key not in exact_index:
                exact_index[key] = len(exact_index)
            return exact_index[key]

        def special_slot(leaf: SpecialLeaf) -> int:
            ident = leaf.identity
            if ident not in special_index:
                special_index[ident] = len(self.special_leaves)
                self.special_leaves.append(leaf)
            return special_index[ident]

        def group_slot(name: str) -> int:
            if name not in group_index:
                # every MatchLeaf name was validated during ordering: it is
                # either a namespace (which takes precedence) or a rule name.
                if name in ordered_namespaces:
                    members = ordered_namespaces[name]
                else:
                    members = [self.name_to_idx[name]]
                group_index[name] = len(self.match_groups)
                self.match_groups.append(sorted(members))
                self.match_group_names.append(name)
            return group_index[name]

        node_op: list[int] = []
        node_arg: list[int] = []
        node_min: list[int] = []
        node_max: list[int] = []
        node_children: list[int] = []
        child_offsets: list[int] = [0]
        rule_offsets = [0]

        # n_count_leaves is fixed only after every rule is walked, but match
        # group slots are addressed relative to it. Emit a placeholder base of
        # 0 first, then shift match-leaf args in a final pass.
        match_arg_nodes: list[int] = []

        def emit(node) -> int:
            """Append `node` and its subtree (postfix); return its global index."""
            child_ids: list[int] = []
            if isinstance(node, Not):
                child_ids = [emit(node.child)]
            elif isinstance(node, (And, Or, Some)):
                child_ids = [emit(c) for c in node.children]

            idx = len(node_op)
            if isinstance(node, And):
                node_op.append(OP_AND)
                node_arg.append(0)
            elif isinstance(node, Or):
                node_op.append(OP_OR)
                node_arg.append(0)
            elif isinstance(node, Not):
                node_op.append(OP_NOT)
                node_arg.append(0)
            elif isinstance(node, Some):
                node_op.append(OP_SOME)
                node_arg.append(node.count)
            elif isinstance(node, Range):
                node_op.append(OP_RANGE)
                node_arg.append(exact_slot(node.feature))
            elif isinstance(node, Leaf):
                node_op.append(OP_LEAF)
                node_arg.append(exact_slot(node.feature))
            elif isinstance(node, SpecialLeaf):
                node_op.append(OP_LEAF)
                # special slots live above the exact vocabulary; shifted below
                node_arg.append(-(special_slot(node) + 1))
                match_arg_nodes.append(idx)
            elif isinstance(node, MatchLeaf):
                node_op.append(OP_LEAF)
                node_arg.append(group_slot(node.name))
                match_arg_nodes.append(idx)
            else:  # pragma: no cover - defensive
                raise InvalidRuleError(f"unexpected node {node!r}")
            node_min.append(node.min if isinstance(node, Range) else 0)
            node_max.append(node.max if isinstance(node, Range) else 0)
            node_children.extend(child_ids)
            child_offsets.append(len(node_children))
            return idx

        for rule in self.rules:
            emit(rule.statement)
            rule_offsets.append(len(node_op))

        n_exact = len(exact_index)
        n_special = len(self.special_leaves)
        n_groups = len(self.match_groups)
        self.n_exact_leaves = n_exact
        self.n_count_leaves = n_exact + n_special
        self.n_match_groups = n_groups
        #: counts slots = exact leaves + special leaves + one slot per match
        #: group (carries the presence of a ("match", name) input feature).
        self.n_count_slots = self.n_count_leaves + n_groups

        # Fix up LEAF args: special leaves sit at [n_exact, n_count_leaves),
        # match groups at [n_count_leaves, n_count_leaves + n_groups).
        for idx in match_arg_nodes:
            arg = node_arg[idx]
            if arg < 0:
                node_arg[idx] = n_exact + (-arg - 1)
            else:
                node_arg[idx] = self.n_count_leaves + arg

        self.exact_index = exact_index
        self.exact_keys = [None] * n_exact
        for key, slot in exact_index.items():
            self.exact_keys[slot] = key

        self.rule_node_offsets = np.array(rule_offsets, dtype=np.int64)
        self.node_op = np.array(node_op, dtype=np.int32)
        self.node_child_offsets = np.array(child_offsets, dtype=np.int64)
        self.node_children = np.array(node_children, dtype=np.int32)
        self.node_arg = np.array(node_arg, dtype=np.int32)
        self.node_range_min = np.array(node_min, dtype=np.uint64)
        self.node_range_max = np.array(node_max, dtype=np.uint64)

        group_offsets = [0]
        group_rules: list[int] = []
        for members in self.match_groups:
            group_rules.extend(members)
            group_offsets.append(len(group_rules))
        self.match_group_offsets = np.array(group_offsets, dtype=np.int64)
        self.match_group_rules = np.array(group_rules, dtype=np.int32)

        # Precompiled regex patterns for the special leaves (substring/bytes
        # need no compilation). Plain values are kept for dict-key identity.
        self._special_regexes = [
            compile_regex(leaf.value) if leaf.kind == "regex" else None
            for leaf in self.special_leaves
        ]

    def __len__(self) -> int:
        return len(self.rules)
