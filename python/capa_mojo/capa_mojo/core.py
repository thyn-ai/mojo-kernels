"""Public API: compile capa-style rules and match them against feature maps.

    import capa_mojo

    rules = capa_mojo.load_rules([rule_yaml_text, ...])
    results = rules.match({("api", "CreateFileA"): {0x401010}, ...})          # one scope
    results = rules.match([(0x401000, feature_map), (0x402000, feature_map)]) # many scopes

or the one-shot form: ``capa_mojo.match(rules, features)``.

Evaluation runs on the native Mojo kernel when its shared library is
available (macOS arm64 / Linux x86_64 wheels) and transparently falls back to
the pure-Python reference evaluator otherwise. Both backends implement the
same documented semantics; the differential test suite asserts that both
agree with the reference engine (`capa.engine.match` from flare-capa) on the
supported rule subset, rule for rule and location for location.

Scope: this package accelerates *rule evaluation only*. Feature extraction
(disassembly, vivisect, etc.) is entirely out of scope and untouched: callers
bring feature maps in the documented ``{(name, value): {locations}}`` form.
"""

from __future__ import annotations

import numpy as np

from capa_mojo import _reference
from capa_mojo._native import NativeRuleSet, NativeUnavailable
from capa_mojo._reference import INT32_MAX, ScopeView
from capa_mojo.compiler import CompiledRules
from capa_mojo.features import InvalidRuleError
from capa_mojo.rules import ParsedRule, parse_rule_dict, parse_rule_yaml

_BLOCK_SCOPES = 64  # the native kernel evaluates one 64-scope block per call


class ScopeMatches:
    """Match results for one scope: rule name -> detail tree."""

    __slots__ = ("addr", "matches", "backend")

    def __init__(self, addr, matches, backend):
        self.addr = addr
        self.matches = matches
        self.backend = backend

    @property
    def rule_names(self) -> list[str]:
        """Names of the rules that matched this scope (evaluation order)."""
        return list(self.matches.keys())

    def __contains__(self, rule_name: str) -> bool:
        return rule_name in self.matches

    def __len__(self) -> int:
        return len(self.matches)

    def __repr__(self):
        return f"ScopeMatches(addr={self.addr!r}, rules={len(self.matches)}, backend={self.backend!r})"


def _coerce_rule(rule) -> ParsedRule:
    if isinstance(rule, ParsedRule):
        return rule
    if isinstance(rule, str):
        return parse_rule_yaml(rule)
    if isinstance(rule, dict):
        return parse_rule_dict(rule)
    raise InvalidRuleError(
        f"rules must be YAML strings, parsed dicts, or ParsedRule objects; got {type(rule)}"
    )


def _coerce_scopes(features):
    """Normalize the scopes argument into (addr, feature_map) pairs.

    Accepts a single feature map (addr None), a list of feature maps, or a
    list of (addr, feature_map) pairs. Returns (pairs, was_single).
    """
    if isinstance(features, dict):
        return [(None, features)], True
    pairs = []
    for item in features:
        if isinstance(item, dict):
            pairs.append((None, item))
        else:
            addr, fmap = item
            pairs.append((addr, fmap))
    return pairs, False


class RuleSet:
    """A compiled rule set: parse once, match against many scopes."""

    def __init__(self, rules):
        parsed = [_coerce_rule(r) for r in rules]
        self._compiled = CompiledRules(parsed)
        self._regex_by_identity = {
            leaf.identity: pattern
            for leaf, pattern in zip(self._compiled.special_leaves, self._compiled._special_regexes)
        }
        self._match_slot_by_name = {
            name: self._compiled.n_count_leaves + i
            for i, name in enumerate(self._compiled.match_group_names)
        }
        try:
            self._native = NativeRuleSet(self._compiled)
            self._backend = "native"
        except NativeUnavailable:
            self._native = None
            self._backend = "fallback"

    @property
    def backend(self) -> str:
        """Which evaluator serves this instance: "native" or "fallback"."""
        return self._backend

    @property
    def rules(self):
        """The compiled rules in evaluation (topological) order."""
        return self._compiled.rules

    def __len__(self) -> int:
        return len(self._compiled)

    # -- evaluation ---------------------------------------------------------

    def _regex_of(self, node):
        return self._regex_by_identity[node.identity]

    def match(self, features):
        """Match every rule against one or more scopes.

        `features`: a single feature map, a list of feature maps, or a list
        of ``(addr, feature_map)`` pairs. A feature map maps ``(name, value)``
        keys to iterables of integer locations (a feature may be present with
        an empty location set, which counts as present for leaf matching and
        as count 0 for ``count(...)``).

        Returns a :class:`ScopeMatches` for a single input map, otherwise a
        list of them, one per scope, in input order.
        """
        pairs, was_single = _coerce_scopes(features)
        views = [ScopeView(addr, fmap) for addr, fmap in pairs]
        if self._native is not None:
            decisions = self._decide_native(views)
        else:
            decisions = self._decide_fallback(views)
        results = [
            self._build_scope_matches(view, matched_indices)
            for view, matched_indices in zip(views, decisions)
        ]
        if was_single:
            return results[0]
        return results

    def _decide_fallback(self, views: list[ScopeView]) -> list[list[int]]:
        """Per scope, the indices of matched rules (pure-Python decision)."""
        compiled = self._compiled
        all_matched: list[list[int]] = []
        for view in views:
            matched_indices: list[int] = []
            published: set[str] = set()
            for idx, rule in enumerate(compiled.rules):
                if _reference.eval_bool(rule.statement, view, published, self._regex_of):
                    matched_indices.append(idx)
                    published.update(compiled.rule_publishes[idx])
            all_matched.append(matched_indices)
        return all_matched

    def _decide_native(self, views: list[ScopeView]) -> list[list[int]]:
        """Per scope, the indices of matched rules (native kernel decision)."""
        compiled = self._compiled
        n_slots = compiled.n_count_slots
        n_exact = compiled.n_exact_leaves
        exact_index = compiled.exact_index
        special_leaves = compiled.special_leaves
        match_slots = self._match_slot_by_name
        decisions: list[list[int]] = [[] for _ in views]
        for block_start in range(0, len(views), _BLOCK_SCOPES):
            block = views[block_start : block_start + _BLOCK_SCOPES]
            n_lanes = len(block)
            counts = np.zeros((n_slots, n_lanes), dtype=np.int32)
            present = np.zeros(n_slots, dtype=np.uint64)
            for lane, view in enumerate(block):
                bit = np.uint64(1) << np.uint64(lane)
                for key, locs in view.features.items():
                    slot = exact_index.get(key)
                    if slot is not None:
                        counts[slot, lane] = min(len(locs), INT32_MAX)
                        present[slot] |= bit
                    elif key[0] == "match":
                        gslot = match_slots.get(key[1])
                        if gslot is not None:
                            counts[gslot, lane] = min(len(locs), INT32_MAX)
                            present[gslot] |= bit
                for si, leaf in enumerate(special_leaves):
                    regex = self._regex_by_identity.get(leaf.identity)
                    if view.special(leaf, regex)[0]:
                        present[n_exact + si] |= bit
            words = self._native.eval(counts, present)
            for lane in range(n_lanes):
                bit = 1 << lane
                decisions[block_start + lane] = [
                    r for r in range(len(compiled)) if int(words[r]) & bit
                ]
        return decisions

    def _build_scope_matches(self, view: ScopeView, matched_indices: list[int]) -> ScopeMatches:
        """Reconstruct the per-rule detail trees for one scope's matches.

        Only matched rules get a detail tree; `published` accumulates the
        names (rule names and namespace prefixes) that earlier matches make
        visible to later rules' ``match:`` leaves.
        """
        compiled = self._compiled
        matches = {}
        published: set[str] = set()
        for idx in matched_indices:
            rule = compiled.rules[idx]
            detail = _reference.eval_detail(rule.statement, view, published, self._regex_of)
            # the decision pass and the detail pass must agree
            assert detail[1] is True
            matches[rule.name] = detail
            published.update(compiled.rule_publishes[idx])
        return ScopeMatches(view.addr, matches, self._backend)


def load_rules(rules) -> RuleSet:
    """Compile a list of rules (YAML strings, parsed dicts, or ParsedRule)."""
    return RuleSet(rules)


def match(rules, features):
    """One-shot form: compile `rules` and match them against `features`."""
    return RuleSet(rules).match(features)
