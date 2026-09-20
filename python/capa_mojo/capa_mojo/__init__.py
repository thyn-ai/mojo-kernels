"""capa-mojo: an accelerated rule-evaluation engine for capa-style rule sets.

Given a rule set (capa's documented YAML format) and a feature map per scope,
produce match results identical to the reference engine's
``capa.engine.match`` for the supported rule subset — powered by a clean-room
Mojo batch boolean rule-tree evaluator where the platform supports it (macOS
arm64, Linux x86_64), with a pure-Python fallback everywhere else.

    import capa_mojo

    rules = capa_mojo.load_rules([open("rule.yml").read()])
    result = rules.match({("api", "CreateFileA"): {0x401010}})
    assert "my rule" in result

Feature extraction (disassembly etc.) is out of scope: callers bring feature
maps. Set CAPA_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from capa_mojo._native import backend_info, native_available
from capa_mojo.core import RuleSet, ScopeMatches, load_rules, match
from capa_mojo.features import InvalidRuleError, UnsupportedRuleError
from capa_mojo.rules import ParsedRule, parse_rule_dict, parse_rule_yaml

__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "InvalidRuleError",
    "ParsedRule",
    "RuleSet",
    "ScopeMatches",
    "UnsupportedRuleError",
    "backend_info",
    "load_rules",
    "match",
    "native_available",
    "parse_rule_dict",
    "parse_rule_yaml",
    "__version__",
]
