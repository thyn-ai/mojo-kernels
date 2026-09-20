"""Vendored pure-Python JMESPath implementation (clean-room).

Written fresh from the JMESPath language specification (jmespath.org) plus
black-box behavioural probes of the reference package (PyPI jmespath 1.0.1).
No third-party code is used or adapted.

This module is the fallback backend of ``jmespath_mojo`` — selected on
platforms without a native build — and the wrapper's correctness oracle: any
call the native kernel cannot answer with reference-identical semantics is
recomputed here. The differential test suite runs the whole expression x
document matrix through this module on both backends and asserts structural
equality (and error-class equality) against the reference package.

Semantic notes (all verified against jmespath 1.0.1):

- arrays are exactly ``list`` and objects exactly ``dict`` (a tuple is not an
  array); field access on a non-dict is ``None``.
- truthiness: ``false``, ``null``, ``""``, ``[]`` and ``{}`` are falsy;
  numbers (including ``0``) are truthy.
- booleans are never numbers: ``1 == True`` is False, and ``abs``/``sum``/
  ``sort``/... reject booleans with ``JMESPathTypeError``.
- ordering comparisons on non-numbers are ``None`` — except string/number
  mixes, which propagate Python's ``TypeError`` like the reference.
- projections (list/object wildcards, slices, flatten, filters) drop ``null``
  results; ``map`` keeps them.
- ``to_string`` renders with ``json.dumps(..., separators=(",", ":"),
  ensure_ascii=True)``; ``ceil``/``floor`` return ints; ``avg`` returns a
  float; ``sum`` of all-int arrays returns an int.
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Exceptions (same class names/hierarchy as jmespath.exceptions)
# ---------------------------------------------------------------------------


class JMESPathError(Exception):
    pass


class EmptyExpressionError(JMESPathError):
    pass


class LexerError(JMESPathError):
    pass


class ParseError(JMESPathError):
    pass


class IncompleteExpressionError(LexerError):
    pass


class UnknownFunctionError(JMESPathError):
    pass


class ArityError(ParseError):
    pass


class VariadictArityError(ArityError):
    pass


class JMESPathTypeError(JMESPathError):
    pass


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_WHITESPACE = " \t\n\r"
_IDENTIFIER_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
_IDENTIFIER_CHARS = _IDENTIFIER_START + "0123456789"

_SIMPLE_TOKENS = {
    ".": "dot",
    ",": "comma",
    "[": "lbracket",
    "]": "rbracket",
    "(": "lparen",
    ")": "rparen",
    "{": "lbrace",
    "}": "rbrace",
    ":": "colon",
    "*": "star",
    "@": "current",
    "?": "question",
}


def _tokenize(expression: str) -> list[tuple[str, object]]:
    tokens: list[tuple[str, object]] = []
    i = 0
    n = len(expression)
    while i < n:
        c = expression[i]
        if c in _WHITESPACE:
            i += 1
            continue
        if c in _SIMPLE_TOKENS:
            tokens.append((_SIMPLE_TOKENS[c], c))
            i += 1
        elif c == "|":
            if expression[i + 1 : i + 2] == "|":
                tokens.append(("or", "||"))
                i += 2
            else:
                tokens.append(("pipe", "|"))
                i += 1
        elif c == "&":
            if expression[i + 1 : i + 2] == "&":
                tokens.append(("and", "&&"))
                i += 2
            else:
                tokens.append(("expref", "&"))
                i += 1
        elif c == "!":
            if expression[i + 1 : i + 2] == "=":
                tokens.append(("ne", "!="))
                i += 2
            else:
                tokens.append(("not", "!"))
                i += 1
        elif c == "=":
            if expression[i + 1 : i + 2] == "=":
                tokens.append(("eq", "=="))
                i += 2
            else:
                raise LexerError(f"Bad jmespath expression: Bad token {c}:\n{expression}\n^")
        elif c == "<":
            if expression[i + 1 : i + 2] == "=":
                tokens.append(("le", "<="))
                i += 2
            else:
                tokens.append(("lt", "<"))
                i += 1
        elif c == ">":
            if expression[i + 1 : i + 2] == "=":
                tokens.append(("ge", ">="))
                i += 2
            else:
                tokens.append(("gt", ">"))
                i += 1
        elif c == "`":
            value, i = _consume_literal(expression, i + 1)
            tokens.append(("literal", value))
        elif c == '"':
            value, i = _consume_quoted(expression, i + 1)
            tokens.append(("quoted_identifier", value))
        elif c == "'":
            value, i = _consume_raw_string(expression, i + 1)
            tokens.append(("raw_string", value))
        elif c == "-" and expression[i + 1 : i + 2].isdigit() or c.isdigit():
            j = i + (1 if c == "-" else 0)
            while j < n and expression[j].isdigit():
                j += 1
            tokens.append(("number", int(expression[i:j])))
            i = j
        elif c in _IDENTIFIER_START:
            j = i + 1
            while j < n and expression[j] in _IDENTIFIER_CHARS:
                j += 1
            tokens.append(("unquoted_identifier", expression[i:j]))
            i = j
        else:
            raise LexerError(
                f"Bad jmespath expression: Unknown token {c}:\n{expression}\n^"
            )
    tokens.append(("eof", None))
    return tokens


def _consume_quoted(expression: str, i: int) -> tuple[str, int]:
    """JSON string semantics for "quoted identifiers" (delegated to
    json.loads so escape and surrogate handling matches the reference)."""
    start = i
    n = len(expression)
    while i < n:
        c = expression[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            body = expression[start:i]
            try:
                return json.loads('"' + body + '"'), i + 1
            except ValueError:
                raise LexerError(
                    f"Bad jmespath expression: Bad token:\n{expression}\n^"
                ) from None
        i += 1
    raise LexerError(
        f"Bad jmespath expression: Unclosed ` delimiter:\n{expression}\n    ^"
    )


def _consume_raw_string(expression: str, i: int) -> tuple[str, int]:
    """Raw strings: \\X -> X when X is a quote, else both chars kept."""
    out: list[str] = []
    n = len(expression)
    while i < n:
        c = expression[i]
        if c == "'":
            return "".join(out), i + 1
        if c == "\\":
            if i + 1 >= n:
                break
            nxt = expression[i + 1]
            if nxt == "'":
                out.append("'")
            else:
                out.append("\\")
                out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    raise LexerError(
        f"Bad jmespath expression: Unclosed ` delimiter:\n{expression}\n    ^"
    )


def _consume_literal(expression: str, i: int) -> tuple[object, int]:
    """Backtick literal: parse as JSON; else as a JSON string body; else error."""
    out: list[str] = []
    n = len(expression)
    while i < n:
        c = expression[i]
        if c == "`":
            i += 1
            break
        if c == "\\":
            if i + 1 >= n:
                raise LexerError(
                    f"Bad jmespath expression: Unclosed ` delimiter:\n{expression}\n    ^"
                )
            nxt = expression[i + 1]
            if nxt == "`":
                out.append("`")
            else:
                out.append("\\")
                out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    else:
        raise LexerError(
            f"Bad jmespath expression: Unclosed ` delimiter:\n{expression}\n    ^"
        )
    content = "".join(out).lstrip()
    try:
        return json.loads(content), i
    except ValueError:
        pass
    try:
        return json.loads('"' + content + '"'), i
    except ValueError:
        raise LexerError(
            f"Bad jmespath expression: Bad token {content}:\n`{content}`\n^"
        ) from None


# ---------------------------------------------------------------------------
# Parser (Pratt; binding powers verified against the reference)
# ---------------------------------------------------------------------------

# AST nodes are tuples:
#   ("field", name) ("dot", l, r) ("index", src, i)
#   ("project", kind, src, rhs, (start, stop, step))
#   ("filter", src, pred, rhs) ("pipe", l, r) ("or", l, r) ("and", l, r)
#   ("not", e) ("cmp", op, l, r) ("lit", value) ("func", name, args)
#   ("expref", e) ("current",) ("multi_list", elems) ("multi_hash", pairs)

_BP_PIPE = 5
_BP_OR = 10
_BP_AND = 15
_BP_CMP = 20
_BP_FLATTEN = 30
_BP_NOT = 40
_BP_PROJ = 40
_BP_BRACKET = 55
_BP_DOT = 60

_LED_BP = {
    "dot": _BP_DOT,
    "lbracket": _BP_BRACKET,
    "eq": _BP_CMP,
    "ne": _BP_CMP,
    "lt": _BP_CMP,
    "le": _BP_CMP,
    "gt": _BP_CMP,
    "ge": _BP_CMP,
    "and": _BP_AND,
    "or": _BP_OR,
    "pipe": _BP_PIPE,
}


class _Parser:
    def __init__(self, tokens: list[tuple[str, object]], source: str):
        self.toks = tokens
        self.pos = 0
        self.source = source

    def cur(self) -> str:
        return self.toks[self.pos][0]

    def cur_value(self):
        return self.toks[self.pos][1]

    def advance(self):
        if self.pos < len(self.toks) - 1:
            self.pos += 1

    def parse(self):
        node = self.parse_expr(0)
        if self.cur() != "eof":
            raise ParseError(
                f"Unexpected token: {self.cur_value()}: Parse error, for expression:\n"
                f'"{self.source}"'
            )
        return node

    def led_bp(self) -> int:
        tk = self.cur()
        if tk == "lbracket" and self.toks[self.pos + 1][0] == "rbracket":
            # `[]` (flatten) has its own low binding power: `a == b[]` is
            # `a == (b[])` but `!a[]` is `(!a)[]`, and a projection's right
            # side never swallows a flatten (`matrix[][]` flattens again).
            return _BP_FLATTEN
        return _LED_BP.get(tk, 0)

    def parse_expr(self, min_bp: int):
        node = self.parse_nud()
        while True:
            bp = self.led_bp()
            if bp == 0 or bp < min_bp:
                return node
            node = self.parse_led(node)

    def parse_nud(self):
        tk = self.cur()
        val = self.cur_value()
        if tk == "eof":
            raise IncompleteExpressionError(
                f"Invalid jmespath expression: Incomplete expression:\n"
                f'"{self.source}"\n{" " * len(self.source)}^'
            )
        if tk == "lbracket":
            # parse_bracket consumes the '[' itself
            return self.parse_bracket(("current",))
        self.advance()
        if tk == "unquoted_identifier":
            if self.cur() == "lparen":
                return self.parse_function(val)
            return ("field", val)
        if tk == "quoted_identifier":
            return ("field", val)
        if tk == "raw_string":
            return ("lit", val)
        if tk == "literal":
            return ("lit", val)
        if tk == "current":
            return ("current",)
        if tk == "star":
            src = ("current",)
            rhs = self.parse_continuation()
            return ("project", "objwildcard", src, rhs, None)
        if tk == "lparen":
            inner = self.parse_expr(0)
            self.expect("rparen")
            return inner
        if tk == "not":
            return ("not", self.parse_expr(_BP_NOT))
        if tk == "expref":
            return ("expref", self.parse_expr(0))
        if tk == "lbrace":
            return self.parse_multi_hash()
        raise ParseError(
            f"invalid token: Parse error at column 0, token {val!r} ({tk}), "
            f'for expression:\n"{self.source}"'
        )

    def parse_led(self, left):
        tk = self.cur()
        if tk == "dot":
            return self.parse_dot(left)
        if tk == "lbracket":
            return self.parse_bracket(left)
        self.advance()
        if tk == "pipe":
            return ("pipe", left, self.parse_expr(_BP_PIPE + 1))
        if tk == "or":
            return ("or", left, self.parse_expr(_BP_OR + 1))
        if tk == "and":
            return ("and", left, self.parse_expr(_BP_AND + 1))
        # comparisons (eq/ne/lt/le/gt/ge), left-associative
        return ("cmp", tk, left, self.parse_expr(_BP_CMP + 1))

    def parse_dot(self, left):
        self.advance()  # consume '.'
        tk = self.cur()
        val = self.cur_value()
        if tk == "unquoted_identifier":
            self.advance()
            if self.cur() == "lparen":
                return ("dot", left, self.parse_function(val))
            return ("dot", left, ("field", val))
        if tk == "quoted_identifier":
            self.advance()
            return ("dot", left, ("field", val))
        if tk == "star":
            self.advance()
            rhs = self.parse_continuation()
            return ("project", "objwildcard", left, rhs, None)
        if tk == "lbracket":
            self.advance()
            elems = [self.parse_expr(0)]
            while self.cur() == "comma":
                self.advance()
                elems.append(self.parse_expr(0))
            self.expect("rbracket")
            return ("dot", left, ("multi_list", elems))
        if tk == "lbrace":
            self.advance()
            node = self.parse_multi_hash_body()
            return ("dot", left, node)
        raise ParseError(
            f"Expecting: ['quoted_identifier', 'unquoted_identifier', "
            f"'lbracket', 'lbrace'], got: {tk}: Parse error, for expression:\n"
            f'"{self.source}"'
        )

    def parse_multi_hash(self):
        return self.parse_multi_hash_body()

    def parse_multi_hash_body(self):
        pairs = []
        while True:
            tk = self.cur()
            if tk not in ("unquoted_identifier", "quoted_identifier"):
                raise ParseError(
                    f"Expecting hash key, got: {tk}: for expression:\n"
                    f'"{self.source}"'
                )
            key = self.cur_value()
            self.advance()
            self.expect("colon")
            pairs.append((key, self.parse_expr(0)))
            if self.cur() == "comma":
                self.advance()
                continue
            break
        self.expect("rbrace")
        return ("multi_hash", pairs)

    def parse_bracket(self, source):
        self.advance()  # consume '['
        tk = self.cur()
        if tk == "eof":
            raise IncompleteExpressionError(
                f"Invalid jmespath expression: Incomplete expression:\n"
                f'"{self.source}"\n{" " * len(self.source)}^'
            )
        if tk == "star":
            self.advance()
            self.expect("rbracket")
            rhs = self.parse_continuation()
            return ("project", "listwildcard", source, rhs, None)
        if tk == "rbracket":
            self.advance()
            rhs = self.parse_continuation()
            return ("project", "flatten", source, rhs, None)
        if tk == "question":
            self.advance()
            pred = self.parse_expr(0)
            self.expect("rbracket")
            rhs = self.parse_continuation()
            return ("filter", source, pred, rhs)
        if tk in ("number", "colon"):
            start = stop = step = None
            if tk == "number":
                start = self.cur_value()
                self.advance()
            if self.cur() == "rbracket" and start is not None:
                self.advance()
                return ("index", source, start)
            self.expect("colon")
            if self.cur() == "number":
                stop = self.cur_value()
                self.advance()
            if self.cur() == "colon":
                self.advance()
                if self.cur() == "number":
                    step = self.cur_value()
                    self.advance()
            self.expect("rbracket")
            rhs = self.parse_continuation()
            return ("project", "slice", source, rhs, (start, stop, step))
        raise ParseError(
            f"Expecting: star, got: {tk}: Parse error, for expression:\n"
            f'"{self.source}"'
        )

    def parse_continuation(self):
        if self.cur() in ("dot", "lbracket"):
            return self.parse_leds_from(("current",), _BP_PROJ)
        return ("current",)

    def parse_leds_from(self, node, min_bp):
        while True:
            bp = self.led_bp()
            if bp == 0 or bp < min_bp:
                return node
            node = self.parse_led(node)

    def parse_function(self, name):
        self.advance()  # consume '('
        args = []
        if self.cur() != "rparen":
            while True:
                args.append(self.parse_expr(0))
                if self.cur() == "comma":
                    self.advance()
                    # a trailing comma before ')' is accepted by the reference
                    if self.cur() == "rparen":
                        break
                    continue
                break
        self.expect("rparen")
        return ("func", name, args)

    def expect(self, kind):
        if self.cur() == kind:
            self.advance()
            return
        if self.cur() == "eof":
            raise IncompleteExpressionError(
                f"Invalid jmespath expression: Incomplete expression:\n"
                f'"{self.source}"\n{" " * len(self.source)}^'
            )
        raise ParseError(
            f"Expecting: {kind}, got: {self.cur()}: Parse error, for expression:\n"
            f'"{self.source}"'
        )


def _parse(expression: str):
    return _Parser(_tokenize(expression), expression).parse()


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


class _Expression:
    """An &expression reference (mirrors the reference's expref object)."""

    def __init__(self, expr):
        self.expr = expr


def _is_num(v) -> bool:
    # booleans are never numbers in JMESPath (verified against the reference)
    return type(v) in (int, float)


def _truthy(v) -> bool:
    if v is None or v is False:
        return False
    if type(v) is str:
        return len(v) > 0
    if type(v) in (list, dict):
        return len(v) > 0
    return True  # numbers (including 0) and True are truthy


def _equals(a, b) -> bool:
    ta, tb = type(a), type(b)
    if ta is bool or tb is bool:
        return ta is bool and tb is bool and a == b
    if _is_num(a) and _is_num(b):
        return a == b
    if ta is not tb:
        return False
    if ta is str:
        return a == b
    if a is None:
        return True
    if ta is list:
        return len(a) == len(b) and all(_equals(x, y) for x, y in zip(a, b))
    if ta is dict:
        return len(a) == len(b) and all(k in b and _equals(v, b[k]) for k, v in a.items())
    return a == b


_CMP_OPS = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def _eval(node, current):
    kind = node[0]
    if kind == "current":
        return current
    if kind == "lit":
        return node[1]
    if kind == "field":
        if type(current) is not dict:
            return None
        return current.get(node[1])
    if kind == "dot":
        return _eval(node[2], _eval(node[1], current))
    if kind == "index":
        base = _eval(node[1], current)
        if type(base) is not list:
            return None
        i = node[2]
        if i < 0:
            i += len(base)
        if i < 0 or i >= len(base):
            return None
        return base[i]
    if kind == "project":
        return _eval_project(node, current)
    if kind == "filter":
        return _eval_filter(node, current)
    if kind == "pipe":
        return _eval(node[2], _eval(node[1], current))
    if kind == "or":
        left = _eval(node[1], current)
        return left if _truthy(left) else _eval(node[2], current)
    if kind == "and":
        left = _eval(node[1], current)
        return left if not _truthy(left) else _eval(node[2], current)
    if kind == "not":
        return not _truthy(_eval(node[1], current))
    if kind == "cmp":
        return _eval_compare(node[1], _eval(node[2], current), _eval(node[3], current))
    if kind == "expref":
        return _Expression(node[1])
    if kind == "func":
        return _eval_function(node[1], node[2], current)
    if kind == "multi_list":
        if current is None:
            return None
        return [_eval(e, current) for e in node[1]]
    if kind == "multi_hash":
        if current is None:
            return None
        return {k: _eval(e, current) for k, e in node[1]}
    raise ParseError(f"unknown AST node: {node!r}")


def _eval_project(node, current):
    _, proj_kind, src, rhs, slice_parts = node
    base = _eval(src, current)
    if proj_kind == "objwildcard":
        if type(base) is not dict:
            return None
        items = list(base.values())
    else:
        if type(base) is not list:
            return None
        if proj_kind == "listwildcard":
            items = base
        elif proj_kind == "flatten":
            items = []
            for el in base:
                if type(el) is list:
                    items.extend(el)
                else:
                    items.append(el)
        elif proj_kind == "slice":
            start, stop, step = slice_parts
            items = base[slice(start, stop, step)]  # ValueError on step 0
        else:  # pragma: no cover
            raise ParseError(f"unknown projection: {proj_kind}")
    # projection drops null results (including the identity form)
    out = []
    for el in items:
        r = _eval(rhs, el) if rhs != ("current",) else el
        if r is not None:
            out.append(r)
    return out


def _eval_filter(node, current):
    _, src, pred, rhs = node
    base = _eval(src, current)
    if type(base) is not list:
        return None
    out = []
    for el in base:
        if _truthy(_eval(pred, el)):
            r = _eval(rhs, el) if rhs != ("current",) else el
            if r is not None:
                out.append(r)
    return out


def _eval_compare(op, left, right):
    if op in ("eq", "ne"):
        r = _equals(left, right)
        return r if op == "eq" else not r
    lnum, rnum = _is_num(left), _is_num(right)
    lstr, rstr = type(left) is str, type(right) is str
    if (lnum and rnum) or (lstr and rstr):
        return _CMP_OPS[op](left, right)
    if (lstr and rnum) or (lnum and rstr):
        # the reference propagates Python's TypeError for string/number mixes
        _ = left < right  # raises TypeError
    return None  # ordering on bool/null/array/object is null


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _type_name(v) -> str:
    if v is None:
        return "null"
    if type(v) is bool:
        return "boolean"
    if _is_num(v):
        return "number"
    if type(v) is str:
        return "string"
    if type(v) is list:
        return "array"
    if type(v) is dict:
        return "object"
    if type(v) is _Expression:
        return "expref"
    return "unknown"


def _err(name, value, expected):
    raise JMESPathTypeError(
        f"In function {name}(), invalid type for value: {value!r}, "
        f"expected one of: {expected}, received: \"{_type_name(value)}\""
    )


def _require_number_array(name, arr, allow_strings=False):
    if type(arr) is not list:
        _err(name, arr, ["array"])
    cls = None
    for el in arr:
        this = "number" if _is_num(el) else ("string" if type(el) is str else None)
        if this is None or (this == "string" and not allow_strings):
            _err(name, el, ["array-number"] + (["array-string"] if allow_strings else []))
        if cls is None:
            cls = this
        elif cls != this:
            _err(name, el, ["array-number", "array-string"])
    return cls


def _eval_function(name, arg_nodes, current):
    fn = _FUNCTIONS.get(name)
    if fn is None:
        raise UnknownFunctionError(f"Unknown function: {name}()")
    arity, variadic = fn[1]
    if not variadic and len(arg_nodes) != arity:
        raise ArityError(
            f"Expected {arity} argument{'s' if arity != 1 else ''} for function "
            f"{name}(), received {len(arg_nodes)}"
        )
    if variadic and len(arg_nodes) < arity:
        raise VariadictArityError(
            f"Expected at least {arity} argument{'s' if arity != 1 else ''} for "
            f"function {name}(), received {len(arg_nodes)}"
        )
    args = [
        _Expression(a[1]) if a[0] == "expref" else _eval(a, current) for a in arg_nodes
    ]
    return fn[0](name, args)


def _fn_abs(name, args):
    v = args[0]
    if not _is_num(v):
        _err(name, v, ["number"])
    return abs(v)


def _fn_avg(name, args):
    arr = args[0]
    _require_number_array(name, arr)
    if len(arr) == 0:
        return None
    return sum(arr) / len(arr)


def _fn_ceil(name, args):
    v = args[0]
    if not _is_num(v):
        _err(name, v, ["number"])
    return math.ceil(v)


def _fn_floor(name, args):
    v = args[0]
    if not _is_num(v):
        _err(name, v, ["number"])
    return math.floor(v)


def _fn_contains(name, args):
    hay, needle = args
    if type(hay) is list:
        return any(_equals(el, needle) for el in hay)
    if type(hay) is str:
        # `in` raises TypeError for a non-string needle, like the reference
        return needle in hay
    _err(name, hay, ["array", "string"])


def _fn_ends_with(name, args):
    s, suffix = args
    if type(s) is not str:
        _err(name, s, ["string"])
    if type(suffix) is not str:
        _err(name, suffix, ["string"])
    return s.endswith(suffix)


def _fn_starts_with(name, args):
    s, prefix = args
    if type(s) is not str:
        _err(name, s, ["string"])
    if type(prefix) is not str:
        _err(name, prefix, ["string"])
    return s.startswith(prefix)


def _fn_join(name, args):
    sep, arr = args
    if type(sep) is not str:
        _err(name, sep, ["string"])
    if type(arr) is not list:
        _err(name, arr, ["array"])
    for el in arr:
        if type(el) is not str:
            _err(name, el, ["array-string"])
    return sep.join(arr)


def _fn_keys(name, args):
    if type(args[0]) is not dict:
        _err(name, args[0], ["object"])
    return list(args[0].keys())


def _fn_values(name, args):
    if type(args[0]) is not dict:
        _err(name, args[0], ["object"])
    return list(args[0].values())


def _fn_length(name, args):
    v = args[0]
    if type(v) in (str, list, dict):
        return len(v)
    _err(name, v, ["string", "array", "object"])


def _fn_map(name, args):
    expref, arr = args
    if type(expref) is not _Expression:
        _err(name, expref, ["expref"])
    if type(arr) is not list:
        _err(name, arr, ["array"])
    return [_eval(expref.expr, el) for el in arr]  # map keeps nulls


def _fn_max_min(want_max):
    def fn(name, args):
        arr = args[0]
        _require_number_array(name, arr, allow_strings=True)
        if len(arr) == 0:
            return None
        best = arr[0]
        for el in arr[1:]:
            if (el > best) if want_max else (el < best):
                best = el
        return best

    return fn


def _eval_keys_for(name, expref, arr):
    keys = []
    cls = None
    for el in arr:
        key = _eval(expref.expr, el)
        this = "number" if _is_num(key) else ("string" if type(key) is str else None)
        if this is None:
            _err(name, key, ["number", "string"])
        if cls is None:
            cls = this
        elif cls != this:
            _err(name, key, ["number", "string"])
        keys.append(key)
    return keys


def _fn_max_min_by(want_max):
    def fn(name, args):
        arr, expref = args
        if type(arr) is not list:
            _err(name, arr, ["array"])
        if type(expref) is not _Expression:
            _err(name, expref, ["expref"])
        if len(arr) == 0:
            return None
        keys = _eval_keys_for(name, expref, arr)
        best = 0
        for i in range(1, len(arr)):
            if (keys[i] > keys[best]) if want_max else (keys[i] < keys[best]):
                best = i
        return arr[best]

    return fn


def _fn_merge(name, args):
    out = {}
    for d in args:
        if type(d) is not dict:
            _err(name, d, ["object"])
        out.update(d)  # overwrite keeps first position (Python dict)
    return out


def _fn_not_null(name, args):
    for v in args:
        if v is not None:
            return v
    return None


def _fn_reverse(name, args):
    v = args[0]
    if type(v) in (str, list):
        return v[::-1]
    _err(name, v, ["string", "array"])


def _fn_sort(name, args):
    arr = args[0]
    _require_number_array(name, arr, allow_strings=True)
    return sorted(arr)  # stable; nan matches the reference (same runtime)


def _fn_sort_by(name, args):
    arr, expref = args
    if type(arr) is not list:
        _err(name, arr, ["array"])
    if type(expref) is not _Expression:
        _err(name, expref, ["expref"])
    keys = _eval_keys_for(name, expref, arr)
    order = sorted(range(len(arr)), key=lambda i: keys[i])  # stable
    return [arr[i] for i in order]


def _fn_sum(name, args):
    arr = args[0]
    _require_number_array(name, arr)
    return sum(arr)


def _fn_to_array(name, args):
    v = args[0]
    return v if type(v) is list else [v]


def _fn_to_number(name, args):
    v = args[0]
    if _is_num(v):
        return v
    if type(v) is str:
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _fn_to_string(name, args):
    v = args[0]
    if type(v) is str:
        return v
    return json.dumps(v, separators=(",", ":"), ensure_ascii=True)


def _fn_type(name, args):
    v = args[0]
    if type(v) is _Expression:
        return None
    return _type_name(v) if _type_name(v) != "unknown" else None


# name -> (implementation, (arity, variadic))
_FUNCTIONS = {
    "abs": (_fn_abs, (1, False)),
    "avg": (_fn_avg, (1, False)),
    "ceil": (_fn_ceil, (1, False)),
    "contains": (_fn_contains, (2, False)),
    "ends_with": (_fn_ends_with, (2, False)),
    "floor": (_fn_floor, (1, False)),
    "join": (_fn_join, (2, False)),
    "keys": (_fn_keys, (1, False)),
    "values": (_fn_values, (1, False)),
    "length": (_fn_length, (1, False)),
    "map": (_fn_map, (2, False)),
    "max": (_fn_max_min(True), (1, False)),
    "max_by": (_fn_max_min_by(True), (2, False)),
    "merge": (_fn_merge, (1, True)),
    "min": (_fn_max_min(False), (1, False)),
    "min_by": (_fn_max_min_by(False), (2, False)),
    "not_null": (_fn_not_null, (1, True)),
    "reverse": (_fn_reverse, (1, False)),
    "sort": (_fn_sort, (1, False)),
    "sort_by": (_fn_sort_by, (2, False)),
    "starts_with": (_fn_starts_with, (2, False)),
    "sum": (_fn_sum, (1, False)),
    "to_array": (_fn_to_array, (1, False)),
    "to_number": (_fn_to_number, (1, False)),
    "to_string": (_fn_to_string, (1, False)),
    "type": (_fn_type, (1, False)),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search(expression, data):
    """Vendored pure-Python equivalent of ``jmespath.search``."""
    if expression is None:
        raise EmptyExpressionError(
            "Invalid JMESPath expression: cannot be empty."
        )
    if isinstance(expression, bytes):
        # the reference's lexer iterates bytes into ints and rejects them
        raise LexerError(
            f"Bad jmespath expression: Unknown token {expression[:1]!r}:\n"
            f"{expression!r}\n^"
        )
    if not isinstance(expression, str):
        raise TypeError(f"'{type(expression).__name__}' object is not iterable")
    if expression == "":
        raise EmptyExpressionError(
            "Invalid JMESPath expression: cannot be empty."
        )
    return _eval(_parse(expression), data)
