"""Vendored pure-Python JSONPath engine (h2non/jsonpath-ng ext-dialect semantics).

This is a clean-room reimplementation of the observable behavior of
``jsonpath_ng.ext.parse(expr).find(data)`` (PyPI jsonpath_ng 1.7.0), written
from a black-box semantics contract (see the package README and the
differential test suite). It serves two roles:

* the transparent fallback on platforms without the native Mojo kernel, and
* the in-repo semantics reference for the native kernel.

Supported grammar:

    child steps        $.a  $['a']  $.a.b  (dot-number children: $.a.0)
    field unions       $['a', 'b']  (expression order, duplicates repeated)
    wildcards          $.*  $[*]
    recursive descent  $..a  $..*  $..[0]  $..['a']  (DFS pre-order)
    index              $.a[0]  $.a[-1]
    slices             $.a[1:5:2]  $.a[::-1]  (Python slice semantics)
    filters            $.a[?(@.x > 2)]  with ==, !=, <, <=, >, >=, `&`
                       conjunctions, `@` paths, arith (+, -, *), and bare
                       existence tests `?(@.x)`

Find results are ``(value, path)`` pairs in the oracle's order, with paths
rendered exactly like the oracle's ``str(DatumInContext.full_path)``.

The observable error contract (raised for invalid queries or hostile data)
mirrors the oracle: ``JsonPathError`` for parse/lex errors, and the same
builtin ``TypeError`` / ``IndexError`` / ``KeyError`` / ``ValueError`` /
``NotImplementedError`` the oracle raises at evaluation time, with matching
messages (CPython produces most of them for free).
"""

from __future__ import annotations

__all__ = ["JsonPathError", "parse", "find", "render_path"]


class JsonPathError(Exception):
    """Invalid JSONPath expression (parse or lex error)."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_WS = " \t\n\r"
_QUOTE_NEEDED_CHARS = ".,:[]()|&$*~"


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def error(self) -> JsonPathError:
        return JsonPathError(f"Parse error at 1:{self.pos} in {self.text!r}")

    def peek(self, k: int = 0) -> str:
        i = self.pos + k
        return self.text[i] if i < len(self.text) else ""

    def ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _WS:
            self.pos += 1

    def quoted(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        out = []
        while True:
            if self.pos >= len(self.text):
                raise self.error()
            c = self.text[self.pos]
            if c == quote:
                self.pos += 1
                return "".join(out)
            if c == "\\":
                # a backslash escapes the next character literally
                self.pos += 1
                if self.pos >= len(self.text):
                    raise self.error()
                out.append(self.text[self.pos])
                self.pos += 1
            else:
                out.append(c)
                self.pos += 1

    def ident(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isascii()
            and (self.text[self.pos].isalnum() or self.text[self.pos] == "_")
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def uint(self) -> int:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == start:
            raise self.error()
        return int(self.text[start : self.pos])

    def number_token(self):
        """INT or FLOAT literal (no exponents, digits on both sides of '.')."""
        neg = self.peek() == "-"
        if neg:
            self.pos += 1
        if not self.peek().isdigit():
            raise self.error()
        iv = self.uint()
        if self.peek() == ".":
            self.pos += 1
            if not self.peek().isdigit():
                raise self.error()
            frac_start = self.pos
            frac = self.uint()
            dv = float(iv) + float(frac) / 10 ** (self.pos - frac_start)
            return ("dbl", -dv if neg else dv)
        if self.peek() and (self.peek().isalpha() or self.peek() == "_"):
            # e.g. 1e2 — the oracle lexes 'e2' as an ID and fails at parse time
            raise self.error()
        return ("int", -iv if neg else iv)

    def dot_child(self):
        """After '.': '*' | ID | digits (digits become a field name)."""
        self.ws()
        c = self.peek()
        if c == "*":
            self.pos += 1
            return ("wild",)
        if c.isdigit():
            return ("child", (str(self.uint()),))
        name = self.ident()
        if not name:
            raise self.error()
        return ("child", (name,))

    def index_or_slice_body(self):
        """Inside brackets: INT | [start]:[end][:step]."""
        has_start = False
        start_v = 0
        neg = self.peek() == "-"
        if neg:
            self.pos += 1
        if self.peek().isdigit():
            start_v = self.uint()
            has_start = True
            if neg:
                start_v = -start_v
        elif neg:
            raise self.error()
        self.ws()
        if self.peek() != ":":
            if not has_start:
                raise self.error()
            if self.peek() != "]":
                raise self.error()
            self.pos += 1
            return ("index", start_v)
        self.pos += 1
        self.ws()
        end_v = None
        neg = self.peek() == "-"
        if neg:
            self.pos += 1
        if self.peek().isdigit():
            end_v = self.uint()
            if neg:
                end_v = -end_v
        elif neg:
            raise self.error()
        self.ws()
        step_v = None
        if self.peek() == ":":
            self.pos += 1
            self.ws()
            neg = self.peek() == "-"
            if neg:
                self.pos += 1
            if self.peek().isdigit():
                step_v = self.uint()
                if neg:
                    step_v = -step_v
            elif neg:
                raise self.error()
            self.ws()
        if self.peek() != "]":
            raise self.error()
        self.pos += 1
        return ("slice", start_v if has_start else None, end_v, step_v)

    def bracket(self, after_descend: bool):
        self.ws()
        c = self.peek()
        if c == "*":
            self.pos += 1
            self.ws()
            if self.peek() != "]":
                raise self.error()
            self.pos += 1
            return ("idxwild",)
        if c == "?":
            if after_descend:
                raise self.error()  # the oracle rejects `..[?(...)]`
            self.pos += 1
            filt = self.filter_expr()
            self.ws()
            if self.peek() != "]":
                raise self.error()
            self.pos += 1
            return ("filter", filt)
        if c in "'\"":
            names = [self.quoted()]
            self.ws()
            while self.peek() == ",":
                self.pos += 1
                self.ws()
                if self.peek() not in "'\"":
                    raise self.error()
                names.append(self.quoted())
                self.ws()
            if self.peek() != "]":
                raise self.error()
            self.pos += 1
            # a quoted '*' — alone or inside a union — is a dict wildcard
            if "*" in names:
                return ("wild",)
            return ("child", tuple(names))
        if c == "-" or c.isdigit() or c == ":":
            return self.index_or_slice_body()
        raise self.error()

    def descend_substep(self):
        self.ws()
        c = self.peek()
        if c == "*":
            self.pos += 1
            return ("wild",)
        if c == "[":
            self.pos += 1
            return self.bracket(after_descend=True)
        if c.isdigit():
            return ("child", (str(self.uint()),))
        name = self.ident()
        if not name:
            raise self.error()
        return ("child", (name,))

    # --- filter expressions ---

    def fpath(self):
        """The step sequence after `@` in a filter (empty = current node)."""
        steps = []
        while True:
            c = self.peek()
            if c == ".":
                if self.peek(1) == ".":
                    break
                self.pos += 1
                self.ws()
                nc = self.peek()
                if nc == "*":
                    self.pos += 1
                    steps.append(("wild",))
                elif nc.isdigit():
                    steps.append(("field", str(self.uint())))
                else:
                    name = self.ident()
                    if not name:
                        raise self.error()
                    steps.append(("field", name))
            elif c == "[":
                self.pos += 1
                self.ws()
                nc = self.peek()
                if nc == "*":
                    self.pos += 1
                    self.ws()
                    if self.peek() != "]":
                        raise self.error()
                    self.pos += 1
                    steps.append(("idxwild",))
                elif nc in "'\"":
                    names = [self.quoted()]
                    self.ws()
                    while self.peek() == ",":
                        self.pos += 1
                        self.ws()
                        if self.peek() not in "'\"":
                            raise self.error()
                        names.append(self.quoted())
                        self.ws()
                    if self.peek() != "]":
                        raise self.error()
                    self.pos += 1
                    if "*" in names:
                        steps.append(("wild",))
                    elif len(names) == 1:
                        steps.append(("field", names[0]))
                    else:
                        steps.append(("union", tuple(names)))
                elif nc == "-" or nc.isdigit() or nc == ":":
                    steps.append(self.index_or_slice_body())
                else:
                    raise self.error()
            else:
                break
        return ("path", steps)

    def factor(self):
        self.ws()
        c = self.peek()
        if c == "@":
            self.pos += 1
            return self.fpath()
        if c in "'\"":
            return ("str", self.quoted())
        if c == "(":
            self.pos += 1
            inner = self.arith()
            self.ws()
            if self.peek() != ")":
                raise self.error()
            self.pos += 1
            return inner
        if c == "-" or c.isdigit():
            return self.number_token()
        if c.isascii() and (c.isalpha() or c == "_"):
            word = self.ident()
            if word == "true":
                return ("int", 1)
            if word == "false":
                return ("int", 0)
            # any other bare word (null, None, foo, ...) is a string literal
            return ("str", word)
        raise self.error()

    def term(self):
        left = self.factor()
        while True:
            self.ws()
            if self.peek() != "*":
                return left
            self.pos += 1
            left = ("mul", left, self.factor())

    def arith(self):
        left = self.term()
        while True:
            self.ws()
            c = self.peek()
            if c not in "+-":
                return left
            self.pos += 1
            left = ("add" if c == "+" else "sub", left, self.term())

    def filter_literal(self):
        """The right operand of a filter comparison: a single literal —
        number, quoted string, true/false, or bare word (no parens, no @,
        no arith), matching the oracle grammar."""
        self.ws()
        c = self.peek()
        if c == "-" or c.isdigit():
            return self.number_token()
        if c in "'\"":
            return ("str", self.quoted())
        if c.isascii() and (c.isalpha() or c == "_"):
            word = self.ident()
            if word == "true":
                return ("int", 1)
            if word == "false":
                return ("int", 0)
            return ("str", word)
        raise self.error()

    def filter_expr(self):
        if self.peek() != "(":
            raise self.error()
        self.pos += 1
        conjs = []
        while True:
            left = self.arith()
            self.ws()
            two = self.text[self.pos : self.pos + 2]
            one = self.text[self.pos]
            op = None
            if two in ("==", "!=", "<=", ">="):
                op = two
                self.pos += 2
            elif one in "<>":
                op = one
                self.pos += 1
            if op is not None:
                right = self.filter_literal()
                conjs.append(("cmp", left, op, right))
            else:
                conjs.append(("exists", left))
            self.ws()
            if self.peek() == "&":
                self.pos += 1
                continue
            if self.peek() == ")":
                self.pos += 1
                return conjs
            raise self.error()

    def expression(self):
        """Parse a whole JSONPath expression into a list of plan steps."""
        self.ws()
        if self.peek() == "$":
            self.pos += 1
        steps = []
        while True:
            self.ws()
            c = self.peek()
            if not c:
                return steps
            if c == ".":
                if self.peek(1) == ".":
                    self.pos += 2
                    steps.append(("desc", self.descend_substep()))
                else:
                    self.pos += 1
                    steps.append(self.dot_child())
            elif c == "[":
                self.pos += 1
                steps.append(self.bracket(after_descend=False))
            elif not steps and (c == "*" or c in "'\"" or c.isascii() and (c.isalpha() or c == "_")):
                # implicit root: the oracle accepts a leading step without '$'
                if c == "*":
                    self.pos += 1
                    steps.append(("wild",))
                elif c in "'\"":
                    steps.append(("child", (self.quoted(),)))
                else:
                    steps.append(("child", (self.ident(),)))
            else:
                raise self.error()


def _arith_has_path(arith) -> bool:
    kind = arith[0]
    if kind == "path":
        return True
    if kind in ("add", "sub", "mul"):
        return _arith_has_path(arith[1]) or _arith_has_path(arith[2])
    return False


def parse(expr: str):
    """Compile a JSONPath expression to a plan (list of steps)."""
    if not isinstance(expr, str):
        raise TypeError(f"expression must be a str, not {type(expr).__name__}")
    parser = _Parser(expr)
    steps = parser.expression()
    parser.ws()
    if parser.pos != len(expr):
        raise parser.error()
    return tuple(steps)


# ---------------------------------------------------------------------------
# Path rendering (matches the oracle's str(full_path) byte for byte)
# ---------------------------------------------------------------------------

# step kinds in result paths: ("f", name) object field; ("i", idx) sequence
# index; ("d", pos) dict-position (filter/`[*]` on dicts); ("s", idx)
# render-only index that keeps the current node ([*] on dicts/str/scalars).


def _render_field(name: str) -> str:
    # NB: `ch in name` (not `ch in _QUOTE_NEEDED`) — for non-string keys this
    # raises TypeError "argument of type 'int' is not iterable", the oracle's
    # exact error when a path containing such a key is rendered.
    if any(ch in name for ch in _QUOTE_NEEDED_CHARS):
        return f"'{name}'"
    return name


def render_path(steps) -> str:
    """Render a result path like the oracle's ``str(full_path)``."""
    if not steps:
        return "$"
    pieces = []
    for kind, val in steps:
        if kind == "f":
            pieces.append(_render_field(val))
        else:
            pieces.append(f"[{val}]")
    return pieces[0] + "".join("." + p for p in pieces[1:])


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _child(value, names):
    if not isinstance(value, dict):
        return
    for name in names:
        if name in value:
            yield value[name], ("f", name)


def _wild(value):
    if not isinstance(value, dict):
        return
    for key, val in value.items():
        yield val, ("f", key)


def _index(value, idx):
    if not value:
        return
    if isinstance(value, (list, tuple, str)):
        if idx >= 0:
            if idx < len(value):
                yield value[idx], ("i", idx)
        else:
            # out-of-range negatives raise IndexError (CPython message)
            yield value[idx], ("i", idx)
        return
    if isinstance(value, dict):
        if idx >= 0 and idx >= len(value):
            return
        # value[idx] on a dict raises KeyError(idx) for JSON-domain keys
        yield value[idx], ("i", idx)
        return
    # truthy scalar: len() raises TypeError ("object of type 'int' has no len()")
    len(value)
    return  # unreachable


def _idxwild(value):
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield item, ("i", i)
        return
    if not value:
        return
    if isinstance(value, float):
        len(value)  # TypeError: object of type 'float' has no len()
    # truthy dict/str/int/bool/other-sized-object: the node itself at [0]
    yield value, ("s", 0)


def _slice(value, start, end, step):
    if not value:
        return
    step_v = step if step is not None else 1
    if step_v == 0:
        raise ValueError("slice step cannot be zero")
    if not isinstance(value, (list, tuple)):
        return
    for i in range(*slice(start, end, step).indices(len(value))):
        yield value[i], ("i", i)


def _descend(substep, value, path, out):
    """DFS pre-order: apply the substep at the node, then descend."""
    for val, step in _apply(substep, value):
        out.append((val, path + (step,)))
    if isinstance(value, dict):
        children = ((v, ("f", k)) for k, v in value.items())
    elif isinstance(value, list):
        children = ((v, ("i", i)) for i, v in enumerate(value))
    else:
        return
    for child, step in children:
        _descend(substep, child, path + (step,), out)


def _filter(filt, value):
    if isinstance(value, list):
        candidates = ((v, ("i", i)) for i, v in enumerate(value))
    elif isinstance(value, dict):
        candidates = ((v, ("d", i)) for i, v in enumerate(value.values()))
    else:
        return
    for cand, step in candidates:
        if _eval_filter(filt, cand):
            yield cand, step


def _apply(step, value):
    kind = step[0]
    if kind == "child":
        yield from _child(value, step[1])
    elif kind == "wild":
        yield from _wild(value)
    elif kind == "index":
        yield from _index(value, step[1])
    elif kind == "idxwild":
        yield from _idxwild(value)
    elif kind == "slice":
        yield from _slice(value, step[1], step[2], step[3])
    elif kind == "filter":
        yield from _filter(step[1], value)
    else:  # desc is handled by _descend, never here
        raise AssertionError(f"unexpected step {step!r}")


def _eval_filter(conjs, at) -> bool:
    """Evaluate all conjuncts (the oracle does NOT short-circuit `&`: later
    conjuncts still run and their TypeErrors propagate) and AND the results.
    The comparison cross product is likewise fully evaluated."""
    all_ok = True
    for i, conj in enumerate(conjs):
        if conj[0] == "exists":
            if i == 0 and len(conjs) > 1:
                # `?(@.a & ...)`: existence as the LEFT operand of `&`
                raise NotImplementedError()
            ok = _arith_has_path(conj[1]) and bool(_eval_arith_safe(conj[1], at))
        else:
            _, left, op, right = conj
            lefts = _eval_arith_safe(left, at)
            rights = _eval_arith_safe(right, at)
            ok = False
            for lv in lefts:
                for rv in rights:
                    if _compare(lv, op, rv):
                        ok = True
        all_ok = all_ok and ok
    return all_ok


def _eval_arith(arith, at) -> list:
    kind = arith[0]
    if kind in ("int", "dbl", "str"):
        return [arith[1]]
    if kind == "path":
        values = [at]
        for fpstep in arith[1]:
            nxt = []
            for node in values:
                nxt.extend(_apply_fpath(fpstep, node))
            values = nxt
        return values
    # binary arithmetic: cross product. A pair that would raise in Python
    # poisons the WHOLE expression (see _eval_arith_safe). Note that path
    # evaluation errors (TypeError/KeyError/IndexError/ValueError from index
    # and slice steps) are raised OUTSIDE this try and propagate, exactly
    # like the oracle.
    lefts = _eval_arith(arith[1], at)
    rights = _eval_arith(arith[2], at)
    out = []
    for lv in lefts:
        for rv in rights:
            try:
                if kind == "add":
                    out.append(lv + rv)
                elif kind == "sub":
                    out.append(lv - rv)
                else:
                    out.append(lv * rv)
            except (TypeError, ValueError):
                raise _ArithPoison from None
    return out


class _ArithPoison(Exception):
    """An arithmetic pair raised TypeError/ValueError: the whole operand
    yields no values (the oracle treats the conjunct as a non-match)."""


def _eval_arith_safe(arith, at) -> list:
    """Evaluate an arith operand; an arithmetic TypeError/ValueError anywhere
    in it makes the whole operand yield no values. Errors from PATH steps
    (indexing scalars, out-of-range negatives, ...) propagate."""
    try:
        return _eval_arith(arith, at)
    except _ArithPoison:
        return []


def _apply_fpath(step, value):
    kind = step[0]
    if kind == "field":
        return [v for v, _ in _child(value, (step[1],))]
    if kind == "union":
        return [v for v, _ in _child(value, step[1])]
    if kind == "wild":
        return [v for v, _ in _wild(value)]
    if kind == "index":
        return [v for v, _ in _index(value, step[1])]
    if kind == "slice":
        return [v for v, _ in _slice(value, step[1], step[2], step[3])]
    if kind == "idxwild":
        return [v for v, _ in _idxwild(value)]
    raise AssertionError(f"unexpected filter-path step {step!r}")


def _compare(left, op, right) -> bool:
    """One filter comparison under the oracle's coercion contract."""
    if type(right) is int:
        # int() coercion of the left operand; ValueError is caught (no
        # match), TypeError propagates (CPython's message, like the oracle)
        try:
            lv = int(left)
        except ValueError:
            return False
        if op == "==":
            return lv == right
        if op == "!=":
            return lv != right
        if op == "<":
            return lv < right
        if op == "<=":
            return lv <= right
        if op == ">":
            return lv > right
        return lv >= right
    # float / str (incl. bare-word) right: direct comparison; mixed-type
    # ordering raises TypeError with CPython's message, like the oracle
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right


def evaluate(plan, data):
    """Evaluate a parsed plan; return [(value, path_steps), ...]."""
    current = [(data, ())]
    for step in plan:
        kind = step[0]
        nxt = []
        if kind == "desc":
            for value, path in current:
                _descend(step[1], value, path, nxt)
        else:
            for value, path in current:
                for val, pstep in _apply(step, value):
                    nxt.append((val, path + (pstep,)))
        current = nxt
    return current


def find(plan, data):
    """Evaluate a parsed plan and return ``[(value, rendered_path), ...]``."""
    return [(value, render_path(path)) for value, path in evaluate(plan, data)]
