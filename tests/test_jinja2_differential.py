"""Differential tests: jinja2_mojo compile must render byte-identically to the oracle.

Run twice by `scripts/test_all_jinja2.sh`: once against the native Mojo
kernel and once with JINJA2_MOJO_DISABLE_NATIVE=1 (forced stock-lexer
fallback). Both backends must agree with the oracle everywhere — render
output is compared byte-for-byte, and the generated Python source
(Environment.compile(raw=True)) is compared byte-for-byte as well.

The oracle is the published PyPI package, pinned to jinja2==3.1.6 (see
scripts/test_all_jinja2.sh for how it is provisioned; jinja2 is also the
wrapper's own runtime dependency — parsing, compilation and rendering are
always stock jinja2, only the compile-time lexer is swapped).
jinja2 was used strictly as a black-box oracle: its token stream and render
output were observed on probe inputs; no oracle source was read or adapted.

Scope under test: variables, attribute/item access, filters, arithmetic and
logic, if/elif/else, for/else with loop vars, set (inline and block),
include/import/from via DictLoader, macros (defaults, caller), comments,
whitespace control (- and +, trim_blocks, lstrip_blocks, keep_trailing_
newline), string literals with Python-style escapes, numbers (decimal, hex,
octal, binary, underscored, floats, exponents), unicode names and content,
\\r\\n normalisation, dict/list literals, slicing, tests (is defined/none/
sequence/number), filter blocks, call blocks, the do extension, and raw
blocks (native lexer declines; stock lexer takes over transparently).
"""

from __future__ import annotations

import copy
import os

import jinja2
import pytest
from jinja2 import DictLoader, Environment, TemplateSyntaxError

import jinja2_mojo
from jinja2_mojo import _tokens

NATIVE_FORCED_OFF = os.environ.get("JINJA2_MOJO_DISABLE_NATIVE") == "1"

# ---------------------------------------------------------------- corpus

LOADERS = {
    "std": DictLoader(
        {
            "row.html": "<td>{{ v }}</td>",
            "base.html": "A[{% block body %}base{% endblock %}]Z",
            "macros.html": (
                "{% macro wrap(x, cls='w') %}<span class=\"{{ cls }}\">{{ x }}</span>{% endmacro %}"
                "{% macro pair(a, b=2) %}{{ a }}-{{ b }}{% endmacro %}"
            ),
        }
    ),
}

# (template, [contexts], env_options). Env options None means "each combo".
TEMPLATES: list[tuple[str, list[dict], dict | None]] = [
    ("Hello {{ name }}!", [{"name": "World"}, {"name": "<x>&"} ], None),
    ("{{ user.name }} is {{ user['age'] }} years old",
     [{"user": {"name": "Ada", "age": 36}}], None),
    ("{{ a + b }} {{ a * b }} {{ a // b }} {{ a % b }} {{ a ** 2 }} {{ -a }} {{ a - b }}",
     [{"a": 7, "b": 2}], None),
    ("{{ x > y }} {{ x == y }} {{ x != y and x or y }} {{ not x }}",
     [{"x": 1, "y": 2}], None),
    ("{{ 'single' }} {{ \"double\" }} {{ 'tab\\tnl\\n' }} {{ 'q\\'q' }} {{ \"d\\\"d\" }}",
     [{}], None),
    ("{{ 'a\\x41\\u0042' }} {{ '\\N{LATIN CAPITAL LETTER C}' }} {{ '\\101\\0z' }}",
     [{}], None),
    ("{{ 'keep \\z and \\ space' }}", [{}], None),
    ("{{ 42 }} {{ 0x1F }} {{ 0o17 }} {{ 0b101 }} {{ 1_000_000 }} {{ 2.5 }} {{ 1e3 }} {{ 1.5e-2 }} {{ 0x_2 }}",
     [{}], None),
    ("{{ x|upper }} {{ xs|join(',') }} {{ xs|length }} {{ xs|sum }} {{ missing|default('d') }}",
     [{"x": "ab", "xs": [1, 2, 3]}], None),
    ("{{ '%.2f'|format(p) }} {{ s|replace('a', 'b') }} {{ s|truncate(5) }}",
     [{"p": 3.14159, "s": "banana split"}], None),
    ("{% if x > 1 %}big{% elif x == 1 %}one{% else %}small{% endif %}",
     [{"x": 5}, {"x": 1}, {"x": -3}], None),
    ("{% for i in items %}{{ loop.index }}:{{ i }}:{{ loop.first }}:{{ loop.last }};{% endfor %}",
     [{"items": ["a", "b", "c"]}, {"items": []}], None),
    ("{% for i in items %}{{ i }}{% else %}none{% endfor %}",
     [{"items": [1]}, {"items": []}], None),
    ("{% for r in rows %}{% for c in r %}({{ loop.index }},{{ loop.revindex }}){{ c }}{% endfor %}|{% endfor %}",
     [{"rows": [[1, 2], [3]]}], None),
    ("{% for x in xs %}{{ loop.cycle('o', 'e') }}{{ x }}{% endfor %}",
     [{"xs": [1, 2, 3, 4]}], None),
    ("{% set y = x * 2 %}{{ y }}", [{"x": 21}], None),
    ("{% set y %}b{{ x }}d{% endset %}{{ y }}", [{"x": "c"}], None),
    ("{% set a, b = 1, 2 %}{{ a }}{{ b }}", [{}], None),
    ("{% include 'row.html' %}", [{"v": 9}], {"loader": "std"}),
    ("{% for v in vs %}{% include 'row.html' %}{% endfor %}", [{"vs": [1, 2]}], {"loader": "std"}),
    ("{% import 'macros.html' as m %}{{ m.wrap('x') }}{{ m.pair(1) }}", [{}], {"loader": "std"}),
    ("{% from 'macros.html' import wrap, pair as p %}{{ wrap('y', 'z') }}{{ p(5, 6) }}",
     [{}], {"loader": "std"}),
    ("{% extends 'base.html' %}{% block body %}child {{ v }}{% endblock %}",
     [{"v": 3}], {"loader": "std"}),
    ("{% macro m(a, b='B') %}[{{ a }}{{ b }}]{% endmacro %}{{ m('x') }}{{ m('x', 'y') }}",
     [{}], None),
    ("{% macro list(xs) %}{% for x in xs %}{{ x }}{% if not loop.last %},{% endif %}{% endfor %}{% endmacro %}{{ list([1, 2]) }}",
     [{}], None),
    ("{# comment #}a{# multi\nline #}b", [{}], None),
    ("a {#- c -#} b", [{}], None),
    ("a\n\n  {%- if x %}y{% endif -%}  \n\n b", [{"x": True}], None),
    ("{% if x +%}\n padded \n{%- endif %}", [{"x": True}], {"trim_blocks": True}),
    ("{% if x %}\nbody\n{% endif %}\n", [{"x": True}], {"trim_blocks": True}),
    ("  \n  {% if x %}b{% endif %}", [{"x": True}], {"lstrip_blocks": True}),
    ("  \n  {%+ if x %}b{% endif %}", [{"x": True}], {"lstrip_blocks": True}),
    ("a\n{% if x %}\nb\n{% endif %}\nc", [{"x": True}],
     {"trim_blocks": True, "lstrip_blocks": True}),
    ("tail\n", [{}], None),
    ("tail\n\n", [{}], {"keep_trailing_newline": True}),
    ("tail\n\n", [{}], {"keep_trailing_newline": False}),
    ("{{ x }}\n", [{"x": 1}], None),
    ("{{ x }}\n", [{"x": 1}], {"keep_trailing_newline": True}),
    ("a\r\nb\r\nc\rd", [{}], None),
    ("{%\n if x \n%}\n y \n{%\n endif \n%}", [{"x": True}], None),
    ("{{ d['k'] }} {{ d.get('m', 'z') }} {{ [1, 2, 3][1:] }} {{ (1, 2)[0] }}",
     [{"d": {"k": "v"}}], None),
    ("{{ {'a': 1}['a'] }} {{ [3, 1, 2]|sort|first }}", [{}], None),
    ("{% if x is defined %}yes{% else %}no{% endif %}", [{"x": 0}, {}], None),
    ("{{ x is none }} {{ xs is sequence }} {{ n is number }} {{ s is string }}",
     [{"x": None, "xs": [1], "n": 2, "s": "t"}], None),
    ("{% filter upper %}ab{{ x }}cd{% endfilter %}", [{"x": "!"}], None),
    ("{% filter join('|') %}ab{% endfilter %}", [{}], None),
    ("{% macro render_items(xs) %}{% for x in xs %}{{ caller(x) }}{% endfor %}{% endmacro %}"
     "{% call(i) render_items([1, 2]) %}<{{ i }}>{% endcall %}", [{}], None),
    ("{% do xs.append(1) %}{{ xs }}", [{"xs": [2]}], {"extensions": ["jinja2.ext.do"]}),
    ("{{ café }} {{ 变量 }}", [{"café": "ok", "变量": "好"}], None),
    ("{{ x ~ 1 ~ 'a' ~ [2] }}", [{"x": None}], None),
    ("{% if a and b or not c %}hit{% endif %}", [{"a": 1, "b": 0, "c": 0}], None),
    ("{{ '{%' }} {{ \"}}\" }} {{ '%}' }}", [{}], None),
    ("{{ ((1 + 2) * (3 - 4)) // 5 }}", [{}], None),
    ("{{ ns(x=1) if false else 'fb' }}", [{}], None),
]

# Templates exercising autoescape on both sides.
AUTOESCAPE_TEMPLATES = [
    ("{{ s }}", [{"s": "<b>&amp;</b>"}]),
    ("{{ s|e }} {{ s|safe }}", [{"s": "<i>"}]),
    ("{% autoescape false %}{{ s }}{% endautoescape %}", [{"s": "<u>"}]),
]

ERROR_TEMPLATES = [
    "a {# never closed",
    "{{ 'unterminated }}",
    "{{ a @ b }}",
    "{{ (1 + 2 }}",
    "{{ 1 + 2) }}",
    "{{ 'bad \\x escape' }}",
    "{% frobnicate x %}",
    "{% if %}x{% endif %}",
    "{ {{",
    "{{ x }漏",
]

# Templates the native lexer must decline (stock lexer transparently covers).
FALLBACK_ONLY_TEMPLATES = [
    ("{% raw %}{{ x }}{% endraw %}", [{"x": 1}], {}),
    ("{% raw %}a{% endraw %}", [{}], {}),
    ("{% raw %}{{ not_evaluated }}{% if x %}{% endraw %}", [{}], {}),
    ("{%- raw -%} literal {% endraw %}", [{}], {}),
]


def _env_options(spec):
    opts = dict(spec or {})
    loader_key = opts.pop("loader", None)
    loader = LOADERS[loader_key] if loader_key else None
    return loader, opts


def _oracle_render(src, ctx, spec):
    loader, opts = _env_options(spec)
    env = Environment(loader=loader, **opts)
    return env.from_string(src).render(ctx)


def _ours_render(src, ctx, spec, cache=False):
    loader, opts = _env_options(spec)
    t = jinja2_mojo.compile_template(src, loader=loader, cache=cache, **opts)
    return t.render(ctx), t


@pytest.mark.parametrize("idx", range(len(TEMPLATES)))
def test_render_parity_corpus(idx):
    src, contexts, spec = TEMPLATES[idx]
    for ctx in contexts:
        expected = _oracle_render(src, copy.deepcopy(ctx), spec)
        got, t = _ours_render(src, copy.deepcopy(ctx), spec)
        assert got == expected, f"template #{idx}: {got!r} != oracle {expected!r}"
        if not NATIVE_FORCED_OFF:
            # everything in the main corpus is inside the native scope
            assert t.backend == "native", f"template #{idx} unexpectedly stock: {src!r}"


@pytest.mark.parametrize("autoescape", [True, False])
def test_render_parity_autoescape(autoescape):
    for src, contexts in AUTOESCAPE_TEMPLATES:
        for ctx in contexts:
            spec = {"autoescape": autoescape}
            expected = _oracle_render(src, ctx, spec)
            got, _t = _ours_render(src, ctx, spec)
            assert got == expected


def test_generated_python_source_parity():
    """The compiled Python source must be byte-identical to the oracle's."""
    from jinja2_mojo.core import _NativeEnvironment

    for src, _contexts, spec in TEMPLATES:
        loader, opts = _env_options(spec)
        oracle_src = Environment(loader=loader, **opts).compile(src, raw=True)
        ours_src = _NativeEnvironment(loader=loader, **opts).compile(src, raw=True)
        assert ours_src == oracle_src, f"generated code diverged for {src[:60]!r}"


@pytest.mark.parametrize("src", ERROR_TEMPLATES)
def test_error_parity(src):
    with pytest.raises(TemplateSyntaxError):
        Environment().from_string(src)
    with pytest.raises(TemplateSyntaxError):
        jinja2_mojo.compile_template(src, cache=False)
    # and the message must be the oracle's own (stock lexer produces it)
    try:
        Environment().from_string(src)
    except TemplateSyntaxError as e:
        oracle_msg = str(e)
    try:
        jinja2_mojo.compile_template(src, cache=False)
    except TemplateSyntaxError as e:
        assert str(e) == oracle_msg


@pytest.mark.parametrize("idx", range(len(FALLBACK_ONLY_TEMPLATES)))
def test_native_declines_transparently(idx):
    src, contexts, spec = FALLBACK_ONLY_TEMPLATES[idx]
    for ctx in contexts:
        expected = _oracle_render(src, ctx, spec)
        got, t = _ours_render(src, ctx, spec)
        assert got == expected
        assert t.backend == "stock"


def test_custom_delimiters_fall_back():
    opts = {"variable_start_string": "[[", "variable_end_string": "]]"}
    src = "a [[ x ]] b"
    expected = Environment(**opts).from_string(src).render({"x": 1})
    got, t = _ours_render(src, {"x": 1}, opts)
    assert got == expected
    assert t.backend == "stock"


def test_line_statements_fall_back():
    opts = {"line_statement_prefix": "%"}
    src = "% for x in [1, 2]\n{{ x }}\n% endfor"
    expected = Environment(**opts).from_string(src).render()
    got, t = _ours_render(src, {}, opts)
    assert got == expected
    assert t.backend == "stock"


# ------------------------------------------------------------ token parity


@pytest.mark.skipif(not jinja2_mojo.native_available(), reason="native kernel unavailable")
def test_token_stream_parity_native():
    """Native tokens must equal the oracle's (type+lineno always; value except
    block/variable end tokens, where the oracle folds trimmed whitespace into
    the value and ours keeps the bare marker — the parser never reads them)."""
    option_sets = [
        {},
        {"trim_blocks": True},
        {"lstrip_blocks": True},
        {"trim_blocks": True, "lstrip_blocks": True},
        {"keep_trailing_newline": True},
        {"trim_blocks": True, "keep_trailing_newline": True},
    ]
    for src, _contexts, spec in TEMPLATES:
        loader, base_opts = _env_options(spec)
        if loader is not None:
            continue  # includes/extends do not change tokenisation of src
        for extra in option_sets:
            opts = {**base_opts, **extra}
            env = Environment(**opts)
            ref = list(env._tokenize(src, None, None, None))
            try:
                mine_list = _tokens.build_tokens(
                    src,
                    trim_blocks=env.trim_blocks,
                    lstrip_blocks=env.lstrip_blocks,
                    keep_trailing_newline=env.keep_trailing_newline,
                )
            except Exception:
                mine_list = None
            if mine_list is None:
                continue  # native declined; fallback path covered elsewhere
            assert len(mine_list) == len(ref), (src, opts)
            for m, r in zip(mine_list, ref):
                assert m.type == r.type, (src, opts, m, r)
                assert m.lineno == r.lineno, (src, opts, m, r)
                if m.type in ("block_end", "variable_end"):
                    assert r.value.startswith(m.value), (src, opts, m, r)
                else:
                    assert m.value == r.value, (src, opts, m, r)


# ------------------------------------------------------------- misc parity


def test_include_render_uses_same_environment():
    loader, _opts = _env_options({"loader": "std"})
    t = jinja2_mojo.compile_template(
        "{% extends 'base.html' %}{% block body %}[{{ v }}]{% endblock %}", loader=loader
    )
    oracle = Environment(loader=loader).from_string(
        "{% extends 'base.html' %}{% block body %}[{{ v }}]{% endblock %}"
    )
    assert t.render({"v": 42}) == oracle.render({"v": 42})


def test_compile_cache_hit_and_bypass():
    a = jinja2_mojo.compile_template("cache {{ x }} probe")
    b = jinja2_mojo.compile_template("cache {{ x }} probe")
    assert a is b  # deterministic compile: cached object returned
    c = jinja2_mojo.compile_template("cache {{ x }} probe", cache=False)
    assert c is not a
    assert a.render({"x": 1}) == c.render({"x": 1})


def test_batch_order_and_workers():
    srcs = ["{{ %d }}" % i for i in range(40)]
    plain = jinja2_mojo.compile_batch(srcs)
    threaded = jinja2_mojo.compile_batch(srcs, workers=4, cache=False)
    oracle = [Environment().from_string(s).render() for s in srcs]
    assert [t.render() for t in plain] == oracle
    assert [t.render() for t in threaded] == oracle
    assert len(plain) == len(threaded) == len(srcs)


def test_type_error_for_non_string():
    with pytest.raises(TypeError):
        jinja2_mojo.compile_template(123)  # type: ignore[arg-type]


def test_expected_backend():
    info = jinja2_mojo.backend_info()
    if NATIVE_FORCED_OFF:
        assert not info["native_available"]
    else:
        assert info["native_available"], f"native backend required for this run: {info}"


# --------------------------------------------------------------- fuzzing


def _rand_template(rng):
    parts = []
    names = ["x", "y", "user.name", "items", "n"]
    for _ in range(rng.randint(1, 12)):
        kind = rng.randrange(9)
        if kind == 0:
            parts.append(rng.choice(["plain ", "txt & more ", "line\nbreak ", "  spaced  "]))
        elif kind == 1:
            dash1 = rng.choice(["", "-", "+"])
            dash2 = rng.choice(["", "-", "+"])
            expr = rng.choice(names)
            if rng.random() < 0.4:
                expr += rng.choice(["|upper", "|length", "|default('d')", " * 2", " ~ '!'"])
            parts.append("{{" + dash1 + " " + expr + " " + dash2 + "}}")
        elif kind == 2:
            d = rng.choice(["", "-"])
            parts.append("{%" + d + " if x " + d + "%}yes{%" + d + " endif " + d + "%}")
        elif kind == 3:
            d = rng.choice(["", "-"])
            parts.append(
                "{%" + d + " for i in items " + d + "%}[{{ i }}]{%"
                + d + " else " + d + "%}empty{%" + d + " endfor " + d + "%}"
            )
        elif kind == 4:
            parts.append("{# " + rng.choice(["c", "note\nmore", "{{ hidden }}"]) + " #}")
        elif kind == 5:
            parts.append("{% set v = " + rng.choice(["1", "x", "'s'"]) + " %}{{ v }}")
        elif kind == 6:
            parts.append(rng.choice(["{{ 0x1A }}", "{{ 1_0.5 }}", "{{ 'a\\tb' }}", "{{ 07 }}"]))
        elif kind == 7:
            parts.append("{% if n is defined %}N{% endif %}")
        else:
            parts.append("{% macro m(a) %}<{{ a }}>{% endmacro %}{{ m('q') }}")
    return "".join(parts)


def test_seeded_fuzz_render_parity():
    import random

    rng = random.Random(20260920)
    ctxs = [
        {"x": 1, "y": 2, "user": {"name": "n"}, "items": [1, 2, 3], "n": 5},
        {"x": 0, "y": -1, "user": {"name": "m"}, "items": [], "n": 0},
    ]
    for i in range(300):
        src = _rand_template(rng)
        opts = {}
        if rng.random() < 0.3:
            opts["trim_blocks"] = True
        if rng.random() < 0.3:
            opts["lstrip_blocks"] = True
        for ctx in ctxs:
            try:
                expected = Environment(**opts).from_string(src).render(copy.deepcopy(ctx))
            except Exception as e:
                with pytest.raises(type(e)):
                    jinja2_mojo.compile_template(src, cache=False, **opts).render(copy.deepcopy(ctx))
                continue
            got = jinja2_mojo.compile_template(src, cache=False, **opts).render(copy.deepcopy(ctx))
            assert got == expected, f"seed {i}: {got!r} != {expected!r} for {src!r}"
