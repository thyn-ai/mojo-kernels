"""`python -m jinja2_mojo` — quickstart smoke for end users and CI.

Compiles and renders a handful of templates covering the supported corpus
(variables, if/for/set, include via DictLoader, macros, filters, comments,
whitespace control), prints the outputs and a checksum, and reports the
active backend. Both the native and the forced-fallback
(JINJA2_MOJO_DISABLE_NATIVE=1) runs must print the same checksum.
"""

from __future__ import annotations

import jinja2
import jinja2_mojo

QUICKSTART = [
    ("Hello {{ name }}!", {"name": "World"}),
    ("{% for i in items %}{{ i }}{% if not loop.last %}, {% endif %}{% endfor %}",
     {"items": [1, 2, 3]}),
    ("{% set x = 2 * 21 %}{{ x }}", {}),
    ("{% macro badge(t) %}<b>{{ t|upper }}</b>{% endmacro %}{{ badge('hi') }}", {}),
    ("a  \n  {%- if ok %}trimmed{% endif -%}  \n b", {"ok": True}),
    ("{% include 'row' %}", {"v": 7}),
    ("{# a comment #}{{ 'x\\ty' ~ \" \" ~ 'z' }}", {}),
    ("{{ [1, 2, 3]|sum }} {{ 0x10 }} {{ 2.5 * 4 }}", {}),
]

LOADER = jinja2.DictLoader({"row": "<td>{{ v }}</td>"})


def main() -> None:
    outputs = []
    for src, ctx in QUICKSTART:
        t = jinja2_mojo.compile_template(src, loader=LOADER)
        outputs.append(t.render(ctx))
    for o in outputs:
        print(o)
    checksum = sum(sum(map(ord, o)) for o in outputs)
    print(f"checksum: {checksum}")
    info = jinja2_mojo.backend_info()
    print(f"native: {info['native_available']} ({info.get('native_source') or info.get('error')})")


if __name__ == "__main__":
    main()
