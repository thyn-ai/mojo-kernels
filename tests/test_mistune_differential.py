"""Differential tests: mistune_mojo must match mistune.markdown byte-for-byte.

Run twice by `scripts/test_all_mistune.sh`: once against the native Mojo
kernel and once with MISTUNE_MOJO_DISABLE_NATIVE=1 (forced pure-Python
engine). Both backends must agree with the published `mistune` package
(3.3.4) byte-for-byte on every case.

Corpus:
1. The full CommonMark 0.31.2 spec corpus (tests/test_mistune_spec.json,
   652 cases, vendored from https://spec.commonmark.org/0.31.2/spec.json).
   Note the oracle is `mistune.markdown`, not the spec's expected HTML:
   mistune's default configuration deliberately deviates from CommonMark
   (no raw HTML, no entity decoding in text, code-block trailing newlines,
   `<li>` rendering); both engines mirror those deviations, and this suite
   pins agreement with mistune on every spec input.
2. A seeded corpus of generated documents (nested lists, tables-of-contents
   style links, code spans/fences, block quotes, reference links, emphasis
   edge cases, HTML-escaping cases).

The oracle is the published PyPI package, pinned to mistune==3.3.4.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

mistune = pytest.importorskip("mistune", reason="oracle package mistune not installed")

import mistune_mojo
from mistune_mojo import _reference

SPEC_JSON = Path(__file__).with_name("test_mistune_spec.json")
SPEC_CASES = json.loads(SPEC_JSON.read_text())

if mistune.__version__ != "3.3.4":
    pytest.skip(
        f"oracle pin is mistune==3.3.4, found {mistune.__version__}",
        allow_module_level=True,
    )


def _expected_backend() -> str:
    return "fallback" if os.environ.get("MISTUNE_MOJO_DISABLE_NATIVE") == "1" else "native"


@pytest.mark.parametrize("case", SPEC_CASES, ids=[f"ex{c['example']}" for c in SPEC_CASES])
def test_spec_case_matches_mistune(case):
    """Both backends must be byte-identical to mistune.markdown."""
    want = mistune.markdown(case["markdown"])
    got = mistune_mojo.markdown(case["markdown"])
    assert got == want, (
        f"spec example {case['example']} ({case['section']}):\n"
        f"input: {case['markdown']!r}\nwant:  {want!r}\ngot:   {got!r}"
    )


def test_native_backend_used_when_enabled():
    """In the native run the kernel must actually serve renders."""
    if _expected_backend() == "native":
        assert mistune_mojo.native_available(), mistune_mojo.backend_info()


@pytest.mark.parametrize("case", SPEC_CASES, ids=[f"ex{c['example']}" for c in SPEC_CASES])
def test_backends_agree_with_each_other(case):
    """Native kernel and pure-Python engine agree on every input."""
    native_out = mistune_mojo.markdown(case["markdown"])
    fallback_out = _reference.markdown(case["markdown"])
    assert native_out == fallback_out, f"spec example {case['example']}"


def _generated_docs() -> list[str]:
    rng = random.Random(42)
    frags = [
        "# Title\n\n",
        "## Section with *emphasis*\n\n",
        "Some **bold** and _em_ text with `code` spans.\n\n",
        "- item one\n- item two with [a link](http://example.com)\n  - nested\n\n",
        "1. first\n2. second\n\n",
        "> a quote line\n> more quote\n\n",
        "```python\ndef f(x):\n    return x & 1\n```\n\n",
        "    indented code\n\n",
        "para with <div>inline html</div> and <script>alert()</script> tags.\n\n",
        '[ref link][r1] and [shortcut]\n\n[r1]: /url "title"\n[shortcut]: https://x.y/z\n\n',
        "Hard break  \nhere and soft\nbreak.\n\n",
        "***\n\n",
        "Text with &copy; entity and &#35; numeric.\n\n",
        '![image](/img.png "alt text")\n\n',
        "<div>\nblock html with **no emphasis**\n</div>\n\n",
        "Mixed *em **strong** em* end.\n\n",
        "Setext heading\n==============\n\n",
        "Another setext\n--------------\n\n",
        "- [ ] task\n- [x] done\n\n",
        "Autolinks <http://a.b/c?d=e&f=g> and <a@b.cd>.\n\n",
        "Escapes: \\*not em\\* and \\[not link\\].\n\n",
        "Line with trailing spaces   \nnext.\n\n",
        "Unicode text: привет, ὐ, ẞ, £50.\n\n",
        "Nested> quote\n> > deep\n\n",
        "- a\n  - b\n    - c\n      - d\n\n",
        "Tabs\there and\n\n\tindented\n",
        "[multi\nline link](/url \"multi\nline title\")\n\n",
    ]
    docs = []
    for _ in range(300):
        docs.append("".join(rng.choice(frags) for _ in range(rng.randint(1, 12))))
    return docs


GENERATED_DOCS = _generated_docs()


@pytest.mark.parametrize("doc", GENERATED_DOCS, ids=[f"doc{i}" for i in range(len(GENERATED_DOCS))])
def test_generated_docs_match_mistune(doc):
    want = mistune.markdown(doc)
    got = mistune_mojo.markdown(doc)
    assert got == want, f"input: {doc!r}\nwant: {want!r}\ngot:  {got!r}"


def test_empty_input():
    assert mistune_mojo.markdown("") == mistune.markdown("") == ""


def test_call_contract_matches_mistune():
    # None/"" -> "" on both; other non-str input fails identically
    for value in [None, ""]:
        assert mistune_mojo.markdown(value) == mistune.markdown(value)  # type: ignore[arg-type]
    for value, exc in [(0, AttributeError), (False, AttributeError), (123, AttributeError), (b"bytes", TypeError)]:
        with pytest.raises(exc):
            mistune_mojo.markdown(value)  # type: ignore[arg-type]
        with pytest.raises(exc):
            mistune.markdown(value)  # type: ignore[arg-type]


def test_crlf_and_cr():
    doc = "a\r\nb\r\nc\rd"
    assert mistune_mojo.markdown(doc) == mistune.markdown(doc)
