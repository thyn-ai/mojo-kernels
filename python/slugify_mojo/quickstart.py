"""slugify-mojo quickstart: exercise the package end to end.

Run with the installed package (see python/slugify_mojo/README.md):

    python quickstart.py

Prints a deterministic checksum so CI can assert the native and forced
fallback backends produce byte-identical output.
"""

from __future__ import annotations

import hashlib
import os
import sys

# When this file is invoked by path (python .../slugify_mojo/quickstart.py),
# sys.path[0] is the package source directory, which would shadow the
# installed wheel with the repository source. Pop it so the quickstart always
# exercises the installed wheel (the end-user path).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _SCRIPT_DIR:
    sys.path.pop(0)

import slugify_mojo  # noqa: E402  (import after the sys.path fix is the point)

TEXTS = [
    "Hello World, This is a Test!",
    "Déjà Vu — Café naïve résumé",
    "Москва столица России",
    "日本語のテキストです",
    "Fish &amp; Chips &lt;tag&gt; &#65;&#x42;",
    "emoji 😀🎉 are dropped",
    "1,000,000 dollar question's",
    "Καλημέρα κόσμε, ϗamp; entities",
]


def main() -> None:
    info = slugify_mojo.backend_info()
    print(f"slugify-mojo {slugify_mojo.__version__} "
          f"(backend: {'native' if info['native_available'] else 'fallback'})")

    digest = hashlib.sha256()
    for text in TEXTS:
        slug = slugify_mojo.slugify(text)
        print(f"  {text!r} -> {slug!r}")
        digest.update(slug.encode("utf-8"))

    column = slugify_mojo.slugify_column(TEXTS[:4])
    for slug in column:
        digest.update(slug.encode("utf-8"))
    print(f"  column: {column}")

    options = slugify_mojo.slugify(
        "A very long title indeed", max_length=16, word_boundary=True, separator="_"
    )
    digest.update(options.encode("utf-8"))
    print(f"  options: {options!r}")
    print(f"slug checksum: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
