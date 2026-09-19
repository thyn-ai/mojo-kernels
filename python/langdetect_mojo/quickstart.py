"""End-user quickstart for the langdetect-mojo wheel (also used by CI smoke).

Prints detection results for a handful of texts plus a deterministic checksum
line, so CI can compare the native and forced-fallback outputs of the same
installed wheel.
"""

from __future__ import annotations

import langdetect_mojo

TEXTS = [
    "This is a sample English sentence for language detection.",
    "Bonjour le monde, ceci est un texte francais pour le test.",
    "これは日本語のテキストです。言語検出のテストをしています。",
    "Dies ist ein deutscher Beispielsatz für die Spracherkennung.",
]


def main() -> None:
    print(f"langdetect_mojo {langdetect_mojo.__version__}")
    info = langdetect_mojo.backend_info()
    print(f"backend: {'native' if info['native_available'] else 'fallback'}")
    checksum = 0.0
    for text in TEXTS:
        langs = langdetect_mojo.detect_langs(text)
        print(f"{langdetect_mojo.detect(text)} <- {text[:48]!r}")
        print(f"  {langs}")
        checksum += sum(item.prob for item in langs)
    print(f"detection checksum: {checksum:.15e}")


if __name__ == "__main__":
    main()
