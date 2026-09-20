#!/usr/bin/env python3
"""Compose the body of a GitHub Release from CHANGELOG.md.

The body of the Release for version X.Y.Z is that version's entry in
CHANGELOG.md -- the ``## [X.Y.Z]...`` heading (release-please writes
``## [X.Y.Z](compare-url) (YYYY-MM-DD)``, the hand-written 0.1.0 entry
``## [0.1.0] - 2026-09-20``; both match) and everything up to the next
``## `` heading, minus the link-reference definitions Keep a Changelog puts
at the end of the file -- followed by GitHub's generated "What's Changed"
list (the ``generate-notes`` API) when one is supplied.

On the supported path release-please publishes the Release, with the
CHANGELOG entry as its notes, before ``release.yml`` runs, and those notes
are kept. This composer supplies the body everywhere there are no such
notes -- a Release whose body is blank, a same-tag draft, a hand-pushed tag
with no release -- so a draft's body is never what gets published. The
composed body contains the entry's heading whenever there is an entry to
compose from: that is what both subcommands check.

    release_notes.py check   --changelog CHANGELOG.md --version 0.1.0
    release_notes.py compose --changelog CHANGELOG.md --version 0.1.0 \\
        [--generated generated.md] [--allow-missing-section] --out notes.md

``check`` prints the heading it found and exits 1 with a message on stderr
when there is no entry for the version or the entry is empty; ``preflight``
runs it before anything is built. ``compose`` writes the body and the
``release`` job runs it with ``--allow-missing-section``: a release is not
worth blocking on a changelog, because release-please publishes the Release
*before* this workflow runs, so refusing to build would leave an
already-published Release without its assets, signatures and provenance --
permanently, since a re-run builds the tag's own tree and would refuse
again. With that flag a missing or empty entry degrades to the generated
notes alone, or to a one-line body when there are none either, and says so
on stderr; it never fails. Standard library only, Python 3.9+.

Run the tests with ``python3 -m unittest discover -s tests -p test_release_notes.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A Keep a Changelog version heading: `## [0.1.0] - 2026-09-20`, `## [0.2.0-rc.1]`,
# `## [Unreleased]`. Group 1 is what is inside the brackets.
HEADING_RE = re.compile(r"^## \[([^\]]+)\](.*)$")
# Any second-level heading ends a section, whether or not it is a version.
SECTION_END_RE = re.compile(r"^## ")
# A link-reference definition line: `[0.1.0]: https://...`. Keep a Changelog
# collects the version links in one block at the very end of the file, which
# makes them the tail of the last section; only that trailing block is
# dropped. A definition that a section's own text refers to
# (`[text][bug-42]` ... `[bug-42]: https://...`) is content and stays.
LINK_DEFINITION_RE = re.compile(r"^\[[^\]]+\]:\s+\S+\s*$")


class ChangelogError(ValueError):
    """CHANGELOG.md has no usable section for the requested version."""


def find_heading(changelog: str, version: str) -> str:
    """Return the heading line of the section for `version`."""
    section = extract_section(changelog, version)
    return section.splitlines()[0]


def extract_section(changelog: str, version: str) -> str:
    """Return the CHANGELOG.md section for `version`, heading included.

    The section runs from its heading to the line before the next `## `
    heading (or the end of the file). A block of link-reference definitions
    that ends the section (the file-level `[X.Y.Z]: https://...` list Keep a
    Changelog puts last) is dropped, as are surrounding blank lines; a
    definition followed by more content is part of the section and kept.
    Raises ChangelogError if there is no heading for exactly this version or
    the section has no content.
    """
    if version.lower() == "unreleased":
        raise ChangelogError("the [Unreleased] section is not a release; cut it as a [X.Y.Z] section first")
    lines = changelog.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match and match.group(1) == version:
            start = index
            break
    if start is None:
        raise ChangelogError(f"CHANGELOG.md has no `## [{version}]` section")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if SECTION_END_RE.match(lines[index]):
            end = index
            break
    body = _drop_trailing_link_definitions(lines[start + 1 : end])
    body_text = "\n".join(body).strip("\n")
    if not body_text.strip():
        raise ChangelogError(f"the `## [{version}]` section of CHANGELOG.md is empty")
    return lines[start].rstrip() + "\n\n" + body_text + "\n"


def _drop_trailing_link_definitions(body: list[str]) -> list[str]:
    """Remove the link-reference definitions (and blank lines) that end `body`.

    Scans backwards from the end while every line is blank or a definition,
    so only the trailing block goes; the first line of real content stops the
    scan and everything before it, definitions included, is kept.
    """
    end = len(body)
    while end > 0 and (not body[end - 1].strip() or LINK_DEFINITION_RE.match(body[end - 1])):
        end -= 1
    return body[:end]


def compose(
    changelog: str,
    version: str,
    generated: str | None = None,
    require_section: bool = True,
) -> str:
    """Compose the Release body: the CHANGELOG entry, then the generated notes.

    `generated` is the body GitHub's generate-notes API returned (or None /
    blank, in which case the entry stands alone). When the entry is present
    the result always contains its heading; that invariant is asserted here
    so a caller cannot publish a body without it.

    `require_section=False` (the `release` job) never raises for a missing
    or empty entry, so a release is never blocked on the changelog: the body
    degrades to the generated notes alone, or, when there are none either,
    to one line naming the release and saying why it carries no notes. Both
    are reported on stderr.
    """
    try:
        section = extract_section(changelog, version)
    except ChangelogError as error:
        if require_section:
            raise
        if generated and generated.strip():
            print(
                f"release_notes: warning: {error}; composing from the generated notes alone",
                file=sys.stderr,
            )
            return generated.strip("\n") + "\n"
        print(
            f"release_notes: warning: {error}, and there are no generated notes either; "
            "writing a one-line body",
            file=sys.stderr,
        )
        return (
            f"Release {version}. No `## [{version}]` entry was found in CHANGELOG.md at "
            "the tagged commit and GitHub generated no notes, so this release carries "
            "no notes; the assets, signatures and provenance below are unaffected.\n"
        )
    # The entry leads the body by construction; the tests assert it.
    parts = [section.rstrip("\n")]
    if generated and generated.strip():
        parts.append(generated.strip("\n"))
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="verify CHANGELOG.md has a non-empty section for the version")
    check.add_argument("--changelog", type=Path, required=True)
    check.add_argument("--version", required=True, help="X.Y.Z, without the leading v")

    comp = sub.add_parser("compose", help="write the Release body for the version")
    comp.add_argument("--changelog", type=Path, required=True)
    comp.add_argument("--version", required=True, help="X.Y.Z, without the leading v")
    comp.add_argument("--generated", type=Path, help="file holding GitHub's generated notes (optional)")
    comp.add_argument(
        "--allow-missing-section",
        action="store_true",
        help="compose from the generated notes alone instead of failing when CHANGELOG.md "
             "has no entry for the version (what the release job passes)",
    )
    comp.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    changelog = args.changelog.read_text(encoding="utf-8")
    try:
        if args.command == "check":
            print(find_heading(changelog, args.version))
        else:
            generated = args.generated.read_text(encoding="utf-8") if args.generated else None
            body = compose(
                changelog, args.version, generated,
                require_section=not args.allow_missing_section,
            )
            args.out.write_text(body, encoding="utf-8")
            print(f"wrote {args.out} ({len(body.splitlines())} lines; heading: {body.splitlines()[0]})")
    except ChangelogError as error:
        print(f"release_notes: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
