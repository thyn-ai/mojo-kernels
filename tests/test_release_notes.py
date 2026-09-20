"""Unit tests for scripts/release_notes.py, the composer of GitHub Release bodies.

Standard library only (``python3 -m unittest discover -s tests -p test_release_notes.py``):
the release workflow's ``preflight`` job runs this file with the runner's
interpreter before it checks CHANGELOG.md for the version being released.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release_notes.py"

spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(release_notes)

CHANGELOG = """\
# Changelog

All notable changes are documented here.

## [Unreleased]

### Fixed

- something not yet released

## [0.2.0] - 2026-10-01

### Added

- `foo-mojo` (Python) — a new kernel, measured 10x vs `foo` 1.0.0.

### Fixed

- `bm25-mojo`: a fix ([#15](https://github.com/thyn-ai/mojo-kernels/issues/15)).

## [0.1.0] - 2026-09-20

### Added

- the first release.

[Unreleased]: https://github.com/thyn-ai/mojo-kernels/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thyn-ai/mojo-kernels/releases/tag/v0.1.0
"""

GENERATED = """\
## What's Changed
* fix(bm25): evaluate a term in reference order by @0xamlab in https://github.com/thyn-ai/mojo-kernels/pull/28
* docs(contributing): the main ruleset requires every kernel workflow job by @0xamlab in https://github.com/thyn-ai/mojo-kernels/pull/25


**Full Changelog**: https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.2.0
"""

# The body the v0.1.0 release ended up with before it was repaired by hand:
# a placeholder generated when no pull request had landed since the notes
# were last written. It is the regression this composer exists to prevent,
# and it must never be what the composer produces.
PLACEHOLDER_BODY = "## What's Changed\n\n* No changes\n"


class ExtractSection(unittest.TestCase):
    def test_middle_section_runs_to_the_next_heading(self):
        section = release_notes.extract_section(CHANGELOG, "0.2.0")
        lines = section.splitlines()
        self.assertEqual(lines[0], "## [0.2.0] - 2026-10-01")
        self.assertIn("- `foo-mojo` (Python) — a new kernel, measured 10x vs `foo` 1.0.0.", lines)
        self.assertIn("- `bm25-mojo`: a fix ([#15](https://github.com/thyn-ai/mojo-kernels/issues/15)).", lines)
        # Nothing from the neighbouring sections leaks in.
        self.assertNotIn("## [0.1.0] - 2026-09-20", lines)
        self.assertNotIn("- the first release.", lines)
        self.assertNotIn("- something not yet released", lines)
        self.assertNotIn("## [Unreleased]", lines)

    def test_last_section_drops_the_link_reference_definitions(self):
        section = release_notes.extract_section(CHANGELOG, "0.1.0")
        self.assertEqual(section.splitlines()[0], "## [0.1.0] - 2026-09-20")
        self.assertIn("- the first release.", section)
        self.assertNotIn("[0.1.0]: https://", section)
        self.assertNotIn("[Unreleased]: https://", section)
        self.assertNotIn("[0.2.0]: https://", section)
        self.assertTrue(section.endswith("- the first release.\n"))

    def test_reference_style_links_used_inside_a_section_are_kept(self):
        changelog = (
            "## [1.1.0] - 2026-02-01\n\n"
            "### Fixed\n\n"
            "- a fix, see [the report][bug-42] and [the follow-up][bug-43].\n\n"
            "[bug-42]: https://github.com/thyn-ai/mojo-kernels/issues/42\n"
            "[bug-43]: https://github.com/thyn-ai/mojo-kernels/issues/43\n\n"
            "- another entry after the definitions\n\n"
            "## [1.0.0] - 2026-01-01\n\n- first\n\n"
            "[1.1.0]: https://github.com/thyn-ai/mojo-kernels/compare/v1.0.0...v1.1.0\n"
            "[1.0.0]: https://github.com/thyn-ai/mojo-kernels/releases/tag/v1.0.0\n"
        )
        section = release_notes.extract_section(changelog, "1.1.0")
        # Definitions the section's own text refers to are content.
        self.assertIn("[bug-42]: https://github.com/thyn-ai/mojo-kernels/issues/42", section)
        self.assertIn("[bug-43]: https://github.com/thyn-ai/mojo-kernels/issues/43", section)
        self.assertTrue(section.endswith("- another entry after the definitions\n"))
        # The last section still loses only the file-level block at its tail,
        # even when that block is the sole thing after its content.
        last = release_notes.extract_section(changelog, "1.0.0")
        self.assertEqual(last, "## [1.0.0] - 2026-01-01\n\n- first\n")

    def test_only_the_trailing_definition_block_is_dropped(self):
        changelog = (
            "## [1.0.0] - 2026-01-01\n\n"
            "- uses [a ref][r1] then ends\n\n"
            "[r1]: https://example.invalid/r1\n\n"
            "[1.0.0]: https://github.com/thyn-ai/mojo-kernels/releases/tag/v1.0.0\n"
        )
        # `[r1]` is referenced by the text but nothing but definitions follows
        # it, so it is part of the trailing block and goes with it: the rule
        # is positional (Keep a Changelog's file-level list), not semantic.
        section = release_notes.extract_section(changelog, "1.0.0")
        self.assertEqual(section, "## [1.0.0] - 2026-01-01\n\n- uses [a ref][r1] then ends\n")

    def test_heading_and_body_are_separated_by_one_blank_line(self):
        section = release_notes.extract_section(CHANGELOG, "0.2.0")
        self.assertTrue(section.startswith("## [0.2.0] - 2026-10-01\n\n### Added\n"))

    def test_version_must_match_exactly(self):
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(CHANGELOG, "0.2")
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(CHANGELOG, "v0.2.0")
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(CHANGELOG, "0.3.0")

    def test_unreleased_is_refused(self):
        for spelling in ("Unreleased", "unreleased"):
            with self.assertRaises(release_notes.ChangelogError):
                release_notes.extract_section(CHANGELOG, spelling)

    def test_empty_section_is_refused(self):
        changelog = "## [1.0.0] - 2026-01-01\n\n\n## [0.9.0] - 2025-12-01\n\n- old\n"
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(changelog, "1.0.0")
        # Link definitions alone do not count as content either.
        changelog = "## [1.0.0] - 2026-01-01\n\n[1.0.0]: https://example.invalid/v1.0.0\n"
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(changelog, "1.0.0")

    def test_release_please_heading_format(self):
        # release-type "simple" writes `## [X.Y.Z](compare-url) (date)`; the
        # first release, with nothing to compare against, gets `## X.Y.Z (date)`
        # only when there is no previous tag, which this repository has.
        changelog = (
            "# Changelog\n\n"
            "## [0.1.1](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.1.1) (2026-09-21)\n\n\n"
            "### Fixed\n\n"
            "* **bm25:** evaluate a BM25Plus term in reference order when its floor overflows ([#28](https://github.com/thyn-ai/mojo-kernels/issues/28)) ([1381151](https://github.com/thyn-ai/mojo-kernels/commit/1381151))\n\n"
            "## [0.1.0] - 2026-09-20\n\n### Added\n\n- first\n\n"
            "[0.1.0]: https://github.com/thyn-ai/mojo-kernels/releases/tag/v0.1.0\n"
        )
        section = release_notes.extract_section(changelog, "0.1.1")
        lines = section.splitlines()
        self.assertEqual(lines[0], "## [0.1.1](https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.1.1) (2026-09-21)")
        self.assertIn("### Fixed", lines)
        self.assertIn("* **bm25:** evaluate a BM25Plus term in reference order when its floor overflows ([#28](https://github.com/thyn-ai/mojo-kernels/issues/28)) ([1381151](https://github.com/thyn-ai/mojo-kernels/commit/1381151))", lines)
        self.assertNotIn("## [0.1.0] - 2026-09-20", lines)
        self.assertNotIn("- first", lines)
        # Selecting by the bracketed version must not be fooled by the
        # compare URL, which also names both versions.
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.extract_section(changelog, "v0.1.1")

    def test_prerelease_versions_and_headings_without_dates(self):
        changelog = "## [0.2.0-rc.1]\n\n- candidate\n\n## [0.1.0] - 2026-09-20\n\n- first\n"
        section = release_notes.extract_section(changelog, "0.2.0-rc.1")
        self.assertEqual(section, "## [0.2.0-rc.1]\n\n- candidate\n")

    def test_find_heading(self):
        self.assertEqual(release_notes.find_heading(CHANGELOG, "0.2.0"), "## [0.2.0] - 2026-10-01")


class Compose(unittest.TestCase):
    def test_body_contains_the_section_heading_then_the_generated_notes(self):
        body = release_notes.compose(CHANGELOG, "0.2.0", GENERATED)
        lines = body.splitlines()
        self.assertEqual(lines[0], "## [0.2.0] - 2026-10-01")
        self.assertIn("## What's Changed", lines)
        self.assertLess(lines.index("## [0.2.0] - 2026-10-01"), lines.index("## What's Changed"))
        self.assertIn("**Full Changelog**: https://github.com/thyn-ai/mojo-kernels/compare/v0.1.0...v0.2.0", lines)
        # Exactly one blank line between the section and the generated notes,
        # and a single trailing newline.
        self.assertIn(
            "- `bm25-mojo`: a fix ([#15](https://github.com/thyn-ai/mojo-kernels/issues/15)).\n\n## What's Changed\n",
            body,
        )
        self.assertTrue(body.endswith("...v0.2.0\n"))
        self.assertFalse(body.endswith("\n\n"))

    def test_generated_notes_are_optional(self):
        for generated in (None, "", "   \n\n"):
            body = release_notes.compose(CHANGELOG, "0.2.0", generated)
            self.assertEqual(body, release_notes.extract_section(CHANGELOG, "0.2.0"))

    def test_body_is_never_a_placeholder(self):
        # Even if the "generated" input were the placeholder body, the
        # CHANGELOG entry leads the body, so the placeholder can never be
        # the whole of it.
        body = release_notes.compose(CHANGELOG, "0.2.0", PLACEHOLDER_BODY)
        self.assertNotEqual(body.strip(), PLACEHOLDER_BODY.strip())
        self.assertTrue(body.startswith("## [0.2.0] - 2026-10-01\n"))

    def test_missing_section_fails_before_anything_is_composed(self):
        with self.assertRaises(release_notes.ChangelogError):
            release_notes.compose(CHANGELOG, "9.9.9", GENERATED)

    def test_missing_section_degrades_to_the_generated_notes_when_allowed(self):
        # The release job's mode: a release must never be blocked on the
        # changelog, because the Release is already published by then.
        body = release_notes.compose(CHANGELOG, "9.9.9", GENERATED, require_section=False)
        self.assertEqual(body, GENERATED.strip("\n") + "\n")

    def test_empty_section_degrades_to_the_generated_notes_when_allowed(self):
        # `Release-As: X.Y.Z` can force a release whose commits are all
        # hidden types, leaving a heading with no body under it.
        changelog = "## [1.0.0] - 2026-01-01\n\n\n## [0.9.0] - 2025-12-01\n\n- old\n"
        body = release_notes.compose(changelog, "1.0.0", GENERATED, require_section=False)
        self.assertEqual(body, GENERATED.strip("\n") + "\n")

    def test_nothing_to_compose_fails_even_when_a_missing_section_is_allowed(self):
        for generated in (None, "", "  \n"):
            with self.assertRaises(release_notes.ChangelogError):
                release_notes.compose(CHANGELOG, "9.9.9", generated, require_section=False)


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.changelog = self.dir / "CHANGELOG.md"
        self.changelog.write_text(CHANGELOG, encoding="utf-8")
        self.generated = self.dir / "generated.md"
        self.generated.write_text(GENERATED, encoding="utf-8")
        self.out = self.dir / "notes.md"

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = release_notes.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_check_prints_the_heading(self):
        code, out, err = self.run_main("check", "--changelog", str(self.changelog), "--version", "0.2.0")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.strip(), "## [0.2.0] - 2026-10-01")

    def test_check_fails_for_a_version_without_a_section(self):
        code, out, err = self.run_main("check", "--changelog", str(self.changelog), "--version", "0.3.0")
        self.assertEqual(code, 1)
        self.assertIn("no `## [0.3.0]` section", err)
        self.assertEqual(out, "")

    def test_compose_writes_the_body_and_reports_the_heading(self):
        code, out, err = self.run_main(
            "compose", "--changelog", str(self.changelog), "--version", "0.2.0",
            "--generated", str(self.generated), "--out", str(self.out),
        )
        self.assertEqual((code, err), (0, ""))
        self.assertIn("heading: ## [0.2.0] - 2026-10-01", out)
        body = self.out.read_text(encoding="utf-8")
        self.assertEqual(body, release_notes.compose(CHANGELOG, "0.2.0", GENERATED))

    def test_compose_without_generated_notes(self):
        code, _, err = self.run_main(
            "compose", "--changelog", str(self.changelog), "--version", "0.1.0", "--out", str(self.out),
        )
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.out.read_text(encoding="utf-8"), release_notes.extract_section(CHANGELOG, "0.1.0"))

    def test_compose_fails_and_writes_nothing_for_a_missing_section(self):
        code, _, err = self.run_main(
            "compose", "--changelog", str(self.changelog), "--version", "0.3.0", "--out", str(self.out),
        )
        self.assertEqual(code, 1)
        self.assertIn("no `## [0.3.0]` section", err)
        self.assertFalse(self.out.exists())

    def test_compose_with_allow_missing_section_writes_the_generated_notes(self):
        code, out, err = self.run_main(
            "compose", "--changelog", str(self.changelog), "--version", "0.3.0",
            "--generated", str(self.generated), "--allow-missing-section", "--out", str(self.out),
        )
        self.assertEqual(code, 0)
        self.assertIn("warning", err)
        self.assertIn("no usable `## [0.3.0]` entry", err)
        self.assertIn("heading: ## What's Changed", out)
        self.assertEqual(self.out.read_text(encoding="utf-8"), GENERATED.strip("\n") + "\n")

    def test_compose_with_allow_missing_section_still_fails_with_nothing_to_write(self):
        code, _, err = self.run_main(
            "compose", "--changelog", str(self.changelog), "--version", "0.3.0",
            "--allow-missing-section", "--out", str(self.out),
        )
        self.assertEqual(code, 1)
        self.assertIn("nothing to compose", err)
        self.assertFalse(self.out.exists())


class RepositoryChangelog(unittest.TestCase):
    """The real CHANGELOG.md must satisfy the composer for every cut version."""

    def test_every_cut_version_has_a_composable_section(self):
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        versions = [
            m.group(1)
            for m in (release_notes.HEADING_RE.match(line) for line in changelog.splitlines())
            if m and m.group(1).lower() != "unreleased"
        ]
        self.assertTrue(versions, "CHANGELOG.md has no [X.Y.Z] section")
        for version in versions:
            body = release_notes.compose(changelog, version, GENERATED)
            self.assertEqual(body.splitlines()[0], release_notes.find_heading(changelog, version))


if __name__ == "__main__":
    unittest.main()
