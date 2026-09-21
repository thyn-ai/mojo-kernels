"""Unit tests for scripts/version_locations.py, the version lockstep check.

Standard library only (``python3 -m unittest discover -s tests -p test_version_locations.py``):
release.yml's ``preflight`` job and readme-install.yml run this file with the
runner's interpreter before they run the check itself. The last test runs the
check against this very tree, so a tree whose locations disagree fails here
as well as in the workflows.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "version_locations.py"

spec = importlib.util.spec_from_file_location("version_locations", SCRIPT)
assert spec is not None and spec.loader is not None, f"could not load {SCRIPT}"
version_locations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_locations)

MARKER = version_locations.MARKER


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_tree(root: Path, version: str = "1.2.3") -> None:
    """The smallest tree that exercises every kind of location the check knows."""
    write(root, "pixi.toml", f'[workspace]\nname = "t"\nversion = "{version}"\n')
    write(root, "version.txt", f"{version}\n")
    write(root, ".release-please-manifest.json", json.dumps({".": version}))
    write(
        root,
        "release-please-config.json",
        json.dumps(
            {
                "packages": {
                    ".": {
                        "extra-files": [
                            {"type": "generic", "path": "python/*/*/__init__.py", "glob": True},
                            {"type": "generic", "path": "typescript/*/tests/unit.test.cjs", "glob": True},
                            {"type": "generic", "path": "README.md"},
                        ]
                    }
                }
            }
        ),
    )
    write(root, "python/pkg_mojo/pyproject.toml", f'[project]\nname = "pkg-mojo"\nversion = "{version}"\n')
    write(root, "python/pkg_mojo/pkg_mojo/__init__.py", f'__version__ = "{version}"  # {MARKER}\n')
    write(root, "typescript/pkg-mojo/package.json", json.dumps({"version": version}))
    write(
        root,
        "typescript/pkg-mojo/packages/core/package.json",
        json.dumps({"version": version, "optionalDependencies": {"@pkg-mojo/darwin-arm64": version}}),
    )
    write(
        root,
        "typescript/pkg-mojo/package-lock.json",
        json.dumps(
            {
                "version": version,
                "packages": {
                    "": {"version": version},
                    "packages/core": {"version": version, "optionalDependencies": {"@pkg-mojo/darwin-arm64": version}},
                },
            }
        ),
    )
    # A unit test that asserts the version, and one that does not (not a location).
    write(root, "typescript/pkg-mojo/tests/unit.test.cjs", f"assert.equal(pkg.version, '{version}') // {MARKER}\n")
    write(root, "typescript/other-mojo/tests/unit.test.cjs", "assert.ok(true)\n")
    write(
        root,
        "README.md",
        "# t\n\n## Install\n\n```bash\n"
        f"V={version}  # {MARKER}\n"
        'pip install "https://example.invalid/v$V/pkg_mojo-$V-py3-none-any.whl"\n'
        "```\n\nFuse.js 7.1.0 is the oracle; bm25s 0.3.11 too.\n",
    )


def run(root: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = version_locations.main(["--root", str(root), *argv])
    return code, out.getvalue(), err.getvalue()


class VersionLocationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_tree(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_agreeing_tree_prints_the_version(self) -> None:
        code, out, err = run(self.root)
        self.assertEqual((code, out.strip()), (0, "1.2.3"), err)
        self.assertIn(f"README.md:6 {MARKER}", err)
        self.assertIn("typescript/pkg-mojo/tests/unit.test.cjs:1", err)
        self.assertNotIn("other-mojo", err, "a globbed file with no marker is not a location")

    def test_expected_version_is_enforced(self) -> None:
        code, _, err = run(self.root, "--expect", "1.2.3")
        self.assertEqual(code, 0, err)
        code, _, err = run(self.root, "--expect", "1.2.4")
        self.assertEqual(code, 1)
        self.assertIn("Tag names version 1.2.4", err)

    def test_readme_install_block_left_behind_is_named(self) -> None:
        readme = self.root / "README.md"
        readme.write_text(readme.read_text().replace(f"V=1.2.3  # {MARKER}", f"V=1.2.2  # {MARKER}"))
        code, _, err = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("Package versions disagree (1.2.2, 1.2.3)", err)
        self.assertIn(f"Odd ones out: README.md:6 {MARKER}", err)

    def test_readme_without_its_marker_is_missing(self) -> None:
        readme = self.root / "README.md"
        readme.write_text(readme.read_text().replace(f"  # {MARKER}", ""))
        code, _, err = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn(f"<missing>      README.md {MARKER}", err)

    def test_two_semvers_on_a_marker_line_is_refused(self) -> None:
        # release-please replaces only the first semver on a marker line; a
        # second one would survive the bump, so the check refuses the line.
        readme = self.root / "README.md"
        readme.write_text(readme.read_text().replace(f"V=1.2.3  # {MARKER}", f"V=1.2.3  # {MARKER} (was 1.2.2)"))
        code, _, err = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("<2 versions on the line>", err)

    def test_missing_key_reads_missing(self) -> None:
        (self.root / "pixi.toml").write_text('[workspace]\nname = "t"\n')
        code, _, err = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("<missing>      pixi.toml [workspace].version", err)

    def test_error_values_never_become_the_reference(self) -> None:
        # Three broken locations and two correct ones: the reference is still
        # the real version, so the message names the broken locations.
        seen = {"a": "<missing>", "b": "<missing>", "c": "<missing>", "d": "1.2.3", "e": "1.2.3"}
        self.assertEqual(version_locations.reference_version(seen), "1.2.3")
        only_errors = {"a": "<missing>", "b": "<2 versions on the line>", "c": "<missing>"}
        self.assertEqual(version_locations.reference_version(only_errors), "<missing>")

    def test_this_repository_is_in_lockstep(self) -> None:
        code, out, err = run(REPO_ROOT)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), (REPO_ROOT / "version.txt").read_text().strip())
        for readme in ("README.md", "python/cclib_mojo/README.md", "typescript/fuse-mojo/README.md"):
            self.assertIn(f"{readme}:", err, f"{readme} carries no install block marker")


if __name__ == "__main__":
    unittest.main()
