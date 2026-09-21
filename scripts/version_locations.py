#!/usr/bin/env python3
"""Every version location in the tree carries the same version, or this fails naming the odd one.

Standard library only. Prints the agreed version on stdout and the full
listing (one line per location) on stderr; exits 1 when the locations
disagree or when EXPECTED_VERSION (or --expect) names a different version.

    python3 scripts/version_locations.py               # print the tree's version
    EXPECTED_VERSION=0.1.2 python3 scripts/version_locations.py

The locations are the ones release-please-config.json bumps, discovered by
the same globs, plus release-please's own anchors (version.txt, the
manifest) and pixi.toml. A location whose key is gone reads "<missing>"
rather than raising: it then disagrees with the rest and is named in the
error, which is the point of listing every location. An updater whose path
stopped matching logs a warning and leaves the file untouched, so checking
the whole set is what turns a silently skipped rewrite into a refusal that
names the file, instead of version drift that surfaces releases later.

Lines annotated `x-release-please-version` (the generic updater's marker)
must carry exactly one semver: release-please replaces the FIRST semver on
such a line, so a second one would be left behind and drift.

release.yml's `preflight` job runs this against the tag; readme-install.yml
runs it on every release pull request, so a location the release pull
request forgot fails before the merge, not at the tag.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import tomllib

MARKER = "x-release-please-version"
# release-please's own VERSION_REGEX (src/updaters/generic.ts), without the
# named groups it uses internally.
SEMVER = re.compile(r"\d+\.\d+\.\d+(?:-[\w.]+)?(?:\+[-\w.]+)?")


def toml_version(path: pathlib.Path, *keys: str) -> str:
    node = tomllib.loads(path.read_text())
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return "<missing>"
        node = node[key]
    return str(node)


def json_load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def json_at(data: dict, *keys: str) -> str:
    node = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return "<missing>"
        node = node[key]
    return str(node)


def generic_marker_files(root: pathlib.Path) -> list[tuple[pathlib.Path, bool]]:
    """Files release-please-config.json hands to the generic (line marker) updater.

    Each is paired with whether it was named explicitly (True) or reached by a
    glob (False): a file named explicitly exists to carry a marker line, so
    having none is drift; a globbed file with no marker (a unit test that
    does not assert the version) is simply not a version location.
    """
    config = json_load(root / "release-please-config.json")
    files: list[tuple[pathlib.Path, bool]] = []
    for package in config.get("packages", {}).values():
        for entry in package.get("extra-files", []):
            if not isinstance(entry, dict) or entry.get("type") != "generic":
                continue
            if entry.get("glob"):
                files.extend((path, False) for path in sorted(root.glob(entry["path"])))
            else:
                files.append((root / entry["path"], True))
    return files


def marker_lines(path: pathlib.Path) -> dict[int, str]:
    """{line number: version} for every marker line; a line with != 1 semver reads as an error value."""
    found: dict[int, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if MARKER not in line:
            continue
        versions = SEMVER.findall(line)
        if len(versions) == 1:
            found[number] = versions[0]
        else:
            found[number] = f"<{len(versions)} versions on the line>"
    return found


def collect(root: pathlib.Path) -> dict[str, str]:
    seen: dict[str, str] = {}
    seen["pixi.toml [workspace].version"] = toml_version(root / "pixi.toml", "workspace", "version")
    # release-please's own anchor and the version it recorded as last released.
    seen["version.txt"] = (root / "version.txt").read_text().strip()
    seen[".release-please-manifest.json ."] = json_at(json_load(root / ".release-please-manifest.json"), ".")
    for path in sorted(root.glob("python/*/pyproject.toml")):
        seen[f"{path.relative_to(root)} [project].version"] = toml_version(path, "project", "version")
    for path in sorted(root.glob("python/*/*/__init__.py")):
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', path.read_text(), re.M)
        seen[f"{path.relative_to(root)} __version__"] = match.group(1) if match else "<missing>"
    for path in sorted(root.glob("typescript/*/package.json")):
        seen[f"{path.relative_to(root)} version"] = json_at(json_load(path), "version")
    for path in sorted(root.glob("typescript/*/packages/*/package.json")):
        data = json_load(path)
        rel = path.relative_to(root)
        seen[f"{rel} version"] = json_at(data, "version")
        for name, pin in data.get("optionalDependencies", {}).items():
            seen[f"{rel} optionalDependencies[{name}]"] = pin
    for path in sorted(root.glob("typescript/*/package-lock.json")):
        lock = json_load(path)
        rel = path.relative_to(root)
        seen[f"{rel} version"] = json_at(lock, "version")
        for key in ("", "packages/core"):
            entry = lock.get("packages", {}).get(key)
            if entry is None:
                continue
            label = key or '""'
            seen[f"{rel} packages[{label}].version"] = json_at(entry, "version")
            for name, pin in entry.get("optionalDependencies", {}).items():
                seen[f"{rel} packages[{label}].optionalDependencies[{name}]"] = pin
    # Every line the generic updater rewrites, in every file the config hands
    # it: the __version__ lines, the unit tests' version assertions and the
    # README install blocks' `V=X.Y.Z` lines.
    for path, explicit in generic_marker_files(root):
        rel = path.relative_to(root)
        lines = marker_lines(path) if path.is_file() else {}
        if not lines and explicit:
            seen[f"{rel} {MARKER}"] = "<missing>"
        for number, value in lines.items():
            seen[f"{rel}:{number} {MARKER}"] = value
    return seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root (default: the current directory)")
    parser.add_argument(
        "--expect",
        default=os.environ.get("EXPECTED_VERSION", ""),
        help="the version every location must carry (default: $EXPECTED_VERSION; empty = any, as long as they agree)",
    )
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)

    seen = collect(root)
    for label, value in seen.items():
        print(f"  {value:<14} {label}", file=sys.stderr)
    versions = sorted(set(seen.values()))
    if len(versions) != 1:
        odd = [label for label, value in seen.items() if value != max(versions, key=list(seen.values()).count)]
        print(
            f"::error::Package versions disagree ({', '.join(versions)}); bump them in lockstep (see RELEASING.md). "
            f"Odd ones out: {', '.join(odd)}",
            file=sys.stderr,
        )
        return 1
    version = versions[0]
    if args.expect and args.expect != version:
        print(
            f"::error::Tag names version {args.expect} but every package in the tree is at {version}; refusing to release a mismatched tag.",
            file=sys.stderr,
        )
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
