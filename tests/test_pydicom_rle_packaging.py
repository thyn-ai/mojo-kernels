"""Check pydicom attribution in source and built wheels without network access.

The pydicom license hash is pinned to upstream pydicom v3.0.2 at commit
20aa4b73aef76bffb6fbc46a8827588dfa9bb89d.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "python" / "pydicom_mojo"
EXPECTED_LICENSE_HASHES = {
    "LICENSE": "b813a226207a0d1beccf73cfb645e94b0c9817655ef20e692655c0d4c9e2f546",
    "LICENSE.pydicom": "39b3259cb15151df25167119087df1fef33624765e639af42447ba6fe86dada3",
}
FUSE_ATTRIBUTION = """Fuse.js is additionally vendored as fuse-mojo's fallback backend under its
own Apache-2.0 license; see typescript/fuse-mojo/packages/core/NOTICE."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _license_archive_path(metadata_path: str, license_file: str) -> str:
    relative_path = PurePosixPath(license_file)
    assert not relative_path.is_absolute(), license_file
    assert ".." not in relative_path.parts, license_file
    dist_info = metadata_path.rsplit("/", 1)[0]
    return f"{dist_info}/licenses/{relative_path}"


def assert_wheel_attribution(wheel_path: str | Path) -> None:
    wheel_path = Path(wheel_path)
    assert wheel_path.is_file(), f"wheel does not exist: {wheel_path}"
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1, metadata_paths
        metadata_path = metadata_paths[0]
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        assert metadata["License-Expression"] == "Apache-2.0"

        license_files = metadata.get_all("License-File", failobj=[])
        assert len(license_files) == len(set(license_files)), license_files
        assert set(EXPECTED_LICENSE_HASHES) <= set(license_files), license_files
        for license_file in license_files:
            assert _license_archive_path(metadata_path, license_file) in names, license_file

        for license_file, expected_hash in EXPECTED_LICENSE_HASHES.items():
            archive_path = _license_archive_path(metadata_path, license_file)
            actual_hash = _sha256(archive.read(archive_path))
            assert actual_hash == expected_hash, (license_file, actual_hash)


def test_source_licenses_match_pinned_text():
    for license_file, expected_hash in EXPECTED_LICENSE_HASHES.items():
        actual_hash = _sha256((PACKAGE_ROOT / license_file).read_bytes())
        assert actual_hash == expected_hash, (license_file, actual_hash)


def test_notice_scopes_pydicom_port_and_preserves_fuse_attribution():
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "except for the pydicom-derived Python" in notice
    assert "python/pydicom_mojo/LICENSE.pydicom" in notice
    assert FUSE_ATTRIBUTION in notice


def test_pure_wheel_contains_pinned_license_metadata_and_text(tmp_path):
    env = os.environ.copy()
    env["PYDICOM_MOJO_ALLOW_PURE_WHEEL"] = "1"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(PACKAGE_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, wheels
    assert_wheel_attribution(wheels[0])


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} WHEEL [WHEEL ...]")
    for wheel_path in sys.argv[1:]:
        assert_wheel_attribution(wheel_path)
        print(f"wheel attribution OK: {wheel_path}")  # noqa: T201


if __name__ == "__main__":
    main()
