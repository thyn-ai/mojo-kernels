"""Hatch build hook: bundle the compiled bioparse kernel into a platform wheel.

The wheel for the current platform force-includes the shared library compiled
by `kernels/bioparse/build.sh` (run it before building the wheel) and is
tagged `py3-none-<platform>`.

Escape hatch: `BIO_MOJO_ALLOW_PURE_WHEEL=1` builds a pure-Python
`py3-none-any` wheel with no native library — the resulting package uses the
vendored fallback on every platform. This exists for metadata smoke tests and
for platforms with no Mojo toolchain (e.g. a Windows wheel), never for the
primary macOS/Linux release wheels.
"""

from __future__ import annotations

import os
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import sys_tags

_LIB_NAMES = {
    "darwin": "libbioparse.dylib",
    "linux": "libbioparse.so",
    "win32": "bioparse.dll",
}


def _lib_name() -> str:
    return _LIB_NAMES.get(sys.platform, "libbioparse.so")


def _platform_wheel_tag() -> str:
    """The wheel tag must describe the compiled library, not the build host.

    - macOS: the pixi workspace pins a macOS 14.0 floor and the kernel build
      exports MACOSX_DEPLOYMENT_TARGET accordingly; honor it here.
    - Linux: the pixi workspace pins glibc 2.35, so the wheel is manylinux_2_35.
    If the shared library ever fails to load on a real system, the wrapper
    falls back to pure Python, so these tags only ever widen compatibility.
    """
    if sys.platform == "darwin":
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0")
        major, minor = (target.split(".") + ["0"])[:2]
        return f"py3-none-macosx_{major}_{minor}_arm64"
    if sys.platform.startswith("linux"):
        machine = os.uname().machine
        arch = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(machine, machine)
        return f"py3-none-manylinux_2_35_{arch}"
    platform_tag = next(tag.platform for tag in sys_tags())
    return f"py3-none-{platform_tag}"


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "bio-mojo-native"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: ARG002
        if os.environ.get("BIO_MOJO_ALLOW_PURE_WHEEL") == "1":
            build_data["pure_python"] = True
            build_data["infer_tag"] = False
            build_data["tag"] = "py3-none-any"
            return

        src = os.environ.get("BIO_MOJO_NATIVE_SRC") or os.path.abspath(
            os.path.join(self.root, "..", "..", "kernels", "bioparse", "build", _lib_name())
        )
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"native kernel not found at {src}. Compile it first "
                "(`bash kernels/bioparse/build.sh` via the repo pixi toolchain), or set "
                "BIO_MOJO_ALLOW_PURE_WHEEL=1 to build a fallback-only wheel."
            )

        force_include = build_data.setdefault("force_include", {})
        force_include[src] = f"bio_mojo/_native/{os.path.basename(src)}"

        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = _platform_wheel_tag()
