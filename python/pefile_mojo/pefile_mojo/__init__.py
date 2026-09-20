"""pefile-mojo: hot-loop accelerators for the `pefile` package, powered by a
clean-room Mojo kernel — with a vendored pure-Python fallback for platforms
without a native build (including Windows).

    import pefile_mojo

    blob = open("module.dll", "rb").read()
    offset = pefile_mojo.checksum_field_offset(blob)
    digest = pefile_mojo.generate_checksum(blob, offset)   # == pefile's
    for desc in pefile_mojo.parse_imports(blob):           # == pefile's
        print(desc.dll, [s.name or s.ordinal for s in desc.imports])

This package is NOT a full drop-in for `pefile` (it does not parse whole PE
images); it is a bit-exact, faster replacement for pefile's two hot loops —
`PE.generate_checksum` and the import-directory walk behind
`PE.parse_data_directories` / `PE.parse_import_directory`. See the README for
the exact drop-in recipe and the unsupported scope.

Set PEFILE_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from pefile_mojo._native import backend_info, native_available
from pefile_mojo.core import (
    ImportDescriptor,
    ImportSymbol,
    PEFormatError,
    checksum_field_offset,
    generate_checksum,
    parse_imports,
)

__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "ImportDescriptor",
    "ImportSymbol",
    "PEFormatError",
    "backend_info",
    "checksum_field_offset",
    "generate_checksum",
    "native_available",
    "parse_imports",
    "__version__",
]
