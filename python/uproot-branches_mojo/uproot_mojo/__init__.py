"""uproot-mojo: fast ROOT TBranch reading for vector<T> and string branches.

A minimal, Mojo-accelerated reader for split-level-0 ``std::vector<int32>``,
``std::vector<int64>``, ``std::vector<float>``, ``std::vector<double>``,
``std::string`` and ``std::vector<std::string>`` TBranches in ROOT files,
returning exactly the offsets+content buffers that uproot's interpretations
produce — powered by a Mojo kernel where the platform supports it (macOS
arm64, Linux x86_64), with a vendored pure-Python fallback everywhere else
(including Windows).

    import uproot_mojo

    result = uproot_mojo.read_branch("events.root", "pt", dtype="float64")
    result.offsets   # int64[n_entries + 1], cumulative items per entry
    result.content   # native float64 item buffer, identical to uproot's
    result.to_list() # plain Python list-of-lists

    strings = uproot_mojo.read_branch("events.root", "labels")  # std::string
    strings.to_list()  # list of str

Set UPROOT_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from uproot_mojo._native import backend_info, native_available
from uproot_mojo._rootfile import (
    BasketDataError,
    RootFileError,
    UnsupportedBranchError,
)
from uproot_mojo.core import JaggedArray, JaggedStringArray, read_branch

__version__ = "0.1.1"  # x-release-please-version
__all__ = [
    "BasketDataError",
    "JaggedArray",
    "JaggedStringArray",
    "RootFileError",
    "UnsupportedBranchError",
    "backend_info",
    "native_available",
    "read_branch",
    "__version__",
]
