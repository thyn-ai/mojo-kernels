"""ipaddress-mojo: drop-in replacements for the stdlib `ipaddress` module
whose batch hot loops (bulk parse, bulk membership, bulk collapse) are
powered by a clean-room Mojo kernel — with a pure-Python/NumPy fallback for
platforms without a native build (including Windows).

    import ipaddress_mojo as ipm

    net = ipm.ip_network("10.0.0.0/8")          # like ipaddress.ip_network
    addr = ipm.ip_address("10.1.2.3")           # like ipaddress.ip_address
    addr in net                                  # identical semantics

    # Vectorized batch API for log analytics / ACL tooling:
    addrs = ipm.parse_many(list_of_strings)      # bulk parse
    hits = ipm.contains_many(networks, addrs)    # bool per address (any net)
    merged = ipm.collapse_batch(networks)        # canonical minimal cover

Single-object behavior (parse, format, contains, subnets, supernet,
overlaps, collapse, summarize, exception types/messages) mirrors CPython
3.12's ipaddress exactly; the differential suite asserts parity with the
stdlib oracle on both the native and the fallback backend.

Set IPADDRESS_MOJO_DISABLE_NATIVE=1 to force the fallback.
"""

from ipaddress_mojo._native import backend_info, native_available
from ipaddress_mojo.core import (
    AddressValueError,
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
    NetmaskValueError,
    collapse_addresses,
    collapse_batch,
    contains_many,
    get_mixed_type_key,
    ip_address,
    ip_interface,
    ip_network,
    parse_many,
    summarize_address_range,
    v4_int_to_packed,
    v6_int_to_packed,
)

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "AddressValueError",
    "IPv4Address",
    "IPv4Interface",
    "IPv4Network",
    "IPv6Address",
    "IPv6Interface",
    "IPv6Network",
    "NetmaskValueError",
    "backend_info",
    "collapse_addresses",
    "collapse_batch",
    "contains_many",
    "get_mixed_type_key",
    "ip_address",
    "ip_interface",
    "ip_network",
    "native_available",
    "parse_many",
    "summarize_address_range",
    "v4_int_to_packed",
    "v6_int_to_packed",
    "__version__",
]
