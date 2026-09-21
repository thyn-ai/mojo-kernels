"""Drop-in IP address/network classes, API-compatible with stdlib ipaddress.

`IPv4Address`, `IPv6Address`, `IPv4Network`, `IPv6Network`, `IPv4Interface`,
`IPv6Interface` and the module-level helpers (`ip_address`, `ip_network`,
`ip_interface`, `collapse_addresses`, `summarize_address_range`,
`get_mixed_type_key`) accept the same inputs, return the same values, and
raise the same exception types with the same messages as the CPython 3.12
standard-library module — the differential suite asserts all of this against
the stdlib oracle on both backends (native Mojo kernel and pure-Python
fallback).

On top of the drop-in surface, the package adds three vectorized batch
operations for log-analytics/ACL workloads — `parse_many`, `contains_many`,
and `collapse_batch` — which run on the native Mojo kernel when available and
on a NumPy-based fallback otherwise (identical results either way).

Single-object parsing/validation always runs the pure-Python reference: one
FFI call cannot beat in-process parsing, and keeping one code path makes
backend divergence impossible.
"""

from __future__ import annotations

import functools

from ipaddress import AddressValueError, NetmaskValueError

from ipaddress_mojo import _native, _reference

__all__ = [
    "AddressValueError",
    "NetmaskValueError",
    "IPv4Address",
    "IPv6Address",
    "IPv4Network",
    "IPv6Network",
    "IPv4Interface",
    "IPv6Interface",
    "ip_address",
    "ip_network",
    "ip_interface",
    "collapse_addresses",
    "summarize_address_range",
    "get_mixed_type_key",
    "v4_int_to_packed",
    "v6_int_to_packed",
    "parse_many",
    "contains_many",
    "collapse_batch",
]

_IPV4_BITS = 32
_IPV6_BITS = 128
_V4_ALL_ONES = (1 << _IPV4_BITS) - 1
_V6_ALL_ONES = (1 << _IPV6_BITS) - 1


def _version_type_error(self, other) -> TypeError:
    return TypeError("%s and %s are not of the same version" % (self, other))


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------


@functools.total_ordering
class IPv4Address:
    """Drop-in replacement for ipaddress.IPv4Address."""

    __slots__ = ("_ip",)

    _version = 4
    version = 4
    max_prefixlen = _IPV4_BITS
    _ALL_ONES = _V4_ALL_ONES
    _scope_id = None  # v4 has no zones; uniform access for shared helpers

    def __init__(self, address):
        self._ip = _reference.parse_ipv4(address)

    # -- representation -----------------------------------------------------
    @property
    def packed(self) -> bytes:
        return self._ip.to_bytes(4, "big")

    @property
    def compressed(self) -> str:
        return _reference.ipv4_to_str(self._ip)

    @property
    def exploded(self) -> str:
        return _reference.ipv4_to_str(self._ip)

    @property
    def reverse_pointer(self) -> str:
        return _reference.ipv4_reverse_pointer(self._ip)

    def __str__(self) -> str:
        return _reference.ipv4_to_str(self._ip)

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    def __int__(self) -> int:
        return self._ip

    # -- comparisons --------------------------------------------------------
    def __eq__(self, other) -> bool:
        if isinstance(other, _BaseAddress):
            return self._version == other._version and self._ip == other._ip
        return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, _BaseAddress):
            return NotImplemented
        if self._version != other._version:
            raise _version_type_error(self, other)
        return self._ip < other._ip

    def __hash__(self) -> int:
        # stdlib v4 hashes the hex-string form (salted per process, same value).
        return hash(hex(self._ip))

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(self._ip + other)

    def __sub__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(self._ip - other)

    # -- boolean classification (registries in _reference) ------------------
    @property
    def is_private(self) -> bool:
        return _reference.v4_is_private(self._ip)

    @property
    def is_global(self) -> bool:
        return _reference.v4_is_global(self._ip)

    @property
    def is_multicast(self) -> bool:
        return 0xE0000000 <= self._ip <= 0xEFFFFFFF

    @property
    def is_reserved(self) -> bool:
        return self._ip >= 0xF0000000

    @property
    def is_loopback(self) -> bool:
        return (self._ip >> 24) == 0x7F

    @property
    def is_link_local(self) -> bool:
        return (self._ip >> 16) == 0xA9FE

    @property
    def is_unspecified(self) -> bool:
        return self._ip == 0


@functools.total_ordering
class IPv6Address:
    """Drop-in replacement for ipaddress.IPv6Address (incl. %scope ids)."""

    __slots__ = ("_ip", "_scope_id")

    _version = 6
    version = 6
    max_prefixlen = _IPV6_BITS
    _ALL_ONES = _V6_ALL_ONES

    def __init__(self, address):
        value, scope = _reference.parse_ipv6(address)
        self._ip = value
        self._scope_id = scope

    @classmethod
    def _from_parts(cls, value: int, scope_id=None):
        obj = cls.__new__(cls)
        obj._ip = value
        obj._scope_id = scope_id
        return obj

    @property
    def scope_id(self):
        return self._scope_id

    # -- representation -----------------------------------------------------
    @property
    def packed(self) -> bytes:
        return self._ip.to_bytes(16, "big")

    @property
    def compressed(self) -> str:
        return _reference.ipv6_to_str(self._ip, self._scope_id)

    @property
    def exploded(self) -> str:
        # stdlib derives this by re-parsing the canonical string *with* the
        # scope id through the scope-unaware parser, so a scoped address
        # raises AddressValueError here (3.12 behavior, mirrored verbatim).
        value = _reference._ipv6_int_from_string(
            _reference.ipv6_to_str(self._ip, self._scope_id)
        )
        return _reference.ipv6_exploded_str(value)

    @property
    def reverse_pointer(self) -> str:
        value = _reference._ipv6_int_from_string(
            _reference.ipv6_to_str(self._ip, self._scope_id)
        )
        return _reference.ipv6_reverse_pointer(value)

    def __str__(self) -> str:
        return _reference.ipv6_to_str(self._ip, self._scope_id)

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    def __int__(self) -> int:
        return self._ip

    # -- comparisons --------------------------------------------------------
    def __eq__(self, other) -> bool:
        if isinstance(other, _BaseAddress):
            return (
                self._version == other._version
                and self._ip == other._ip
                and self._scope_id == other._scope_id
            )
        return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, _BaseAddress):
            return NotImplemented
        if self._version != other._version:
            raise _version_type_error(self, other)
        # 3.12 compares the integer only (scope ids do not order).
        return self._ip < other._ip

    def __hash__(self) -> int:
        return hash((self._ip, self._scope_id))

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(self._ip + other)

    def __sub__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(self._ip - other)

    # -- boolean classification ---------------------------------------------
    # A v4-mapped address (::ffff:0:0/96) delegates every classification to
    # the embedded v4 address (3.12 behavior, boundary-verified).
    def _mapped_v4(self):
        if (self._ip >> 32) != 0xFFFF:
            return None
        return IPv4Address(self._ip & _V4_ALL_ONES)

    @property
    def is_private(self) -> bool:
        m = self._mapped_v4()
        return m.is_private if m is not None else _reference.v6_is_private(self._ip)

    @property
    def is_global(self) -> bool:
        m = self._mapped_v4()
        return m.is_global if m is not None else _reference.v6_is_global(self._ip)

    @property
    def is_multicast(self) -> bool:
        m = self._mapped_v4()
        return m.is_multicast if m is not None else (self._ip >> 120) == 0xFF

    @property
    def is_reserved(self) -> bool:
        m = self._mapped_v4()
        if m is not None:
            return m.is_reserved
        h = self._ip >> 112
        return h <= 0x1FFF or 0x4000 <= h <= 0xFBFF or 0xFE00 <= h <= 0xFE7F

    @property
    def is_loopback(self) -> bool:
        m = self._mapped_v4()
        return m.is_loopback if m is not None else self._ip == 1

    @property
    def is_link_local(self) -> bool:
        m = self._mapped_v4()
        return m.is_link_local if m is not None else (self._ip >> 118) == 0x3FA

    @property
    def is_site_local(self) -> bool:
        return (self._ip >> 118) == 0x3FB

    @property
    def is_unspecified(self) -> bool:
        m = self._mapped_v4()
        return m.is_unspecified if m is not None else self._ip == 0

    # -- mapped/transition views --------------------------------------------
    @property
    def ipv4_mapped(self):
        if (self._ip >> 32) != 0xFFFF:
            return None
        return IPv4Address(self._ip & _V4_ALL_ONES)

    @property
    def sixtofour(self):
        if (self._ip >> 112) != 0x2002:
            return None
        return IPv4Address((self._ip >> 80) & _V4_ALL_ONES)

    @property
    def teredo(self):
        if (self._ip >> 96) != 0x20010000:
            return None
        return (
            IPv4Address((self._ip >> 64) & _V4_ALL_ONES),
            IPv4Address((self._ip ^ _V4_ALL_ONES) & _V4_ALL_ONES),
        )


_BaseAddress = (IPv4Address, IPv6Address)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _split_network_input(address, version: int):
    """Split a network constructor input into (address_part, prefix_part).

    Mirrors stdlib constructor semantics: int/bytes -> /<max>; tuple ->
    (addr, prefix); everything else is str()'d and split on '/'.
    """
    maxp = _IPV4_BITS if version == 4 else _IPV6_BITS
    if isinstance(address, tuple):
        return address  # Python unpacks (arity ValueErrors match stdlib)
    if isinstance(address, (bytes, int)):
        return address, maxp
    s = str(address)
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            raise AddressValueError("Only one '/' permitted in %r" % s)
        return parts[0], parts[1]
    return s, maxp


def _prefix_from_tuple_arg(prefix_cache: dict, prefix_in, bits: int, version: int) -> int:
    """Tuple/int prefix handling, with stdlib's netmask cache semantics:
    successful int prefixes are cached per class, and later int-*equal*
    values (floats, bools — dict keys unify them) hit the cache instead of
    raising AttributeError on the string path."""
    if isinstance(prefix_in, int):
        prefixlen = _reference.prefix_from_int(prefix_in, bits)
        prefix_cache[prefix_in] = prefixlen
        return prefixlen
    if prefix_in in prefix_cache:
        return prefix_cache[prefix_in]
    return _reference.prefix_from_string(prefix_in, bits, version)


def _build_network(cls_addr, prefix_cache, address, strict, version: int):
    """Shared constructor logic for both network classes."""
    addr_in, prefix_in = _split_network_input(address, version)
    if version == 4:
        addr_int = _reference.parse_ipv4(addr_in)
        scope = None
        bits = _IPV4_BITS
    else:
        addr_int, scope = _reference.parse_ipv6(addr_in)
        bits = _IPV6_BITS
    prefixlen = _prefix_from_tuple_arg(prefix_cache, prefix_in, bits, version)
    mask = _reference.netmask_int(prefixlen, bits)
    net_int = addr_int & mask
    if net_int != addr_int:
        if strict:
            addr_str = (
                _reference.ipv4_to_str(addr_int)
                if version == 4
                else _reference.ipv6_to_str(addr_int, scope)
            )
            raise ValueError("%s/%d has host bits set" % (addr_str, prefixlen))
        network_address = cls_addr(net_int)  # masked: scope id is dropped
    else:
        if version == 6:
            network_address = cls_addr._from_parts(addr_int, scope)
        else:
            network_address = cls_addr(addr_int)
    return network_address, prefixlen, mask, bits


@functools.total_ordering
class IPv4Network:
    """Drop-in replacement for ipaddress.IPv4Network."""

    __slots__ = (
        "network_address",
        "broadcast_address",
        "netmask",
        "hostmask",
        "_prefixlen",
    )

    _version = 4
    version = 4
    max_prefixlen = _IPV4_BITS
    _AddressClass = IPv4Address
    _prefix_cache: dict = {}

    def __init__(self, address, strict=True):
        na, plen, mask, _ = _build_network(
            IPv4Address, IPv4Network._prefix_cache, address, strict, 4
        )
        self.network_address = na
        self._prefixlen = plen
        self.netmask = IPv4Address(mask)
        self.hostmask = IPv4Address(mask ^ _V4_ALL_ONES)
        self.broadcast_address = IPv4Address(na._ip | (mask ^ _V4_ALL_ONES))

    # -- identity ------------------------------------------------------------
    @property
    def prefixlen(self) -> int:
        return self._prefixlen

    @property
    def num_addresses(self) -> int:
        return self.broadcast_address._ip - self.network_address._ip + 1

    def __str__(self) -> str:
        return "%s/%d" % (self.network_address, self._prefixlen)

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    @property
    def with_prefixlen(self) -> str:
        return "%s/%d" % (self.network_address, self._prefixlen)

    @property
    def with_netmask(self) -> str:
        return "%s/%s" % (self.network_address, self.netmask)

    @property
    def with_hostmask(self) -> str:
        return "%s/%s" % (self.network_address, self.hostmask)

    # -- comparisons ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, _BaseNetwork):
            return NotImplemented
        return (
            self._version == other._version
            and self.network_address._ip == other.network_address._ip
            and self.netmask._ip == other.netmask._ip
        )

    def __lt__(self, other):
        if not isinstance(other, _BaseNetwork):
            return NotImplemented
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (self.network_address._ip, self.netmask._ip) < (
            other.network_address._ip,
            other.netmask._ip,
        )

    def __hash__(self) -> int:
        return hash((self.network_address._ip, self.netmask._ip))

    # -- membership / iteration ----------------------------------------------
    def __contains__(self, other) -> bool:
        # Attribute access order matters: stdlib touches other._version first
        # (an int operand raises AttributeError there).
        if other._version != self._version:
            return False
        if isinstance(other, _BaseNetwork):
            return False
        return (
            self.network_address._ip <= other._ip <= self.broadcast_address._ip
        )

    def __iter__(self):
        lo = self.network_address._ip
        hi = self.broadcast_address._ip
        cls = self._AddressClass
        return (cls(v) for v in range(lo, hi + 1))

    def __getitem__(self, n):
        if n >= 0:
            if self.network_address._ip + n > self.broadcast_address._ip:
                raise IndexError("address out of range")
        else:
            n += self.num_addresses
            if n < 0:
                raise IndexError("address out of range")
        return self._AddressClass(self.network_address._ip + n)

    def hosts(self):
        lo = self.network_address._ip
        hi = self.broadcast_address._ip
        cls = self._AddressClass
        return (cls(v) for v in range(lo, hi + 1))

    # -- boolean classification ----------------------------------------------
    @property
    def is_private(self) -> bool:
        return (
            self.network_address.is_private and self.broadcast_address.is_private
        )

    @property
    def is_global(self) -> bool:
        return self.network_address.is_global and self.broadcast_address.is_global

    @property
    def is_multicast(self) -> bool:
        return (
            self.network_address.is_multicast and self.broadcast_address.is_multicast
        )

    @property
    def is_loopback(self) -> bool:
        return (
            self.network_address.is_loopback and self.broadcast_address.is_loopback
        )

    @property
    def is_link_local(self) -> bool:
        return (
            self.network_address.is_link_local
            and self.broadcast_address.is_link_local
        )

    @property
    def is_reserved(self) -> bool:
        return (
            self.network_address.is_reserved and self.broadcast_address.is_reserved
        )

    @property
    def is_unspecified(self) -> bool:
        return (
            self.network_address.is_unspecified
            and self.broadcast_address.is_unspecified
        )

    # -- relational ops --------------------------------------------------------
    def overlaps(self, other) -> bool:
        if other._version != self._version:
            return False
        return (
            self.network_address._ip <= other.broadcast_address._ip
            and other.network_address._ip <= self.broadcast_address._ip
        )

    def subnet_of(self, other) -> bool:
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (
            other.network_address._ip
            <= self.network_address._ip
            <= self.broadcast_address._ip
            <= other.broadcast_address._ip
        )

    def supernet_of(self, other) -> bool:
        if self._version != other._version:
            # stdlib's message names (other, self) here (subnet_of reversed).
            raise _version_type_error(other, self)
        return (
            self.network_address._ip
            <= other.network_address._ip
            <= other.broadcast_address._ip
            <= self.broadcast_address._ip
        )

    def compare_networks(self, other) -> int:
        if self._version != other._version:
            raise TypeError("%s and %s are not of the same type" % (self, other))
        a = (self.network_address._ip, self.netmask._ip)
        b = (other.network_address._ip, other.netmask._ip)
        return (a > b) - (a < b)

    # -- subdivision ------------------------------------------------------------
    def subnets(self, prefixlen_diff=1, new_prefix=None):
        if self._prefixlen == self.max_prefixlen:
            # A host route has no smaller subnets: stdlib yields itself
            # before any argument validation.
            yield self
            return
        if new_prefix is not None:
            if new_prefix == self._prefixlen:
                yield self
                return
            if new_prefix < self._prefixlen:
                raise ValueError("new prefix must be longer")
            prefixlen_diff = new_prefix - self._prefixlen
        if prefixlen_diff < 0:
            raise ValueError("prefix length diff must be > 0")
        new_prefixlen = self._prefixlen + prefixlen_diff
        if new_prefixlen > self.max_prefixlen:
            raise ValueError(
                "prefix length diff %d is invalid for netblock %s"
                % (new_prefixlen, self)
            )
        if prefixlen_diff == 0:
            yield self
            return
        # stdlib computes the block layout with shifts; a non-int
        # prefixlen_diff fails here with its TypeError verbatim.
        hostmask = self._ALL_ONES >> new_prefixlen
        size = hostmask + 1
        base = self.network_address._ip
        cls = type(self)
        for i in range(1 << prefixlen_diff):
            yield cls((base + i * size, new_prefixlen))

    def supernet(self, prefixlen_diff=1, new_prefix=None):
        if self._prefixlen == 0:
            return self
        if new_prefix is not None:
            if new_prefix == self._prefixlen:
                return self
            if new_prefix > self._prefixlen:
                raise ValueError("new prefix must be shorter")
            prefixlen_diff = self._prefixlen - new_prefix
        if prefixlen_diff == 0:
            return self
        if prefixlen_diff > self._prefixlen:
            raise ValueError(
                "current prefixlen is %d, cannot have a prefixlen_diff of %d"
                % (self._prefixlen, prefixlen_diff)
            )
        # stdlib derives the supernet mask by shifting the current netmask;
        # a negative prefixlen_diff fails that shift with 'negative shift
        # count' (a raw ValueError), mirrored here.
        new_mask = (self.netmask._ip << prefixlen_diff) & self._ALL_ONES
        return type(self)((self.network_address._ip & new_mask,
                           self._prefixlen - prefixlen_diff))

    def address_exclude(self, other):
        if not other.subnet_of(self):
            raise ValueError("%s not contained in %s" % (other, self))
        if other == self:
            return []
        result = []
        current = self
        target_lo = other.network_address._ip
        while current._prefixlen < other._prefixlen:
            left, right = current.subnets()
            if target_lo >= right.network_address._ip:
                result.append(left)
                current = right
            else:
                result.append(right)
                current = left
        return result


@functools.total_ordering
class IPv6Network:
    """Drop-in replacement for ipaddress.IPv6Network."""

    __slots__ = (
        "network_address",
        "broadcast_address",
        "netmask",
        "hostmask",
        "_prefixlen",
    )

    _version = 6
    version = 6
    max_prefixlen = _IPV6_BITS
    _AddressClass = IPv6Address
    _prefix_cache: dict = {}

    def __init__(self, address, strict=True):
        na, plen, mask, _ = _build_network(
            IPv6Address, IPv6Network._prefix_cache, address, strict, 6
        )
        self.network_address = na
        self._prefixlen = plen
        self.netmask = IPv6Address(mask)
        self.hostmask = IPv6Address(mask ^ _V6_ALL_ONES)
        self.broadcast_address = IPv6Address(na._ip | (mask ^ _V6_ALL_ONES))

    # Shared implementations with IPv4Network (kept explicit per class so the
    # public class shape mirrors stdlib's independent classes).
    @property
    def prefixlen(self) -> int:
        return self._prefixlen

    @property
    def num_addresses(self) -> int:
        return self.broadcast_address._ip - self.network_address._ip + 1

    def __str__(self) -> str:
        return "%s/%d" % (self.network_address, self._prefixlen)

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    @property
    def with_prefixlen(self) -> str:
        return "%s/%d" % (self.network_address, self._prefixlen)

    @property
    def with_netmask(self) -> str:
        return "%s/%s" % (self.network_address, self.netmask)

    @property
    def with_hostmask(self) -> str:
        return "%s/%s" % (self.network_address, self.hostmask)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _BaseNetwork):
            return NotImplemented
        return (
            self._version == other._version
            and self.network_address._ip == other.network_address._ip
            and self.netmask._ip == other.netmask._ip
        )

    def __lt__(self, other):
        if not isinstance(other, _BaseNetwork):
            return NotImplemented
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (self.network_address._ip, self.netmask._ip) < (
            other.network_address._ip,
            other.netmask._ip,
        )

    def __hash__(self) -> int:
        return hash((self.network_address._ip, self.netmask._ip))

    def __contains__(self, other) -> bool:
        if other._version != self._version:
            return False
        if isinstance(other, _BaseNetwork):
            return False
        return (
            self.network_address._ip <= other._ip <= self.broadcast_address._ip
        )

    def __iter__(self):
        lo = self.network_address._ip
        hi = self.broadcast_address._ip
        cls = self._AddressClass
        return (cls(v) for v in range(lo, hi + 1))

    def __getitem__(self, n):
        if n >= 0:
            if self.network_address._ip + n > self.broadcast_address._ip:
                raise IndexError("address out of range")
        else:
            n += self.num_addresses
            if n < 0:
                raise IndexError("address out of range")
        return self._AddressClass(self.network_address._ip + n)

    def hosts(self):
        # v6 excludes only the subnet-router anycast address (no broadcast);
        # for /127 every address is a host. stdlib returns a *list* for the
        # single-address case and an iterator otherwise.
        lo = self.network_address._ip
        hi = self.broadcast_address._ip
        cls = self._AddressClass
        if hi == lo:
            return [cls(lo)]
        if hi - lo == 1:
            return (cls(v) for v in range(lo, hi + 1))
        return (cls(v) for v in range(lo + 1, hi + 1))

    @property
    def is_private(self) -> bool:
        return (
            self.network_address.is_private and self.broadcast_address.is_private
        )

    @property
    def is_global(self) -> bool:
        return self.network_address.is_global and self.broadcast_address.is_global

    @property
    def is_multicast(self) -> bool:
        return (
            self.network_address.is_multicast and self.broadcast_address.is_multicast
        )

    @property
    def is_loopback(self) -> bool:
        return (
            self.network_address.is_loopback and self.broadcast_address.is_loopback
        )

    @property
    def is_link_local(self) -> bool:
        return (
            self.network_address.is_link_local
            and self.broadcast_address.is_link_local
        )

    @property
    def is_site_local(self) -> bool:
        return (
            self.network_address.is_site_local
            and self.broadcast_address.is_site_local
        )

    @property
    def is_reserved(self) -> bool:
        return (
            self.network_address.is_reserved and self.broadcast_address.is_reserved
        )

    @property
    def is_unspecified(self) -> bool:
        return (
            self.network_address.is_unspecified
            and self.broadcast_address.is_unspecified
        )

    def overlaps(self, other) -> bool:
        if other._version != self._version:
            return False
        return (
            self.network_address._ip <= other.broadcast_address._ip
            and other.network_address._ip <= self.broadcast_address._ip
        )

    def subnet_of(self, other) -> bool:
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (
            other.network_address._ip
            <= self.network_address._ip
            <= self.broadcast_address._ip
            <= other.broadcast_address._ip
        )

    def supernet_of(self, other) -> bool:
        if self._version != other._version:
            # stdlib's message names (other, self) here (subnet_of reversed).
            raise _version_type_error(other, self)
        return (
            self.network_address._ip
            <= other.network_address._ip
            <= other.broadcast_address._ip
            <= self.broadcast_address._ip
        )

    def compare_networks(self, other) -> int:
        if self._version != other._version:
            raise TypeError("%s and %s are not of the same type" % (self, other))
        a = (self.network_address._ip, self.netmask._ip)
        b = (other.network_address._ip, other.netmask._ip)
        return (a > b) - (a < b)

    def subnets(self, prefixlen_diff=1, new_prefix=None):
        if self._prefixlen == self.max_prefixlen:
            # A host route has no smaller subnets: stdlib yields itself
            # before any argument validation.
            yield self
            return
        if new_prefix is not None:
            if new_prefix == self._prefixlen:
                yield self
                return
            if new_prefix < self._prefixlen:
                raise ValueError("new prefix must be longer")
            prefixlen_diff = new_prefix - self._prefixlen
        if prefixlen_diff < 0:
            raise ValueError("prefix length diff must be > 0")
        new_prefixlen = self._prefixlen + prefixlen_diff
        if new_prefixlen > self.max_prefixlen:
            raise ValueError(
                "prefix length diff %d is invalid for netblock %s"
                % (new_prefixlen, self)
            )
        if prefixlen_diff == 0:
            yield self
            return
        hostmask = self._ALL_ONES >> new_prefixlen
        size = hostmask + 1
        base = self.network_address._ip
        cls = type(self)
        for i in range(1 << prefixlen_diff):
            yield cls((base + i * size, new_prefixlen))

    def supernet(self, prefixlen_diff=1, new_prefix=None):
        if self._prefixlen == 0:
            return self
        if new_prefix is not None:
            if new_prefix == self._prefixlen:
                return self
            if new_prefix > self._prefixlen:
                raise ValueError("new prefix must be shorter")
            prefixlen_diff = self._prefixlen - new_prefix
        if prefixlen_diff == 0:
            return self
        if prefixlen_diff > self._prefixlen:
            raise ValueError(
                "current prefixlen is %d, cannot have a prefixlen_diff of %d"
                % (self._prefixlen, prefixlen_diff)
            )
        new_mask = (self.netmask._ip << prefixlen_diff) & self._ALL_ONES
        return type(self)((self.network_address._ip & new_mask,
                           self._prefixlen - prefixlen_diff))

    def address_exclude(self, other):
        if not other.subnet_of(self):
            raise ValueError("%s not contained in %s" % (other, self))
        if other == self:
            return []
        result = []
        current = self
        target_lo = other.network_address._ip
        while current._prefixlen < other._prefixlen:
            left, right = current.subnets()
            if target_lo >= right.network_address._ip:
                result.append(left)
                current = right
            else:
                result.append(right)
                current = left
        return result


_BaseNetwork = (IPv4Network, IPv6Network)

# The v4 network's hosts()/supernet() need the same v4-specific rules the v6
# class already customizes; v4 hosts exclude network AND broadcast below /31.
_IPv4Network_hosts = IPv4Network.hosts


def _ipv4_hosts(self):
    # v4 excludes network AND broadcast below /31; for /31 and /32 every
    # address is a host. stdlib returns a *list* for the single-address case
    # and an iterator otherwise.
    lo = self.network_address._ip
    hi = self.broadcast_address._ip
    cls = self._AddressClass
    if hi == lo:
        return [cls(lo)]
    if hi - lo == 1:
        return (cls(v) for v in range(lo, hi + 1))
    return (cls(v) for v in range(lo + 1, hi))


IPv4Network.hosts = _ipv4_hosts

# subnets()/supernet() shift by ALL_ONES per version.
IPv4Network._ALL_ONES = _V4_ALL_ONES
IPv6Network._ALL_ONES = _V6_ALL_ONES


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


@functools.total_ordering
class IPv4Interface(IPv4Address):
    """Drop-in replacement for ipaddress.IPv4Interface."""

    __slots__ = ("network",)

    def __init__(self, address):
        addr_in, prefix_in = _split_network_input(address, 4)
        self._ip = _reference.parse_ipv4(addr_in)
        self.network = IPv4Network(address, strict=False)

    # -- representation -----------------------------------------------------
    @property
    def ip(self) -> IPv4Address:
        return IPv4Address(self._ip)

    @property
    def netmask(self) -> IPv4Address:
        return self.network.netmask

    @property
    def hostmask(self) -> IPv4Address:
        return self.network.hostmask

    @property
    def with_prefixlen(self) -> str:
        return "%s/%d" % (self.ip, self.network.prefixlen)

    @property
    def with_netmask(self) -> str:
        return "%s/%s" % (self.ip, self.netmask)

    @property
    def with_hostmask(self) -> str:
        return "%s/%s" % (self.ip, self.hostmask)

    @property
    def exploded(self) -> str:
        return "%s/%d" % (self.ip.exploded, self.network.prefixlen)

    @property
    def compressed(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "%s/%d" % (self.ip, self.network.prefixlen)

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    # -- comparisons ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, _BaseInterface):
            return False
        return (
            self._version == other._version
            and self._ip == other._ip
            and self.network == other.network
        )

    def __lt__(self, other):
        if not isinstance(other, _BaseInterface):
            return False
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (
            self._ip,
            self.network.prefixlen,
            self.network.network_address._ip,
        ) < (other._ip, other.network.prefixlen, other.network.network_address._ip)

    def __hash__(self) -> int:
        return hash((self._ip, self.network.prefixlen, self.network.network_address._ip))


@functools.total_ordering
class IPv6Interface(IPv6Address):
    """Drop-in replacement for ipaddress.IPv6Interface."""

    __slots__ = ("network",)

    def __init__(self, address):
        addr_in, prefix_in = _split_network_input(address, 6)
        self._ip, self._scope_id = _reference.parse_ipv6(addr_in)
        # The network is built from the original input: a scoped address
        # that survives masking keeps its zone in the network (3.12 parity).
        self.network = IPv6Network(address, strict=False)

    # -- representation -----------------------------------------------------
    @property
    def ip(self) -> IPv6Address:
        return IPv6Address(self._ip)  # scope id intentionally dropped

    @property
    def netmask(self) -> IPv6Address:
        return self.network.netmask

    @property
    def hostmask(self) -> IPv6Address:
        return self.network.hostmask

    @property
    def with_prefixlen(self) -> str:
        return "%s/%d" % (self.ip, self.network.prefixlen)

    @property
    def with_netmask(self) -> str:
        return "%s/%s" % (self.ip, self.netmask)

    @property
    def with_hostmask(self) -> str:
        return "%s/%s" % (self.ip, self.hostmask)

    @property
    def exploded(self) -> str:
        return "%s/%d" % (self.ip.exploded, self.network.prefixlen)

    @property
    def compressed(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "%s/%d" % (
            _reference.ipv6_to_str(self._ip, self._scope_id),
            self.network.prefixlen,
        )

    def __repr__(self) -> str:
        return "%s('%s')" % (type(self).__name__, self)

    # -- comparisons ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, _BaseInterface):
            return False
        return (
            self._version == other._version
            and self._ip == other._ip
            and self._scope_id == other._scope_id
            and self.network == other.network
        )

    def __lt__(self, other):
        if not isinstance(other, _BaseInterface):
            return False
        if self._version != other._version:
            raise _version_type_error(self, other)
        return (
            self._ip,
            self.network.prefixlen,
            self.network.network_address._ip,
        ) < (other._ip, other.network.prefixlen, other.network.network_address._ip)

    def __hash__(self) -> int:
        return hash((self._ip, self.network.prefixlen, self.network.network_address._ip))


_BaseInterface = (IPv4Interface, IPv6Interface)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def ip_address(address):
    """Drop-in replacement for ipaddress.ip_address."""
    try:
        return IPv4Address(address)
    except (AddressValueError, NetmaskValueError):
        pass
    try:
        return IPv6Address(address)
    except (AddressValueError, NetmaskValueError):
        pass
    raise ValueError("%r does not appear to be an IPv4 or IPv6 address" % (address,))


def ip_network(address, strict=True):
    """Drop-in replacement for ipaddress.ip_network."""
    try:
        return IPv4Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass
    try:
        return IPv6Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass
    raise ValueError("%r does not appear to be an IPv4 or IPv6 network" % (address,))


def ip_interface(address):
    """Drop-in replacement for ipaddress.ip_interface."""
    try:
        return IPv4Interface(address)
    except (AddressValueError, NetmaskValueError):
        pass
    try:
        return IPv6Interface(address)
    except (AddressValueError, NetmaskValueError):
        pass
    raise ValueError("%r does not appear to be an IPv4 or IPv6 interface" % (address,))


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def collapse_addresses(addresses):
    """Drop-in replacement for ipaddress.collapse_addresses.

    Returns an iterator of the canonical collapsed networks. Mixed-version
    inputs raise the same TypeError as stdlib (same ordering of the offending
    pair in the message).
    """
    addrs = []
    nets = []
    addr_versions = set()
    net_versions = set()
    for item in addresses:
        if isinstance(item, _BaseAddress):
            addr_versions.add(item._version)
            addrs.append(item)
        elif isinstance(item, _BaseNetwork):
            net_versions.add(item._version)
            nets.append(item)
        else:
            raise TypeError(
                "%r is not an IP address or network" % (item,)
            )
    if len(addr_versions | net_versions) > 1:
        # Mixed versions: replicate stdlib's error exactly. Sorting each
        # family with the classes' own __lt__ raises the mixed-version
        # TypeError with the offending pair in stdlib's comparison order.
        addrs = sorted(addrs)
        nets = sorted(nets)
        if addr_versions and net_versions:
            addr_net = _net_from_parts(addrs[0]._version, addrs[0]._ip,
                                       addrs[0].max_prefixlen)
            raise _version_type_error(addr_net, nets[0])
        # Unreachable in practice (the sorts above raise first), kept for
        # completeness of the error contract.
        raise _version_type_error(addrs[0], nets[0])
    if addrs:
        version = addrs[0]._version
        bits = addrs[0].max_prefixlen
    elif nets:
        version = nets[0]._version
        bits = nets[0].max_prefixlen
    else:
        return iter([])
    # Same-version fast path: the canonical cover is order-independent, so
    # there is no need to sort Python objects — the kernel (or the reference
    # merge) sorts plain (base, prefixlen) integer pairs instead.
    items = [(a._ip, bits) for a in addrs]
    items.extend((n.network_address._ip, n._prefixlen) for n in nets)
    merged = _native.collapse(items, bits, version)
    return (_net_from_parts(version, base, plen) for base, plen in merged)


def _net_from_parts(version: int, base: int, prefixlen: int):
    if version == 4:
        return IPv4Network((base, prefixlen))
    return IPv6Network((base, prefixlen))


def collapse_batch(networks):
    """Vectorized collapse: same result as collapse_addresses, computed in bulk.

    Accepts an iterable of network objects (addresses are taken as /32 and
    /128 routes, exactly like collapse_addresses). Mixed versions raise the
    same TypeError as collapse_addresses.
    """
    return list(collapse_addresses(networks))


def summarize_address_range(first, last):
    """Drop-in replacement for ipaddress.summarize_address_range."""
    if first._version != last._version:
        raise _version_type_error(first, last)
    if first._ip > last._ip:
        raise ValueError("last IP address must be greater than first")
    bits = first.max_prefixlen
    version = first._version
    return (
        _net_from_parts(version, base, plen)
        for base, plen in _reference.summarize_range(first._ip, last._ip, bits)
    )


def get_mixed_type_key(obj):
    """Drop-in replacement for ipaddress.get_mixed_type_key."""
    if isinstance(obj, _BaseNetwork):
        return (obj._version, obj.network_address, obj.netmask)
    if isinstance(obj, _BaseAddress):
        return (obj._version, obj)
    return NotImplemented


def v4_int_to_packed(value: int) -> bytes:
    return value.to_bytes(4, "big")


def v6_int_to_packed(value: int) -> bytes:
    return value.to_bytes(16, "big")


# ---------------------------------------------------------------------------
# Vectorized batch API (native kernel + NumPy fallback; identical results)
# ---------------------------------------------------------------------------


def parse_many(addresses):
    """Bulk `ip_address`: parse a list of strings/ints/bytes in one pass.

    Returns a list of IPv4Address/IPv6Address objects, element-for-element
    identical to ``[ip_address(a) for a in addresses]``; the first malformed
    item raises exactly the exception ip_address() would raise for it.
    """
    items = list(addresses)
    native = _native.try_load()
    if native is not None:
        return _native.parse_many(native, items)
    return [ip_address(a) for a in items]


def contains_many(networks, addresses):
    """Bulk membership: for each address, is it in ANY of the networks?

    Returns a numpy bool array of shape ``(len(addresses),)``. Cross-version
    pairs never match (mirroring ``in``). Element-for-element identical to
    ``[any(a in n for n in networks) for a in addresses]``.
    """
    return _native.contains_many(list(networks), list(addresses))
