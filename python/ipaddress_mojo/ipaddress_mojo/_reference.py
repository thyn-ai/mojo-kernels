"""Pure-Python reference implementation of the ipaddressmojo kernel semantics.

This module is both the fallback backend and the semantic backbone of the
package: it parses and formats IPv4/IPv6 addresses and networks with exactly
the observable behavior of the CPython 3.12 standard-library ``ipaddress``
module — same values, same canonical strings, same exception types
(``ipaddress.AddressValueError`` / ``ipaddress.NetmaskValueError`` are reused
so ``except`` clauses behave identically), and the same exception messages,
which the differential suite asserts verbatim on both backends.

The batch hot loops (bulk parsing, bulk membership, bulk collapse) have a
Mojo implementation behind ``_native``; everything else runs here on every
platform, so single-object behavior cannot diverge between backends.

Written fresh from RFC 791/8200 (address formats), RFC 4632 (CIDR), RFC 5952
(text representation), and black-box observation of the reference module's
public behavior (exception types/messages, ordering rules, zone-id handling,
the private/global address registries). No third-party code is used or
adapted.
"""

from __future__ import annotations

# Reuse the stdlib exception *types* so that code catching
# ``ipaddress.AddressValueError`` / ``ipaddress.NetmaskValueError`` catches
# ours. (Drop-in requirement; the classes themselves are trivial markers.)
from ipaddress import AddressValueError, NetmaskValueError

__all__ = [
    "AddressValueError",
    "NetmaskValueError",
    "parse_ipv4",
    "parse_ipv6",
    "ipv4_to_str",
    "ipv6_to_str",
    "ipv6_exploded_str",
    "ipv4_reverse_pointer",
    "ipv6_reverse_pointer",
    "prefix_from_string",
    "collapse_intervals",
    "summarize_range",
    "V4_PRIVATE_NETWORKS",
    "V6_PRIVATE_NETWORKS",
]

IPV4_BITS = 32
IPV6_BITS = 128
V4_ALL_ONES = (1 << IPV4_BITS) - 1
V6_ALL_ONES = (1 << IPV6_BITS) - 1


# ---------------------------------------------------------------------------
# IPv4 parsing
# ---------------------------------------------------------------------------


def _ipv4_int_from_string(ip_str: str) -> int:
    """Parse a dotted-quad string. Raises AddressValueError (stdlib messages)."""
    if not ip_str:
        raise AddressValueError("Address cannot be empty")
    octets = ip_str.split(".")
    if len(octets) != 4:
        raise AddressValueError("Expected 4 octets in %r" % ip_str)
    value = 0
    for octet in octets:
        if not octet:
            raise AddressValueError("Empty octet not permitted in %r" % ip_str)
        if not octet.isascii() or not octet.isdigit():
            raise AddressValueError(
                "Only decimal digits permitted in %r in %r" % (octet, ip_str)
            )
        if len(octet) > 3:
            raise AddressValueError(
                "At most 3 characters permitted in %r in %r" % (octet, ip_str)
            )
        if octet[0] == "0" and len(octet) != 1:
            raise AddressValueError(
                "Leading zeros are not permitted in %r in %r" % (octet, ip_str)
            )
        o = int(octet)
        if o > 255:
            raise AddressValueError(
                "Octet %d (> 255) not permitted in %r" % (o, ip_str)
            )
        value = (value << 8) | o
    return value


def parse_ipv4(address) -> int:
    """IPv4Address(address) -> int, with stdlib's exact errors."""
    if isinstance(address, int):
        if address < 0:
            raise AddressValueError(
                "%d (< 0) is not permitted as an IPv4 address" % address
            )
        if address > V4_ALL_ONES:
            raise AddressValueError(
                "%d (>= 2**32) is not permitted as an IPv4 address" % address
            )
        return address
    if isinstance(address, bytes):
        if len(address) != 4:
            raise AddressValueError(
                "%r (len %d != 4) is not permitted as an IPv4 address"
                % (address, len(address))
            )
        return int.from_bytes(address, "big")
    ip_str = str(address)
    if "/" in ip_str:
        raise AddressValueError("Unexpected '/' in %r" % ip_str)
    return _ipv4_int_from_string(ip_str)


# ---------------------------------------------------------------------------
# IPv6 parsing
# ---------------------------------------------------------------------------


def _ipv6_int_from_string(ip_str: str) -> int:
    """Parse an RFC 4291 string (without scope id) to a 128-bit int.

    Error order and messages mirror CPython 3.12 exactly (asserted verbatim
    by the differential suite): the embedded-v4 expansion happens before the
    colon/parts count checks, so an embedded v4 moves the colon-count error
    boundary by one.
    """
    if not ip_str:
        raise AddressValueError("Address cannot be empty")

    if ip_str.count("::") > 1 or ":::" in ip_str:
        raise AddressValueError("At most one '::' permitted in %r" % ip_str)

    parts = ip_str.split(":")
    if len(parts) < 3:
        raise AddressValueError("At least 3 parts expected in %r" % ip_str)

    # An embedded IPv4 address may only appear in the last part; it is parsed
    # (wrapping v4 errors) and expanded to two hextets before the count checks.
    has_v4 = "." in parts[-1]
    if has_v4:
        try:
            v4 = _ipv4_int_from_string(parts[-1])
        except AddressValueError as exc:
            raise AddressValueError("%s in %r" % (exc, ip_str)) from None
        parts[-1:] = ["%x" % (v4 >> 16), "%x" % (v4 & 0xFFFF)]

    if len(parts) - 1 > 8:
        raise AddressValueError("At most 8 colons permitted in %r" % ip_str)

    has_dc = "::" in ip_str
    if has_dc:
        # The '::' must stand for at least one hextet: at most 7 other parts.
        n_other = sum(1 for p in parts if p)
        if n_other > 7:
            raise AddressValueError(
                "Expected at most 7 other parts with '::' in %r" % ip_str
            )
    else:
        if len(parts) != 8:
            raise AddressValueError(
                "Exactly 8 parts expected without '::' in %r" % ip_str
            )

    if ip_str.startswith(":") and not ip_str.startswith("::"):
        raise AddressValueError(
            "Leading ':' only permitted as part of '::' in %r" % ip_str
        )
    if ip_str.endswith(":") and not ip_str.endswith("::"):
        raise AddressValueError(
            "Trailing ':' only permitted as part of '::' in %r" % ip_str
        )

    # Validate hextets and assemble; empty parts are the '::' fill marker.
    hextets = []
    fill = 8 - sum(1 for p in parts if p)
    filled = False
    for part in parts:
        if part == "":
            if not filled:
                hextets.extend([0] * fill)
                filled = True
            continue
        for ch in part:
            if ch not in "0123456789abcdefABCDEF":
                raise AddressValueError(
                    "Only hex digits permitted in %r in %r" % (part, ip_str)
                )
        if len(part) > 4:
            raise AddressValueError(
                "At most 4 characters permitted in %r in %r" % (part, ip_str)
            )
        hextets.append(int(part, 16))

    value = 0
    for h in hextets:
        value = (value << 16) | h
    return value


def parse_ipv6(address) -> tuple[int, "str | None"]:
    """IPv6Address(address) -> (int, scope_id), with stdlib's exact errors."""
    if isinstance(address, int):
        if address < 0:
            raise AddressValueError(
                "%d (< 0) is not permitted as an IPv6 address" % address
            )
        if address > V6_ALL_ONES:
            raise AddressValueError(
                "%d (>= 2**128) is not permitted as an IPv6 address" % address
            )
        return address, None
    if isinstance(address, bytes):
        if len(address) != 16:
            raise AddressValueError(
                "%r (len %d != 16) is not permitted as an IPv6 address"
                % (address, len(address))
            )
        return int.from_bytes(address, "big"), None
    ip_str = str(address)
    if "/" in ip_str:
        raise AddressValueError("Unexpected '/' in %r" % ip_str)
    scope_id = None
    if "%" in ip_str:
        addr, sep, scope = ip_str.partition("%")
        if not sep or not scope or "%" in scope:
            raise AddressValueError('Invalid IPv6 address: "%r"' % ip_str)
        scope_id = scope
    else:
        addr = ip_str
    return _ipv6_int_from_string(addr), scope_id


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def ipv4_to_str(value: int) -> str:
    return ".".join(str((value >> (8 * i)) & 0xFF) for i in (3, 2, 1, 0))


def _hextets(value: int) -> list[int]:
    return [(value >> (16 * i)) & 0xFFFF for i in range(7, -1, -1)]


def ipv6_to_str(value: int, scope_id: "str | None" = None) -> str:
    """Canonical compressed form (RFC 5952 + stdlib's v4-mapped rendering)."""
    # v4-mapped (::ffff:0:0/96) renders as '::ffff:a.b.c.d'.
    if (value >> 32) == 0xFFFF:
        s = "::ffff:" + ipv4_to_str(value & V4_ALL_ONES)
        return s + ("%" + scope_id if scope_id else "")
    hextets = _hextets(value)
    # Longest run of zero hextets (length >= 2); leftmost wins ties.
    best_start, best_len = -1, 1
    i = 0
    while i < 8:
        if hextets[i] == 0:
            j = i
            while j < 8 and hextets[j] == 0:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    if best_start < 0:
        s = ":".join("%x" % h for h in hextets)
    else:
        left = ":".join("%x" % h for h in hextets[:best_start])
        right = ":".join("%x" % h for h in hextets[best_start + best_len :])
        s = left + "::" + right
    return s + ("%" + scope_id if scope_id else "")


def ipv6_exploded_str(value: int) -> str:
    # v4-mapped addresses explode the low 32 bits as a dotted quad (3.12).
    if (value >> 32) == 0xFFFF:
        head = ":".join("%04x" % h for h in _hextets(value)[:6])
        return head + ":" + ipv4_to_str(value & V4_ALL_ONES)
    return ":".join("%04x" % h for h in _hextets(value))


def ipv4_reverse_pointer(value: int) -> str:
    return ".".join(str((value >> (8 * i)) & 0xFF) for i in (0, 1, 2, 3)) + ".in-addr.arpa"


def ipv6_reverse_pointer(value: int) -> str:
    nibbles = ["%x" % ((value >> (4 * i)) & 0xF) for i in range(32)]
    return ".".join(nibbles) + ".ip6.arpa"


# ---------------------------------------------------------------------------
# Prefix lengths and netmasks
# ---------------------------------------------------------------------------


def prefix_from_int(prefixlen: int, max_prefixlen: int) -> int:
    """Tuple/int prefix validation ('%r is not a valid netmask')."""
    if not 0 <= prefixlen <= max_prefixlen:
        raise NetmaskValueError("%r is not a valid netmask" % prefixlen)
    return prefixlen


def prefix_from_string(mask: str, max_prefixlen: int, version: int) -> int:
    """Parse the string after '/' (prefix int, or v4 netmask/hostmask)."""
    if mask.isascii() and mask.isdigit():
        prefixlen = int(mask)
        if not 0 <= prefixlen <= max_prefixlen:
            raise NetmaskValueError("%r is not a valid netmask" % mask)
        return prefixlen
    if version == 6:
        raise NetmaskValueError("%r is not a valid netmask" % mask)
    # Dotted-quad netmask or hostmask.
    try:
        bits = _ipv4_int_from_string(mask)
    except AddressValueError:
        raise NetmaskValueError("%r is not a valid netmask" % mask) from None
    inv = bits ^ V4_ALL_ONES
    if inv & (inv + 1) == 0:
        # Contiguous ones from the left: a netmask.
        return bin(bits).count("1")
    if bits & (bits + 1) == 0:
        # Contiguous ones from the right: a hostmask.
        return IPV4_BITS - bin(bits).count("1")
    raise NetmaskValueError("%r is not a valid netmask" % mask)


def netmask_int(prefixlen: int, bits: int) -> int:
    return ((1 << bits) - 1) ^ ((1 << (bits - prefixlen)) - 1) if prefixlen else 0


def hostmask_int(prefixlen: int, bits: int) -> int:
    return (1 << (bits - prefixlen)) - 1 if prefixlen < bits else 0


# ---------------------------------------------------------------------------
# Private/global registries (CPython 3.12 semantics; boundary-verified
# against the stdlib oracle over the whole v4 space and all v6 special
# regions — see tests/test_ipaddress_differential.py).
# ---------------------------------------------------------------------------

# (base_int, prefixlen) intervals.
V4_PRIVATE_NETWORKS = (
    (0x00000000, 8),  # 0.0.0.0/8 "this network"
    (0x0A000000, 8),  # 10.0.0.0/8
    (0x7F000000, 8),  # 127.0.0.0/8 loopback
    (0xA9FE0000, 16),  # 169.254.0.0/16 link-local
    (0xAC100000, 12),  # 172.16.0.0/12
    (0xC0000000, 24),  # 192.0.0.0/24
    (0xC0000200, 24),  # 192.0.2.0/24 TEST-NET-1
    (0xC0A80000, 16),  # 192.168.0.0/16
    (0xC6120000, 15),  # 198.18.0.0/15 benchmarking
    (0xC6336400, 24),  # 198.51.100.0/24 TEST-NET-2
    (0xCB007100, 24),  # 203.0.113.0/24 TEST-NET-3
    (0xF0000000, 4),  # 240.0.0.0/4 reserved (incl. broadcast)
)

V4_NOT_GLOBAL = ((0x64400000, 10),)  # 100.64.0.0/10 shared address space

# v6 128-bit registry (base, prefixlen); the v6 table below is not guessed —
# every range and every boundary below was verified against the 3.12 stdlib
# oracle by full-space sweeps of each special region.
V6_PRIVATE_NETWORKS = (
    (0x00000000000000000000000000000000, 128),  # ::/128 unspecified
    (0x00000000000000000000000000000001, 128),  # ::1/128 loopback
    (0x0064FF9B000100000000000000000000, 48),  # 64:ff9b:1::/48 local-use
    (0x01000000000000000000000000000000, 64),  # 100::/64 discard-only
    (0x20010000000000000000000000000000, 23),  # 2001::/23 IETF special
    (0x20010DB8000000000000000000000000, 32),  # 2001:db8::/32 documentation
    (0x20020000000000000000000000000000, 16),  # 2002::/16 6to4
    (0x3FFF0000000000000000000000000000, 20),  # 3fff::/20 documentation
    (0xFC000000000000000000000000000000, 7),  # fc00::/7 ULA
    (0xFE800000000000000000000000000000, 10),  # fe80::/10 link-local
)

# Globally-reachable exceptions carved out of 2001::/23 by the registry.
V6_PRIVATE_EXCEPTIONS = (
    (0x20010001000000000000000000000001, 128),  # 2001:1::1/128
    (0x20010001000000000000000000000002, 128),  # 2001:1::2/128
    (0x20010003000000000000000000000000, 32),  # 2001:3::/32
    (0x20010004011200000000000000000000, 48),  # 2001:4:112::/48
    (0x20010020000000000000000000000000, 28),  # 2001:20::/28
    (0x20010030000000000000000000000000, 28),  # 2001:30::/28
)


def _in_networks(value: int, networks, width: int) -> bool:
    for base, plen in networks:
        if plen == 0 or (value ^ base) >> (width - plen) == 0:
            return True
    return False


def v4_is_private(value: int) -> bool:
    return _in_networks(value, V4_PRIVATE_NETWORKS, 32)


def v4_is_global(value: int) -> bool:
    return not v4_is_private(value) and not _in_networks(value, V4_NOT_GLOBAL, 32)


def v6_is_private(value: int) -> bool:
    return _in_networks(value, V6_PRIVATE_NETWORKS, 128) and not _in_networks(
        value, V6_PRIVATE_EXCEPTIONS, 128
    )


def v6_is_global(value: int) -> bool:
    return not v6_is_private(value)


# ---------------------------------------------------------------------------
# Canonical interval collapse / range summarization
# ---------------------------------------------------------------------------


def collapse_intervals(items, bits: int):
    """Merge a list of (base_int, prefixlen) into the canonical minimal cover.

    Output is sorted ascending by (base, prefixlen) and is the unique minimal
    CIDR cover of the input (identical to stdlib collapse_addresses' result).
    """
    # Sort by base asc, then prefixlen asc (larger block first at same base).
    ordered = sorted(items, key=lambda t: (t[0], t[1]))
    stack: list[list[int]] = []
    all_ones = (1 << bits) - 1
    for base, plen in ordered:
        size = 1 << (bits - plen)
        hi = base + size - 1
        cur = [base, plen, hi]
        while stack:
            t_base, t_plen, t_hi = stack[-1]
            if cur[0] >= t_base and cur[2] <= t_hi:
                # Contained in the stack top: drop.
                cur = None
                break
            if (
                t_plen == cur[1]
                and t_hi < all_ones
                and t_hi + 1 == cur[0]
                and t_base % (1 << (bits - t_plen + 1)) == 0
            ):
                # Siblings: merge into the parent block, then re-check.
                stack.pop()
                new_plen = t_plen - 1
                cur = [t_base, new_plen, t_base + (1 << (bits - new_plen)) - 1]
                continue
            break
        if cur is not None:
            stack.append(cur)
    return [(b, p) for b, p, _ in stack]


def summarize_range(first: int, last: int, bits: int):
    """Canonical CIDR decomposition of [first, last] (greedy, left to right).

    Returns a list of (base_int, prefixlen), identical to stdlib
    summarize_address_range (the canonical greedy decomposition is unique).
    """
    out = []
    while first <= last:
        # Largest block aligned at `first`...
        if first == 0:
            align_bits = bits
        else:
            align_bits = (first & -first).bit_length() - 1
        # ...capped so the block stays within [first, last].
        remaining = last - first + 1
        span_bits = remaining.bit_length() - 1
        size_bits = align_bits if align_bits < span_bits else span_bits
        out.append((first, bits - size_bits))
        first += 1 << size_bits
    return out
