"""Differential suite: ipaddress_mojo vs the CPython stdlib ipaddress oracle.

Runs twice (see scripts/test_all_ipaddress.sh): once against the native Mojo
kernel, once with IPADDRESS_MOJO_DISABLE_NATIVE=1 forcing the pure-Python
fallback. Both backends must agree with the oracle exactly:

* parsing/formatting of addresses, networks and interfaces: identical values,
  identical canonical strings, and identical exception types *and messages*
  for malformed input (zero tolerance),
* operators (ordering, hashing, arithmetic, membership, subnets/supernet/
  overlaps/collapse/summarize): identical results,
* the vectorized batch API (parse_many / contains_many / collapse_batch):
  identical to the equivalent stdlib loops,
* the private/global address registries: boundary sweep + seeded sampling.

The oracle is the standard library of the interpreter running the tests, so
parity is pinned to that exact CPython version (3.12 in the repo's pixi env).
"""

from __future__ import annotations

import random

import pytest

import ipaddress as ref
import ipaddress_mojo as mine
from ipaddress_mojo import _native

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def outcome(fn):
    """Run fn; return ("ok", value) or ("err", type_name, message)."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def check(label, ref_fn, mine_fn):
    r1, r2 = outcome(ref_fn), outcome(mine_fn)
    if r1[0] == "err" or r2[0] == "err":
        assert r1 == r2, f"{label}:\n  ref ={r1}\n  mine={r2}"
    else:
        v1, v2 = r1[1], r2[1]
        assert type(v1) is type(v2) or type(v1).__name__ == type(v2).__name__, (
            f"{label}: type {type(v1)} vs {type(v2)}"
        )
        assert v1 == v2 and repr(v1) == repr(v2), (
            f"{label}:\n  ref ={v1!r}\n  mine={v2!r}"
        )


def check_exc(label, ref_fn, mine_fn):
    """For operations whose results are exceptions on both sides."""
    r1, r2 = outcome(ref_fn), outcome(mine_fn)
    assert r1 == r2, f"{label}:\n  ref ={r1}\n  mine={r2}"


# ---------------------------------------------------------------------------
# Address parsing corpus (valid + malformed)
# ---------------------------------------------------------------------------

V4_STRINGS = [
    "0.0.0.0", "1.2.3.4", "255.255.255.255", "10.0.0.1", "127.0.0.1",
    "192.168.1.1", "8.8.8.8", "100.64.0.1", "169.254.1.1", "224.0.0.1",
    "240.0.0.1",
    # malformed
    "", "1.2.3", "1.2.3.4.5", "1..2.3", "01.2.3.4", "1.2.3.256", "a.2.3.4",
    "1.2.3.4 ", " 1.2.3.4", "0x10.1.2.3", "1.2.3.4\n", "+1.2.3.4", "-1.2.3.4",
    "1.2.3.04", "00.0.0.0", "1,2,3,4", "1.2.3.4/24", "٣.2.3.4", "1.2.3.４",
    "1.2.3.99999999999999999999", "....", "1.2.3.", ".1.2.3", "1.2.3.4'",
    "1.2\n3", "1.2.3.²", "1.2.3.4\x00", "1.2.3.4/24/25",
]

V6_STRINGS = [
    "::", "::1", "1::", "1::2", "1:2:3:4:5:6:7:8", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    "2001:db8::1", "aBcd::1", "0:0:0:0:0:0:0:0", "1:2:3:4:5:6:7:0",
    "1:2:3:4:5:6::8", "1:2:3:4:5::7:8", "0:0:0:0:0:ffff:1.2.3.4",
    "::ffff:1.2.3.4", "::ffff:0.0.0.0", "::1.2.3.4", "1:2:3:4:5:6:1.2.3.4",
    "::ffff:0:1.2.3.4", "64:ff9b::1.2.3.4", "fe80::1%eth0", "fe80::%eth0",
    "::%0", "fe80::1%et h0", "fe80::1%25eth0", "1:2:3:4:5:6:7::",
    "1:2:3:4:5:6:7:0",
    # malformed
    "", "1::2::3", "1:2:3:4:5:6:7:8:9", "1:2:3:4:5:6:7", "12345::", "gg::",
    "::ffff:1.2.3", "::ffff:1.2.3.256", "::ffff:01.2.3.4", "1.2.3.4::",
    "fe80::1%", "1::2%", "1:2:3:4:5:6:7:1.2.3.4", ":1:2:3:4:5:6:7:8",
    "1:2:3:4:5:6:7:8:", "1::2%%3", "1:::2", "::::", ":", ":x", "x",
    "1:2:3:4:5:6:7:", ":1:2:3:4:5:6:7", "::1:2:3:4:5:6:7:8", "1:2:3:4:5:6:7:8::",
    "1:2:3:4:5:6:7:8:9:10", "1:2:3::4:5:6:7:8", "1::2:3:4:5:6:7:8",
    "::ffff:1.2.3.4.5", "1.2.3.4:5::6", "::x", "1.2.3.4", "123456789::",
    "12345x::", "x2345::", "0x1::", " 1::", "1:: ", "%eth0", "%", "%%",
    "fe80::1%eth/0", "::ffff:1.2.3%eth0", "::ffff:999.1.1.1",
    "1:2:3:4:5:6:7:999.1.1.1", ":1:2:3:4:5:6:1.2.3.4", "1:2:3:4:5:6::1.2.3.4",
    "1:2:3:4:5:6:7::1.2.3.4", "1:2:3:4:5::1.2.3.4", "::ffff:0:0:1.2.3.4",
]

ADDRESS_INPUTS = [
    0, 1, 2**32 - 1, 2**32, 2**33, 2**128 - 1, 2**128, -1, -100,
    b"\x01\x02\x03\x04", b"\x00" * 16, b"\x01\x02\x03", b"\x00" * 5,
    b"\x00" * 17, 1.5, None, True, False, ["1.2.3.4"],
]


@pytest.mark.parametrize("s", V4_STRINGS)
def test_ipv4address_parse(s):
    check(f"IPv4Address({s!r})", lambda: ref.IPv4Address(s), lambda: mine.IPv4Address(s))


@pytest.mark.parametrize("s", V6_STRINGS)
def test_ipv6address_parse(s):
    check(f"IPv6Address({s!r})", lambda: ref.IPv6Address(s), lambda: mine.IPv6Address(s))


@pytest.mark.parametrize("s", V4_STRINGS + V6_STRINGS)
def test_ip_address_factory_strings(s):
    check(f"ip_address({s!r})", lambda: ref.ip_address(s), lambda: mine.ip_address(s))


@pytest.mark.parametrize("v", ADDRESS_INPUTS)
def test_address_inputs(v):
    check(f"IPv4Address({v!r})", lambda: ref.IPv4Address(v), lambda: mine.IPv4Address(v))
    check(f"IPv6Address({v!r})", lambda: ref.IPv6Address(v), lambda: mine.IPv6Address(v))
    check(f"ip_address({v!r})", lambda: ref.ip_address(v), lambda: mine.ip_address(v))


# ---------------------------------------------------------------------------
# Formatting grid
# ---------------------------------------------------------------------------

FORMAT_ADDRS = [
    "0.0.0.0", "1.2.3.4", "255.255.255.255", "10.0.0.1",
    "::", "::1", "1::", "1:2:3:4:5:6:7:8", "2001:db8::1", "::ffff:1.2.3.4",
    "::ffff:0:0", "::1.2.3.4", "64:ff9b::1.2.3.4", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    "1:0:0:2:0:0:0:3", "1:0:0:2:0:0:3:4", "fe80::1%eth0", "fe80::%eth0",
]


@pytest.mark.parametrize("s", FORMAT_ADDRS)
def test_formatting(s):
    for attr in ["str", "repr", "int", "packed", "compressed", "exploded", "reverse_pointer"]:
        def get(obj, attr=attr):
            if attr == "str":
                return str(obj)
            if attr == "repr":
                return repr(obj)
            if attr == "int":
                return int(obj)
            return getattr(obj, attr)
        check(f"{attr}({s!r})",
              lambda s=s, get=get: get(ref.ip_address(s)),
              lambda s=s, get=get: get(mine.ip_address(s)))


@pytest.mark.parametrize("s", FORMAT_ADDRS)
def test_address_properties(s):
    props = ["version", "max_prefixlen", "is_private", "is_global", "is_multicast",
             "is_reserved", "is_loopback", "is_link_local", "is_unspecified"]
    for prop in props:
        check(f"{prop}({s!r})",
              lambda s=s, prop=prop: getattr(ref.ip_address(s), prop),
              lambda s=s, prop=prop: getattr(mine.ip_address(s), prop))


def test_v6_specific_properties():
    for s in ["::ffff:1.2.3.4", "::1.2.3.4", "2002:c000:0201::1",
              "2001:0000:4136:e378:8000:63bf:3fff:fdd2", "2001:db8::1", "fec0::1"]:
        for prop in ["ipv4_mapped", "sixtofour", "teredo", "is_site_local"]:
            check(f"{prop}({s})",
                  lambda s=s, prop=prop: getattr(ref.ip_address(s), prop),
                  lambda s=s, prop=prop: getattr(mine.ip_address(s), prop))


def test_scope_id_property():
    for s in ["fe80::1%eth0", "fe80::%eth0", "::1", "::%0"]:
        check(f"scope_id({s})",
              lambda s=s: ref.ip_address(s).scope_id,
              lambda s=s: mine.ip_address(s).scope_id)


# ---------------------------------------------------------------------------
# Address operators
# ---------------------------------------------------------------------------

OPS_ADDRS = [
    "0.0.0.0", "1.2.3.4", "10.0.0.1", "255.255.255.255", "192.168.1.1",
    "::", "::1", "2001:db8::1", "fe80::1", "fe80::1%eth0", "fe80::1%eth1", "fe80::2",
]


@pytest.mark.parametrize("a", OPS_ADDRS)
@pytest.mark.parametrize("b", OPS_ADDRS)
def test_address_comparison(a, b):
    for op in ["==", "!=", "<", "<=", ">", ">="]:
        check(f"{a} {op} {b}",
              lambda a=a, b=b, op=op: _apply(op, ref.ip_address(a), ref.ip_address(b)),
              lambda a=a, b=b, op=op: _apply(op, mine.ip_address(a), mine.ip_address(b)))


def _apply(op, x, y):
    if op == "==":
        return x == y
    if op == "!=":
        return x != y
    if op == "<":
        return x < y
    if op == "<=":
        return x <= y
    if op == ">":
        return x > y
    return x >= y


@pytest.mark.parametrize("a", OPS_ADDRS)
def test_address_hash(a):
    check(f"hash({a})", lambda: hash(ref.ip_address(a)), lambda: hash(mine.ip_address(a)))


@pytest.mark.parametrize("a", OPS_ADDRS)
@pytest.mark.parametrize("n", [-3, -1, 0, 1, 2, 100, 2**31, 2**32, -2**32, 1.5, "1"])
def test_address_arithmetic(a, n):
    check(f"{a} + {n!r}", lambda: ref.ip_address(a) + n, lambda: mine.ip_address(a) + n)
    check(f"{a} - {n!r}", lambda: ref.ip_address(a) - n, lambda: mine.ip_address(a) - n)


def test_address_vs_other_types():
    for other in [5, "1.2.3.4", None, [1]]:
        check(f"a4 == {other!r}", lambda other=other: ref.ip_address("1.2.3.4") == other,
              lambda other=other: mine.ip_address("1.2.3.4") == other)
        check(f"a4 < {other!r}", lambda other=other: ref.ip_address("1.2.3.4") < other,
              lambda other=other: mine.ip_address("1.2.3.4") < other)


def test_address_sorting():
    for addrs in [["10.0.0.2", "10.0.0.1", "1.2.3.4"],
                  ["fe80::1%eth1", "fe80::1", "fe80::1%eth0", "fe80::2"]]:
        check(f"sorted{addrs}",
              lambda addrs=addrs: [str(x) for x in sorted(ref.ip_address(a) for a in addrs)],
              lambda addrs=addrs: [str(x) for x in sorted(mine.ip_address(a) for a in addrs)])


# ---------------------------------------------------------------------------
# Network parsing corpus
# ---------------------------------------------------------------------------

NET4_STRINGS = [
    "192.168.1.0/24", "192.168.1.1/24", "10.0.0.0/8", "1.2.3.4/32", "0.0.0.0/0",
    "1.2.3.4/255.255.255.0", "1.2.3.4/0.0.0.255", "1.2.3.4/255.255.0.255",
    "1.2.3.4/254.0.0.0", "1.2.3.4/255.255.255.255", "1.2.3.4/0.0.0.0",
    "1.2.3.4/127.0.0.0", "1.2.3.4/255.255.255.254", "1.2.3.4/128.0.0.0",
    "1.2.3.4/0.0.0.127", "192.168.1.1", "1.2.3.4/024", "1.2.3.4/33", "1.2.3.4/-1",
    "1.2.3.4/", "1.2.3.4/24.5", "1.2.3.4/0x18", "1.2.3.4/ 24", "1.2.3.4 /24",
    "1.2.3.4/24 ", "1.2.3.4/+24", "1.2.3.4/٢٤", "1.2.3.4/²", "1.2.3.4/２４",
    "1.2.3.4/24/25", "1.2.3.4/a.b.c.d", "bad/33", "999.1.1.1/33", "x/24",
]

NET6_STRINGS = [
    "::1/64", "2001:db8::/32", "::/0", "::1/128", "fe80::1%eth0/64", "fe80::%eth0/64",
    "::ffff:1.2.3.4/120", "::1/0", "::1/129", "::1/-1", "::1/", "::1/ffff:ffff::",
    "::1/64x", "fe80::1%eth/0/64", "::1/64/65",
]

NET_INPUTS = [
    (0xC0A80101, 24), (0xC0A80100, 33), ("192.168.1.0", 24), (0xC0A80100, -1),
    (1, 2, 3), 12345, 2**32, 2**128 - 1, None, 1.5, b"\xc0\xa8\x01\x00",
    b"\x00" * 16, b"\x00" * 5, (b"\xc0\xa8\x01\x01", 24), ("::1", "64"),
    (0xC0A80100, None), (0xC0A80100, 24.0), (2**32, 24),
]


@pytest.mark.parametrize("s", NET4_STRINGS)
def test_ipv4network_parse(s):
    for strict in (True, False):
        check(f"IPv4Network({s!r}, strict={strict})",
              lambda s=s, st=strict: ref.IPv4Network(s, st),
              lambda s=s, st=strict: mine.IPv4Network(s, st))


@pytest.mark.parametrize("s", NET6_STRINGS)
def test_ipv6network_parse(s):
    for strict in (True, False):
        check(f"IPv6Network({s!r}, strict={strict})",
              lambda s=s, st=strict: ref.IPv6Network(s, st),
              lambda s=s, st=strict: mine.IPv6Network(s, st))


@pytest.mark.parametrize("s", NET4_STRINGS + NET6_STRINGS)
def test_ip_network_factory(s):
    for strict in (True, False):
        check(f"ip_network({s!r}, strict={strict})",
              lambda s=s, st=strict: ref.ip_network(s, st),
              lambda s=s, st=strict: mine.ip_network(s, st))


@pytest.mark.parametrize("v", NET_INPUTS)
def test_network_inputs(v):
    for strict in (True, False):
        check(f"IPv4Network({v!r}, strict={strict})",
              lambda v=v, st=strict: ref.IPv4Network(v, st),
              lambda v=v, st=strict: mine.IPv4Network(v, st))
        check(f"IPv6Network({v!r}, strict={strict})",
              lambda v=v, st=strict: ref.IPv6Network(v, st),
              lambda v=v, st=strict: mine.IPv6Network(v, st))
        check(f"ip_network({v!r}, strict={strict})",
              lambda v=v, st=strict: ref.ip_network(v, st),
              lambda v=v, st=strict: mine.ip_network(v, st))


# ---------------------------------------------------------------------------
# Network semantics grid
# ---------------------------------------------------------------------------

NET_GRID = [
    "10.0.0.0/8", "10.0.0.0/16", "10.0.1.0/24", "10.0.0.0/24", "10.0.0.0/31",
    "10.0.0.1/32", "0.0.0.0/0", "192.168.0.0/16", "192.168.0.0/9",
    "2001:db8::/32", "2001:db8::/48", "2001:db8::1/128", "::/0", "fe80::/10",
    "fe80::/126", "fe80::/127", "fe80::1/128",
]

NET_ATTRS = [
    "str", "repr", "with_prefixlen", "with_netmask", "with_hostmask", "prefixlen",
    "num_addresses", "version", "max_prefixlen",
    "network_address", "broadcast_address", "netmask", "hostmask",
]


@pytest.mark.parametrize("s", NET_GRID)
def test_network_attributes(s):
    for attr in NET_ATTRS:
        def get(obj, attr=attr):
            v = getattr(obj, attr) if attr not in ("str", "repr") else (
                str(obj) if attr == "str" else repr(obj))
            return str(v) if attr in ("network_address", "broadcast_address", "netmask", "hostmask") else v
        check(f"{attr}({s})",
              lambda s=s, get=get: get(ref.ip_network(s, strict=False)),
              lambda s=s, get=get: get(mine.ip_network(s, strict=False)))


@pytest.mark.parametrize("s", NET_GRID)
def test_network_hash(s):
    check(f"hash({s})",
          lambda: hash(ref.ip_network(s, strict=False)),
          lambda: hash(mine.ip_network(s, strict=False)))


@pytest.mark.parametrize("a", NET_GRID)
@pytest.mark.parametrize("b", NET_GRID)
def test_network_relations(a, b):
    na = ref.ip_network(a, strict=False)
    nb = ref.ip_network(b, strict=False)
    ma = mine.ip_network(a, strict=False)
    mb = mine.ip_network(b, strict=False)
    for op in ["==", "!=", "<", "<=", ">", ">="]:
        check(f"{a} {op} {b}", lambda op=op: _apply(op, na, nb), lambda op=op: _apply(op, ma, mb))
    check(f"{a}.overlaps({b})", lambda: na.overlaps(nb), lambda: ma.overlaps(mb))
    check(f"{a}.subnet_of({b})", lambda: na.subnet_of(nb), lambda: ma.subnet_of(mb))
    check(f"{a}.supernet_of({b})", lambda: na.supernet_of(nb), lambda: ma.supernet_of(mb))
    check(f"{a}.compare_networks({b})", lambda: na.compare_networks(nb), lambda: ma.compare_networks(mb))


@pytest.mark.parametrize("s", NET_GRID)
def test_network_hosts_iter(s):
    def hosts_head(net):
        it = net.hosts()
        return [str(next(it)) for _ in range(min(3, net.num_addresses))]

    def iter_head(net):
        it = iter(net)
        return [str(next(it)) for _ in range(min(3, net.num_addresses))]
    # hosts() count is only enumerable for small networks
    for fn, name in [(hosts_head, "hosts_head"), (iter_head, "iter_head")]:
        check(f"{name}({s})",
              lambda s=s, fn=fn: fn(ref.ip_network(s, strict=False)),
              lambda s=s, fn=fn: fn(mine.ip_network(s, strict=False)))
    if ref.ip_network(s, strict=False).num_addresses <= 16:
        check(f"hosts({s})",
              lambda s=s: [str(x) for x in ref.ip_network(s, strict=False).hosts()],
              lambda s=s: [str(x) for x in mine.ip_network(s, strict=False).hosts()])


@pytest.mark.parametrize("s", NET_GRID)
@pytest.mark.parametrize("i", [0, 1, -1, 3, 4, -5, 2**100, 1.5, "a"])
def test_network_getitem(s, i):
    check(f"({s})[{i!r}]",
          lambda s=s, i=i: ref.ip_network(s, strict=False)[i],
          lambda s=s, i=i: mine.ip_network(s, strict=False)[i])


@pytest.mark.parametrize("s", NET_GRID)
@pytest.mark.parametrize("diff,np_", [(1, None), (2, None), (0, None), (-1, None), (9, None),
                                    (None, 0), (None, 16), (None, 24), (None, 25), (None, 129),
                                    (None, -1), (25, None), (1.5, None), (None, 33)])
def test_network_subnets_supernet(s, diff, np_):
    rn = ref.ip_network(s, strict=False)
    mn = mine.ip_network(s, strict=False)
    kw = {}
    if diff is not None:
        kw["prefixlen_diff"] = diff
    if np_ is not None:
        kw["new_prefix"] = np_
    from itertools import islice

    def head(gen):
        # Do NOT materialize the whole generator: /0 networks can yield
        # billions of subnets (and that is the behavior under test).
        return [str(x) for x in islice(gen, 5)]
    check(f"({s}).subnets({kw})",
          lambda: head(rn.subnets(**kw)),
          lambda: head(mn.subnets(**kw)))
    check(f"({s}).supernet({kw})",
          lambda: str(rn.supernet(**kw)), lambda: str(mn.supernet(**kw)))


def test_network_contains():
    nets = ["10.0.0.0/8", "192.168.0.0/16", "::/0", "2001:db8::/32"]
    addrs = ["10.1.2.3", "11.0.0.1", "192.168.1.1", "::1", "2001:db8::1", "fe80::1%eth0"]
    for ns in nets:
        for a in addrs:
            check(f"{a} in {ns}",
                  lambda ns=ns, a=a: ref.ip_address(a) in ref.ip_network(ns),
                  lambda ns=ns, a=a: mine.ip_address(a) in mine.ip_network(ns))
    # network-in-network and int-in-network edge cases
    check("net in net", lambda: ref.ip_network("10.1.0.0/16") in ref.ip_network("10.0.0.0/8"),
          lambda: mine.ip_network("10.1.0.0/16") in mine.ip_network("10.0.0.0/8"))
    check("int in net", lambda: 5 in ref.ip_network("10.0.0.0/8"),
          lambda: 5 in mine.ip_network("10.0.0.0/8"))


def test_address_exclude():
    cases = [
        ("10.0.0.0/24", "10.0.0.64/26"), ("10.0.0.0/24", "10.0.0.0/26"),
        ("10.0.0.0/24", "10.0.0.128/25"), ("10.0.0.0/24", "10.0.0.0/24"),
        ("10.0.0.0/24", "10.1.0.0/16"), ("10.0.0.1/32", "10.0.0.1/32"),
        ("2001:db8::/32", "2001:db8:8000::/33"),
    ]
    for a, b in cases:
        check(f"({a}).address_exclude({b})",
              lambda a=a, b=b: [str(x) for x in ref.ip_network(a).address_exclude(ref.ip_network(b))],
              lambda a=a, b=b: [str(x) for x in mine.ip_network(a).address_exclude(mine.ip_network(b))])


def test_network_is_properties():
    for ns in ["192.168.0.0/16", "192.168.0.0/9", "8.0.0.0/8", "224.0.0.0/4",
               "127.0.0.0/8", "fd00::/8", "2001:db8::/32", "100.64.0.0/10"]:
        for prop in ["is_private", "is_global", "is_multicast", "is_loopback", "is_link_local"]:
            check(f"{prop}({ns})",
                  lambda ns=ns, prop=prop: getattr(ref.ip_network(ns, strict=False), prop),
                  lambda ns=ns, prop=prop: getattr(mine.ip_network(ns, strict=False), prop))


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

IFACE_INPUTS = [
    "192.168.1.1/24", "1.2.3.4/32", "10.0.0.1/8", "192.168.1.1/255.255.255.0",
    "1.2.3.4", "::1/64", "2001:db8::1/32", "fe80::1%eth0/64", "fe80::%eth0/64",
    "1.2.3.4/33", "bad/33", "1.2.3.4/24/25", (0xC0A80101, 24), 0xC0A80101,
    b"\xc0\xa8\x01\x01", ("::1", 64),
]


@pytest.mark.parametrize("v", IFACE_INPUTS)
def test_interfaces(v):
    check(f"IPv4Interface({v!r})", lambda v=v: ref.IPv4Interface(v), lambda v=v: mine.IPv4Interface(v))
    check(f"IPv6Interface({v!r})", lambda v=v: ref.IPv6Interface(v), lambda v=v: mine.IPv6Interface(v))
    check(f"ip_interface({v!r})", lambda v=v: ref.ip_interface(v), lambda v=v: mine.ip_interface(v))


IFACE_ATTRS = ["str", "repr", "with_prefixlen", "with_netmask", "with_hostmask",
               "exploded", "compressed", "prefixlen"]


@pytest.mark.parametrize("s", [v for v in IFACE_INPUTS if isinstance(v, str) and "/" in v])
def test_interface_attributes(s):
    for attr in IFACE_ATTRS:
        def get(obj, attr=attr):
            return str(obj) if attr == "str" else (repr(obj) if attr == "repr" else getattr(obj, attr))
        check(f"{attr}(iface {s})",
              lambda s=s, get=get: get(ref.ip_interface(s)),
              lambda s=s, get=get: get(mine.ip_interface(s)))
    for attr in ["ip", "network", "netmask", "hostmask"]:
        check(f"{attr}(iface {s})",
              lambda s=s, attr=attr: str(getattr(ref.ip_interface(s), attr)),
              lambda s=s, attr=attr: str(getattr(mine.ip_interface(s), attr)))


def test_interface_ops():
    pairs = [("192.168.1.1/24", "192.168.1.2/24"), ("192.168.1.1/24", "192.168.1.1/25"),
             ("10.0.0.1/8", "10.0.0.1/16"), ("fe80::1%eth0/64", "fe80::1%eth0/64"),
             ("fe80::1%eth0/64", "fe80::1/64"), ("::1/64", "1.2.3.4/24")]
    for a, b in pairs:
        for op in ["==", "!=", "<", "<=", ">", ">="]:
            check(f"iface {a} {op} {b}",
                  lambda a=a, b=b, op=op: _apply(op, ref.ip_interface(a), ref.ip_interface(b)),
                  lambda a=a, b=b, op=op: _apply(op, mine.ip_interface(a), mine.ip_interface(b)))
    for s in ["192.168.1.1/24", "fe80::1%eth0/64", "::1/64"]:
        check(f"hash(iface {s})", lambda s=s: hash(ref.ip_interface(s)), lambda s=s: hash(mine.ip_interface(s)))
        check(f"iface {s} + 1", lambda s=s: ref.ip_interface(s) + 1, lambda s=s: mine.ip_interface(s) + 1)
        check(f"iface {s} - 1", lambda s=s: ref.ip_interface(s) - 1, lambda s=s: mine.ip_interface(s) - 1)
    check("iface == addr", lambda: ref.ip_interface("192.168.1.1/24") == ref.ip_address("192.168.1.1"),
          lambda: mine.ip_interface("192.168.1.1/24") == mine.ip_address("192.168.1.1"))
    check("addr == iface", lambda: ref.ip_address("192.168.1.1") == ref.ip_interface("192.168.1.1/24"),
          lambda: mine.ip_address("192.168.1.1") == mine.ip_interface("192.168.1.1/24"))
    check("iface < addr", lambda: ref.ip_interface("192.168.1.1/24") < ref.ip_address("192.168.1.2"),
          lambda: mine.ip_interface("192.168.1.1/24") < mine.ip_address("192.168.1.2"))
    check("iface in net", lambda: ref.ip_interface("192.168.1.1/24") in ref.ip_network("192.168.1.0/24"),
          lambda: mine.ip_interface("192.168.1.1/24") in mine.ip_network("192.168.1.0/24"))
    check("net contains iface?", lambda: ref.ip_address("192.168.1.1") in ref.ip_interface("192.168.1.1/24"),
          lambda: mine.ip_address("192.168.1.1") in mine.ip_interface("192.168.1.1/24"))


# ---------------------------------------------------------------------------
# collapse_addresses / summarize_address_range
# ---------------------------------------------------------------------------


def _rng_v4nets(rng, n, base_lo=0, base_hi=2**32 - 1, plen_lo=8, plen_hi=32):
    out = []
    for _ in range(n):
        out.append((rng.randrange(base_lo, base_hi), rng.randrange(plen_lo, plen_hi + 1)))
    return out


def test_collapse_seeded_v4():
    rng = random.Random(20260920)
    for case, (lo, hi, pl, ph) in enumerate([
        (0, 2**32 - 1, 0, 32), (0x0A000000, 0x0B000000, 8, 32),
        (0, 2**24, 16, 32), (0, 2**16, 24, 32), (0, 2**32 - 1, 30, 32),
    ]):
        items = _rng_v4nets(rng, 300, lo, hi, pl, ph)
        rn = [ref.IPv4Network(x, strict=False) for x in items]
        mn = [mine.IPv4Network(x, strict=False) for x in items]
        check(f"collapse v4 case {case}",
              lambda rn=rn: [str(x) for x in ref.collapse_addresses(rn)],
              lambda mn=mn: [str(x) for x in mine.collapse_addresses(mn)])


def test_collapse_seeded_v6():
    rng = random.Random(20260921)
    for case in range(4):
        items = [(rng.randrange(0, 2 ** (32 + 16 * case)), rng.randrange(32, 129))
                 for _ in range(300)]
        rn = [ref.IPv6Network(x, strict=False) for x in items]
        mn = [mine.IPv6Network(x, strict=False) for x in items]
        check(f"collapse v6 case {case}",
              lambda rn=rn: [str(x) for x in ref.collapse_addresses(rn)],
              lambda mn=mn: [str(x) for x in mine.collapse_addresses(mn)])


def test_collapse_with_addresses_and_duplicates():
    items = ["10.0.0.1", "10.0.0.2", "10.0.0.0/24", "10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24"]
    check("collapse addrs+dups",
          lambda: [str(x) for x in ref.collapse_addresses([ref.ip_network(i) if "/" in i else ref.ip_address(i) for i in items])],
          lambda: [str(x) for x in mine.collapse_addresses([mine.ip_network(i) if "/" in i else mine.ip_address(i) for i in items])])
    check("collapse empty", lambda: list(ref.collapse_addresses([])), lambda: list(mine.collapse_addresses([])))


MIXED_COLLAPSE = [
    ["10.0.0.0/8", "::/0"], ["::/0", "10.0.0.0/8"], ["10.0.0.1", "::1"],
    ["::1", "10.0.0.1"], ["10.0.0.1", "::/0"], ["::/0", "10.0.0.1"],
    ["10.0.0.0/8", "::1"], ["::1", "10.0.0.0/8"],
    ["10.0.0.0/8", "::/0", "11.0.0.0/8"], ["10.0.0.0/8", "10.0.0.1", "::/0"],
    ["10.0.0.1", "10.0.0.0/8", "::/0"], ["2001:db8::/32", "10.0.0.0/8"],
    ["::1", "10.0.0.1", "::/0"], ["10.0.0.0/8", "::/0", "::1"],
]


@pytest.mark.parametrize("items", MIXED_COLLAPSE)
def test_collapse_mixed_version_errors(items):
    def build(mod, xs):
        return [mod.ip_network(x) if "/" in x else mod.ip_address(x) for x in xs]
    check(f"collapse mixed {items}",
          lambda items=items: list(ref.collapse_addresses(build(ref, items))),
          lambda items=items: list(mine.collapse_addresses(build(mine, items))))


def test_summarize_address_range():
    cases = [
        ("10.0.0.0", "10.0.0.255"), ("10.0.0.1", "10.0.0.6"), ("10.0.0.5", "10.0.0.5"),
        ("0.0.0.0", "255.255.255.255"), ("10.0.0.2", "10.0.0.1"), ("10.0.0.1", "::1"),
        ("::1", "::8"), ("::", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"),
        ("2001:db8::", "2001:db8::ffff"),
    ]
    for a, b in cases:
        check(f"summarize({a}, {b})",
              lambda a=a, b=b: [str(x) for x in ref.summarize_address_range(ref.ip_address(a), ref.ip_address(b))],
              lambda a=a, b=b: [str(x) for x in mine.summarize_address_range(mine.ip_address(a), mine.ip_address(b))])


def test_summarize_seeded():
    rng = random.Random(99)
    for case in range(30):
        v4 = case % 2 == 0
        bits = 32 if v4 else 128
        a = rng.randrange(0, 1 << bits)
        b = a + rng.randrange(0, 1 << rng.randrange(1, 20))
        b = min(b, (1 << bits) - 1)
        mk = ref.IPv4Address if v4 else ref.IPv6Address
        mm = mine.IPv4Address if v4 else mine.IPv6Address
        check(f"summarize seeded {case}",
              lambda a=a, b=b, mk=mk: [str(x) for x in ref.summarize_address_range(mk(a), mk(b))],
              lambda a=a, b=b, mm=mm: [str(x) for x in mine.summarize_address_range(mm(a), mm(b))])


def test_get_mixed_type_key():
    for v in ["10.0.0.1", "::1"]:
        check(f"get_mixed_type_key({v})",
              lambda v=v: ref.get_mixed_type_key(ref.ip_address(v)),
              lambda v=v: mine.get_mixed_type_key(mine.ip_address(v)))
    for v in ["10.0.0.0/8", "::/0"]:
        check(f"get_mixed_type_key({v})",
              lambda v=v: ref.get_mixed_type_key(ref.ip_network(v)),
              lambda v=v: mine.get_mixed_type_key(mine.ip_network(v)))
    check("get_mixed_type_key(42)", lambda: ref.get_mixed_type_key(42), lambda: mine.get_mixed_type_key(42))


# ---------------------------------------------------------------------------
# Private/global registry sweeps
# ---------------------------------------------------------------------------


def test_v4_registry_sweep():
    rng = random.Random(1234)
    points = set()
    # Boundaries of every special range, +/-1.
    for base in [0x00000000, 0x0A000000, 0x64400000, 0x64800000, 0x7F000000,
                 0xA9FE0000, 0xAC100000, 0xAC200000, 0xC0000000, 0xC0000100,
                 0xC0000200, 0xC0000300, 0xC0A80000, 0xC0A90000, 0xC6120000,
                 0xC6140000, 0xC6336400, 0xC6336500, 0xCB007100, 0xCB007200,
                 0xE0000000, 0xF0000000, 0xFFFFFFFF]:
        for d in (-1, 0, 1):
            v = base + d
            if 0 <= v <= 2**32 - 1:
                points.add(v)
    points.update(rng.randrange(0, 2**32) for _ in range(3000))
    mismatches = []
    for v in sorted(points):
        ra = ref.IPv4Address(v)
        ma = mine.IPv4Address(v)
        for prop in ("is_private", "is_global"):
            if getattr(ra, prop) != getattr(ma, prop):
                mismatches.append((v, prop, getattr(ra, prop), getattr(ma, prop)))
    assert not mismatches, mismatches[:10]


def test_v6_registry_sweep():
    rng = random.Random(4321)
    points = set()
    ranges = [
        (0x0, 128), (0x1, 128), (0x00000000000000000000FFFF00000000, 96),
        (0x0064FF9B000100000000000000000000, 48), (0x01000000000000000000000000000000, 64),
        (0x20010000000000000000000000000000, 23), (0x20010DB8000000000000000000000000, 32),
        (0x20020000000000000000000000000000, 16), (0x3FFF0000000000000000000000000000, 20),
        (0xFC000000000000000000000000000000, 7), (0xFE800000000000000000000000000000, 10),
        (0x20010001000000000000000000000001, 128), (0x20010001000000000000000000000002, 128),
        (0x20010003000000000000000000000000, 32), (0x20010004011200000000000000000000, 48),
        (0x20010020000000000000000000000000, 28), (0x20010030000000000000000000000000, 28),
    ]
    for base, plen in ranges:
        lo = base
        hi = base + (1 << (128 - plen)) - 1
        for v in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1):
            if 0 <= v < 1 << 128:
                points.add(v)
    # Coarse sweep of the whole space plus denser sweeps of special regions.
    for i in range(0, 1 << 16, 257):
        points.add(i << 112)
    for i in range(0, 1 << 16, 13):
        points.add(0x2001_0000_0000_0000_0000_0000_0000_0000 + (i << 96))
    for i in range(0, 1 << 16, 7):
        points.add(0x0064_FF9B_0000_0000_0000_0000_0000_0000 + (i << 80))
    points.update(rng.randrange(0, 1 << 128) for _ in range(2000))
    mismatches = []
    for v in sorted(points):
        ra = ref.IPv6Address(v)
        ma = mine.IPv6Address(v)
        for prop in ("is_private", "is_global"):
            if getattr(ra, prop) != getattr(ma, prop):
                mismatches.append((ra.exploded, prop, getattr(ra, prop), getattr(ma, prop)))
    assert not mismatches, mismatches[:10]


# ---------------------------------------------------------------------------
# Batch API (vectorized) — parity with stdlib loops
# ---------------------------------------------------------------------------


def test_parse_many_seeded():
    rng = random.Random(555)
    items = []
    for _ in range(3000):
        kind = rng.randrange(4)
        if kind == 0:
            items.append(".".join(str(rng.randrange(256)) for _ in range(4)))
        elif kind == 1:
            items.append(":".join("%x" % rng.randrange(0x10000) for _ in range(8)))
        elif kind == 2:
            parts = ["%x" % rng.randrange(0x10000) for _ in range(rng.randrange(0, 8))]
            s = ":".join(parts) + "::" + ":".join("%x" % rng.randrange(0x10000) for _ in range(rng.randrange(0, 8 - len(parts))))
            items.append(s)
        else:
            items.append("::ffff:" + ".".join(str(rng.randrange(256)) for _ in range(4)))
    # sprinkle scopes
    for i in range(0, len(items), 17):
        if ":" in items[i]:
            items[i] = items[i] + "%eth" + str(i % 3)
    ours = mine.parse_many(items)
    theirs = [ref.ip_address(s) for s in items]
    assert len(ours) == len(theirs)
    for i, (o, t) in enumerate(zip(ours, theirs)):
        assert type(o).__name__ == type(t).__name__, (i, items[i])
        assert int(o) == int(t), (i, items[i])
        assert str(o) == str(t), (i, items[i])
        assert getattr(o, "scope_id", None) == getattr(t, "scope_id", None), (i, items[i])


def test_parse_many_mixed_types():
    items = ["1.2.3.4", 42, b"\x01\x02\x03\x04", "::1", 2**33, b"\x00" * 16]
    ours = mine.parse_many(items)
    theirs = [ref.ip_address(x) for x in items]
    for o, t in zip(ours, theirs):
        assert type(o).__name__ == type(t).__name__
        assert int(o) == int(t)


@pytest.mark.parametrize("bad", ["01.2.3.4", "1.2.3.256", "1::2::3", "gg::", "fe80::1%",
                                 "1.2.3.4/24", "x", "", "1.2.3", "1.2.3.4/24/25",
                                 "::ffff:1.2.3.4.5", "bad/33"])
def test_parse_many_error_parity(bad):
    check_exc(f"parse_many([{bad!r}])",
              lambda: ref.ip_address(bad),
              lambda: mine.parse_many([bad]))


def test_contains_many_seeded():
    rng = random.Random(777)
    for case in range(4):
        n_nets = [0, 1, 50, 2000][case]
        v4 = case % 2 == 0
        if v4:
            nets = [mine.IPv4Network((rng.randrange(0, 2**32), rng.randrange(0, 33)), strict=False)
                    for _ in range(n_nets)]
            addrs = [mine.IPv4Address(rng.randrange(0, 2**32)) for _ in range(500)]
            rnets = [ref.IPv4Network((int(n.network_address), n.prefixlen)) for n in nets]
            raddrs = [ref.IPv4Address(int(a)) for a in addrs]
        else:
            nets = [mine.IPv6Network((rng.randrange(0, 2**128), rng.randrange(0, 129)), strict=False)
                    for _ in range(n_nets)]
            addrs = [mine.IPv6Address(rng.randrange(0, 2**128)) for _ in range(500)]
            rnets = [ref.IPv6Network((int(n.network_address), n.prefixlen)) for n in nets]
            raddrs = [ref.IPv6Address(int(a)) for a in addrs]
        got = mine.contains_many(nets, addrs)
        want = [any(a in n for n in rnets) for a in raddrs]
        assert list(got) == want, f"case {case}"


def test_contains_many_overlapping_and_mixed():
    nets = [mine.ip_network(s) for s in
            ["10.0.0.0/8", "10.1.0.0/16", "10.1.2.0/24", "10.0.0.0/16",
             "2001:db8::/32", "2001:db8:1::/48", "fe80::/10"]]
    addrs = [mine.ip_address(s) for s in
             ["10.1.2.3", "10.1.3.1", "10.0.5.1", "11.0.0.1",
              "2001:db8:1::1", "2001:db8:2::1", "fe80::1", "fec0::1"]]
    got = mine.contains_many(nets, addrs)
    want = [any(a in n for n in [ref.ip_network(s) for s in
            ["10.0.0.0/8", "10.1.0.0/16", "10.1.2.0/24", "10.0.0.0/16",
             "2001:db8::/32", "2001:db8:1::/48", "fe80::/10"]])
            for a in [ref.ip_address(s) for s in
                      ["10.1.2.3", "10.1.3.1", "10.0.5.1", "11.0.0.1",
                       "2001:db8:1::1", "2001:db8:2::1", "fe80::1", "fec0::1"]]]
    assert list(got) == want


def test_collapse_batch_seeded():
    rng = random.Random(888)
    for v4 in (True, False):
        items = []
        for _ in range(1000):
            if v4:
                items.append((rng.randrange(0, 2**32), rng.randrange(0, 33)))
            else:
                items.append((rng.randrange(0, 2**128), rng.randrange(0, 129)))
        if v4:
            nets = [mine.IPv4Network(x, strict=False) for x in items]
            rnets = [ref.IPv4Network(x, strict=False) for x in items]
        else:
            nets = [mine.IPv6Network(x, strict=False) for x in items]
            rnets = [ref.IPv6Network(x, strict=False) for x in items]
        got = [str(x) for x in mine.collapse_batch(nets)]
        want = [str(x) for x in ref.collapse_addresses(rnets)]
        assert got == want, f"v4={v4}"


def test_batch_backend_consistency():
    """Native and fallback must agree element-for-element (this runs on
    whichever backend the suite is using; compare against a forced flip)."""
    if not _native.native_available():
        pytest.skip("native kernel not built in this environment")
    nets = [mine.IPv4Network((0x0A000000 + i * 256, 24)) for i in range(100)]
    addrs = [mine.IPv4Address(0x0A000000 + i * 128) for i in range(400)]
    native_res = mine.contains_many(nets, addrs)
    import os
    os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"] = "1"
    try:
        _native._LIB = None
        fb_res = mine.contains_many(nets, addrs)
    finally:
        del os.environ["IPADDRESS_MOJO_DISABLE_NATIVE"]
        _native._LIB = None
    assert list(native_res) == list(fb_res)
