"""Differential tests: pymavlink_mojo must match pymavlink message-for-message.

Run twice by ``scripts/test_all_pymavlink.sh``: once against the native Mojo
kernel and once with PYMAVLINK_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). Both backends must agree with the oracle exactly — every compared
field is an int/float/str/bytes value, so the parity tolerance is *exact
equality* (0 ULP: both sides decode the same wire bytes into the same
IEEE-754 values).

The oracle is the published PyPI package (pinned: pymavlink==2.4.47, see the
test script). All logs are generated locally with the oracle's own message
writer from explicit seeds — no network, no unseeded randomness — so the
suite is bit-reproducible.

Compared tuple per message (the assignment's parity contract):
  decoded message : (type, header msgId, mlen, incompat_flags, compat_flags,
                     seq, srcSystem, srcComponent, crc, signed-verified flag,
                     wire-signature-present flag, every field value in
                     fieldnames order, raw msgbuf, timestamp)
  MAVLink_unknown : (type "UNKNOWN_<id>", raw msgbuf, timestamp)
  MAVLink_bad_data: (type "BAD_DATA", data bytes, reason string, timestamp)
plus the stream-level error count (pymavlink's total_receive_errors) and, for
complete streams, the consumed byte count.

Oracle drivers (both are pymavlink's own batch paths):
  raw streams  -- dialect MAVLink.parse_buffer(bytes) with robust_parsing
  .tlog framing -- mavutil.mavlink_connection(path) + recv_match loop (the
                  mavlogdump path), which attaches file timestamps
"""

from __future__ import annotations

import os
import random
import string
import struct

import pytest

import pymavlink_mojo

common = pytest.importorskip("pymavlink.dialects.v20.common", reason="oracle not installed")

# Pin the oracle to its MAVLink 2.0 dialect modules (pymavlink.dialects.v20):
# without this, mavutil's set_dialect picks the legacy v1.0 modules
# (pymavlink.dialects.v10), whose tables lack every extension field. This
# matches running mavlogdump with MAVLINK20=1 — the standard modern-log
# setting — and the v20 dialect module used for the raw-stream driver.
os.environ["MAVLINK20"] = "1"

mavutil = pytest.importorskip("pymavlink.mavutil", reason="oracle not installed")


# ---------------------------------------------------------------------------
# Seeded log generation (using the oracle's own writer)
# ---------------------------------------------------------------------------

_INT_RANGES = {
    "int8_t": (-128, 127),
    "uint8_t": (0, 255),
    "uint8_t_mavlink_version": (0, 255),
    "int16_t": (-32768, 32767),
    "uint16_t": (0, 65535),
    "int32_t": (-(1 << 31), (1 << 31) - 1),
    "uint32_t": (0, (1 << 32) - 1),
    "int64_t": (-(1 << 63), (1 << 63) - 1),
    "uint64_t": (0, (1 << 64) - 1),
}


def _gen_char(rng: random.Random, n: int) -> bytes:
    kind = rng.randrange(4)
    if kind == 0:  # printable, full length
        raw = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))
        return raw.encode()
    if kind == 1:  # interior NULs (exercise the first-NUL truncation rule)
        return bytes(rng.randrange(0, 256) for _ in range(n))
    if kind == 2:  # all NULs -> empty string
        return b"\x00" * n
    # short value with trailing NUL padding
    cut = rng.randrange(0, n) if n else 0
    body = "".join(rng.choice(string.ascii_letters) for _ in range(cut)).encode()
    return (body + b"\x00" * n)[:n]


def _gen_scalar(rng: random.Random, ftype: str):
    if ftype in ("float", "double"):
        return rng.choice(
            (
                0.0,
                -0.0,
                1.0,
                -1.5,
                3.141592653589793,
                1e30,
                -1e-30,
                rng.uniform(-1e6, 1e6),
                rng.uniform(-1e-3, 1e-3),
            )
        )
    lo, hi = _INT_RANGES[ftype]
    return rng.choice((lo, hi, max(lo, 0), rng.randint(lo, hi)))


def gen_message(rng: random.Random, cls):
    """One oracle message instance with seeded values for every field."""
    values = []
    for i, _name in enumerate(cls.fieldnames):
        ftype = cls.fieldtypes[i]
        arr = cls.array_lengths[cls.orders[i]]
        if ftype == "char":
            values.append(_gen_char(rng, arr if arr else 1))
        elif arr:
            values.append([_gen_scalar(rng, ftype) for _ in range(arr)])
        else:
            values.append(_gen_scalar(rng, ftype))
    return cls(*values)


def all_message_classes():
    return [common.mavlink_map[mid] for mid in sorted(common.mavlink_map)]


def build_stream(rng: random.Random, signed: bool = False) -> bytes:
    """Every common.xml message at least once, v1 (id<=255) and v2 mixed."""
    sender = common.MAVLink(None, srcSystem=rng.randrange(1, 255), srcComponent=rng.randrange(1, 255))
    if signed:
        sender.signing.secret_key = b"pymavlink-mojo-test-key"
        sender.signing.sign_outgoing = True
    out = bytearray()
    for cls in all_message_classes():
        msg = gen_message(rng, cls)
        force_v1 = cls.id <= 255 and rng.random() < 0.5
        out += msg.pack(sender, force_mavlink1=force_v1)
        sender.seq = (sender.seq + 1) % 256
        # a second copy with fresh values, opposite wire version when possible
        if rng.random() < 0.25:
            msg2 = gen_message(rng, cls)
            out += msg2.pack(sender, force_mavlink1=not force_v1 and cls.id <= 255)
            sender.seq = (sender.seq + 1) % 256
    return bytes(out)


def build_garbage_stream(rng: random.Random) -> bytes:
    """Frames interleaved with every corruption the robust parser handles."""
    sender = common.MAVLink(None, srcSystem=51, srcComponent=9)
    out = bytearray()
    out += bytes(rng.randrange(0, 256) for _ in range(7))  # leading garbage
    out += b"\xfe"  # lone v1 magic (incomplete header at this point)
    classes = [c for c in all_message_classes() if c.id <= 255][:25]
    for cls in classes:
        out += gen_message(rng, cls).pack(sender, force_mavlink1=True)
        sender.seq = (sender.seq + 1) % 256
        out += gen_message(rng, cls).pack(sender)
        sender.seq = (sender.seq + 1) % 256
        pick = rng.randrange(5)
        if pick == 0:
            out += b"\x00\x01\xfe"  # garbage incl. a dangling magic
        elif pick == 1:
            # valid v2 frame with a corrupted CRC
            f = bytearray(gen_message(rng, cls).pack(sender))
            sender.seq = (sender.seq + 1) % 256
            f[-1] ^= 0x5A
            out += f
        elif pick == 2:
            # v2 frame with an unknown incompat flag (0x40)
            f = bytearray(gen_message(rng, cls).pack(sender))
            sender.seq = (sender.seq + 1) % 256
            f[2] |= 0x40
            out += f
        elif pick == 3:
            # unknown message id (not in common.xml), v2 and v1
            out += _unknown_frame(rng, sender, 5000, v2=True)
            out += _unknown_frame(rng, sender, 8, v2=False)
        else:
            # garbage that starts with 0xFD: huge fake length, dies on CRC
            out += bytes((0xFD, 250, 0)) + bytes(rng.randrange(0, 256) for _ in range(262))
    out += bytes((0xFD, 200, 0, 0))  # truncated tail: not a full frame
    return bytes(out)


def _unknown_frame(rng: random.Random, sender, msgid: int, v2: bool) -> bytes:
    payload = bytes(rng.randrange(0, 256) for _ in range(rng.randrange(0, 20)))
    if v2:
        header = bytes(
            (0xFD, len(payload), 0, 0, sender.seq & 0xFF, sender.srcSystem & 0xFF,
             sender.srcComponent & 0xFF, msgid & 0xFF, (msgid >> 8) & 0xFF,
             (msgid >> 16) & 0xFF)
        )
    else:
        header = bytes((0xFE, len(payload), sender.seq & 0xFF, sender.srcSystem & 0xFF,
                        sender.srcComponent & 0xFF, msgid & 0xFF))
    sender.seq = (sender.seq + 1) % 256
    # pymavlink does not CRC-check unknown messages: any two bytes pass.
    return header + payload + b"\x00\x00"


def build_tlog(rng: random.Random, corrupt: bool = False) -> bytes:
    """tlog framing: 8-byte big-endian microsecond timestamp per message."""
    sender = common.MAVLink(None, srcSystem=77, srcComponent=1)
    out = bytearray()
    ts = 1_757_000_000_000_000  # plausible epoch microseconds
    classes = all_message_classes()
    for cls in classes[:80]:
        out += struct.pack(">Q", ts)
        ts += rng.randrange(50_000, 400_000)
        out += gen_message(rng, cls).pack(sender, force_mavlink1=rng.random() < 0.4 and cls.id <= 255)
        sender.seq = (sender.seq + 1) % 256
        if corrupt and rng.random() < 0.1:
            # a garbage stretch between units (resync exercise)
            out += bytes(rng.randrange(0, 256) for _ in range(rng.randrange(1, 12)))
    if corrupt:
        # out-of-range candidate timestamp right after bad data, then a valid
        # unit -- exercises pymavlink's scan_timestamp byte-wise resync.
        out += b"\xfe\xfd"  # bad bytes -> BAD_DATA, framed under the next ts
        out += struct.pack(">Q", 42)  # 1970: out of the 3-day window
        out += gen_message(rng, common.MAVLink_heartbeat_message).pack(sender)
        out += struct.pack(">Q", ts)
        ts += 100_000
        out += gen_message(rng, common.MAVLink_attitude_message).pack(sender)
        out += b"\x12\x34\x56"  # trailing partial timestamp (dropped)
    return bytes(out)


# ---------------------------------------------------------------------------
# Oracle drivers
# ---------------------------------------------------------------------------


def oracle_raw(data: bytes):
    """pymavlink's own batch API on the raw stream (robust parsing)."""
    mav = common.MAVLink(None, srcSystem=255, srcComponent=0)
    mav.robust_parsing = True
    msgs = mav.parse_buffer(bytearray(data)) or []
    return msgs, mav.total_receive_errors


def oracle_tlog(path: str):
    """The mavlogdump parse path on a tlog file (MAVLINK20=1 pinned, i.e.
    pymavlink's v2.0 dialect modules).

    Uses mavutil.mavlogfile directly — the same recv_msg/parse machinery
    mavlink_connection drives — but single-pass: mavlink_connection hands
    back a mavmmaplog whose init_arrays indexing re-parses the whole file
    and double-counts total_receive_errors, which would corrupt the
    stream-level error-count comparison (the message stream itself is
    identical either way, and is asserted pairwise below).
    """
    os.environ["MAVLINK20"] = "1"
    mavutil.set_dialect("common")
    conn = mavutil.mavlogfile(path, robust_parsing=True, notimestamps=False)
    msgs = []
    while True:
        m = conn.recv_msg()
        if m is None:
            break
        msgs.append(m)
    errors = conn.mav.total_receive_errors
    conn.close()
    return msgs, errors


# ---------------------------------------------------------------------------
# Tuple extraction (identical shape on both sides)
# ---------------------------------------------------------------------------

_NO_TS = object()  # raw-stream messages carry no timestamp on either side


def oracle_tuple(m, with_ts: bool):
    ts = m._timestamp if with_ts else _NO_TS
    mtype = m.get_type()
    if mtype == "BAD_DATA":
        return ("BAD_DATA", bytes(m.data), m.reason, ts)
    if m.get_msgId() == -2:  # MAVLink_unknown
        return (mtype, bytes(m.get_msgbuf()), ts)
    header = m.get_header()
    fields = tuple((name, getattr(m, name)) for name in m.get_fieldnames())
    return (
        mtype,
        m.get_msgId(),
        header.mlen,
        header.incompat_flags,
        header.compat_flags,
        m.get_seq(),
        m.get_srcSystem(),
        m.get_srcComponent(),
        m.get_crc(),
        bool(m.get_signed()),
        bool(header.incompat_flags & 1),  # signature block present on the wire
        fields,
        bytes(m.get_msgbuf()),
        ts,
    )


def our_tuple(m, with_ts: bool):
    ts = m._timestamp if with_ts else _NO_TS
    mtype = m.get_type()
    if mtype == "BAD_DATA":
        return ("BAD_DATA", bytes(m.data), m.reason, ts)
    if m.get_msgId() == -2:
        return (mtype, bytes(m.get_msgbuf()), ts)
    header = m.get_header()
    fields = tuple((name, getattr(m, name)) for name in m.get_fieldnames())
    return (
        mtype,
        m.get_msgId(),
        header.mlen,
        header.incompat_flags,
        header.compat_flags,
        m.get_seq(),
        m.get_srcSystem(),
        m.get_srcComponent(),
        m.get_crc(),
        bool(m.get_signed()),
        bool(header.incompat_flags & 1),
        fields,
        bytes(m.get_msgbuf()),
        ts,
    )


def assert_parity(data: bytes, oracle_msgs, oracle_errors, ours, with_ts: bool, label: str):
    ours_t = [our_tuple(m, with_ts) for m in ours.messages]
    oracle_t = [oracle_tuple(m, with_ts) for m in oracle_msgs]
    if ours_t != oracle_t:  # pragma: no cover - diagnostic path
        assert len(ours_t) == len(oracle_t), (
            f"{label}: message count differs: ours={len(ours_t)} oracle={len(oracle_t)}"
        )
        for i, (a, b) in enumerate(zip(ours_t, oracle_t)):
            if a != b:
                raise AssertionError(
                    f"{label}: first mismatch at message {i}:\n ours:  {a!r}\n oracle: {b!r}"
                )
    assert ours.error_count == oracle_errors, (
        f"{label}: error count ours={ours.error_count} oracle={oracle_errors}"
    )


# ---------------------------------------------------------------------------
# Dialect table parity (meta-test: our bundled tables == the oracle's)
# ---------------------------------------------------------------------------


def test_dialect_table_parity():
    from pymavlink_mojo._dialect import MESSAGES

    _STRUCT = {
        0: "b", 1: "B", 2: "h", 3: "H", 4: "i", 5: "I", 6: "q", 7: "Q",
        8: "f", 9: "d",
    }
    assert len(MESSAGES) == len(common.mavlink_map)
    for msgid, name, crc_extra, csize, fields in MESSAGES:
        cls = common.mavlink_map[msgid]
        assert name == cls.msgname
        assert crc_extra == cls.crc_extra
        assert csize == cls.unpacker.size
        assert [f[0] for f in fields] == cls.fieldnames
        # rebuild the wire struct format from our tables and compare;
        # orders[i] is the wire position of XML field i
        wire_order = sorted(range(len(fields)), key=lambda i: cls.orders[i])
        fmt = "<"
        for i in wire_order:
            _name, tcode, alen, _off = fields[i]
            if tcode == 10:
                fmt += f"{alen}s"
            elif alen:
                fmt += f"{alen}{_STRUCT[tcode]}"
            else:
                fmt += _STRUCT[tcode]
        assert fmt == cls.unpacker.format, (msgid, name, fmt, cls.unpacker.format)


# ---------------------------------------------------------------------------
# Raw-stream differential tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", (20260919, 424242, 7))
def test_clean_stream_all_messages(seed):
    rng = random.Random(seed)
    data = build_stream(rng)
    oracle_msgs, oracle_errors = oracle_raw(data)
    ours = pymavlink_mojo.parse_buffer(data)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                  label=f"clean stream seed={seed}")
    assert ours.consumed == len(data)
    # the stream really covers the whole dialect and both wire versions
    types = {m.get_msgId() for m in oracle_msgs}
    assert types == set(common.mavlink_map)


@pytest.mark.parametrize("seed", (1, 2))
def test_clean_stream_signed(seed):
    rng = random.Random(seed * 99991)
    data = build_stream(rng, signed=True)
    oracle_msgs, oracle_errors = oracle_raw(data)
    ours = pymavlink_mojo.parse_buffer(data)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                  label=f"signed stream seed={seed}")
    assert ours.consumed == len(data)
    assert any(m.get_type() != "BAD_DATA" and m.get_header().incompat_flags & 1
               for m in ours.messages if m.get_msgId() >= 0)


@pytest.mark.parametrize("seed", (11, 22, 33))
def test_garbage_resync_stream(seed):
    rng = random.Random(seed)
    data = build_garbage_stream(rng)
    oracle_msgs, oracle_errors = oracle_raw(data)
    ours = pymavlink_mojo.parse_buffer(data)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                  label=f"garbage stream seed={seed}")
    kinds = {m.get_type() for m in ours.messages}
    assert "BAD_DATA" in kinds
    assert any(t.startswith("UNKNOWN_") for t in kinds)


def test_garbage_only_stream():
    data = bytes(range(256)) * 3  # contains 0xFE/0xFD bytes: mixed outcomes
    oracle_msgs, oracle_errors = oracle_raw(data)
    ours = pymavlink_mojo.parse_buffer(data)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                  label="garbage only")


def test_partial_tail_consumed():
    rng = random.Random(5)
    data = build_stream(rng)
    cut = len(data) - 3  # drop the last 3 bytes of the final frame
    oracle_msgs, oracle_errors = oracle_raw(data[:cut])
    ours = pymavlink_mojo.parse_buffer(data[:cut])
    assert_parity(data[:cut], oracle_msgs, oracle_errors, ours, with_ts=False,
                  label="partial tail")
    assert ours.consumed < cut
    # the unconsumed tail is exactly the start of the incomplete final frame
    assert data[ours.consumed] in (0xFE, 0xFD)


def test_seeded_random_buffers():
    """Fuzz-ish resync sweep: seeded random buffers (some seeded with frame
    fragments) must produce identical message streams on both sides."""
    rng = random.Random(0xC0FFEE)
    sender = common.MAVLink(None, srcSystem=3, srcComponent=1)
    for case in range(60):
        n = rng.randrange(0, 400)
        buf = bytearray(rng.randrange(0, 256) for _ in range(n))
        # sprinkle valid frames and frame prefixes into the noise
        for _ in range(rng.randrange(0, 4)):
            if not buf:
                break
            at = rng.randrange(0, len(buf))
            cls = rng.choice(all_message_classes())
            frag = gen_message(rng, cls).pack(sender, force_mavlink1=rng.random() < 0.5 and cls.id <= 255)
            frag = frag[: rng.randrange(1, len(frag) + 1)]
            buf[at : at] = frag
        data = bytes(buf)
        oracle_msgs, oracle_errors = oracle_raw(data)
        ours = pymavlink_mojo.parse_buffer(data)
        assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                      label=f"random buffer case={case}")


@pytest.mark.parametrize("data", (b"", b"\xfe", b"\x00", b"\xfd\x09", b"\xfe\x09\x00"))
def test_tiny_inputs(data):
    oracle_msgs, oracle_errors = oracle_raw(data)
    ours = pymavlink_mojo.parse_buffer(data)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=False,
                  label=f"tiny {data!r}")


# ---------------------------------------------------------------------------
# .tlog differential tests (timestamps framing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", (101, 202))
def test_tlog_clean(seed, tmp_path):
    rng = random.Random(seed)
    data = build_tlog(rng)
    path = tmp_path / "clean.tlog"
    path.write_bytes(data)
    oracle_msgs, oracle_errors = oracle_tlog(str(path))
    ours = pymavlink_mojo.parse_buffer(data, timestamps=True)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=True,
                  label=f"clean tlog seed={seed}")
    assert ours.consumed == len(data)
    assert all(m._timestamp is not None for m in ours.messages)


@pytest.mark.parametrize("seed", (303, 404))
def test_tlog_corrupt(seed, tmp_path):
    rng = random.Random(seed)
    data = build_tlog(rng, corrupt=True)
    path = tmp_path / "corrupt.tlog"
    path.write_bytes(data)
    oracle_msgs, oracle_errors = oracle_tlog(str(path))
    ours = pymavlink_mojo.parse_buffer(data, timestamps=True)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=True,
                  label=f"corrupt tlog seed={seed}")


@pytest.mark.parametrize("data", (b"", b"\x00", b"\x00\x06=\xfbp", b"\x01" * 7))
def test_tlog_tiny_inputs(data, tmp_path):
    path = tmp_path / "tiny.tlog"
    path.write_bytes(data)
    oracle_msgs, oracle_errors = oracle_tlog(str(path))
    ours = pymavlink_mojo.parse_buffer(data, timestamps=True)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=True,
                  label=f"tiny tlog {data!r}")
    assert ours.consumed == 0  # nothing complete: resume from the start


def test_tlog_from_oracle_writer_roundtrip(tmp_path):
    """End-to-end: a tlog written by pymavlink's own mavlogfile writer."""
    rng = random.Random(777)
    path = tmp_path / "written.tlog"
    path.touch()  # mavlink_connection treats an existing path as a logfile
    conn = mavutil.mavlink_connection(str(path), dialect="common", write=True)
    for cls in all_message_classes()[:40]:
        conn.write(gen_message(rng, cls).pack(conn.mav))
    conn.close()
    data = path.read_bytes()
    oracle_msgs, oracle_errors = oracle_tlog(str(path))
    ours = pymavlink_mojo.parse_buffer(data, timestamps=True)
    assert_parity(data, oracle_msgs, oracle_errors, ours, with_ts=True,
                  label="oracle-written tlog")
    assert ours.consumed == len(data)


# ---------------------------------------------------------------------------
# Cross-backend agreement (independent of the oracle)
# ---------------------------------------------------------------------------


def test_backends_agree():
    from pymavlink_mojo import _native, _reference

    if not _native.native_available():
        pytest.skip("native kernel not built")
    rng = random.Random(987654)
    for data, tlog in (
        (build_stream(random.Random(1)), False),
        (build_garbage_stream(random.Random(2)), False),
        (build_tlog(random.Random(3), corrupt=True), True),
    ):
        n_recs, n_fi, n_ff, n_fu, n_src, n_cons, n_err = _native.parse(data, tlog)
        f_recs, f_fi, f_ff, f_fu, f_src, f_cons, f_err = _reference.parse(data, tlog)
        assert list(n_recs) == f_recs
        assert list(n_fi) == f_fi
        assert list(n_ff) == f_ff
        assert bytes(n_fu) == f_fu
        assert n_src == f_src
        assert n_cons == f_cons
        assert n_err == f_err
