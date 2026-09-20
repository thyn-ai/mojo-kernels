"""Public batch-decode API: ``parse_buffer``.

Decodes a whole buffer of MAVLink v1/v2 frames in one call — the shape of
offline telemetry-log (``.tlog``) batch analysis — on the active backend:
the native Mojo kernel when its shared library is available (macOS arm64 /
Linux x86_64 wheels), otherwise the vendored pure-Python parser. Both
backends produce the identical flat record layout (see
:mod:`pymavlink_mojo._layout`), so every normalization rule below applies
identically regardless of backend; the differential suite asserts
pymavlink-oracle equality on both paths.

Decoding semantics mirror pymavlink's robust-parsing state machine (the
path ``mavlogdump`` uses through ``mavutil.mavlogfile``):

  * garbage bytes at a frame boundary become one-byte ``BadData`` records
    ("Bad prefix"); scanning resynchronizes at the next byte;
  * frames with a bad CRC, or MAVLink v2 frames with unknown
    incompat_flags, become whole-frame ``BadData`` records with
    pymavlink's exact reason strings;
  * frames whose message id is outside the bundled common.xml dialect
    become ``UnknownMessage`` records (no CRC check — pymavlink's
    ``MAVLink_unknown`` behaviour);
  * valid frames decode into ``MAVLinkMessage`` objects with pymavlink's
    field values (zero-padded payloads, wire-order -> XML-order field
    mapping, char fields truncated at the first NUL and decoded
    ascii/replace).

With ``timestamps=True`` the buffer is read in ``.tlog`` framing: an
8-byte big-endian microsecond timestamp precedes the bytes of every
message, with pymavlink's ``scan_timestamp`` resync heuristic.

Set ``PYMAVLINK_MOJO_DISABLE_NATIVE=1`` to force the fallback.
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple, Union

from pymavlink_mojo import _layout as L
from pymavlink_mojo import _reference
from pymavlink_mojo._dialect import BY_ID
from pymavlink_mojo._native import NativeUnavailable, _load

__all__ = [
    "parse_buffer",
    "ParseResult",
    "MAVLinkMessage",
    "BadData",
    "UnknownMessage",
    "Header",
    "backend",
]

_U64_MASK = (1 << 64) - 1
_U32_MASK = (1 << 32) - 1


class Header(NamedTuple):
    """pymavlink-parity frame header (``msg.get_header()`` equivalent)."""

    msgid: int
    mlen: int
    incompat_flags: int
    compat_flags: int
    seq: int
    src_system: int
    src_component: int


class MAVLinkMessage:
    """A decoded MAVLink message (pymavlink's ``MAVLink_<name>_message``
    equivalent). Field values are also attributes: ``msg.roll`` works."""

    __slots__ = (
        "_type",
        "_msgid",
        "_mlen",
        "_incompat",
        "_compat",
        "_seq",
        "_sysid",
        "_compid",
        "_fields",
        "_crc",
        "_version",
        "_signed",
        "_timestamp",
        "_msgbuf",
        "_payload",
    )

    def __init__(
        self,
        type: str,
        msgid: int,
        mlen: int,
        incompat_flags: int,
        compat_flags: int,
        seq: int,
        src_system: int,
        src_component: int,
        fields: dict[str, Any],
        crc: int,
        version: int,
        signature_present: bool,
        timestamp: float | None,
        msgbuf: bytes,
        payload: bytes,
    ) -> None:
        self._type = type
        self._msgid = msgid
        self._mlen = mlen
        self._incompat = incompat_flags
        self._compat = compat_flags
        self._seq = seq
        self._sysid = src_system
        self._compid = src_component
        self._fields = fields
        self._crc = crc
        self._version = version
        self._signed = signature_present
        self._timestamp = timestamp
        self._msgbuf = msgbuf
        self._payload = payload

    # pymavlink-compatible accessors -------------------------------------
    def get_type(self) -> str:
        return self._type

    def get_msgId(self) -> int:
        return self._msgid

    def get_header(self) -> Header:
        return Header(
            msgid=self._msgid,
            mlen=self._mlen,
            incompat_flags=self._incompat,
            compat_flags=self._compat,
            seq=self._seq,
            src_system=self._sysid,
            src_component=self._compid,
        )

    def get_srcSystem(self) -> int:
        return self._sysid

    def get_srcComponent(self) -> int:
        return self._compid

    def get_seq(self) -> int:
        return self._seq

    def get_crc(self) -> int:
        return self._crc

    def get_msgbuf(self) -> bytes:
        return self._msgbuf

    def get_payload(self) -> bytes:
        """The on-wire payload bytes (``len == header.mlen``).

        Note: pymavlink's private ``m._payload`` slices v2 frames with a
        hardcoded v1 header length (a known upstream quirk); this accessor
        returns the actual payload instead.
        """
        return self._payload

    def get_fieldnames(self) -> list[str]:
        return list(self._fields)

    def get_signed(self) -> bool:
        """pymavlink parity: whether the frame's signature was *verified*.

        pymavlink-mojo performs no signature verification (offline log
        analysis has no signing key configured, and pymavlink itself accepts
        signed frames without one), so this is always False — exactly what
        pymavlink returns when parsing signed frames without a key. Use
        :meth:`get_signature_present` for the on-wire fact.
        """
        return False

    def get_signature_present(self) -> bool:
        """Whether the frame carries a MAVLink v2 signature block."""
        return self._signed

    def get_timestamp(self) -> float | None:
        """The tlog timestamp (seconds), or None for raw-stream parses.

        Also available as ``msg._timestamp`` for pymavlink code parity.
        """
        return self._timestamp

    def get_version(self) -> int:
        """The wire protocol version of the frame: 1 or 2."""
        return self._version

    def get_fields(self) -> dict[str, Any]:
        """A copy of the decoded field mapping (XML field order)."""
        return dict(self._fields)

    def __getattr__(self, name: str) -> Any:
        # Field values are attributes, pymavlink-style (msg.roll). Only
        # reached when normal lookup fails, so the get_* surface above can
        # never shadow a field — and no common.xml field is named like a
        # get_* method (asserted by the differential suite).
        try:
            return object.__getattribute__(self, "_fields")[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_dict(self) -> dict[str, Any]:
        out = {"mavpackettype": self._type}
        out.update(self._fields)
        return out

    def __repr__(self) -> str:
        body = ", ".join(f"{k}: {v!r}" for k, v in self._fields.items())
        return f"{self._type} {{{body}}}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MAVLinkMessage):
            return NotImplemented
        return (
            self._type == other._type
            and self._msgid == other._msgid
            and self._mlen == other._mlen
            and self._incompat == other._incompat
            and self._compat == other._compat
            and self._seq == other._seq
            and self._sysid == other._sysid
            and self._compid == other._compid
            and self._fields == other._fields
            and self._crc == other._crc
            and self._version == other._version
            and self._signed == other._signed
            and self._timestamp == other._timestamp
            and self._msgbuf == other._msgbuf
        )

    __hash__ = None  # type: ignore[assignment]


class BadData:
    """A piece of bad data in the stream (pymavlink's ``MAVLink_bad_data``)."""

    __slots__ = ("_data", "_reason", "_timestamp")

    def __init__(self, data: bytes, reason: str, timestamp: float | None) -> None:
        self._data = data
        self._reason = reason
        self._timestamp = timestamp

    def get_type(self) -> str:
        return "BAD_DATA"

    def get_msgId(self) -> int:
        return L.MSGID_BAD_DATA

    def get_msgbuf(self) -> bytes:
        return self._data

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def reason(self) -> str:
        return self._reason

    def get_timestamp(self) -> float | None:
        """The tlog timestamp (seconds), or None for raw-stream parses."""
        return self._timestamp

    def __repr__(self) -> str:
        hexstr = ["{:x}".format(i) for i in self._data]
        return f"BAD_DATA {{{self._reason}, data:{hexstr}}}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BadData):
            return NotImplemented
        return (
            self._data == other._data
            and self._reason == other._reason
            and self._timestamp == other._timestamp
        )

    __hash__ = None  # type: ignore[assignment]


class UnknownMessage:
    """A frame whose message id is outside the bundled dialect (pymavlink's
    ``MAVLink_unknown``). No CRC validation is performed, matching the
    oracle; ``data`` carries the whole raw frame."""

    __slots__ = ("_wire_msgid", "_data", "_timestamp")

    def __init__(self, wire_msgid: int, data: bytes, timestamp: float | None) -> None:
        self._wire_msgid = wire_msgid
        self._data = data
        self._timestamp = timestamp

    def get_type(self) -> str:
        return f"UNKNOWN_{self._wire_msgid}"

    def get_msgId(self) -> int:
        return L.MSGID_UNKNOWN

    @property
    def wire_msgid(self) -> int:
        return self._wire_msgid

    def get_msgbuf(self) -> bytes:
        return self._data

    @property
    def data(self) -> bytes:
        return self._data

    def get_timestamp(self) -> float | None:
        """The tlog timestamp (seconds), or None for raw-stream parses."""
        return self._timestamp

    def __repr__(self) -> str:
        hexstr = ["{:x}".format(i) for i in self._data]
        return f"{self.get_type()} {{data:{hexstr}}}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnknownMessage):
            return NotImplemented
        return (
            self._wire_msgid == other._wire_msgid
            and self._data == other._data
            and self._timestamp == other._timestamp
        )

    __hash__ = None  # type: ignore[assignment]


ParsedMessage = Union[MAVLinkMessage, BadData, UnknownMessage]


class ParseResult(NamedTuple):
    """The outcome of one :func:`parse_buffer` call."""

    messages: list[ParsedMessage]
    consumed: int
    error_count: int


def _native_backend():
    """The native parse module if usable right now, else None."""
    try:
        _load()
    except NativeUnavailable:
        return None
    from pymavlink_mojo import _native

    return _native


def backend() -> str:
    """Which backend serves parses right now: "native" or "fallback"."""
    return "native" if _native_backend() is not None else "fallback"


def _reason_string(code: int, a: int, b: int, c: int) -> str:
    """pymavlink's exact MAVError reason strings (robust-parsing path)."""
    if code == L.REASON_BAD_PREFIX:
        return "Bad prefix"
    if code == L.REASON_BAD_INCOMPAT:
        return "invalid incompat_flags 0x%x 0x%x %u" % (a, b, c)
    if code == L.REASON_BAD_CRC:
        return "invalid MAVLink CRC in msgID %u 0x%04x should be 0x%04x" % (a, b, c)
    raise ValueError(f"unknown reason code {code}")


def _materialize(
    data: bytes,
    recs,
    fi,
    ff,
    fu: bytes,
    timestamps: bool,
) -> list[ParsedMessage]:
    """Turn the shared flat arena layout into message objects.

    Hot path for >100k-message logs: the arenas are converted to plain
    Python lists once up front (one C-level pass each) so the per-message
    loop only deals in plain ints/floats.
    """
    if hasattr(recs, "tolist"):
        recs = recs.tolist()
        fi = fi.tolist()
        ff = ff.tolist()
    out: list[ParsedMessage] = []
    n = len(recs) // L.REC_STRIDE
    stride = L.REC_STRIDE
    by_id = BY_ID
    mask_u64 = _U64_MASK
    mask_u32 = _U32_MASK
    tc_char = L.TC_CHAR
    tc_f32 = L.TC_F32
    tc_u32 = L.TC_U32
    tc_u64 = L.TC_U64
    for i in range(n):
        base = i * stride
        kind = recs[base]
        # the ts slot carries raw µs bits (tlog mode) or -1 (raw mode)
        timestamp = (recs[base + L.R_TS_USEC] & mask_u64) * 1e-6 if timestamps else None
        mb_off = recs[base + L.R_MSGBUF_OFF]
        msgbuf = data[mb_off : mb_off + recs[base + L.R_MSGBUF_LEN]]
        if kind == L.KIND_BAD:
            code = recs[base + L.R_REASON]
            if code == L.REASON_BAD_PREFIX:
                reason = "Bad prefix"
            elif code == L.REASON_BAD_INCOMPAT:
                reason = "invalid incompat_flags 0x%x 0x%x %u" % (
                    recs[base + L.R_REASON_A],
                    recs[base + L.R_REASON_B],
                    recs[base + L.R_REASON_C],
                )
            else:
                reason = "invalid MAVLink CRC in msgID %u 0x%04x should be 0x%04x" % (
                    recs[base + L.R_REASON_A],
                    recs[base + L.R_REASON_B],
                    recs[base + L.R_REASON_C],
                )
            out.append(BadData(msgbuf, reason, timestamp))
            continue
        wire_msgid = recs[base + L.R_WIRE_MSGID]
        if kind == L.KIND_UNKNOWN:
            out.append(UnknownMessage(wire_msgid, msgbuf, timestamp))
            continue
        # Decoded message: materialize fields from the spill arenas.
        desc = by_id[wire_msgid]
        fi_idx = recs[base + L.R_FI_OFF]
        ff_idx = recs[base + L.R_FF_OFF]
        fu_idx = recs[base + L.R_FU_OFF]
        fields: dict[str, Any] = {}
        for name, tcode, alen, _woff in desc[4]:
            if tcode == tc_char:
                raw = fu[fu_idx : fu_idx + alen]
                fu_idx += alen
                # pymavlink: bytes up to the first NUL, decoded ascii/replace.
                fields[name] = raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")
            elif tcode == tc_f32 or tcode == L.TC_F64:
                if alen:
                    fields[name] = ff[ff_idx : ff_idx + alen]
                    ff_idx += alen
                else:
                    fields[name] = ff[ff_idx]
                    ff_idx += 1
            elif tcode == tc_u64:
                if alen:
                    fields[name] = [v & mask_u64 for v in fi[fi_idx : fi_idx + alen]]
                    fi_idx += alen
                else:
                    fields[name] = fi[fi_idx] & mask_u64
                    fi_idx += 1
            elif tcode == tc_u32:
                if alen:
                    fields[name] = [v & mask_u32 for v in fi[fi_idx : fi_idx + alen]]
                    fi_idx += alen
                else:
                    fields[name] = fi[fi_idx] & mask_u32
                    fi_idx += 1
            else:
                if alen:
                    fields[name] = fi[fi_idx : fi_idx + alen]
                    fi_idx += alen
                else:
                    fields[name] = fi[fi_idx]
                    fi_idx += 1
        p_off = recs[base + L.R_PAYLOAD_OFF]
        out.append(
            MAVLinkMessage(
                type=desc[1],
                msgid=wire_msgid,
                mlen=recs[base + L.R_MLEN],
                incompat_flags=recs[base + L.R_INCOMPAT],
                compat_flags=recs[base + L.R_COMPAT],
                seq=recs[base + L.R_SEQ],
                src_system=recs[base + L.R_SYSID],
                src_component=recs[base + L.R_COMPID],
                fields=fields,
                crc=recs[base + L.R_CRC],
                version=recs[base + L.R_VERSION],
                signature_present=bool(recs[base + L.R_SIGNED]),
                timestamp=timestamp,
                msgbuf=msgbuf,
                payload=data[p_off : p_off + recs[base + L.R_PAYLOAD_LEN]],
            )
        )
    return out


def parse_buffer(data, *, timestamps: bool = False) -> ParseResult:
    """Decode a whole buffer of MAVLink frames in one batch call.

    ``data`` is the raw stream (``bytes``/``bytearray``/``memoryview``);
    with ``timestamps=True`` it is read in ``.tlog`` framing (8-byte
    big-endian microsecond timestamp before every message).

    Returns a :class:`ParseResult`: the decoded messages (``MAVLinkMessage``
    / ``BadData`` / ``UnknownMessage``), the number of leading bytes fully
    accounted for (a trailing partial frame or partial timestamp unit is
    not consumed — carry it over to the next batch), and the number of
    ``BadData`` records (pymavlink's ``total_receive_errors``).
    """
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise TypeError(
            f"expected bytes-like input, got {type(data).__name__}"
        )
    native = _native_backend()
    if native is not None:
        recs, fi, ff, fu, src, consumed, n_errors = native.parse(data, timestamps)
    else:
        recs, fi, ff, fu, src, consumed, n_errors = _reference.parse(data, timestamps)
    messages = _materialize(src, recs, fi, ff, fu, timestamps)
    return ParseResult(messages=messages, consumed=consumed, error_count=n_errors)


def iter_parse(data, *, timestamps: bool = False) -> Iterator[ParsedMessage]:
    """Convenience iterator over :func:`parse_buffer` messages."""
    return iter(parse_buffer(data, timestamps=timestamps).messages)
