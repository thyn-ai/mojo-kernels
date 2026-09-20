"""Record-layout constants shared by the native backend, the pure-Python
fallback, and the materializing wrapper.

Both backends produce the same flat arenas for one ``parse_buffer`` call:

  * ``recs`` -- ``n_messages * REC_STRIDE`` int64 slots (one fixed-stride
    record per emitted message),
  * ``fi``   -- int64 arena for decoded integer field values,
  * ``ff``   -- float64 arena for decoded float/double field values,
  * ``fu``   -- byte arena for decoded char-array field values.

Keeping the layout identical is what makes the two backends interchangeable:
the wrapper materializes message objects from these arenas and nothing else.
"""

from __future__ import annotations

REC_STRIDE = 26

R_KIND = 0
R_WIRE_MSGID = 1
R_VERSION = 2
R_MLEN = 3
R_SEQ = 4
R_SYSID = 5
R_COMPID = 6
R_INCOMPAT = 7
R_COMPAT = 8
R_CRC = 9
R_SIGNED = 10
R_TS_USEC = 11
R_MSGBUF_OFF = 12
R_MSGBUF_LEN = 13
R_PAYLOAD_OFF = 14
R_PAYLOAD_LEN = 15
R_REASON = 16
R_REASON_A = 17
R_REASON_B = 18
R_REASON_C = 19
R_FI_OFF = 20
R_FI_N = 21
R_FF_OFF = 22
R_FF_N = 23
R_FU_OFF = 24
R_FU_N = 25

KIND_MESSAGE = 0
KIND_BAD = 1
KIND_UNKNOWN = 2

REASON_NONE = 0
REASON_BAD_PREFIX = 1
REASON_BAD_INCOMPAT = 2
REASON_BAD_CRC = 3

# Field type codes (shared with the Mojo kernel; see _dialect.py).
TC_I8 = 0
TC_U8 = 1
TC_I16 = 2
TC_U16 = 3
TC_I32 = 4
TC_U32 = 5
TC_I64 = 6
TC_U64 = 7
TC_F32 = 8
TC_F64 = 9
TC_CHAR = 10

# pymavlink's reserved header message ids for synthetic records.
MSGID_BAD_DATA = -1
MSGID_UNKNOWN = -2

# Protocol constants.
MAGIC_V1 = 0xFE
MAGIC_V2 = 0xFD
HLEN_V1 = 6
HLEN_V2 = 10
SIG_LEN = 13
IFLAG_SIGNED = 0x01
SCAN_BOUND_S = 259200.0  # 3 days; pymavlink scan_timestamp bound
