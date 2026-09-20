#!/usr/bin/env python3
"""pymavlink-mojo quickstart: decode a small synthetic tlog in one batch call.

Self-contained: builds two MAVLink frames by hand (a v2 HEARTBEAT and a v1
ATTITUDE, with spec CRC-16/MCRF4XX + crc-extra), frames them as a .tlog
(8-byte big-endian microsecond timestamp per message), and decodes the whole
buffer with `pymavlink_mojo.parse_buffer`. Runs identically on the native
kernel and the pure-Python fallback (PYMAVLINK_MOJO_DISABLE_NATIVE=1).
"""

from __future__ import annotations

import struct

import pymavlink_mojo


def x25crc(data: bytes, extra: int) -> int:
    """CRC-16/MCRF4XX over `data` plus the message's crc-extra byte."""
    crc = 0xFFFF
    for b in data + bytes((extra,)):
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def heartbeat_v2(seq: int, sysid: int, compid: int) -> bytes:
    # HEARTBEAT (id 0, crc_extra 50): custom_mode u32, then 5 u8s.
    payload = struct.pack("<IBBBBB", 0, 2, 3, 4, 5, 3)
    header = struct.pack("<BBBBBBBHB", 0xFD, len(payload), 0, 0, seq, sysid, compid, 0, 0)
    return header + payload + struct.pack("<H", x25crc(header[1:] + payload, 50))


def attitude_v1(seq: int, sysid: int, compid: int) -> bytes:
    # ATTITUDE (id 30, crc_extra 39): time_boot_ms u32 + 6 f32.
    payload = struct.pack("<Iffffff", 1234, 0.5, -0.25, 3.1, 0.01, -0.02, 0.001)
    header = struct.pack("<BBBBBB", 0xFE, len(payload), seq, sysid, compid, 30)
    return header + payload + struct.pack("<H", x25crc(header[1:] + payload, 39))


def main() -> None:
    ts0 = 1_757_000_000_000_000  # epoch microseconds
    tlog = (
        struct.pack(">Q", ts0)
        + heartbeat_v2(seq=0, sysid=42, compid=7)
        + struct.pack(">Q", ts0 + 100_000)
        + attitude_v1(seq=1, sysid=42, compid=7)
    )

    result = pymavlink_mojo.parse_buffer(tlog, timestamps=True)
    for msg in result.messages:
        if msg.get_type() == "HEARTBEAT":
            print(f"[{msg._timestamp:.6f}] HEARTBEAT from sys {msg.get_srcSystem()} "
                  f"comp {msg.get_srcComponent()}: autopilot={msg.autopilot} "
                  f"custom_mode={msg.custom_mode}")
        elif msg.get_type() == "ATTITUDE":
            print(f"[{msg._timestamp:.6f}] ATTITUDE: roll={msg.roll:.4f} "
                  f"pitch={msg.pitch:.4f} yaw={msg.yaw:.4f}")

    info = pymavlink_mojo.backend_info()
    backend = "native" if info["native_available"] else "fallback"
    # Deterministic summary for the wheel smoke test (native == fallback).
    total = sum(len(m.get_msgbuf()) for m in result.messages)
    print(f"backend: {backend}")
    print(f"checksum: {len(result.messages)} messages, {total} frame bytes, "
          f"{result.error_count} errors, consumed {result.consumed}/{len(tlog)}")


if __name__ == "__main__":
    main()
