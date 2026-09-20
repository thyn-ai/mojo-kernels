#!/usr/bin/env python3
"""Reproducible benchmark: pymavlink (mavlogdump path) vs pymavlink_mojo.

Workload = offline tlog batch analysis: whole-log decode of every message
(header + payload fields + CRC validation + resync semantics), the shape
mavlogdump drives through mavutil.mavlogfile. Corpora are generated locally
with the oracle's own message writer from fixed seeds (no network, no
datasets) and written to real files in a temporary directory:

  tlog 100k / 1M   — mixed v1+v2 telemetry (~2.4 MB / ~24 MB)
  tlog 100k noisy  — same with ~1.5% garbage bytes injected (resync load)
  raw 1M           — the same 1M-message stream without tlog timestamps
                     (oracle: pymavlink's own MAVLink.parse_buffer batch API)

Correctness is asserted (message-for-message tuples + error count vs the
oracle) before any timing happens, so the numbers always come from a
verified-correct build. The oracle runs with MAVLINK20=1 (pymavlink's
v2.0 dialect modules — the standard modern-log setting).

Per cell we report:
  cold — first full decode in this process (kernel dlopen + ABI handshake
         + first buffer scan), single measurement
  warm — median of N_WARM_RUNS subsequent full decodes

Run from the repository root:

    PYTHONPATH=python/pymavlink_mojo pixi run python benchmarks/bench_pymavlink.py
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import tempfile
import time

os.environ["MAVLINK20"] = "1"  # oracle: v2.0 dialect modules (see docstring)

N_WARM_RUNS = 5
N_WARM_RUNS_BIG = 3  # for the >=500k-message cells (the oracle is slow)
SEED = 20260919

# telemetry mix: (message class name, relative frequency)
_MIX = (
    ("ATTITUDE", 30),
    ("GPS_RAW_INT", 10),
    ("GLOBAL_POSITION_INT", 20),
    ("SYS_STATUS", 5),
    ("HEARTBEAT", 2),
    ("RC_CHANNELS", 10),
    ("SCALED_IMU", 10),
    ("HIGHRES_IMU", 5),
    ("VFR_HUD", 5),
    ("ATTITUDE_QUATERNION", 3),
)


def _oracle():
    from pymavlink.dialects.v20 import common
    from pymavlink import mavutil

    return common, mavutil


def _gen_values(rng: random.Random, cls):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_pymavlink_differential import gen_message

    return gen_message(rng, cls)


def make_stream(rng: random.Random, n_msgs: int, noisy: bool = False) -> bytes:
    """A realistic mixed v1+v2 MAVLink stream, written by the oracle itself."""
    common, _ = _oracle()
    by_name = {cls.msgname: cls for cls in common.mavlink_map.values()}
    classes = [by_name[name] for name, _ in _MIX]
    weights = [w for _, w in _MIX]
    sender = common.MAVLink(None, srcSystem=42, srcComponent=7)
    out = bytearray()
    for i in range(n_msgs):
        cls = rng.choices(classes, weights)[0]
        msg = _gen_values(rng, cls)
        force_v1 = cls.id <= 255 and rng.random() < 0.4
        out += msg.pack(sender, force_mavlink1=force_v1)
        sender.seq = (sender.seq + 1) % 256
        if noisy and rng.random() < 0.015:
            out += bytes(rng.randrange(0, 256) for _ in range(rng.randrange(1, 6)))
    return bytes(out)


def make_tlog(rng: random.Random, n_msgs: int, noisy: bool = False) -> bytes:
    stream = make_stream(rng, n_msgs, noisy)
    # frame the stream as a tlog: ts before every *frame* (garbage bytes from
    # the noisy variant ride inside the next unit, as in a corrupt capture)
    out = bytearray()
    ts = 1_757_000_000_000_000
    pos = 0
    while pos < len(stream):
        b = stream[pos]
        if b not in (0xFE, 0xFD):
            out += stream[pos : pos + 1]
            pos += 1
            continue
        hlen = 10 if b == 0xFD else 6
        mlen = stream[pos + 1]
        sig = 13 if (b == 0xFD and (stream[pos + 2] & 1)) else 0
        flen = mlen + hlen + 2 + sig
        out += struct.pack(">Q", ts)
        ts += rng.randrange(5_000, 120_000)
        out += stream[pos : pos + flen]
        pos += flen
    return bytes(out)


def to_tuples(msgs, with_ts: bool):
    """The comparison unit: full decoded content per message."""
    out = []
    for m in msgs:
        mtype = m.get_type()
        ts = m._timestamp if with_ts else None
        if mtype == "BAD_DATA":
            out.append(("BAD_DATA", bytes(m.data), m.reason, ts))
        elif m.get_msgId() == -2:
            out.append((mtype, bytes(m.get_msgbuf()), ts))
        else:
            h = m.get_header()
            out.append(
                (
                    mtype,
                    m.get_msgId(),
                    h.mlen,
                    h.incompat_flags,
                    h.compat_flags,
                    m.get_seq(),
                    m.get_srcSystem(),
                    m.get_srcComponent(),
                    m.get_crc(),
                    tuple((name, getattr(m, name)) for name in m.get_fieldnames()),
                    ts,
                )
            )
    return out


def oracle_tlog_msgs(path: str):
    """The mavlogdump-equivalent parse loop (single pass, real file):
    decode the whole log into pymavlink message objects."""
    _, mavutil = _oracle()
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


def ours_tlog_msgs(path: str):
    """Our equivalent: read the file, decode the whole log into objects."""
    import pymavlink_mojo

    with open(path, "rb") as fh:
        data = fh.read()
    res = pymavlink_mojo.parse_buffer(data, timestamps=True)
    return res.messages, res.error_count


def oracle_raw_msgs(path: str):
    """Raw stream: pymavlink's own MAVLink.parse_buffer batch API."""
    common, _ = _oracle()
    with open(path, "rb") as fh:
        data = fh.read()
    mav = common.MAVLink(None, srcSystem=255, srcComponent=0)
    mav.robust_parsing = True
    msgs = mav.parse_buffer(bytearray(data)) or []
    return msgs, mav.total_receive_errors


def ours_raw_msgs(path: str):
    import pymavlink_mojo

    with open(path, "rb") as fh:
        data = fh.read()
    res = pymavlink_mojo.parse_buffer(data)
    return res.messages, res.error_count


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    lines.append(f"- python: {platform.python_version()}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def bench_cell(label, n_bytes, ours_fn, oracle_fn, oracle_arg, n_msgs, with_ts, warm_runs):
    """Correctness gate (untimed, tuple-exact), then cold + warm timings of
    the decode-into-objects workload on both sides."""
    ours_msgs, ours_err = ours_fn()
    oracle_msgs, oracle_err = oracle_fn(oracle_arg)
    ours_t = to_tuples(ours_msgs, with_ts)
    oracle_t = to_tuples(oracle_msgs, with_ts)
    if ours_t != oracle_t or ours_err != oracle_err:
        sys.exit(
            f"correctness gate failed for {label}: "
            f"tuples equal={ours_t == oracle_t} errors {ours_err} vs {oracle_err}"
        )

    t0 = time.perf_counter()
    ours_fn()
    cold_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    oracle_fn(oracle_arg)
    cold_oracle = time.perf_counter() - t0

    warm_ours, warm_oracle = [], []
    for _ in range(warm_runs):
        t0 = time.perf_counter()
        ours_fn()
        warm_ours.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        oracle_fn(oracle_arg)
        warm_oracle.append(time.perf_counter() - t0)
    return {
        "label": label,
        "msgs": n_msgs,
        "bytes": n_bytes,
        "cold_ours": cold_ours,
        "cold_oracle": cold_oracle,
        "warm_ours": statistics.median(warm_ours),
        "warm_oracle": statistics.median(warm_oracle),
    }


def main() -> None:
    import pymavlink_mojo

    info = pymavlink_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- pymavlink_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    from importlib.metadata import version

    print(f"- pymavlink (oracle): {version('pymavlink')}")
    print(f"- seeds: {SEED}; warm = median of {N_WARM_RUNS}; cold = first call in process")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'pymavlink_mojo'")

    rows = []
    with tempfile.TemporaryDirectory(prefix="pymav-mojo-bench-") as tmp:
        for n_msgs, noisy in ((100_000, False), (500_000, False), (100_000, True)):
            rng = random.Random(SEED + n_msgs + (1 if noisy else 0))
            tlog = make_tlog(rng, n_msgs, noisy)
            path = os.path.join(tmp, f"tlog-{n_msgs}{'-noisy' if noisy else ''}.tlog")
            with open(path, "wb") as fh:
                fh.write(tlog)
            label = f"tlog {n_msgs // 1000}k{' noisy' if noisy else ''}"
            rows.append(
                bench_cell(
                    label,
                    len(tlog),
                    lambda p=path: ours_tlog_msgs(p),
                    oracle_tlog_msgs,
                    path,
                    n_msgs,
                    with_ts=True,
                    warm_runs=N_WARM_RUNS if n_msgs <= 100_000 else N_WARM_RUNS_BIG,
                )
            )
        # raw stream variant (500k messages, no tlog framing)
        rng = random.Random(SEED + 500_000)
        raw = make_stream(rng, 500_000)
        raw_path = os.path.join(tmp, "raw-500k.mavlink")
        with open(raw_path, "wb") as fh:
            fh.write(raw)
        rows.append(
            bench_cell("raw 500k", len(raw), lambda: ours_raw_msgs(raw_path),
                       oracle_raw_msgs, raw_path, 500_000, with_ts=False,
                       warm_runs=N_WARM_RUNS_BIG)
        )

        # honest framing: the vendored pure-Python fallback on the 100k tlog
        os.environ["PYMAVLINK_MOJO_DISABLE_NATIVE"] = "1"
        try:
            rows.append(
                bench_cell(
                    "tlog 100k (fallback)",
                    os.path.getsize(os.path.join(tmp, "tlog-100000.tlog")),
                    lambda p=os.path.join(tmp, "tlog-100000.tlog"): ours_tlog_msgs(p),
                    oracle_tlog_msgs,
                    os.path.join(tmp, "tlog-100000.tlog"),
                    100_000,
                    with_ts=True,
                    warm_runs=N_WARM_RUNS,
                )
            )
        finally:
            del os.environ["PYMAVLINK_MOJO_DISABLE_NATIVE"]

    print("\n== whole-log decode into message objects ==")
    print("(cold = first call in process; warm = median of 5 runs for 100k cells, 3 for 500k)")
    header = (
        f"{'workload':>14} | {'size':>7} | {'cold oracle':>11} | {'cold mojo':>10} "
        f"| {'speedup':>7} | {'warm oracle':>11} | {'warm mojo':>10} | {'speedup':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:>14} | {r['bytes'] / 1e6:>6.1f}M "
            f"| {1e3 * r['cold_oracle']:>9.1f}ms | {1e3 * r['cold_ours']:>8.1f}ms "
            f"| {r['cold_oracle'] / r['cold_ours']:>6.1f}x "
            f"| {1e3 * r['warm_oracle']:>9.1f}ms | {1e3 * r['warm_ours']:>8.1f}ms "
            f"| {r['warm_oracle'] / r['warm_ours']:>6.1f}x"
        )

    print("\n== warm throughput (pymavlink_mojo) ==")
    for r in rows:
        mb_s = r["bytes"] / 1e6 / r["warm_ours"]
        msg_s = r["msgs"] / r["warm_ours"]
        print(f"{r['label']:>14}: {mb_s:>7.1f} MB/s, {msg_s:>12,.0f} messages/s")

    print("\n== README paste block ==")
    print("| workload | size | cold pymavlink | cold pymavlink-mojo | cold speedup | warm pymavlink | warm pymavlink-mojo | warm speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['label']} | {r['bytes'] / 1e6:.1f} MB "
            f"| {1e3 * r['cold_oracle']:.1f} ms | {1e3 * r['cold_ours']:.1f} ms "
            f"| {r['cold_oracle'] / r['cold_ours']:.1f}x "
            f"| {1e3 * r['warm_oracle']:.1f} ms | {1e3 * r['warm_ours']:.1f} ms "
            f"| {r['warm_oracle'] / r['warm_ours']:.1f}x |"
        )


if __name__ == "__main__":
    main()
