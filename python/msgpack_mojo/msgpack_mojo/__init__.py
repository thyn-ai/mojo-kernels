"""msgpack-mojo: a drop-in faster replacement for ``msgpack.packb``/``unpackb``.

Same call shapes, same bytes on the wire, same objects back — powered by a
clean-room Mojo byte engine where the platform supports it (macOS arm64,
Linux x86_64), with a vendored pure-Python engine everywhere else
(including Windows). The two backends share one object-graph walker and one
record assembler, so they cannot disagree; the differential suite proves it
against the published ``msgpack`` package.

    import msgpack_mojo

    blob = msgpack_mojo.packb({"hello": "world", "n": [1, 2, 3]})
    assert blob == msgpack.packb({"hello": "world", "n": [1, 2, 3]})
    assert msgpack_mojo.unpackb(blob) == {"hello": "world", "n": [1, 2, 3]}

Set ``MSGPACK_MOJO_DISABLE_NATIVE=1`` to force the pure-Python backend.

Supported option surface (anything else raises ``TypeError`` /
``NotImplementedError`` loudly — no silent deviations):

* packb: ``default``, ``use_bin_type``, ``strict_types``,
  ``use_single_float``, ``unicode_errors``, ``datetime``; the streaming-only
  knobs ``autoreset``/``buf_size`` are accepted and ignored (they cannot
  change one-shot output).
* unpackb: ``raw``, ``use_list``, ``strict_map_key``, ``timestamp`` (0-3),
  ``unicode_errors``, ``ext_hook``, ``max_str_len``, ``max_bin_len``,
  ``max_array_len``, ``max_map_len``, ``max_ext_len``.
* Not supported: ``object_hook``/``list_hook``/``object_pairs_hook`` and the
  streaming ``Packer``/``Unpacker`` classes.
"""

from __future__ import annotations

from msgpack_mojo import _engine_py, _native
from msgpack_mojo._decode import assemble
from msgpack_mojo._encode import encode_stream
from msgpack_mojo._native import NativeUnavailable, backend_info, native_available
from msgpack_mojo.exceptions import (
    BufferFull,
    ExtraData,
    FormatError,
    OutOfData,
    PackException,
    StackError,
    UnpackException,
)
from msgpack_mojo.ext import ExtType, Timestamp

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "BufferFull",
    "ExtType",
    "ExtraData",
    "FormatError",
    "OutOfData",
    "PackException",
    "StackError",
    "Timestamp",
    "UnpackException",
    "backend_info",
    "dump",
    "dumps",
    "load",
    "loads",
    "native_available",
    "packb",
    "unpackb",
    "__version__",
]


def packb(o, **kwargs) -> bytes:
    """Pack ``o`` into MessagePack bytes (``msgpack.packb`` compatible).

    Byte-exact with the reference packer for every supported type and
    option, on both backends (proven by the differential suite).
    """
    default = kwargs.pop("default", None)
    use_bin_type = kwargs.pop("use_bin_type", True)
    strict_types = kwargs.pop("strict_types", False)
    single_float = kwargs.pop("use_single_float", False)
    unicode_errors = kwargs.pop("unicode_errors", None)
    use_datetime = kwargs.pop("datetime", False)
    # Streaming-only knobs: accepted for drop-in compatibility; they cannot
    # change the output of a one-shot pack.
    kwargs.pop("autoreset", None)
    kwargs.pop("buf_size", None)
    if kwargs:
        raise TypeError(
            f"__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
        )
    stream = encode_stream(
        o,
        default=default,
        strict_types=bool(strict_types),
        unicode_errors=unicode_errors,
        use_datetime=bool(use_datetime),
    )
    flags = (_native.PACK_USE_BIN_TYPE if use_bin_type else 0) | (
        _native.PACK_SINGLE_FLOAT if single_float else 0
    )
    try:
        return _native.pack_bytes(stream, flags)
    except NativeUnavailable:
        return _engine_py.pack_engine(stream, flags)


def unpackb(packed, **kwargs):
    """Unpack one MessagePack buffer (``msgpack.unpackb`` compatible)."""
    if kwargs.pop("object_hook", None) is not None:
        raise NotImplementedError("msgpack_mojo.unpackb does not support object_hook")
    if kwargs.pop("list_hook", None) is not None:
        raise NotImplementedError("msgpack_mojo.unpackb does not support list_hook")
    if kwargs.pop("object_pairs_hook", None) is not None:
        raise NotImplementedError(
            "msgpack_mojo.unpackb does not support object_pairs_hook"
        )
    raw = kwargs.pop("raw", False)
    use_list = kwargs.pop("use_list", True)
    strict_map_key = kwargs.pop("strict_map_key", True)
    timestamp = kwargs.pop("timestamp", 0)
    unicode_errors = kwargs.pop("unicode_errors", None)
    ext_hook = kwargs.pop("ext_hook", ExtType)
    max_str_len = kwargs.pop("max_str_len", -1)
    max_bin_len = kwargs.pop("max_bin_len", -1)
    max_array_len = kwargs.pop("max_array_len", -1)
    max_map_len = kwargs.pop("max_map_len", -1)
    max_ext_len = kwargs.pop("max_ext_len", -1)
    if kwargs:
        raise TypeError(
            f"__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
        )
    data = bytes(packed)
    flags = _native.UNPACK_RAW if raw else 0
    try:
        status, pos, rec = _native.unpack_bytes(data, flags)
    except NativeUnavailable:
        status, pos, rec = _engine_py.unpack_engine(data, flags)
    if status == _native.STATUS_FORMAT:
        raise FormatError(f"Unknown header: 0x{data[pos]:02x}")
    if status == _native.STATUS_TRUNCATED:
        raise ValueError("Unpack failed: incomplete input")
    if status == _native.STATUS_EXTRA_DATA:
        raise ExtraData("unpack(b) received extra data.")
    return assemble(
        rec,
        use_list=bool(use_list),
        strict_map_key=bool(strict_map_key),
        timestamp=int(timestamp),
        unicode_errors=unicode_errors,
        ext_hook=ext_hook,
        max_str_len=max_str_len,
        max_bin_len=max_bin_len,
        max_array_len=max_array_len,
        max_map_len=max_map_len,
        max_ext_len=max_ext_len,
    )


def dump(o, fp, **kwargs) -> None:
    """Pack ``o`` and write it to a binary file object (``msgpack.dump``)."""
    fp.write(packb(o, **kwargs))


def load(fp, **kwargs):
    """Read a binary file object fully and unpack it (``msgpack.load``)."""
    return unpackb(fp.read(), **kwargs)


dumps = packb
loads = unpackb
