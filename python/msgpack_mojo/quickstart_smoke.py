"""Quickstart smoke for the msgpack-mojo wheel (used by ci-msgpack.yml)."""

import msgpack_mojo

doc = {
    "project": "msgpack-mojo",
    "version": [0, 1, 0],
    "released": True,
    "rating": 4.9,
    "blob": b"\x00\x01\xfe\xff",
    "tags": ["msgpack", "mojo", "serialization"],
    "nested": {"a": [1, -33, 2**63, -2**63], "b": {"c": None}},
}

blob = msgpack_mojo.packb(doc)
back = msgpack_mojo.unpackb(blob)
assert back == doc, (back, doc)
assert msgpack_mojo.unpackb(msgpack_mojo.dumps(doc)) == doc

info = msgpack_mojo.backend_info()
checksum = blob.hex()
print("msgpack_mojo quickstart OK")
print("backend native:", info["native_available"])
print("project checksum:", checksum)
