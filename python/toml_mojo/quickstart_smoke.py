"""Quickstart smoke for the toml-mojo wheel (used by ci-toml.yml)."""

import toml_mojo

doc = toml_mojo.loads(
    """\
# quickstart
[project]
name = "demo"
version = "0.1.0"
released = 2026-09-19T09:30:00Z
tags = ["toml", "mojo"]
deps = {core = ">=1", extra.level = 2}

[[servers]]
host = "a.example.com"
[[servers]]
host = "b.example.com"
port = 8080
"""
)
assert doc["project"]["name"] == "demo"
assert doc["project"]["deps"] == {"core": ">=1", "extra": {"level": 2}}
assert doc["servers"][1]["port"] == 8080
assert str(doc["project"]["released"]) == "2026-09-19 09:30:00+00:00"

info = toml_mojo.backend_info()
checksum = repr(sorted(doc["project"].items()))
print("toml_mojo quickstart OK")
print("backend native:", info["native_available"])
print("project checksum:", checksum)
