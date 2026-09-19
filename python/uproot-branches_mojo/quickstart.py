#!/usr/bin/env python3
"""uproot-mojo quickstart: write a ROOT file, read branches back, verify.

Creates a small self-contained ROOT file with uproot (zlib-compressed
vector<float64>, vector<int32> and std::string branches), reads every
branch with uproot_mojo, and checks the results against uproot itself.
Works on the native Mojo kernel where supported and on the pure-Python
fallback everywhere else (force it with UPROOT_MOJO_DISABLE_NATIVE=1).

    pip install uproot-mojo uproot   # uproot only to create the demo file
    python quickstart.py
"""

from __future__ import annotations

import os
import tempfile

import uproot_mojo

POINTS = [[1.5, 2.5, 3.5], [], [4.5], [10.0, -20.0], [7.25]]
IDS = [[1, 2, 3], [4], [], [5, 6], [7]]
LABELS = ["muon", "", "electron", "tau", "photon"]


def main() -> None:
    try:
        import awkward as ak
        import uproot
    except ImportError:
        print("this demo creates its fixture with uproot: pip install uproot awkward")
        raise SystemExit(1)

    path = os.path.join(tempfile.mkdtemp(prefix="uproot-mojo-demo-"), "demo.root")
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as f:
        f.mktree("events", {"pt": "var * float64", "id": "var * int32", "label": "string"})
        f["events"].extend({"pt": POINTS, "id": IDS, "label": LABELS})

    info = uproot_mojo.backend_info()
    pt = uproot_mojo.read_branch(path, "pt", dtype="float64")
    ids = uproot_mojo.read_branch(path, "id", dtype="int32")
    labels = uproot_mojo.read_branch(path, "label")

    # Cross-check against uproot's own arrays.
    tree = uproot.open(path)["events"]
    assert pt.to_list() == ak.to_list(tree["pt"].array(library="ak"))
    assert ids.to_list() == ak.to_list(tree["id"].array(library="ak"))
    assert labels.to_list() == ak.to_list(tree["label"].array(library="ak"))

    checksum = sum(sum(entry) for entry in pt.to_list()) + sum(
        sum(entry) for entry in ids.to_list()
    )
    print(f"backend: {pt.backend} (native source: {info['native_source']})")
    print(f"pt:    {pt.to_list()}")
    print(f"id:    {ids.to_list()}")
    print(f"label: {labels.to_list()}")
    print(f"branch checksum: {checksum!r}")
    print("uproot-mojo quickstart OK: results match uproot exactly")


if __name__ == "__main__":
    main()
