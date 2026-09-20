"""volatility3-shaped integration helpers (the component seam).

These helpers duck-type the oracle's objects; volatility3 itself is NOT a
dependency of this package — nothing here imports it. The two seams:

* ``scan_like_pool_header_scanner`` is the batch equivalent of
  ``volatility3.plugins.windows.poolscanner.PoolHeaderScanner.__call__``:
  it takes the oracle's own ``PoolConstraint`` objects (or
  ``vol_mojo.PoolConstraint``) and returns ``(constraint, offset)`` pairs
  exactly as the oracle yields ``(constraint, header)`` — ``offset`` being
  ``header.vol.offset``. The header-layout arguments mirror what
  ``PoolScanner.get_pool_header_table`` derives from the target system:
  x64 Windows (Vista+) -> ``layout="x64", vista_semantics=True``,
  x64 Windows 7 -> ``layout="x64", vista_semantics=False``,
  x86 Vista+ / pre-Vista -> ``layout="x86"`` with the matching flag;
  ``alignment`` is 0x10 for 64-bit and 0x8 for 32-bit symbol tables.
  Inside volatility3 the seam is the scanner instance: a plugin that
  already has the raw chunk bytes can swap the per-hit validation loop for
  this batch call and keep everything downstream unchanged.

* ``walk_hive_tuples`` is the offline equivalent of running
  ``RegistryHive.visit_nodes`` with a visitor that records
  ``(node.vol.offset, type_name, node.get_name())`` — but for a flat .dat
  hive file, without a memory image, kernel symbols, or a context. Inside
  volatility3 the seam is the layer: tools that only need the key-tree
  tuples from a hive file on disk can use this directly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from vol_mojo.core import PoolConstraint, scan_pool_headers, walk_hive


def scan_like_pool_header_scanner(
    data: object,
    constraints: object,
    alignment: int = 0x10,
    *,
    layout: str = "x64",
    vista_semantics: bool = True,
) -> List[Tuple[object, int]]:
    """Batch version of ``PoolHeaderScanner.__call__``.

    Returns ``(constraint, header_offset)`` pairs in scan order, where
    ``constraint`` is the exact object the caller passed (the oracle's
    ``PoolConstraint`` or a ``vol_mojo.PoolConstraint``) and
    ``header_offset`` matches the oracle's ``header.vol.offset``.
    """
    hits = scan_pool_headers(
        data,
        constraints,
        alignment=alignment,
        layout=layout,
        vista_semantics=vista_semantics,
    )
    by_tag = {}
    for c in constraints:  # tags are unique (validated by scan_pool_headers)
        by_tag[bytes(c.tag)] = c
    return [(by_tag[tag], off) for off, tag in hits]


def walk_hive_tuples(
    hive: object,
    *,
    root_cell_offset: Optional[int] = None,
) -> List[Tuple[int, str, str]]:
    """``RegistryHive.visit_nodes``-equivalent tuples for a flat .dat hive.

    The recorded tuple is ``(node.vol.offset, node.vol.type_name's final
    component, node.get_name())`` — what a visitor plugged into
    ``visit_nodes`` observes on the same key tree.
    """
    return walk_hive(hive, root_cell_offset=root_cell_offset)
