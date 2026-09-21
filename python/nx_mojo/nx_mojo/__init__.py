"""nx-mojo: networkx graph algorithms accelerated by a Mojo kernel.

Same function signatures, same return shapes, same exceptions as
`networkx.betweenness_centrality` and `networkx.single_source_dijkstra` —
powered by a Mojo kernel where the platform supports it (macOS arm64,
Linux x86_64), with a vendored pure-Python fallback everywhere else
(including Windows).

    import networkx as nx
    import nx_mojo

    G = nx.karate_club_graph()
    nx_mojo.betweenness_centrality(G)                 # dict keyed by node
    nx_mojo.single_source_dijkstra(G, source=0)       # (dist, paths)

Set NX_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from nx_mojo._native import backend_info, native_available
from nx_mojo.core import betweenness_centrality, single_source_dijkstra

__version__ = "0.1.3"  # x-release-please-version
__all__ = [
    "betweenness_centrality",
    "single_source_dijkstra",
    "backend_info",
    "native_available",
    "__version__",
]
