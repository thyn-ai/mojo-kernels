"""Official networkx backend interface for nx_mojo (out-of-tree backend).

Registering this module's entry point makes `nx_mojo` dispatchable through
networkx's own backend machinery — the pitch this package exists for:

    # pyproject.toml
    [project.entry-points."networkx.backends"]
    nx_mojo = "nx_mojo.backend:BackendInterface"

    [project.entry-points."networkx.backend_info"]
    nx_mojo = "nx_mojo.backend:get_info"

Then:

    import networkx as nx
    G = nx.karate_club_graph()
    nx.betweenness_centrality(G, backend="nx_mojo")     # dispatched to nx_mojo
    nx.single_source_dijkstra(G, 0, backend="nx_mojo")

networkx loads the "networkx.backends" entry point lazily and calls the
object it resolves to (`.load()` without calling, so a class is used as-is);
"networkx.backend_info" is loaded and *called* (`ep.load()()`), hence the
`get_info` function below. `convert_from_nx` returns the graph unchanged —
nx_mojo consumes networkx graphs directly, so no conversion is needed — and
`convert_to_nx` returns results unchanged (they are already networkx-shaped:
dicts keyed by the original nodes). Out-of-scope inputs (multigraphs, `k`
sampling, `endpoints=True`, callable weights) are declined through
`can_run`, so networkx transparently falls back to its own implementation.
"""

from __future__ import annotations

import nx_mojo

__all__ = ["BackendInterface", "get_info"]


def _reason_if_unsupported(name: str, args: tuple, kwargs: dict) -> str | None:
    """Human-readable reason this call is out of scope, or None if runnable."""
    G = args[0] if args else kwargs.get("G")
    if G is None or not hasattr(G, "is_multigraph"):
        return None  # let the networkx implementation produce the error
    if G.is_multigraph():
        return "multigraphs (parallel edges) are not supported"
    weight = kwargs.get("weight")
    if callable(weight):
        return "callable weight functions are not supported"
    if name == "betweenness_centrality":
        if kwargs.get("k") is not None:
            return "sampled approximation (k) is not supported"
        if kwargs.get("endpoints"):
            return "endpoints=True is not supported"
    return None


class BackendInterface:
    """networkx backend object: attribute lookup yields the implementations."""

    # networkx calls getattr(backend, <dispatchable name>) for each
    # algorithm; only the in-scope algorithms are exposed here.
    betweenness_centrality = staticmethod(nx_mojo.betweenness_centrality)
    single_source_dijkstra = staticmethod(nx_mojo.single_source_dijkstra)

    @staticmethod
    def convert_from_nx(G, *args, **kwargs):  # noqa: ARG004
        # nx_mojo reads networkx graphs directly; no conversion needed.
        return G

    @staticmethod
    def convert_to_nx(result, *args, **kwargs):  # noqa: ARG004
        # Results are already networkx-shaped (dicts keyed by nodes).
        return result

    @staticmethod
    def can_run(name, args, kwargs):
        reason = _reason_if_unsupported(name, args, kwargs)
        return reason if reason is not None else True

    @staticmethod
    def should_run(name, args, kwargs):  # noqa: ARG004
        # Never force ourselves ahead of other backends; we are a strict
        # accelerator for the calls we support.
        return True


def get_info() -> dict:
    """networkx.backend_info entry point (loaded and called by networkx).

    Schema expected by networkx's doc builder: "functions" is a dict keyed
    by dispatchable function name, with optional per-function
    "additional_docs" / "extra_parameters" / "url" entries.
    """
    return {
        "backend_name": "nx_mojo",
        "project": "nx-mojo",
        "package": "nx-mojo",
        "url": "https://github.com/thyn-ai/mojo-kernels",
        "short_summary": "Mojo-accelerated networkx algorithms (pure-Python fallback).",
        "functions": {
            "betweenness_centrality": {
                "url": "https://github.com/thyn-ai/mojo-kernels",
                "additional_docs": (
                    "Exact computation only (no k sampling, endpoints=False); "
                    "simple graphs; weight must be an edge attribute name."
                ),
            },
            "single_source_dijkstra": {
                "url": "https://github.com/thyn-ai/mojo-kernels",
                "additional_docs": (
                    "Simple graphs; non-negative weights given as an edge "
                    "attribute name."
                ),
            },
        },
    }
