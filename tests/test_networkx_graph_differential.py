"""Differential tests: nx_mojo must match networkx within 1e-12 everywhere.

Run twice by `scripts/test_all_networkx_graph.sh`: once against the native
Mojo kernel and once with NX_MOJO_DISABLE_NATIVE=1 (forced pure-Python
fallback). The oracle is the published PyPI package, pinned to
networkx==3.5 (the script provisions it into build/nx-test-oracle, outside
the pixi-managed site-packages). In practice both backends agree with the
oracle bit-for-bit on the graphs in this suite.
"""

from __future__ import annotations

import os

import networkx as nx
import pytest

import nx_mojo

ATOL = 1e-12  # documented tolerance; agreement is bit-exact in practice
RTOL = 1e-12


def _expected_native() -> bool:
    # scripts/test_all_networkx_graph.sh runs the suite once per backend.
    return os.environ.get("NX_MOJO_DISABLE_NATIVE") != "1"


def assert_bc_close(ours: dict, ref: dict) -> None:
    assert set(ours) == set(ref)
    assert list(ours) == list(ref)  # dict keyed in G iteration order
    for v in ref:
        assert abs(ours[v] - ref[v]) <= ATOL + RTOL * abs(ref[v]), (
            f"betweenness[{v!r}]: ours={ours[v]!r} oracle={ref[v]!r}"
        )


def assert_dijkstra_close(ours, ref) -> None:
    (d_ours, p_ours), (d_ref, p_ref) = ours, ref
    if isinstance(d_ref, dict):
        assert set(d_ours) == set(d_ref)
        for v in d_ref:
            assert abs(d_ours[v] - d_ref[v]) <= ATOL + RTOL * abs(d_ref[v]), (
                f"dist[{v!r}]: ours={d_ours[v]!r} oracle={d_ref[v]!r}"
            )
        assert p_ours == p_ref  # paths are deterministic; compare exactly
    else:  # target given: (distance, path)
        assert abs(d_ours - d_ref) <= ATOL + RTOL * abs(d_ref)
        assert p_ours == p_ref


def make_weighted(G, seed: int, attr: str = "weight", lo: float = 0.5, span: int = 13):
    """Deterministic pseudo-random float weights in [lo, lo + span)."""
    for u, v in G.edges():
        G[u][v][attr] = ((u * 31 + v * 17 + seed) % span) + lo
    return G


def make_int_weighted(G, seed: int, attr: str = "weight"):
    """Deterministic small integer weights (oracle does int arithmetic)."""
    for u, v in G.edges():
        G[u][v][attr] = (u * 7 + v + seed) % 9 + 1
    return G


# --------------------------------------------------------------------------
# backend identity: the suite must really exercise the intended backend
# --------------------------------------------------------------------------


def test_backend_under_test():
    info = nx_mojo.backend_info()
    assert info["native_available"] is _expected_native(), info


# --------------------------------------------------------------------------
# betweenness_centrality parity
# --------------------------------------------------------------------------

BC_MATRIX = [
    pytest.param(False, None, True, id="undir-unweighted-norm"),
    pytest.param(False, None, False, id="undir-unweighted-raw"),
    pytest.param(False, "weight", True, id="undir-weighted-norm"),
    pytest.param(False, "weight", False, id="undir-weighted-raw"),
    pytest.param(True, None, True, id="dir-unweighted-norm"),
    pytest.param(True, None, False, id="dir-unweighted-raw"),
    pytest.param(True, "weight", True, id="dir-weighted-norm"),
    pytest.param(True, "weight", False, id="dir-weighted-raw"),
]


@pytest.mark.parametrize("directed, weight, normalized", BC_MATRIX)
def test_bc_seeded_graphs(directed, weight, normalized):
    for n, m, seed in ((25, 80, 3), (60, 200, 5), (120, 400, 11)):
        G = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
        if weight:
            make_weighted(G, seed)
        ours = nx_mojo.betweenness_centrality(G, normalized=normalized, weight=weight)
        ref = nx.betweenness_centrality(G, normalized=normalized, weight=weight)
        assert_bc_close(ours, ref)


@pytest.mark.parametrize("directed, weight, normalized", BC_MATRIX)
def test_bc_karate_club(directed, weight, normalized):
    G = nx.karate_club_graph()
    if directed:
        G = nx.DiGraph(G)
    if weight:
        make_weighted(G, 7)
    ours = nx_mojo.betweenness_centrality(G, normalized=normalized, weight=weight)
    ref = nx.betweenness_centrality(G, normalized=normalized, weight=weight)
    assert_bc_close(ours, ref)


@pytest.mark.parametrize(
    "make_graph",
    [
        pytest.param(lambda: nx.Graph(), id="empty"),
        pytest.param(lambda: nx.Graph([(0, 0)]), id="single-selfloop"),
        pytest.param(lambda: nx.path_graph(2), id="two-nodes"),
        pytest.param(lambda: nx.path_graph(9), id="path"),
        pytest.param(lambda: nx.cycle_graph(9), id="cycle"),
        pytest.param(lambda: nx.complete_graph(8), id="complete"),
        pytest.param(lambda: nx.star_graph(7), id="star"),
        pytest.param(
            lambda: nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 3)]), id="dir-cycle-selfloop"
        ),
        pytest.param(
            lambda: nx.Graph([(0, 1), (2, 3), (4, 5), (5, 6)]), id="disconnected"
        ),
    ],
)
def test_bc_special_graphs(make_graph):
    G = make_graph()
    if G.number_of_nodes() and not G.is_directed():
        G.add_node(99)  # isolated node
    for normalized in (True, False):
        ours = nx_mojo.betweenness_centrality(G, normalized=normalized)
        ref = nx.betweenness_centrality(G, normalized=normalized)
        assert_bc_close(ours, ref)


def test_bc_zero_weight_edges():
    # networkx documents zero weights as invalid for betweenness but still
    # computes a deterministic result; we mirror it bit-for-bit.
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)])
    G[0][1]["w"] = 0.0
    G[1][2]["w"] = 0.0
    G[0][2]["w"] = 1.0
    G[2][3]["w"] = 2.0
    G[1][3]["w"] = 1.0
    for normalized in (True, False):
        ours = nx_mojo.betweenness_centrality(G, weight="w", normalized=normalized)
        ref = nx.betweenness_centrality(G, weight="w", normalized=normalized)
        assert_bc_close(ours, ref)


def test_bc_zero_weight_self_loop():
    G = nx.Graph([(0, 1), (1, 2)])
    G.add_edge(1, 1, weight=0.0)
    G[0][1]["weight"] = 1.0
    G[1][2]["weight"] = 2.0
    ours = nx_mojo.betweenness_centrality(G, weight="weight", normalized=False)
    ref = nx.betweenness_centrality(G, weight="weight", normalized=False)
    assert_bc_close(ours, ref)


def test_bc_integer_weights():
    G = nx.gnm_random_graph(40, 120, seed=13)
    make_int_weighted(G, 13)
    ours = nx_mojo.betweenness_centrality(G, weight="weight", normalized=False)
    ref = nx.betweenness_centrality(G, weight="weight", normalized=False)
    assert_bc_close(ours, ref)


def test_bc_custom_weight_attribute_and_missing_values():
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 3)])
    G[0][1]["cost"] = 2.5
    G[1][2]["cost"] = 1.5
    # (2,3) and (0,3) have no "cost" attribute: the oracle defaults them to 1.
    ours = nx_mojo.betweenness_centrality(G, weight="cost", normalized=False)
    ref = nx.betweenness_centrality(G, weight="cost", normalized=False)
    assert_bc_close(ours, ref)


def test_bc_many_equal_shortest_paths():
    # Grids maximize sigma (number of equal shortest paths), stressing the
    # path-counting and the accumulation order.
    G = nx.grid_2d_graph(6, 6)
    G = nx.convert_node_labels_to_integers(G)
    ours = nx_mojo.betweenness_centrality(G, normalized=False)
    ref = nx.betweenness_centrality(G, normalized=False)
    assert_bc_close(ours, ref)


def test_bc_deterministic_repeated_calls():
    G = nx.gnm_random_graph(30, 90, seed=17)
    first = nx_mojo.betweenness_centrality(G)
    for _ in range(3):
        assert nx_mojo.betweenness_centrality(G) == first


# --------------------------------------------------------------------------
# single_source_dijkstra parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", ["float", "int"])
def test_dijkstra_all_sources(directed, weighted):
    G = nx.gnm_random_graph(35, 120, seed=19, directed=directed)
    if weighted == "float":
        make_weighted(G, 19)
    else:
        make_int_weighted(G, 19)
    for s in G.nodes:
        ours = nx_mojo.single_source_dijkstra(G, s, weight="weight")
        ref = nx.single_source_dijkstra(G, s, weight="weight")
        assert_dijkstra_close(ours, ref)


@pytest.mark.parametrize("cutoff", [0, 1, 2.5, 6.25, 100.0])
def test_dijkstra_cutoffs(cutoff):
    G = nx.gnm_random_graph(40, 120, seed=23)
    make_weighted(G, 23, lo=0.25)
    ours = nx_mojo.single_source_dijkstra(G, 0, cutoff=cutoff, weight="weight")
    ref = nx.single_source_dijkstra(G, 0, cutoff=cutoff, weight="weight")
    assert_dijkstra_close(ours, ref)


def test_dijkstra_targets():
    G = nx.gnm_random_graph(40, 130, seed=29)
    make_weighted(G, 29)
    for target in (1, 7, 20, 39):
        ours = nx_mojo.single_source_dijkstra(G, 0, target=target, weight="weight")
        ref = nx.single_source_dijkstra(G, 0, target=target, weight="weight")
        assert_dijkstra_close(ours, ref)


def test_dijkstra_target_is_source():
    G = nx.path_graph(5)
    assert nx_mojo.single_source_dijkstra(G, 2, target=2) == (0, [2])
    assert nx.single_source_dijkstra(G, 2, target=2) == (0, [2])


def test_dijkstra_disconnected_graph():
    G = nx.Graph([(0, 1), (2, 3)])
    G.add_node(4)
    G[0][1]["weight"] = 1.5
    G[2][3]["weight"] = 2.5
    ours = nx_mojo.single_source_dijkstra(G, 0, weight="weight")
    ref = nx.single_source_dijkstra(G, 0, weight="weight")
    assert_dijkstra_close(ours, ref)
    # Source in a singleton component reaches only itself.
    ours = nx_mojo.single_source_dijkstra(G, 4, weight="weight")
    ref = nx.single_source_dijkstra(G, 4, weight="weight")
    assert_dijkstra_close(ours, ref)


def test_dijkstra_unreachable_target_raises_like_oracle():
    G = nx.Graph([(0, 1), (2, 3)])
    with pytest.raises(nx.NetworkXNoPath) as ours_exc:
        nx_mojo.single_source_dijkstra(G, 0, target=2)
    with pytest.raises(nx.NetworkXNoPath) as ref_exc:
        nx.single_source_dijkstra(G, 0, target=2)
    assert str(ours_exc.value) == str(ref_exc.value)


def test_dijkstra_missing_source_raises_like_oracle():
    G = nx.path_graph(4)
    with pytest.raises(nx.NodeNotFound) as ours_exc:
        nx_mojo.single_source_dijkstra(G, 99)
    with pytest.raises(nx.NodeNotFound) as ref_exc:
        nx.single_source_dijkstra(G, 99)
    assert str(ours_exc.value) == str(ref_exc.value)


def test_dijkstra_negative_weights_raise_like_oracle():
    G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    G[0][1]["weight"] = 1.0
    G[1][2]["weight"] = -2.0
    G[0][2]["weight"] = 5.0
    with pytest.raises(ValueError) as ours_exc:
        nx_mojo.single_source_dijkstra(G, 0, weight="weight")
    with pytest.raises(ValueError) as ref_exc:
        nx.single_source_dijkstra(G, 0, weight="weight")
    assert ours_exc.value.args == ref_exc.value.args


def test_dijkstra_zero_weight_edges_and_self_loops():
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 3)])
    G[0][1]["w"] = 0.0
    G[1][2]["w"] = 0.0
    G[0][2]["w"] = 1.0
    G[2][3]["w"] = 2.0
    G.add_edge(1, 1, w=0.0)
    ours = nx_mojo.single_source_dijkstra(G, 0, weight="w")
    ref = nx.single_source_dijkstra(G, 0, weight="w")
    assert_dijkstra_close(ours, ref)


def test_dijkstra_string_labels():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1.5)
    G.add_edge("b", "c", weight=2.5)
    G.add_edge("a", "c", weight=10.0)
    ours = nx_mojo.single_source_dijkstra(G, "a", weight="weight", target="c")
    ref = nx.single_source_dijkstra(G, "a", weight="weight", target="c")
    assert_dijkstra_close(ours, ref)


def test_dijkstra_default_weight_is_one():
    # Edges without the attribute count as weight 1 on both sides.
    G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    G[0][2]["weight"] = 10.0
    ours = nx_mojo.single_source_dijkstra(G, 0, weight="weight")
    ref = nx.single_source_dijkstra(G, 0, weight="weight")
    assert_dijkstra_close(ours, ref)


# --------------------------------------------------------------------------
# documented scope: out-of-scope inputs raise, never silently compute
# --------------------------------------------------------------------------


def test_multigraph_rejected():
    MG = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
    with pytest.raises(NotImplementedError, match="[Mm]ultigraph|simple graphs"):
        nx_mojo.betweenness_centrality(MG)
    with pytest.raises(NotImplementedError, match="[Mm]ultigraph|simple graphs"):
        nx_mojo.single_source_dijkstra(MG, 0)
    MDG = nx.MultiDiGraph([(0, 1), (0, 1)])
    with pytest.raises(NotImplementedError):
        nx_mojo.betweenness_centrality(MDG)
    with pytest.raises(NotImplementedError):
        nx_mojo.single_source_dijkstra(MDG, 0)


def test_sampled_betweenness_rejected():
    G = nx.path_graph(5)
    with pytest.raises(NotImplementedError, match="k=None"):
        nx_mojo.betweenness_centrality(G, k=3, seed=1)


def test_endpoints_rejected():
    G = nx.path_graph(5)
    with pytest.raises(NotImplementedError, match="endpoints"):
        nx_mojo.betweenness_centrality(G, endpoints=True)


def test_callable_weight_rejected():
    G = nx.path_graph(5)
    with pytest.raises(NotImplementedError, match="attribute name"):
        nx_mojo.betweenness_centrality(G, weight=lambda u, v, d: 1.0)
    with pytest.raises(NotImplementedError, match="attribute name"):
        nx_mojo.single_source_dijkstra(G, 0, weight=lambda u, v, d: 1.0)


def test_nonnumeric_weight_rejected():
    G = nx.Graph([(0, 1)])
    G[0][1]["weight"] = "heavy"
    with pytest.raises(ValueError, match="non-numeric"):
        nx_mojo.single_source_dijkstra(G, 0, weight="weight")
