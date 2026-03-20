from __future__ import annotations

import sys
from itertools import pairwise

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mosaic.contour import ContourGraph

# ---------------------------------------------------------------------------
# Hypothesis strategies for structured graph inputs
# ---------------------------------------------------------------------------


@st.composite
def path_graph(draw) -> tuple[ContourGraph, list[int]]:
    """Draw a single path graph: n unique nodes connected in a line."""
    nodes = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000), min_size=2, unique=True
        )
    )
    v1 = np.array(nodes[:-1])
    v2 = np.array(nodes[1:])
    return ContourGraph(v1, v2), nodes


@st.composite
def cycle_graph(draw) -> tuple[ContourGraph, list[int]]:
    """Draw a single cycle graph: n unique nodes connected in a closed loop."""
    nodes = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000), min_size=3, unique=True
        )
    )
    v1 = np.array(nodes)
    v2 = np.roll(v1, -1)
    return ContourGraph(v1, v2), nodes


@st.composite
def multi_component_graph(draw) -> tuple[ContourGraph, list[list[int]]]:
    """Draw 1-4 independent path or cycle components with disjoint node IDs."""
    n_components = draw(st.integers(min_value=1, max_value=4))

    all_v1, all_v2 = [], []
    components = []
    used: set[int] = set()

    for _ in range(n_components):
        size = draw(st.integers(min_value=2, max_value=20))
        is_cycle = draw(st.booleans())

        # Draw node IDs that don't collide with previously used ones
        nodes = draw(
            st.lists(
                st.integers(min_value=0, max_value=100_000).filter(
                    lambda x: x not in used
                ),
                min_size=max(size, 3 if is_cycle else 2),
                max_size=max(size, 3 if is_cycle else 2),
                unique=True,
            )
        )
        used.update(nodes)
        components.append(nodes)

        if is_cycle:
            v1 = np.array(nodes)
            v2 = np.roll(v1, -1)
        else:
            v1 = np.array(nodes[:-1])
            v2 = np.array(nodes[1:])

        all_v1.append(v1)
        all_v2.append(v2)

    g = ContourGraph(
        np.concatenate(all_v1),
        np.concatenate(all_v2),
    )
    return g, components


# ---------------------------------------------------------------------------
# Unit tests - construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_graph_is_falsy(self) -> None:
        g = ContourGraph(np.array([]), np.array([]))
        assert not g

    def test_nonempty_graph_is_truthy(self) -> None:
        g = ContourGraph(np.array([0]), np.array([1]))
        assert g

    def test_single_edge_nodes(self) -> None:
        g = ContourGraph(np.array([10]), np.array([20]))
        assert set(g) == {10, 20}

    def test_adjacency_is_symmetric(self) -> None:
        # Path: 0 - 1 - 2
        g = ContourGraph(np.array([0, 1]), np.array([1, 2]))
        for u in g:
            for v in g._adj[u]:
                assert u in g._adj[v], f"{u} in adj[{v}] expected"

    def test_node_set_equals_unique_vertices(self) -> None:
        v1 = np.array([0, 1, 2])
        v2 = np.array([1, 2, 3])
        g = ContourGraph(v1, v2)
        assert set(g) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Unit tests - degree
# ---------------------------------------------------------------------------


class TestDegree:
    def test_path_endpoints_have_degree_1(self) -> None:
        # Path: 0 - 1 - 2
        g = ContourGraph(np.array([0, 1]), np.array([1, 2]))
        assert g.degree(0) == 1
        assert g.degree(2) == 1

    def test_path_interior_has_degree_2(self) -> None:
        # Path: 0 - 1 - 2
        g = ContourGraph(np.array([0, 1]), np.array([1, 2]))
        assert g.degree(1) == 2

    def test_cycle_all_nodes_have_degree_2(self) -> None:
        # Triangle: 0 - 1 - 2 - 0
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        for node in g:
            assert g.degree(node) == 2


# ---------------------------------------------------------------------------
# Unit tests - components
# ---------------------------------------------------------------------------


class TestComponents:
    def test_single_path_is_one_component(self) -> None:
        # Path: 0 - 1 - 2 - 3
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 3]))
        comps = list(g.components())
        assert len(comps) == 1
        assert comps[0] == {0, 1, 2, 3}

    def test_single_cycle_is_one_component(self) -> None:
        # Triangle: 0 - 1 - 2 - 0
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        comps = list(g.components())
        assert len(comps) == 1
        assert comps[0] == {0, 1, 2}

    def test_two_disconnected_paths(self) -> None:
        # Path A: 0-1-2  Path B: 10-11-12
        g = ContourGraph(
            np.array([0, 1, 10, 11]),
            np.array([1, 2, 11, 12]),
        )
        comps = list(g.components())
        assert len(comps) == 2
        assert {0, 1, 2} in comps
        assert {10, 11, 12} in comps

    def test_components_are_disjoint(self) -> None:
        g = ContourGraph(
            np.array([0, 1, 10, 11]),
            np.array([1, 2, 11, 12]),
        )
        comps = list(g.components())
        all_nodes = [n for c in comps for n in c]
        assert len(all_nodes) == len(set(all_nodes))

    def test_components_cover_all_nodes(self) -> None:
        g = ContourGraph(
            np.array([0, 1, 10, 11]),
            np.array([1, 2, 11, 12]),
        )
        assert {n for c in g.components() for n in c} == set(g)


# ---------------------------------------------------------------------------
# Unit tests - walk
# ---------------------------------------------------------------------------


class TestWalk:
    def test_walk_path_from_endpoint_visits_all_nodes(self) -> None:
        # Path: 0 - 1 - 2 - 3
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 3]))
        walked = g.walk(0)
        assert set(walked) == {0, 1, 2, 3}

    def test_walk_path_starts_at_given_node(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 3]))
        assert g.walk(0)[0] == 0
        assert g.walk(3)[0] == 3

    def test_walk_path_no_repeated_nodes(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 3]))
        walked = g.walk(0)
        assert len(walked) == len(set(walked))

    def test_walk_consecutive_nodes_are_adjacent(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 3]))
        walked = g.walk(0)
        for a, b in pairwise(walked):
            assert b in g._adj[a]

    def test_walk_cycle_visits_all_nodes(self) -> None:
        # Triangle: 0 - 1 - 2 - 0
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        walked = g.walk(0)
        assert set(walked) == {0, 1, 2}

    def test_walk_cycle_no_repeated_nodes(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        walked = g.walk(0)
        assert len(walked) == len(set(walked))


# ---------------------------------------------------------------------------
# Unit tests - to_networkx
# ---------------------------------------------------------------------------


class TestToNetworkx:
    def test_node_sets_match(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        nx_g = g.to_networkx()
        assert set(nx_g.nodes()) == set(g)

    def test_edge_sets_match(self) -> None:
        g = ContourGraph(np.array([0, 1, 2]), np.array([1, 2, 0]))
        nx_g = g.to_networkx()
        expected_edges = {frozenset(e) for e in [(0, 1), (1, 2), (2, 0)]}
        actual_edges = {frozenset(e) for e in nx_g.edges()}
        assert actual_edges == expected_edges

    def test_raises_import_error_when_networkx_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        g = ContourGraph(np.array([0, 1]), np.array([1, 2]))
        monkeypatch.setitem(sys.modules, "networkx", None)
        with pytest.raises(ImportError, match="networkx is required"):
            g.to_networkx()


# ---------------------------------------------------------------------------
# Property tests - structural invariants
# ---------------------------------------------------------------------------


class TestPathGraphProperties:
    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_adjacency_is_symmetric(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, _ = data
        for u in g:
            for v in g._adj[u]:
                assert u in g._adj[v]

    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_endpoints_have_degree_1(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        assert g.degree(nodes[0]) == 1
        assert g.degree(nodes[-1]) == 1

    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_interior_nodes_have_degree_2(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        for node in nodes[1:-1]:
            assert g.degree(node) == 2

    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_walk_covers_all_nodes(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        walked = g.walk(nodes[0])
        assert set(walked) == set(nodes)

    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_walk_length_equals_node_count(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        assert len(g.walk(nodes[0])) == len(nodes)

    @settings(deadline=None, max_examples=200)
    @given(data=path_graph())
    def test_walk_consecutive_nodes_are_adjacent(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        walked = g.walk(nodes[0])
        for a, b in pairwise(walked):
            assert b in g._adj[a]


class TestCycleGraphProperties:
    @settings(deadline=None, max_examples=200)
    @given(data=cycle_graph())
    def test_all_nodes_have_degree_2(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, _ = data
        for node in g:
            assert g.degree(node) == 2

    @settings(deadline=None, max_examples=200)
    @given(data=cycle_graph())
    def test_walk_covers_all_nodes(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        walked = g.walk(nodes[0])
        assert set(walked) == set(nodes)

    @settings(deadline=None, max_examples=200)
    @given(data=cycle_graph())
    def test_walk_no_repeated_nodes(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        g, nodes = data
        walked = g.walk(nodes[0])
        assert len(walked) == len(set(walked))

    @settings(deadline=None, max_examples=200)
    @given(data=cycle_graph())
    def test_walk_last_node_adjacent_to_first(
        self, data: tuple[ContourGraph, list[int]]
    ) -> None:
        """The walk leaves off one step from closing the cycle."""
        g, nodes = data
        walked = g.walk(nodes[0])
        assert walked[0] in g._adj[walked[-1]]


# ---------------------------------------------------------------------------
# Property tests - cross-validation against networkx
# ---------------------------------------------------------------------------


class TestAgainstNetworkx:
    """Verify ContourGraph agrees with networkx on all relevant operations.

    These tests are the primary argument for correctness: for randomly
    generated graphs, ContourGraph and networkx must agree on components
    and node degrees.
    """

    @settings(deadline=None, max_examples=200)
    @given(data=multi_component_graph())
    def test_components_match_networkx(
        self, data: tuple[ContourGraph, list[list[int]]]
    ) -> None:
        g, _ = data
        nx_comps = list(nx.connected_components(g.to_networkx()))

        our_comps = list(g.components())

        assert len(our_comps) == len(nx_comps)
        assert sorted(our_comps, key=min) == sorted(nx_comps, key=min)

    @settings(deadline=None, max_examples=200)
    @given(data=multi_component_graph())
    def test_degrees_match_networkx(
        self, data: tuple[ContourGraph, list[list[int]]]
    ) -> None:
        g, _ = data
        nx_g = g.to_networkx()

        for node in g:
            assert g.degree(node) == nx_g.degree(node)

    @settings(deadline=None, max_examples=200)
    @given(data=multi_component_graph())
    def test_components_partition_node_set(
        self, data: tuple[ContourGraph, list[list[int]]]
    ) -> None:
        g, _ = data
        comps = list(g.components())

        # disjoint
        all_nodes = [n for c in comps for n in c]
        assert len(all_nodes) == len(set(all_nodes))

        # covers
        assert set(all_nodes) == set(g)
