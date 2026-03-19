from __future__ import annotations

import cartopy.crs as ccrs
import networkx as nx
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import mosaic
from mosaic.contour import MPASContourGenerator


class ContourGraphGenerator:
    """
    Generate random field of floats and random contour (filled or unfilled)
    """

    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())

    def _build_contour_cyle_graph(
        self, data: st.DataObject, filled: bool
    ) -> tuple[nx.Graph, set[int]]:
        n_cells = self.descriptor.sizes["nCells"]

        field = data.draw(
            hnp.arrays(
                dtype=float,
                shape=(n_cells,),
                elements=st.floats(
                    min_value=-1e6,
                    max_value=+1e6,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )

        field_min, field_max = np.min(field), np.max(field)
        field_range = field_max - field_min

        # Ensure field has sufficient variation
        assume(field_min < field_max)
        # Ensure at least 1% relative variation, or absolute minimum of 0.01
        min_variation = max(0.01, 0.01 * abs(field_max))
        assume(field_range > min_variation)

        # Generate contour levels strictly between min and max
        if filled:
            lower_level = data.draw(
                st.floats(
                    min_value=field_min,
                    max_value=field_max,
                    exclude_min=True,
                    exclude_max=True,
                )
            )

            upper_level = data.draw(
                st.floats(
                    min_value=lower_level,
                    max_value=field_max,
                    exclude_max=True,
                )
            )
            # Ensure lower < upper
            assume(lower_level < upper_level)

            cell_mask = (field > lower_level) & (field < upper_level)
        else:
            level = data.draw(
                st.floats(
                    min_value=field_min,
                    max_value=field_max,
                    exclude_min=True,
                    exclude_max=True,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )

            cell_mask = field > level

        # Exclude all True and all False masks
        assume(cell_mask.any() and (~cell_mask).any())

        contour_gen = MPASContourGenerator(
            self.descriptor, cell_mask.astype(float)
        )

        graph = contour_gen._create_contour_graph(cell_mask, filled=filled)
        boundary_vertices = set(contour_gen.boundary_vertices)

        return graph, boundary_vertices


class TestFilledContourGraphProperties(ContourGraphGenerator):
    """
    Filled contours should only be cycle graphs

    .. [1] https://en.wikipedia.org/wiki/Cycle_graph
    """

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_node_degrees_are_2(self, data: st.DataObject) -> None:
        """Cycle graphs should only have nodes of degree 2"""
        graph, _ = self._build_contour_cyle_graph(data, filled=True)

        for node, degree in graph.degree():
            assert degree == 2, (
                f"Node {node} has degree {degree}; expected 1 or 2."
            )


class TestUnfilledContourGraphProperties(ContourGraphGenerator):
    """
    Unfilled contours should only be path or cycle graphs

    .. [1] https://en.wikipedia.org/wiki/Path_graph
    .. [2] https://en.wikipedia.org/wiki/Cycle_graph
    """

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_node_degrees_are_1_or_2(self, data: st.DataObject) -> None:
        """Path/cycle graphs should only have nodes of deg. 1/2, respctively"""
        graph, _ = self._build_contour_cyle_graph(data, filled=False)

        for node, degree in graph.degree():
            assert degree in (1, 2), (
                f"Node {node} has degree {degree}; expected 1 or 2."
            )

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_path_endpoints_on_boundary(self, data: st.DataObject) -> None:
        """If component is path graph, should begin & end on mesh boundary"""
        graph, boundary_vertices = self._build_contour_cyle_graph(
            data, filled=False
        )

        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            endpoints = [n for n, d in subgraph.degree() if d == 1]

            if len(endpoints) != 0:
                invalid = set(endpoints) - boundary_vertices

                if len(invalid) > 0:
                    msg = f"Endpoints {invalid} are not boundary vertices."
                    raise AssertionError(msg)


class TestEmptyContours:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())

    def test_unfilled_contour_all_below_level(self) -> None:
        """When all values are below contour level, return empty contour"""
        field = np.ones(self.descriptor.sizes["nCells"]) * 5.0
        level = 10.0

        contour_gen = MPASContourGenerator(self.descriptor, field)
        polys, codes = contour_gen.create_contour(level)

        assert isinstance(polys, list)
        assert isinstance(codes, list)
        assert len(polys) == 0
        assert len(codes) == 0

    def test_unfilled_contour_all_above_level(self) -> None:
        """When all values are above contour level, return empty contour"""
        field = np.ones(self.descriptor.sizes["nCells"]) * 10.0
        level = 5.0

        contour_gen = MPASContourGenerator(self.descriptor, field)
        polys, codes = contour_gen.create_contour(level)

        assert isinstance(polys, list)
        assert isinstance(codes, list)
        assert len(polys) == 0
        assert len(codes) == 0

    def test_filled_contour_levels_all_above(self) -> None:
        """When field is outside fill bounds, return empty contour"""
        field = np.ones(self.descriptor.sizes["nCells"]) * 5.0

        contour_gen = MPASContourGenerator(self.descriptor, field)
        polys, codes = contour_gen.create_filled_contour(
            lower_level=10.0, upper_level=20.0
        )

        assert isinstance(polys, list)
        assert isinstance(codes, list)
        assert len(polys) == 0
        assert len(codes) == 0

    def test_filled_contour_levels_all_below(self) -> None:
        """When field is outside fill bounds, return empty contour"""
        field = np.ones(self.descriptor.sizes["nCells"]) * 20.0

        contour_gen = MPASContourGenerator(self.descriptor, field)
        polys, codes = contour_gen.create_filled_contour(
            lower_level=5.0, upper_level=10.0
        )

        assert isinstance(polys, list)
        assert isinstance(codes, list)
        assert len(polys) == 0
        assert len(codes) == 0
