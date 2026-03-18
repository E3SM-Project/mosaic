from __future__ import annotations

import cartopy.crs as ccrs
import networkx as nx
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import mosaic
from mosaic.contour import MPASContourGenerator


class ContourCycleGraphGenerator:
    """ """

    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())

    def _build_contour_cyle_graph(
        self, data: st.DataObject, filled: bool
    ) -> tuple[nx.Graph, set[int]]:
        """ """

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

        # Ensure field has sufficient variation
        assume(field_min < field_max)
        # TODO: Adjust this threshold based on range of values in field
        assume(field_max - field_min > 1.0)

        if filled:
            # Generate contour levels strictly between min and max
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
            # Generate contour level strictly between min and max
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


class TestFilledContourProperties(ContourCycleGraphGenerator):
    """ """

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_node_degrees_are_2(self, data: st.DataObject) -> None:
        """Test that all nodes in the boundary graph have degree 1 or 2."""
        graph, _ = self._build_contour_cyle_graph(data, filled=True)

        for node, degree in graph.degree():
            assert degree == 2, (
                f"Node {node} has degree {degree}; expected 1 or 2."
            )


class TestUnfilledContourProperties(ContourCycleGraphGenerator):
    """ """

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_node_degrees_are_1_or_2(self, data: st.DataObject) -> None:
        """Test that all nodes in the boundary graph have degree 1 or 2."""
        graph, _ = self._build_contour_cyle_graph(data, filled=False)

        for node, degree in graph.degree():
            assert degree in (1, 2), (
                f"Node {node} has degree {degree}; expected 1 or 2."
            )

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_paths_endpoints_on_boundary(self, data: st.DataObject) -> None:
        """ """
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
