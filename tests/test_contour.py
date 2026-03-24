from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.path as mpath
import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from shapely import LineString, Polygon

import mosaic
from mosaic.contour import MPASContourGenerator, _is_ccw

# TODO:
# - [ ] Need some further investigation of how matplotlib handles nans


def _square_ring(
    cx: float, cy: float, size: float, ccw: bool = True
) -> np.ndarray:
    """Return a closed square ring centered at (cx, cy) with half-width *size*.

    The ring is counter-clockwise by default and clockwise when ccw=False.
    """
    if ccw:
        coords = [
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size],
        ]
    else:
        coords = [
            [cx - size, cy - size],
            [cx - size, cy + size],
            [cx + size, cy + size],
            [cx + size, cy - size],
        ]
    coords.append(coords[0])
    return np.array(coords, dtype=float)


def _make_codes(ring: np.ndarray) -> np.ndarray:
    """Return matplotlib path codes for a single ring (MOVETO + LINETOs)."""
    codes = np.full(len(ring), mpath.Path.LINETO, dtype=mpath.Path.code_type)
    codes[0] = mpath.Path.MOVETO
    return codes


def _parse_rings(
    poly_coords: np.ndarray, codes: np.ndarray
) -> list[np.ndarray]:
    """Split a single output path into constituent rings by MOVETO boundary."""
    starts = np.flatnonzero(codes == mpath.Path.MOVETO)
    rings = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(codes)
        rings.append(poly_coords[start:end])
    return rings


class ContourGenerator:
    """
    Generate random field of floats and random contour (filled or unfilled)
    """

    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())

    def _build_field(self, data: st.DataObject) -> np.ndarray:
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

        return field

    def _build_unfilled_contour_level(
        self, field_min: float, field_max: float, data: st.DataObject
    ) -> float:
        return data.draw(
            st.floats(
                min_value=field_min,
                max_value=field_max,
                exclude_min=True,
                exclude_max=True,
                allow_nan=False,
                allow_infinity=False,
            )
        )

    def _build_filled_contour_levels(
        self, field_min: float, field_max: float, data: st.DataObject
    ) -> tuple[float, float]:
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

        return lower_level, upper_level


class ContourGraphGenerator(ContourGenerator):
    """
    Convert random field and contour to a networkx.Graph

    Also returns unique boundary vertices, needed for proppert testing
    """

    def _build_contour_cycle_graph(
        self, data: st.DataObject, filled: bool
    ) -> tuple[nx.Graph, set[int]]:
        field = self._build_field(data)

        field_min, field_max = np.min(field), np.max(field)

        contour_gen = MPASContourGenerator(self.descriptor, field)

        if filled:
            lower_level, upper_level = self._build_filled_contour_levels(
                field_min=field_min, field_max=field_max, data=data
            )

            cell_mask = (field > lower_level) & (field < upper_level)
        else:
            level = self._build_unfilled_contour_level(
                field_min=field_min, field_max=field_max, data=data
            )

            cell_mask = field > level

        # Exclude all True and all False masks
        assume(cell_mask.any() and (~cell_mask).any())

        graph = contour_gen._create_contour_graph(cell_mask, filled=filled)
        boundary_vertices = set(contour_gen.boundary_vertices)

        return graph.to_networkx(), boundary_vertices


class ContourGeometryGenerator(ContourGenerator):
    """
    Convert contour coordinates to Shapely geometries

    Filled contour --> shapely.Polygon
    Unfilled Conoutr --> shapely.LineString
    """

    def _build_contour_geometries(
        self, data: st.DataObject, filled: bool
    ) -> list[LineString] | list[Polygon]:
        field = self._build_field(data)

        field_min, field_max = np.min(field), np.max(field)

        contour_gen = MPASContourGenerator(self.descriptor, field)

        if filled:
            lower_level, upper_level = self._build_filled_contour_levels(
                field_min=field_min, field_max=field_max, data=data
            )

            polys, codes = contour_gen.create_filled_contour(
                lower_level, upper_level
            )

            geoms = []
            for p, c in zip(polys, codes, strict=True):
                # Split at MOVETO boundaries
                starts = np.flatnonzero(c == mpath.Path.MOVETO)
                # first ring = exterior shell, rest = holes
                rings = [
                    p[
                        starts[i] : starts[i + 1]
                        if i + 1 < len(starts)
                        else len(p)
                    ]
                    for i in range(len(starts))
                ]

                geoms.append(Polygon(shell=rings[0], holes=rings[1:]))

        else:
            level = self._build_unfilled_contour_level(
                field_min=field_min, field_max=field_max, data=data
            )

            polys, _ = contour_gen.create_contour(level)

            geoms = [LineString(p) for p in polys]

        return geoms


class TestFilledContourGraphProperties(ContourGraphGenerator):
    """
    Filled contours should only be cycle graphs

    .. [1] https://en.wikipedia.org/wiki/Cycle_graph
    """

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_node_degrees_are_2(self, data: st.DataObject) -> None:
        """Cycle graphs should only have nodes of degree 2"""
        graph, _ = self._build_contour_cycle_graph(data, filled=True)

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
        graph, _ = self._build_contour_cycle_graph(data, filled=False)

        for node, degree in graph.degree():
            assert degree in (1, 2), (
                f"Node {node} has degree {degree}; expected 1 or 2."
            )

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_path_endpoints_on_boundary(self, data: st.DataObject) -> None:
        """If component is path graph, should begin & end on mesh boundary"""
        graph, boundary_vertices = self._build_contour_cycle_graph(
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


class TestFilledContourPolygonProperties(ContourGeometryGenerator):
    """Cycle graphs produced by a filled contour should be valid Polygons"""

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_no_self_intersection_and_closed(
        self, data: st.DataObject
    ) -> None:
        """Polygon is valid if closed and is not self intersecting"""
        geoms = self._build_contour_geometries(data, filled=True)

        assert all(g.is_valid for g in geoms)


class TestUnfilledContourLineStringProperties(ContourGeometryGenerator):
    """Convert unfilled contours to LineStrings; test for self intersection"""

    @settings(deadline=None, max_examples=200)
    @given(data=st.data())
    def test_no_self_intersection(self, data: st.DataObject) -> None:
        """LineStrings is simple if there is no self intersection"""
        geoms = self._build_contour_geometries(data, filled=False)

        assert all(g.is_simple for g in geoms)


class TestEmptyContours:
    """Test empty contour is well defined"""

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


class TestCheckLevels:
    """Test validation of contour level ordering"""

    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())
    field = np.ones(descriptor.sizes["nCells"])
    contour_gen = MPASContourGenerator(descriptor, field)

    def test_check_levels_valid_increasing(self) -> None:
        """Contour levels with lower < upper should pass validation"""
        lower_level = 5.0
        upper_level = 10.0

        result_lower, result_upper = self.contour_gen.check_levels(
            lower_level, upper_level
        )

        assert result_upper == upper_level
        assert result_lower == lower_level

    def test_check_levels_equal(self) -> None:
        """Contour levels with lower == upper should raise ValueError"""
        level = 5.0

        with pytest.raises(
            ValueError, match="Contour levels must be increasing"
        ):
            self.contour_gen.check_levels(lower_level=level, upper_level=level)

    def test_check_levels_inverted(self) -> None:
        """Contour levels with lower > upper should raise ValueError"""
        lower_level = 10.0
        upper_level = 5.0

        with pytest.raises(
            ValueError, match="Contour levels must be increasing"
        ):
            self.contour_gen.check_levels(
                lower_level=lower_level, upper_level=upper_level
            )


class TestSortFilledContoursUnit:
    """Unit tests for _sort_filled_contours using synthetic ring data.

    Each test constructs explicit coordinate arrays at known nesting depths
    and checks that the output topology (number of paths, ring counts, hole
    containment, and winding order) matches the expected result.
    """

    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, projection=ccrs.PlateCarree())
    field = np.ones(descriptor.sizes["nCells"])
    contour_gen = MPASContourGenerator(descriptor, field)

    def test_single_ring_no_holes(self) -> None:
        """A single ring with no nesting produces one exterior polygon."""
        ring = _square_ring(0, 0, 5)
        polys = [ring]
        codes = [_make_codes(ring)]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 1
        assert np.sum(out_codes[0] == mpath.Path.MOVETO) == 1

    def test_two_disjoint_rings(self) -> None:
        """Two non-overlapping rings produce two separate exterior polygons."""
        polys = [_square_ring(0, 0, 5), _square_ring(20, 0, 5)]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 2
        assert all(np.sum(c == mpath.Path.MOVETO) == 1 for c in out_codes)

    def test_simple_donut(self) -> None:
        """One ring containing another produces one polygon with one hole."""
        outer = _square_ring(0, 0, 10)
        inner = _square_ring(0, 0, 4)
        polys = [outer, inner]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 1
        rings = _parse_rings(out_polys[0], out_codes[0])
        assert len(rings) == 2

    def test_three_level_nesting(self) -> None:
        """A→B→C nesting: exterior A with hole B, and standalone exterior C."""
        ring_a = _square_ring(0, 0, 12)
        ring_b = _square_ring(0, 0, 8)
        ring_c = _square_ring(0, 0, 4)
        polys = [ring_a, ring_b, ring_c]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 2
        moveto_counts = sorted(
            np.sum(c == mpath.Path.MOVETO) for c in out_codes
        )
        assert moveto_counts == [1, 2]

    def test_multiple_holes(self) -> None:
        """One exterior ring containing three disjoint holes."""
        polys = [
            _square_ring(0, 0, 20),
            _square_ring(-10, 0, 3),
            _square_ring(0, 0, 3),
            _square_ring(10, 0, 3),
        ]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 1
        assert np.sum(out_codes[0] == mpath.Path.MOVETO) == 4

    def test_two_separate_donuts(self) -> None:
        """Two independent donuts each produce one polygon with one hole."""
        polys = [
            _square_ring(-15, 0, 8),
            _square_ring(-15, 0, 3),
            _square_ring(15, 0, 8),
            _square_ring(15, 0, 3),
        ]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 2
        assert all(np.sum(c == mpath.Path.MOVETO) == 2 for c in out_codes)

    def test_deep_nesting_four_levels(self) -> None:
        """A→B→C→D: two output polygons, each with one hole."""
        polys = [
            _square_ring(0, 0, 16),
            _square_ring(0, 0, 12),
            _square_ring(0, 0, 8),
            _square_ring(0, 0, 4),
        ]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        assert len(out_polys) == 2
        moveto_counts = sorted(
            np.sum(c == mpath.Path.MOVETO) for c in out_codes
        )
        assert moveto_counts == [2, 2]

    def test_winding_order_correction_same_orientation(self) -> None:
        """A hole with the same winding as its exterior must be reversed."""
        outer = _square_ring(0, 0, 10, ccw=True)
        inner = _square_ring(0, 0, 4, ccw=True)
        polys = [outer, inner]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        rings = _parse_rings(out_polys[0], out_codes[0])
        assert _is_ccw(Polygon(rings[0]))
        assert not _is_ccw(Polygon(rings[1]))

    def test_hole_already_opposite_winding(self) -> None:
        """A hole already opposite to its exterior must not be reversed."""
        outer = _square_ring(0, 0, 10, ccw=True)
        inner = _square_ring(0, 0, 4, ccw=False)
        polys = [outer, inner]
        codes = [_make_codes(r) for r in polys]

        out_polys, out_codes = self.contour_gen._sort_filled_contours(
            polys, codes
        )

        rings = _parse_rings(out_polys[0], out_codes[0])
        assert _is_ccw(Polygon(rings[0]))
        assert not _is_ccw(Polygon(rings[1]))
