from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import mosaic
from mosaic import _geometries

# -----------------------------------------------------------------------
# A small, hand-specified 2-cell mesh used for exact-value unit testing.
#
#   v3(0,1) --e2-- v2(1,1) --e5-- v4(2,0.5)
#     |              |  \            /
#     e3   cell 0    e1   \  cell 1 /
#     |              |      \     /
#   v0(0,0) --e0-- v1(1,0) ---e4--
#
# * cell 0 is a square   (v0, v1, v2, v3) -- no padding
# * cell 1 is a triangle (v1, v4, v2)     -- padded w/ -1 (maxEdges=4)
# * e1 (v1-v2) is the only interior edge, shared by both cells; the other
#   five edges (e0, e2, e3, e4, e5) are boundary edges.
# -----------------------------------------------------------------------


def _make_synthetic_mesh() -> xr.Dataset:
    verticesOnCell = np.array([[0, 1, 2, 3], [1, 4, 2, -1]])
    verticesOnEdge = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 2]])
    cellsOnEdge = np.array(
        [[0, -1], [0, 1], [0, -1], [0, -1], [1, -1], [1, -1]]
    )

    return xr.Dataset(
        data_vars={
            "xVertex": ("nVertices", [0.0, 1.0, 1.0, 0.0, 2.0]),
            "yVertex": ("nVertices", [0.0, 0.0, 1.0, 1.0, 0.5]),
            "xCell": ("nCells", [0.5, 1.3]),
            "yCell": ("nCells", [0.5, 0.5]),
            "xEdge": ("nEdges", [0.5, 1.0, 0.5, 0.0, 1.5, 1.5]),
            "yEdge": ("nEdges", [0.0, 0.5, 1.0, 0.5, 0.25, 0.75]),
            "verticesOnCell": (("nCells", "maxEdges"), verticesOnCell),
            "verticesOnEdge": (("nEdges", "TWO"), verticesOnEdge),
            "cellsOnEdge": (("nEdges", "TWO"), cellsOnEdge),
        }
    )


@pytest.fixture
def synthetic_mesh() -> xr.Dataset:
    return _make_synthetic_mesh()


def _zero_indexed(name: str) -> xr.Dataset:
    """Zero-indexed, minimal dataset for a cached real test mesh.

    ``_geometries`` functions expect zero-indexed connectivity arrays (as
    produced by ``Descriptor``), not the raw, 1-indexed mesh file.
    """
    return mosaic.Descriptor(mosaic.datasets.open_dataset(name)).ds


def _undirected_edges_from_patches(
    patches: np.ndarray, valid_mask: np.ndarray
) -> set:
    """Rounded, undirected vertex-pairs tracing each patch's boundary."""
    edges = set()
    for i in range(patches.shape[0]):
        n_valid = int(valid_mask[i].sum())
        pts = [tuple(np.round(patches[i, j], 8)) for j in range(n_valid)]
        for j in range(n_valid):
            edges.add(frozenset((pts[j], pts[(j + 1) % n_valid])))
    return edges


def _undirected_edges_from_wireframe(x: np.ndarray, y: np.ndarray) -> set:
    """Rounded, undirected vertex-pairs for each wireframe line segment."""
    edges = set()
    for row in range(x.shape[0]):
        a = (round(float(x[row, 0]), 8), round(float(y[row, 0]), 8))
        b = (round(float(x[row, 1]), 8), round(float(y[row, 1]), 8))
        edges.add(frozenset((a, b)))
    return edges


# -----------------------------------------------------------------------
# resolve_coords: the shared coordinate look-up/fallback primitive behind
# every patch and wireframe function. Tested in isolation with tiny inline
# arrays, so the per-location wireframe tests below don't need to
# re-verify every fallback branch themselves.
# -----------------------------------------------------------------------


def test_resolve_coords_fallback_first():
    ds = {
        "xVertex": np.array([0.0, 1.0, 2.0]),
        "yVertex": np.array([0.0, 5.0, 9.0]),
    }
    conn = np.array([[0, 1, -1], [2, 0, -1]])

    x, y = _geometries.resolve_coords(
        ds, conn, source="Vertex", fallback="First"
    )

    # the padded (-1) slot collapses to the first valid node of its row
    np.testing.assert_array_equal(x, [[0.0, 1.0, 0.0], [2.0, 0.0, 2.0]])
    np.testing.assert_array_equal(y, [[0.0, 5.0, 0.0], [9.0, 0.0, 9.0]])


def test_resolve_coords_fallback_location():
    ds = {
        "xCell": np.array([10.0, 20.0, 30.0]),
        "yCell": np.array([1.0, 2.0, 3.0]),
        "xEdge": np.array([100.0, 200.0]),
        "yEdge": np.array([-1.0, -2.0]),
    }
    conn = np.array([[0, -1], [1, 2]])  # row 0 is missing its 2nd neighbor

    x, y = _geometries.resolve_coords(ds, conn, source="Cell", fallback="Edge")

    # the missing cell neighbor falls back to the "Edge" location instead
    np.testing.assert_array_equal(x, [[10.0, 100.0], [20.0, 30.0]])
    np.testing.assert_array_equal(y, [[1.0, -1.0], [2.0, 3.0]])


def test_resolve_coords_no_fallback_raises():
    ds = {"xVertex": np.array([0.0, 1.0]), "yVertex": np.array([0.0, 1.0])}
    conn = np.array([[0, 1], [1, -1]])

    with pytest.raises(ValueError, match="no fallback given"):
        _geometries.resolve_coords(ds, conn, source="Vertex", fallback=None)


def test_resolve_coords_ignore_skips_correction():
    ds = {
        "xVertex": np.array([0.0, 1.0, 2.0]),
        "yVertex": np.array([0.0, 1.0, 2.0]),
    }
    conn = np.array([[0, 1, -1]])

    x, _ = _geometries.resolve_coords(
        ds, conn, source="Vertex", fallback="Ignore"
    )

    # "Ignore" skips fallback correction entirely, so -1 is left to NumPy's
    # default wraparound indexing (silently resolving to the last vertex).
    # This documents that behavior rather than relying on it elsewhere.
    np.testing.assert_array_equal(x, [[0.0, 1.0, 2.0]])


# -----------------------------------------------------------------------
# Structural contract shared by all three compute_*_wireframe functions.
# -----------------------------------------------------------------------


@pytest.mark.parametrize("location", ["Cell", "Edge", "Vertex"])
def test_wireframe_shape_and_nan_contract(synthetic_mesh, location):
    compute = getattr(_geometries, f"compute_{location.lower()}_wireframe")
    x, y = compute(synthetic_mesh)

    assert x.shape == y.shape
    assert x.shape[1] == 3

    # NaN separator in the 3rd column only, so each row plots as one
    # disconnected segment when ravel()'d for ax.plot
    assert np.isnan(x[:, 2]).all()
    assert np.isnan(y[:, 2]).all()
    assert not np.isnan(x[:, :2]).any()
    assert not np.isnan(y[:, :2]).any()
    np.testing.assert_array_equal(np.isnan(x), np.isnan(y))


# -----------------------------------------------------------------------
# Node containment: every segment endpoint drawn by a wireframe must be
# one of the "real" nodes returned by compute_node_coords for that
# location (i.e. the points where markers get drawn).
#
# This holds for "Cell" and "Vertex" locations. It does *not* hold for
# "Edge": the boundary-closing segments in compute_edge_wireframe reuse
# compute_boundary_wireframe's midpoint-split purely for line-drawing
# convenience, intentionally passing through an edge midpoint that isn't
# promoted to a marker node (it's not a true node of the edge patch).
# -----------------------------------------------------------------------


@pytest.mark.parametrize("location", ["Cell", "Vertex"])
def test_node_containment(synthetic_mesh, location):
    compute = getattr(_geometries, f"compute_{location.lower()}_wireframe")
    x, y = compute(synthetic_mesh)
    nx, ny = _geometries.compute_node_coords(synthetic_mesh, location)

    endpoints = set(
        zip(
            np.round(x[:, :2].ravel(), 8),
            np.round(y[:, :2].ravel(), 8),
            strict=False,
        )
    )
    nodes = set(zip(np.round(nx, 8), np.round(ny, 8), strict=False))

    assert endpoints <= nodes


# -----------------------------------------------------------------------
# Node uniqueness: coordinates returned by compute_node_coords must be
# unique so markers are not plotted multiple times at the same location
# (which causes opacity artifacts when alpha < 1).
# -----------------------------------------------------------------------


@pytest.mark.parametrize("location", ["Cell", "Edge", "Vertex"])
def test_node_coords_are_unique(location):
    ds = _zero_indexed("QU.240km")
    x, y = _geometries.compute_node_coords(ds, location)
    coords = np.column_stack([x, y])

    assert len(coords) == len(np.unique(coords, axis=0))


# -----------------------------------------------------------------------
# Boundary / topology counts
# -----------------------------------------------------------------------


def test_boundary_wireframe_empty_without_culled_boundary():
    # every edge of this doubly periodic mesh has two valid cell neighbors
    ds = _zero_indexed("doubly_periodic_4x4")

    x, y = _geometries.compute_boundary_wireframe(ds)

    assert x.shape == (0, 2)
    assert y.shape == (0, 2)


def test_boundary_wireframe_count_matches_boundary_edges(synthetic_mesh):
    x, y = _geometries.compute_boundary_wireframe(synthetic_mesh)

    n_boundary_edges = int(
        np.any(np.asarray(synthetic_mesh.cellsOnEdge) < 0, axis=1).sum()
    )

    assert x.shape == (2 * n_boundary_edges, 2)
    assert y.shape == (2 * n_boundary_edges, 2)


def test_cell_wireframe_segment_count_synthetic(synthetic_mesh):
    x, _ = _geometries.compute_cell_wireframe(synthetic_mesh)
    assert x.shape[0] == synthetic_mesh.sizes["nEdges"]


def test_cell_wireframe_segment_count_real_mesh():
    ds = _zero_indexed("QU.240km")
    x, _ = _geometries.compute_cell_wireframe(ds)
    assert x.shape[0] == ds.sizes["nEdges"]


# -----------------------------------------------------------------------
# Patch <-> wireframe consistency: two independently-coded paths that
# should describe exactly the same mesh geometry.
# -----------------------------------------------------------------------


def test_cell_wireframe_matches_patch_edges_synthetic(synthetic_mesh):
    patches = _geometries.compute_cell_patches(synthetic_mesh)
    valid_mask = np.asarray(synthetic_mesh.verticesOnCell) >= 0
    patch_edges = _undirected_edges_from_patches(patches, valid_mask)

    x, y = _geometries.compute_cell_wireframe(synthetic_mesh)
    wireframe_edges = _undirected_edges_from_wireframe(x, y)

    assert patch_edges == wireframe_edges


def test_cell_wireframe_matches_patch_edges_real_mesh():
    ds = _zero_indexed("QU.240km")

    patches = _geometries.compute_cell_patches(ds)
    valid_mask = np.asarray(ds.verticesOnCell) >= 0
    patch_edges = _undirected_edges_from_patches(patches, valid_mask)

    x, y = _geometries.compute_cell_wireframe(ds)
    wireframe_edges = _undirected_edges_from_wireframe(x, y)

    assert patch_edges == wireframe_edges


# -----------------------------------------------------------------------
# Exact-value / executable-documentation test
# -----------------------------------------------------------------------


def test_cell_wireframe_exact_values(synthetic_mesh):
    x, y = _geometries.compute_cell_wireframe(synthetic_mesh)

    expected_x = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, 1.0, np.nan],
            [1.0, 0.0, np.nan],
            [0.0, 0.0, np.nan],
            [1.0, 2.0, np.nan],
            [2.0, 1.0, np.nan],
        ]
    )
    expected_y = np.array(
        [
            [0.0, 0.0, np.nan],
            [0.0, 1.0, np.nan],
            [1.0, 1.0, np.nan],
            [1.0, 0.0, np.nan],
            [0.0, 0.5, np.nan],
            [0.5, 1.0, np.nan],
        ]
    )

    np.testing.assert_allclose(x, expected_x, equal_nan=True)
    np.testing.assert_allclose(y, expected_y, equal_nan=True)
