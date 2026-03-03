from __future__ import annotations

import numpy as np
import pytest
import shapely

import mosaic.utils


def _compute_cell_patches(
    polygons: list[np.ndarray], max_edges: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """ """
    n_cells = len(polygons)
    max_edges = max_edges or max(p.shape[0] for p in polygons)

    cell_patches = np.empty((n_cells, max_edges, 2), dtype=float)
    vertices_on_cell = -np.ones((n_cells, max_edges), dtype=int)

    for i, p in enumerate(polygons):
        # number of sides in polygon
        K = p.shape[0]
        # need to have at least 3 valid sides (i.e. triangle)
        assert K >= 3
        # start by filling with first vertex
        cell_patches[i, :, :] = p[0]
        # then fill in valid coordinates
        cell_patches[i, :K, :] = p
        # populate connectivity array
        vertices_on_cell[i, :K] = np.arange(K, dtype=np.int64)

    return cell_patches, vertices_on_cell


def random_convex_polygon(rng, scale=1.0, n_points=40):
    pts = rng.normal(0, scale, size=(n_points, 2))
    hull = shapely.MultiPoint(pts).convex_hull
    if hull.geom_type != "Polygon":
        raise RuntimeError()
    return np.asarray(hull.exterior.coords)


def test_triangle_exact():
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    patches, vertices_on_cell = _compute_cell_patches([tri], max_edges=10)
    cx, cy = mosaic.utils.compute_cell_centroid(patches, vertices_on_cell)
    sx, sy = shapely.Polygon(tri).centroid.xy

    np.testing.assert_allclose(
        [cx[0], cy[0]], [2 / 3, 2 / 3], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose([cx, cy], [sx, sy], rtol=1e-12, atol=1e-12)


def test_orientation_invariance():
    rng = np.random.default_rng()
    f_polys = [random_convex_polygon(rng) for _ in range(10)]
    b_polys = [p[::-1] for p in f_polys]
    f_patches, f_verts_on_cell = _compute_cell_patches(f_polys)
    b_patches, b_verts_on_cell = _compute_cell_patches(b_polys)

    f_cx, f_cy = mosaic.utils.compute_cell_centroid(f_patches, f_verts_on_cell)
    b_cx, b_cy = mosaic.utils.compute_cell_centroid(b_patches, b_verts_on_cell)

    sx, sy = np.unstack(
        np.hstack([shapely.Polygon(p).centroid.xy for p in f_polys])
    )

    np.testing.assert_allclose([f_cx, f_cy], [sx, sy], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose([b_cx, b_cy], [sx, sy], rtol=1e-11, atol=1e-11)


def test_ragged_invariance():
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])

    patch_1, verts_on_cell_1 = _compute_cell_patches([poly], max_edges=4)
    patch_2, verts_on_cell_2 = _compute_cell_patches([poly], max_edges=40)

    cx_1, cy_1 = mosaic.utils.compute_cell_centroid(patch_1, verts_on_cell_1)
    cx_2, cy_2 = mosaic.utils.compute_cell_centroid(patch_2, verts_on_cell_2)

    np.testing.assert_allclose([cx_1, cy_1], [cx_2, cy_2], rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", np.logspace(1, 6, 11))
def test_util_vs_shapely_for_random_convex_polygon(scale):
    rng = np.random.default_rng()
    polys = [random_convex_polygon(rng, scale=scale) for _ in range(100)]
    patches, vertices_on_cell = _compute_cell_patches(polys)

    cx, cy = mosaic.utils.compute_cell_centroid(patches, vertices_on_cell)

    sx, sy = np.unstack(
        np.hstack([shapely.Polygon(p).centroid.xy for p in polys])
    )

    np.testing.assert_allclose([cx, cy], [sx, sy], rtol=1e-12, atol=1e-12)


def test_padding_violation_raises_and_reports_cell():
    rng = np.random.default_rng()
    polys = [random_convex_polygon(rng) for _ in range(25)]
    patches, vertices_on_cell = _compute_cell_patches(polys)

    # find a random 1-d index for a ragged node
    idx = rng.choice(np.flatnonzero(vertices_on_cell == -1))

    # map the 1-d index to the row/col of the ragged node
    i, j = np.unravel_index(idx, vertices_on_cell.shape)

    # manually break the padding with first node
    patches[i, j, :] = np.array([999.0, 999.0])

    with pytest.raises(ValueError, match="Padded vertex") as e:
        mosaic.utils.compute_cell_centroid(patches, vertices_on_cell)

    assert str(i) in str(e.value)


# TODO: for a given polygon in lat/lon coorindates, iterate over all cartopy
#       projection and ensure manual centroid and shapely match in projected
#       coordinates
