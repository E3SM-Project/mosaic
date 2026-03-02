from __future__ import annotations

import numpy as np
import shapely
import xarray as xr
from cartopy.crs import CRS
from numpy.typing import ArrayLike

__all__ = ["cull_mesh", "get_invalid_patches"]


def _make_lookup_table(mask: np.ndarray[bool]) -> np.ndarray[np.int64]:
    """
    Build lookup table for remapping connectivity indices after culling.

    Parameters
    ----------
    mask : :py:class:`~numpy.ndarray`
        Array of length N where True and False mean keep and drop, respectively

    Returns
    -------
    :py:class:`~numpy.ndarray`
        An array of length N+1 where kept indices map to [0, M), culled cells
        map to -1, and -1 indices are preserved from the original mesh.

    Notes
    -----
    This function assumes the connectivity array the lookup table will be
    applied to have already been zero-indexed.

    - N: number of cells in the base mesh
    - P: number of cells culled from the base mesh
    - M: N-P (i.e. number of cells in the culled mesh)

    Because an index of `-1` maps to the last element of an array, but a
    connectivity value of `-1` mean "no neighbor", the lookup table is
    constructed with a one-position offset, so that:
        - index 0 represents old id -1
        - index (i + 1) represents old id i  (for i in 0..N-1)

    That is why you apply it as `lut[old_conn + 1]`.
    """
    old_idx = np.flatnonzero(mask)  # 0..N-1
    new_idx = np.arange(old_idx.size, dtype=np.int64)

    # lut[0] reserved for old=-1
    lut = np.full(mask.size + 1, -1, dtype=np.int64)
    lut[old_idx + 1] = new_idx

    return lut


def _remap_conn_array(
    ds: xr.Dataset, conn_array: str, lut: np.ndarray
) -> tuple[tuple[str, str], np.ndarray]:
    dims_dict = {
        "cellsOnEdge": ("nEdges", "TWO"),
        "cellsOnVertex": ("nVertices", "vertexDegree"),
        "verticesOnCell": ("nCells", "maxEdges"),
        "verticesOnEdge": ("nEdges", "TWO"),
        "edgesOnVertex": ("nVertices", "vertexDegree"),
    }

    conn = ds[conn_array].values

    return (dims_dict[conn_array], lut[conn + 1])


def cull_mesh(
    ds_base: xr.Dataset, cells_to_cull: np.ndarray[bool]
) -> xr.Dataset:
    """
    Cull cells from a mesh using a cell mask

    The function removes cells based on the `cells_to_cull` mask, where a
    `True` value means the cell will be culled.

    Parameters
    ----------
    ds_base : :py:class:`~xarray.Dataset`
        Input mesh dataset, with zero-based connectivity arrays

    cells_to_cull : :py:class:`~numpy.ndarray`
        Mask of cells to be culled (True where cells should be culled).

    Returns
    -------
    :py:class:`~xarray.Dataset`
        A culled mesh dataset, where `indexToCellID`, `indexToEdgeID`,
        and `indexToVertexID` arrays have been added, which act as lookup
        tables for data array from the base (ie. unculled) mesh
    """
    cells_to_cull = np.asarray(cells_to_cull, dtype=bool)

    base_nCells = ds_base.sizes["nCells"]

    if cells_to_cull.ndim != 1 or cells_to_cull.size != base_nCells:
        msg = (
            "`cells_to_cull` must be a 1D boolean array of "
            " length ds_base.sizes['nCells']"
        )
        raise ValueError(msg)

    # TODO: add check to make sure `da_base` has been zero-indexed

    # invert mask so that cells to cull are dropped
    cell_mask = ~cells_to_cull

    # pad masks with a False, using `np.r_`, to avoid issues with a
    # connectivity value of `-1`, which means no neighbor NOT the last index
    cell_mask_p = np.r_[False, cell_mask]
    # keep edge is any adjacent cell is kept
    edge_mask = np.any(cell_mask_p[ds_base.cellsOnEdge + 1], axis=1)

    edge_mask_p = np.r_[False, edge_mask]
    # keep vertex is any adjacent vertex is kept
    vert_mask = np.any(edge_mask_p[ds_base.edgesOnVertex + 1], axis=1)

    # indices of kept location from the original mesh
    index_to_cell_id = np.flatnonzero(cell_mask)
    index_to_edge_id = np.flatnonzero(edge_mask)
    index_to_vert_id = np.flatnonzero(vert_mask)

    # lookup tables that map from base to culled indices
    cell_lut = _make_lookup_table(cell_mask)
    edge_lut = _make_lookup_table(edge_mask)
    vert_lut = _make_lookup_table(vert_mask)

    # downsample coordinate/connectivity arrays to culled indices
    ds_culled = ds_base.isel(
        nCells=index_to_cell_id,
        nEdges=index_to_edge_id,
        nVertices=index_to_vert_id,
    ).copy(deep=True)

    ds_culled["cellsOnEdge"] = _remap_conn_array(
        ds_culled, "cellsOnEdge", cell_lut
    )

    ds_culled["cellsOnVertex"] = _remap_conn_array(
        ds_culled, "cellsOnVertex", cell_lut
    )

    ds_culled["verticesOnCell"] = _remap_conn_array(
        ds_culled, "verticesOnCell", vert_lut
    )

    ds_culled["verticesOnEdge"] = _remap_conn_array(
        ds_culled, "verticesOnEdge", vert_lut
    )

    ds_culled["edgesOnVertex"] = _remap_conn_array(
        ds_culled, "edgesOnVertex", edge_lut
    )

    # add lookup tables for indexing into full sized mesh arrays
    ds_culled["indexToCellID"] = ("nCells", index_to_cell_id)
    ds_culled["indexToEdgeID"] = ("nEdges", index_to_edge_id)
    ds_culled["indexToVertexID"] = ("nVertices", index_to_vert_id)

    return ds_culled


def get_radius(projection: CRS) -> float:
    """ """
    x_range = projection.x_limits[1] - projection.x_limits[0]
    y_range = projection.y_limits[1] - projection.y_limits[0]

    return max(x_range, y_range)


def compute_cell_centroid(
    cell_patches: np.ndarray, verticesOnCell: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Centroid of cell patch using shoelace formula

    https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

    Parameters
    ----------
    cell_patches: np.ndarray
        cell patches to calculate the centroids of (nCells, maxEdges, 2)
    verticesOnCell: ArrayLike
        mesh connectivity array with ragged cells denoted by ``-1``

    Returns
    -------
    cx, cy: (np.ndarray, np.ndarray)
        Patch centroids where each array is of length nCells
    """
    mask = verticesOnCell != -1

    r, c = np.nonzero(~mask)
    bad = np.any(cell_patches[r, c, :] != cell_patches[r, 0, :], axis=1)
    if np.any(bad):
        msg = (
            f"Padded vertex does not equal first vertex. Check nCells ="
            f" {np.unique(r[bad])}"
        )
        raise ValueError(msg)

    x0 = cell_patches[..., 0]
    y0 = cell_patches[..., 1]
    x1 = np.roll(x0, -1, axis=1)
    y1 = np.roll(y0, -1, axis=1)

    # need to mask out ragged indices b/c centroid is a weighted sum
    cross = ((x0 * y1) - (x1 * y0)) * mask

    cx = ((x0 + x1) * cross).sum(axis=1) / (3.0 * cross.sum(axis=1))
    cy = ((y0 + y1) * cross).sum(axis=1) / (3.0 * cross.sum(axis=1))

    return cx, cy


def get_invalid_patches(patches: np.ndarray) -> None | np.ndarray:
    """Helper function to identify problematic patches.

    Returns the indices of the problematic patches as determined by
    :py:func:`shapely.is_valid`.

    Parameters
    ----------
    patches : :py:class:`~numpy.ndarray`
        The patch array to check

    Returns
    -------
    :py:class:`~numpy.ndarray` or None
        Indices of invalid patches. If no patches are invalid then returns None
    """

    # TODO: switch to Polygon, which less permissive than LineString
    # convert the patches to a list of shapely geometries
    geoms = [shapely.LineString(patch) for patch in patches]
    # check if the shapely geometries are valid
    valid = shapely.is_valid(geoms)

    if np.all(valid):
        # no invalid patches, so return None
        return None

    return np.flatnonzero(~valid)
