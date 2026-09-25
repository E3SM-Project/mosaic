from __future__ import annotations

from typing import Literal

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from xarray.core.dataset import Dataset

point_types = Literal["Cell", "Edge", "Vertex"]


def resolve_coords(
    ds: Dataset,
    connectivity: ArrayLike,
    source: point_types,
    fallback: point_types | Literal["First", "Ignore"] | None = None,
) -> tuple[ndarray, ndarray]:
    """Resolve (x, y) coords for connectivity array with prescribed fallback.

    Fallback needed for handling culled boundaries and polygon with variable
    number of faces.

    Parameters
    ----------
    ds : Dataset
        Zero-indexed MPAS mesh dataset.
    connectivity : ArrayLike
        Array of indices pointing to source elements.
    source : {"Cell", "Edge", "Vertex"}
        Location to extract primary coordinates from.
    fallback : {"Cell", "Edge", "Vertex", "First", "Ignore"}, optional
        Location to collapse missing (< 0) indices to.
        "First" will repeat first node; ensuring polygon is closed.
        "Ignore" prevents an error being raised when connectivity has ragged
        entries, but no fallback location is provided. Use with caution.

    Returns
    -------
    tuple[ndarray, ndarray]
        Resolved (x, y) coordinates with shape matching connectivity.
    """
    conn = np.asarray(connectivity)
    mask = conn < 0

    # conn may contain negative sentinel values (-1 or -2)
    # will be masked out later using the fallback values
    x_out = np.asarray(ds[f"x{source}"])[conn]
    y_out = np.asarray(ds[f"y{source}"])[conn]

    if not np.any(mask):
        return x_out, y_out

    if fallback is None:
        err_msg = (
            f"{np.count_nonzero(mask)} ragged indices in connectivity, but no "
            "fallback given. Provide a fallback location or 'Ignore'"
        )
        raise ValueError(err_msg)

    if fallback == "Ignore":
        return x_out, y_out

    # Handle ragged padding or collapse to boundary fallback location
    if fallback == "First":
        x_fb, y_fb = x_out[:, 0:1], y_out[:, 0:1]
    else:
        x_fb = np.asarray(ds[f"x{fallback}"])[..., np.newaxis]
        y_fb = np.asarray(ds[f"y{fallback}"])[..., np.newaxis]

    return np.where(mask, x_fb, x_out), np.where(mask, y_fb, y_out)


def compute_cell_patches(ds: Dataset) -> ndarray:
    """Create cell patches (i.e. Primary cells) for an MPAS mesh.

    Ragged indices (cells with fewer than ``maxEdges`` nodes) are padded by
    repeating the first node of the cell.

    Parameters
    ----------
    ds : Dataset
        Zero-indexed MPAS mesh dataset.

    Returns
    -------
    ndarray
        Cell patch coords with shape ``(nCells, maxEdges, 2)``.
    """
    # Collapse ragged padding to first vertex of the cell
    x, y = resolve_coords(
        ds, ds.verticesOnCell, source="Vertex", fallback="First"
    )
    return np.stack((x, y), axis=-1)


def compute_edge_patches(ds: Dataset) -> ndarray:
    """Create edge patches for an MPAS mesh.

    Edge patches have four nodes which typically correspond to the two cell
    centers of ``cellsOnEdge`` and the two vertices of ``verticesOnEdge``.
    For an edge patch along a culled mesh boundary, one of the cell centers is
    missing, so the corresponding node is collapsed to the edge coordinate.

    Parameters
    ----------
    ds : Dataset
        Zero-indexed MPAS mesh dataset.

    Returns
    -------
    ndarray
        Edge patch coords with shape ``(nEdges, 4, 2)``.
    """
    # If only one cell on edge (culled boundary), collapse missing cell to edge
    x_cell, y_cell = resolve_coords(
        ds, ds.cellsOnEdge, source="Cell", fallback="Edge"
    )
    x_vert, y_vert = resolve_coords(
        ds, ds.verticesOnEdge, source="Vertex", fallback=None
    )

    # Alternate between cell centers and vertices
    x = np.stack(
        (x_cell[:, 0], x_vert[:, 0], x_cell[:, 1], x_vert[:, 1]), axis=-1
    )
    y = np.stack(
        (y_cell[:, 0], y_vert[:, 0], y_cell[:, 1], y_vert[:, 1]), axis=-1
    )

    return np.stack((x, y), axis=-1)


def compute_vertex_patches(ds: Dataset) -> ndarray:
    """Create vertex patches (i.e. Dual Cells) for an MPAS mesh.

    Vertex patches have 6 nodes despite the typical dual cell only having
    three nodes (the cell centers of three cells on the vertex) in order to
    simplify patch creation along culled boundaries. Alternates edges and cell
    centers of ``cellsOnVertex``. MPAS Mesh Spec (v1.0 Sec 5.3): "Edges lead
    cells as they move around vertex".

    Along culled boundaries, missing edge or cell center nodes collapse to the
    patch's vertex position.

    Parameters
    ----------
    ds : Dataset
        Zero-indexed MPAS mesh dataset.

    Returns
    -------
    ndarray
        Vertex patch coords with shape ``(nVertices, vertexDegree * 2, 2)``.
    """
    n_vert = ds.sizes["nVertices"]
    vert_deg = ds.sizes["vertexDegree"]

    # 1. Resolve edge and cell coords (collapsing missing nodes to vertex)
    x_edge, y_edge = resolve_coords(
        ds, ds.edgesOnVertex, source="Edge", fallback="Vertex"
    )
    x_cell, y_cell = resolve_coords(
        ds, ds.cellsOnVertex, source="Cell", fallback="Vertex"
    )

    # 2. Interleave: edges lead cells counter-clockwise around vertex
    nodes = np.zeros((n_vert, vert_deg * 2, 2))
    nodes[:, ::2, 0], nodes[:, ::2, 1] = x_edge, y_edge
    nodes[:, 1::2, 0], nodes[:, 1::2, 1] = x_cell, y_cell

    # -------------------------------------------------------------------------
    # NOTE: The condition below will only be true for meshes run through the
    #       MPAS mesh converter after culling. A bug in the converter alters
    #       the ordering of edges, causing problems for vertex patches.
    #
    # If final cell and edge nodes are missing, collapse both back to first
    # edge. Ensures patches cover full kite area and are properly closed.
    # -------------------------------------------------------------------------
    missing_tail = np.asarray((ds.cellsOnVertex < 0) & (ds.edgesOnVertex < 0))[
        :, -1:
    ]
    nodes[:, 4:, :] = np.where(
        missing_tail[..., np.newaxis], nodes[:, 0:1, :], nodes[:, 4:, :]
    )

    return nodes
