from __future__ import annotations

from functools import cached_property
from typing import Literal

import cartopy.crs as ccrs
import numpy as np
import shapely
import xarray as xr
from cartopy.crs import CRS
from numpy import ndarray
from numpy.typing import ArrayLike
from xarray.core.dataset import Dataset

import mosaic.utils

renaming_dict = {
    "lonCell": "xCell",
    "latCell": "yCell",
    "lonEdge": "xEdge",
    "latEdge": "yEdge",
    "lonVertex": "xVertex",
    "latVertex": "yVertex",
}

connectivity_arrays = [
    "cellsOnEdge",
    "cellsOnVertex",
    "verticesOnEdge",
    "verticesOnCell",
    "edgesOnVertex",
]

UNSUPPORTED_SPHERICAL_PROJECTIONS = (
    # EuroPP actually works is a subclass of the generic UTM projection, which
    # does not work, so EuroPP raises an error.
    ccrs.EuroPP,
    ccrs.UTM,
)


def attr_to_bool(attr: str):
    """Format attribute strings and return a boolean value"""
    match attr.strip().upper():
        case "YES":
            return True
        case "NO":
            return False
        case _:
            msg = "f{attr} was unable to be parsed as YES/NO"
            raise ValueError(msg)


class Descriptor:
    """Data structure describing unstructured MPAS meshes.

    Enables visualization of fields defined at cell centers, vertices,
    and edges within ``matplotlib`` through the creation of
    :py:class:`~matplotlib.collections.PolyCollection` objects. Separate patch
    arrays, which are subsequently passed to the
    :py:class:`~matplotlib.collections.PolyCollection` class by the
    :py:class:`polypcolor` method,
    are lazily loaded for each variable location (i.e. cells, edges, and
    vertices) when attribute is first looked up. We use lazily loaded arrays
    because we are rarely plotting variables located at all three locations
    and patch array creation is the most expensive
    process in our visualization procedure.

    The resulting patch arrays properly handle culled mesh boundaries (i.e.
    culled land boundaries for spherical meshes and/or mesh boundary for planar
    non-periodic meshes). Additionally, x and/or y periodicity is properly
    handled for planar periodic mesh and we can correct for patch wrapping over
    the antimeridian for projected spherical meshes.

    Parameters
    ----------
    mesh_ds : DataSet
        A valid MPAS mesh dataset that contains the basic mesh variables (i.e.
        coordinate and connectivity arrays) needed for creating patch arrays.
    projection : cartopy.crs.Projection, optional
        The target projection for plotting.
    transform : cartopy.crs.Projection, optional
        The coordinate system in which the parent mesh coordinates are defined.
    use_latlon : bool, optional
        Whether to use the lat/lon coordinate arrays to construct the patches.

    Notes
    -----
    If both the ``projection`` and ``transform`` parameters are passed, then
    the coordinate arrays in the :attr:`.Descriptor.ds` will be transformed
    prior to patch construction. We do this as matter of efficiency, since
    transforming the one-dimensional coordinate arrays is much faster than
    transforming the multi-dimensional patch arrays. This means the
    :attr:`.Descriptor.transform` will **not** be the values passed to the
    constructor, but instead will be equal to :attr:`.Descriptor.projection`

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import mosaic
    >>>
    >>> ds = mosaic.datasets.open_dataset("QU.240km")
    >>>
    >>> transform = ccrs.PlateCarree()
    >>> projection = ccrs.NorthPolarStereo()
    >>>
    >>> # set `use_latlon` to True b/c our transform expects lat/lon coords
    >>> descriptor = mosaic.Descriptor(
    >>>     ds, projection, transform, use_latlon=True
    >>> )
    """

    def __init__(
        self,
        mesh_ds: Dataset,
        projection: CRS | None = None,
        transform: CRS | None = None,
        use_latlon: bool = False,
    ) -> None:
        #: Boolean whether parent mesh is spherical
        self.is_spherical = attr_to_bool(mesh_ds.on_a_sphere)

        if not self.is_spherical:
            is_periodic = attr_to_bool(mesh_ds.is_periodic)
        else:
            is_periodic = False

        #: Boolean whether parent mesh is (planar) periodic in at least one dim
        self.is_periodic = is_periodic

        # calls attribute setter method
        self.latlon = use_latlon

        # store original mesh dimension sizes
        self.sizes = mesh_ds

        #: :py:class:`~xarray.Dataset` that contains the minimal subset of
        #: coordinate and connectivity arrays from the parent mesh needed to
        #: create patches arrays.
        self.ds = self._create_minimal_dataset(mesh_ds)

        #: The coordinate system in which patch coordinates are defined.
        #:
        #: This *could* have a different value than ``transform`` parameter
        #: passed to the constructor, because we transform the one-dimensional
        #: coordinate arrays at initialization, if both the ``transform`` and
        #: ``projection`` kwargs are provided.
        self.transform = transform

        # calls ``projection.setter`` method, which will transform coordinates
        # if both a projection and transform were provided to the constructor
        self.projection = projection

        # method ensures is periodic, avoiding AttributeErrors if non-periodic
        self.x_period = self._parse_period(mesh_ds, "x")
        self.y_period = self._parse_period(mesh_ds, "y")

    def _create_minimal_dataset(self, ds: Dataset) -> Dataset:
        """
        Create a xarray.Dataset that contains the minimal subset of
        coordinate / connectivity arrays needed to create patches for plotting

        Parameters
        ----------
        ds : DataArray
            A valid MPAS mesh dataset
        """

        def fix_outofbounds_indices(ds: Dataset, array_name: str) -> Dataset:
            """
            Some meshes (e.g. QU240km) don't do masking of ragged indices
            with 0. Instead they use `nInidices+1` as the invalid value,
            which can produce out of bounds errors. Check if that the case,
            and if so set the out of bounds indices to (-1)

            NOTE: Assumes connectivity arrays are zero indexed
            """
            # programmatically create the appropriate dimension name
            dim = "n" + array_name.split("On")[0].title()
            # get the maximum valid size for the array to be indexed too
            maxSize = ds.sizes[dim]
            # get mask of where index is out bounds
            mask = ds[array_name] == maxSize
            # where index is out of bounds, set to invalid (i.e. -1)
            ds[array_name] = xr.where(mask, -1, ds[array_name])

            return ds

        if self.latlon:
            coordinate_arrays = list(renaming_dict.keys())
        else:
            coordinate_arrays = list(renaming_dict.values())

        # list of coordinate / connectivity arrays needed to create patches
        mesh_arrays = coordinate_arrays + connectivity_arrays

        # get subset of arrays from mesh and load into memory if dask arrays
        minimal_ds = ds[mesh_arrays].load()

        # delete the attributes in the minimal dataset to avoid confusion
        minimal_ds.attrs.clear()

        for array in connectivity_arrays:
            # zero index all the connectivity arrays
            minimal_ds[array] = minimal_ds[array] - 1
            # fix any out of bounds indices
            minimal_ds = fix_outofbounds_indices(minimal_ds, array)

        if self.latlon:
            # convert lat/lon coordinates from radian to degrees
            for loc in ["Cell", "Edge", "Vertex"]:
                # shift lon from [0, 360) to [-180, 180)
                minimal_ds[f"lon{loc}"] = (
                    (np.rad2deg(minimal_ds[f"lon{loc}"]) + 180.0) % 360
                ) - 180.0
                # lat is already [-90, 90] so no shift is needed
                minimal_ds[f"lat{loc}"] = np.rad2deg(minimal_ds[f"lat{loc}"])

            # rename the coordinate arrays to be named x.../y... irrespective
            # of whether spherical or Cartesian coordinates are used
            minimal_ds = minimal_ds.rename(renaming_dict)

        return minimal_ds

    @property
    def transform(self) -> CRS | None:
        """The coordinate system in which patch coordinates are defined."""
        return self._transform

    @transform.setter
    def transform(self, transform: CRS | None) -> None:
        if (self.is_spherical and transform is None) or (
            self.latlon and transform is None
        ):
            self._transform = ccrs.Geodetic()

        else:
            self._transform = transform

    @property
    def projection(self) -> CRS:
        """The target projection for plotting."""
        return self._projection

    @projection.setter
    def projection(self, projection: CRS) -> None:
        # small subset of regional projs are not supported for spherical meshes
        if self.is_spherical and isinstance(
            projection, UNSUPPORTED_SPHERICAL_PROJECTIONS
        ):
            msg = (
                f"Invalid projection: {type(projection).__name__} is not "
                f"supported for spherical meshes"
            )
            raise ValueError(msg)

        # b/c cells with vertices outside projection boundary are culled
        if hasattr(self, "_projection"):
            msg = "Reprojecting a descriptor is not supported"
            raise AttributeError(msg)

        self._projection = projection

        if self._projection is None:
            return

        if self.transform is None:
            msg = (
                f"Must specify a transform, in order to reproject"
                f"mesh coordinates to {type(projection).__name__}"
            )
            raise ValueError(msg)

        # blindly reproject coordinate arrays in the minimal dataset
        self._transform_coordinates(projection, self.transform)

        # update the transform attribute following the reprojection
        self.transform = projection

        # compute mask of cells that need to be culled
        cull_mask = _compute_cull_mask(self.ds, self.projection)
        self.ds = mosaic.utils.cull_mesh(self.ds, cull_mask)

    @property
    def sizes(self) -> dict[str, int]:
        """
        :py:class:`dict` of dimension (``nCells``, ``nEdges``, and
        ``nVertices``) sizes
        """
        return self._sizes

    @sizes.setter
    def sizes(self, ds) -> None:
        """ """
        self._sizes = {
            dim: ds.sizes[dim] for dim in ["nCells", "nEdges", "nVertices"]
        }

    @property
    def latlon(self) -> bool:
        """
        Boolean whether the lat/lon coordinate arrays should be used for
        patch construction.
        """
        return self._latlon

    @latlon.setter
    def latlon(self, value) -> None:
        """TODO: check that the passed value is consistent with transform"""
        if self.is_spherical:
            value = True

        self._latlon = value

    def _parse_period(self, ds, dim: Literal["x", "y"]):
        """Parse period attribute, return None for non-periodic meshes"""

        attr = f"{dim}_period"
        if attr in ds.attrs and ds.attrs[attr] is not None:
            period = float(ds.attrs[attr])
        else:
            period = None

        # in the off chance mesh is periodic but does not have period attribute
        if self.is_periodic and attr not in ds.attrs:
            msg = (
                f'Mesh file: "{ds.encoding["source"]}" does not have '
                f"attribute `{attr}` despite being a planar periodic mesh."
            )
            raise AttributeError(msg)
        if period == 0.0:
            return None
        return period

    @property
    def x_period(self) -> float | None:
        """Period along x-dimension, is ``None`` for non-periodic meshes"""
        return self._x_period

    @x_period.setter
    def x_period(self, value) -> None:
        # needed to avoid AttributeError for non-periodic meshes
        if not (self.is_periodic and self.is_spherical):
            self._x_period = None
        if not self.is_periodic and self.is_spherical and self.projection:
            x_limits = self.projection.x_limits
            self._x_period = np.abs(x_limits[1] - x_limits[0])
        else:
            self._x_period = value

    @property
    def y_period(self) -> float | None:
        """Period along y-dimension, is ``None`` for non-periodic meshes"""
        return self._y_period

    @y_period.setter
    def y_period(self, value) -> None:
        # needed to avoid AttributeError for non-periodic meshes
        if not (self.is_periodic and self.is_spherical):
            self._y_period = None
        if not self.is_periodic and self.is_spherical and self.projection:
            y_limits = self.projection.y_limits
            self._y_period = np.abs(y_limits[1] - y_limits[0])
        else:
            self._y_period = value

    @property
    def origin(self) -> tuple[float, float]:
        """Coordinates of bottom left corner of plot"""

        def get_axis_min(self, coord: Literal["x", "y"]) -> float:
            """ """
            period = self.__getattribute__(f"{coord}_period")
            edge_min = float(self.ds[f"{coord}Edge"].min())
            vertex_min = float(self.ds[f"{coord}Vertex"].min())

            # an edge connects two vertices, so a vertices most extreme
            # position should always be more extended than an edge's
            if (period is not None) and (vertex_min > edge_min):
                max = float(self.ds[f"{coord}Vertex"].max())
                min = max - period
            else:
                min = float(self.ds[f"{coord}Vertex"].min())

            return min

        xmin = get_axis_min(self, "x")
        ymin = get_axis_min(self, "y")

        return (xmin, ymin)

    @cached_property
    def cell_patches(self) -> ndarray:
        """:py:class:`~numpy.ndarray` of patch coordinates for cell centered
        values

        **Dimensions**: ``(nCells, maxEdges)``

        Notes
        -----
        The second dimension of the patch array is length ``maxEdges``, even if
        ``nEdgesOnCell`` for the ``i-th`` cell is less than ``maxEdges``. We've
        chosen to deal with ragged indices (i.e. nodes indices greater than
        ``nEdgesOnCell`` for the given cell) by repeating the first node of the
        patch. Nodes are ordered counter clockwise around the cell center.
        """
        patches = _compute_cell_patches(self.ds)

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            patches = self._wrap_patches(patches, "Cell")
            mirrored, mirrored_idxs = self._mirror_patches(patches)

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._cell_mirrored = mirrored
                self._cell_mirrored_idxs = mirrored_idxs

        return patches

    @cached_property
    def edge_patches(self) -> ndarray:
        """:py:class:`~numpy.ndarray` of patch coordinates for edge centered
        values

        **Dimensions**: ``(nEdges, 4)``

        Notes
        -----
        Edge patches have four nodes which typically correspond to the two cell
        centers of the ``cellsOnEdge`` and the two vertices which make up the
        edge. For an edge patch along a culled mesh boundary one of the cell
        centers usually used to construct the patch will be missing, so the
        corresponding node will be collapsed to the edge coordinate.
        """
        patches = _compute_edge_patches(self.ds)

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            patches = self._wrap_patches(patches, "Edge")
            mirrored, mirrored_idxs = self._mirror_patches(patches)

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._edge_mirrored = mirrored
                self._edge_mirrored_idxs = mirrored_idxs

        return patches

    @cached_property
    def vertex_patches(self) -> ndarray:
        """:py:class:`~numpy.ndarray` of patch coordinates for vertex centered
        values

        **Dimensions**: ``(nVertices, 6)``

        Notes
        -----
        Vertex patches have 6 nodes despite the typical dual cell only having
        three nodes (i.e. the cell centers of three cells on the vertex) in
        order to simplify vertex patches creation along culled mesh boundaries.
        The typical vertex patch will be comprised of the edges and cell
        centers of the ``cellsOnVertex``. As the MPAS Mesh Specs
        (version 1.0: Section 5.3) outlines:
        "Edges lead cells as they move around vertex". So, the first node
        in a vertex patch will correspond to an edge (if present).

        For patches along culled boundaries, if an edge and/or cell center is
        missing the corresponding node will be collapsed to the patches vertex
        position.
        """
        patches = _compute_vertex_patches(self.ds)

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            patches = self._wrap_patches(patches, "Vertex")
            mirrored, mirrored_idxs = self._mirror_patches(patches)

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._vertex_mirrored = mirrored
                self._vertex_mirrored_idxs = mirrored_idxs

        return patches

    def _transform_coordinates(self, projection, transform):
        """Blindly transform coordinate arrays"""

        for loc in ["Cell", "Edge", "Vertex"]:
            transformed_coords = projection.transform_points(
                transform, self.ds[f"x{loc}"], self.ds[f"y{loc}"]
            )

            # assign to .data attr b/c RHS is numpy array
            self.ds[f"x{loc}"].data = transformed_coords[:, 0]
            self.ds[f"y{loc}"].data = transformed_coords[:, 1]

    def _wrap_patches(self, patches, loc):
        """Wrap patches for spherical and planar-periodic meshes"""

        def _find_boundary_patches(patches, loc, coord, period):
            """
            Find the patches that cross the periodic boundary and what
            direction they cross the boundary (i.e. their ``sign``). This
            method assumes the patch centroids are not periodic
            """
            # get axis index we are inquiring over
            axis = 0 if coord == "x" else 1
            # get requested coordinate of patch centroids
            center = self.ds[f"{coord}{loc.title()}"].values.reshape(-1, 1)
            # get difference b/w centroid and nodes of patches
            diff = patches[..., axis] - center

            mask = np.abs(diff) > np.abs(period) / (2.0 * np.sqrt(2.0))
            sign = np.sign(diff)

            return mask, sign

        def _wrap_1D(patches, mask, sign, axis, period):
            """Correct patch periodicity along a single dimension"""
            patches[..., axis][mask] -= np.sign(sign[mask]) * period

            # TODO: clip spherical wrapped patches to projection limits
            return patches

        # Stereographic projections do not need wrapping, so exit early
        if isinstance(self.projection, ccrs.Stereographic):
            return patches

        if self.x_period:
            # find the patches that are periodic in x-direction
            x_mask, x_sign = _find_boundary_patches(
                patches, loc, "x", self.x_period
            )

            if np.any(x_mask):
                # using the sign of the difference correct patches x coordinate
                patches = _wrap_1D(patches, x_mask, x_sign, 0, self.x_period)

        if self.y_period:
            # find the patches that are periodic in y-direction
            y_mask, y_sign = _find_boundary_patches(
                patches, loc, "y", self.y_period
            )

            if np.any(y_mask):
                # using the sign of the difference correct patches y coordinate
                patches = _wrap_1D(patches, y_mask, y_sign, 1, self.y_period)

        return patches

    def _mirror_patches(self, patches):
        """Mirror patches across periodic boundary for planar-periodic meshes

        Instead of correcting the periodic nodes of a patch (i.e. like
        ``_wrap_patches`` above), this method treats **all** nodes of a patch
        equivalently. Therefore, it assumes the patches have already had their
        periodicity corrected
        """

        idx_list = []
        mirrored_list = []

        def check_signs(array):
            """Get the sign along an axis (i.e. a single sign per patch)

            If patch has more than one sign will raise an error
            """
            if np.all(array == 0):
                return np.array([0])
            return np.unique(array[array != 0])

        def _find_mirrored(patches, coord, period):
            """Find the patches that need to be mirrored across periodic axis

            Also return the direction the patch cross the boundary (i.e.
            their ``sign``). This method assumes each patch has a unique sign.
            """
            # get axis index we are inquiring over
            axis = 0 if coord == "x" else 1

            # get the minimum for a given axis
            min = self.origin[axis]
            # subtract axis origin to get int number of periods (i.e. -1, 0, 1)
            n_periods = ((patches[..., axis] - min) // period).astype(int)

            # get the sign along axis
            try:
                mirror_sign = np.apply_along_axis(check_signs, 1, n_periods)
            except ValueError as exc:
                error_str = (
                    "A patch is periodic across multiple periods "
                    "(i.e. a patch has a non-unqiue sign). "
                    "Therefore patches cannot bed mirrored"
                )
                raise ValueError(error_str) from exc

            # make a 1D array so broadcasting below will work
            mirror_sign = mirror_sign.squeeze()
            # convert sign into boolean mask
            mirror_mask = mirror_sign.astype(bool)

            return mirror_mask, mirror_sign

        def _mirror_1D(patches, mask, sign, coord, period):
            """Duplicate and mirror patches across a periodic axis

            Also returns the indices of the mirrored patches.
            """
            # get axis index we are inquiring over
            axis = 0 if coord == "x" else 1

            # get subset of patches to be mirrored
            mirrored = patches[mask]
            # correct coordinate so that patches are mirror across axis
            mirrored[..., axis] -= sign[mask, np.newaxis] * period
            # return the indices of the mirror patches for plotting
            idx = np.where(mask)[0]

            return mirrored, idx

        if self.x_period:
            # find the patches that need to be mirrored in x-direction
            x_mask, x_sign = _find_mirrored(patches, "x", self.x_period)

            if np.any(x_mask):
                # using the sign of the difference correct patches x coordinate
                x_mirrored, x_idxs = _mirror_1D(
                    patches, x_mask, x_sign, "x", self.x_period
                )

                # add values to list to concatenated
                idx_list.append(x_idxs)
                mirrored_list.append(x_mirrored)

        if self.y_period:
            # find the patches that need to be mirrored in y-direction
            y_mask, y_sign = _find_mirrored(patches, "y", self.y_period)

            if np.any(y_mask):
                # using the sign of the difference correct patches y coordinate
                y_mirrored, y_idxs = _mirror_1D(
                    patches, y_mask, y_sign, "y", self.y_period
                )

                # add values to list to concatenated
                idx_list.append(y_idxs)
                mirrored_list.append(y_mirrored)

        # doubly periodic mesh
        if self.x_period and self.y_period:
            # find the (single) doubly periodic index
            both_idx = x_idxs[np.isin(x_idxs, y_idxs)]

            if both_idx.size > 0:
                both_x = x_mirrored[np.isin(x_idxs, both_idx), :, 0]
                both_y = y_mirrored[np.isin(y_idxs, both_idx), :, 1]
                both_mirrored = np.dstack([both_x, both_y])

                idx_list.append(both_idx)
                mirrored_list.append(both_mirrored)

        if mirrored_list:
            idxs = np.concat(idx_list)
            mirrored = np.vstack(mirrored_list)
        else:
            idxs = None
            mirrored = None

        return mirrored, idxs

    def _get_array_location(self, array: ArrayLike) -> tuple[str, ArrayLike]:
        """Helper function to find mesh location where array is defined

        If descriptor has been culled, also subset array to culled mesh size.
        """

        dimension_to_location = {
            "nCells": "cell",
            "nEdges": "edge",
            "nVertices": "vertex",
        }

        array = np.asarray(array)

        if array.ndim != 1:
            msg = f"Array should be 1-D, instead has {array.ndim} dims"
            if sum([size > 1 for size in array.shape]) > 1:
                # more than one dimensions is greater than length one
                raise ValueError(msg)
            array = array.squeeze()

        # dict of dim lengths of the original dataset
        origin_rev = {v: k for k, v in self.sizes.items()}
        # dict of dim lengths of the culled dataset
        culled_rev = {self.ds.sizes[d]: d for d in self.sizes}

        # start by trying to find match in original dataset
        dim = origin_rev.get(array.size)
        if dim is not None:
            loc = dimension_to_location[dim]
            array_name = f"indexTo{loc.capitalize()}ID"
            if array_name in self.ds:
                lut = self.ds[array_name]
                return loc, array[lut]
            return loc, array

        # fallback to checking culled dataset
        dim = culled_rev.get(array.size)
        if dim is not None:
            # no need to look up table with culled mesh array
            return dimension_to_location[dim], array

        msg = (
            f"Size of array {array.size} is incompatible with mesh dimensions "
            f"{self.sizes}"
        )
        raise ValueError(msg)


def _compute_cell_patches(ds: Dataset) -> ndarray:
    """Create cell patches (i.e. Primary cells) for an MPAS mesh."""
    # get the maximum number of edges on a cell
    maxEdges = ds.sizes["maxEdges"]
    # connectivity arrays have already been zero indexed
    verticesOnCell = ds.verticesOnCell
    # get a mask of the active vertices
    mask = verticesOnCell == -1

    # tile the first vertices index
    firstVertex = np.tile(verticesOnCell[:, 0], (maxEdges, 1)).T
    # set masked vertices to the first vertex of the cell
    verticesOnCell = np.where(mask, firstVertex, verticesOnCell)

    # reshape/expand the vertices coordinate arrays
    x_nodes = ds.xVertex.values[verticesOnCell]
    y_nodes = ds.yVertex.values[verticesOnCell]

    return np.stack((x_nodes, y_nodes), axis=-1)


def _compute_edge_patches(ds: Dataset) -> ndarray:
    """Create edge patches for an MPAS mesh."""

    # connectivity arrays have already been zero indexed
    cellsOnEdge = ds.cellsOnEdge
    verticesOnEdge = ds.verticesOnEdge
    # condition should only be true once per row or else wouldn't be an edge
    cellMask = cellsOnEdge < 0

    # get subset of cell coordinate arrays corresponding to edge patches
    xCell = ds.xCell.values[cellsOnEdge]
    yCell = ds.yCell.values[cellsOnEdge]
    # get subset of vertex coordinate arrays corresponding to edge patches
    xVertex = ds.xVertex.values[verticesOnEdge]
    yVertex = ds.yVertex.values[verticesOnEdge]

    # if only one cell on edge (i.e. along a culled boundary), then collapse
    # the node corresponding to the missing cell back the edge location
    if np.any(cellMask):
        xCell = np.where(cellMask, ds.xEdge.values[:, np.newaxis], xCell)
        yCell = np.where(cellMask, ds.yEdge.values[:, np.newaxis], yCell)

    x_nodes = np.stack(
        (xCell[:, 0], xVertex[:, 0], xCell[:, 1], xVertex[:, 1]), axis=-1
    )

    y_nodes = np.stack(
        (yCell[:, 0], yVertex[:, 0], yCell[:, 1], yVertex[:, 1]), axis=-1
    )

    return np.stack((x_nodes, y_nodes), axis=-1)


def _compute_vertex_patches(ds: Dataset) -> ndarray:
    """Create vertex patches (i.e. Dual Cells) for an MPAS mesh."""
    nVertices = ds.sizes["nVertices"]
    vertexDegree = ds.sizes["vertexDegree"]

    nodes = np.zeros((nVertices, vertexDegree * 2, 2))
    # connectivity arrays have already been zero indexed
    cellsOnVertex = ds.cellsOnVertex.values
    edgesOnVertex = ds.edgesOnVertex.values
    # get a mask of active nodes
    cellMask = cellsOnVertex == -1
    edgeMask = edgesOnVertex == -1

    # get the coordinates needed to patch construction
    xCell = ds.xCell.values
    yCell = ds.yCell.values
    xEdge = ds.xEdge.values
    yEdge = ds.yEdge.values
    # convert vertex coordinates to column vectors for broadcasting below
    xVertex = ds.xVertex.values[:, np.newaxis]
    yVertex = ds.yVertex.values[:, np.newaxis]

    # if edge is missing collapse edge node to vertex, else leave at edge
    nodes[:, ::2, 0] = np.where(edgeMask, xVertex, xEdge[edgesOnVertex])
    nodes[:, ::2, 1] = np.where(edgeMask, yVertex, yEdge[edgesOnVertex])

    # if cell is missing collapse cell node to vertex, else leave at cell
    nodes[:, 1::2, 0] = np.where(cellMask, xVertex, xCell[cellsOnVertex])
    nodes[:, 1::2, 1] = np.where(cellMask, yVertex, yCell[cellsOnVertex])

    # -------------------------------------------------------------------------
    # NOTE: The condition below will only be true for meshes run through the
    #       MPAS mesh converter after culling. A bug in the converter alters
    #       the ordering of edges, causing problems for vertex patches
    #
    # If final cell and edge nodes are missing collapse both back to first edge
    # Ensures patches encompasses the full kite area and are properly closed.
    # -------------------------------------------------------------------------
    condition = (cellMask & edgeMask)[:, -1:]
    nodes[:, 4:, 0] = np.where(condition, nodes[:, 0:1, 0], nodes[:, 4:, 0])
    nodes[:, 4:, 1] = np.where(condition, nodes[:, 0:1, 1], nodes[:, 4:, 1])

    return nodes


def _compute_cull_mask(ds: xr.Dataset, projection: CRS) -> ndarray[bool]:
    """
    Calculate boolean mask of cells to be culled for a given projection

    cells are culled if any of the following conditions are met:
        - all vertices on cell are outside the projection domain
        - any vertices on cell is nan
        - distance between projected cell center and the patch centroid is too
          large (i.e. due to patch wrapping over the projection boundary)

    Parameters
    ----------
    ds: xr.Dataset
        A zero indexed dataset where coordinate arrays have already been
        transformed to the projection coordinate system

    projection: cartopy.crs.Projection
        Target projection for plotting

    Returns
    -------
    cull_mask: ndarray[bool]
        Boolean mask of cells to be culled for plotting
    """

    # get and prepare projection domain. prepare returns None, so do not assign
    ext_domain = projection.domain
    shapely.prepare(ext_domain)

    cell_patches = _compute_cell_patches(ds)

    # start with no cells to be culled
    cull_mask = np.zeros(ds.sizes["nCells"], dtype=bool)

    xVertex = ds.xVertex.values
    yVertex = ds.yVertex.values

    # connectivity arrays have already been zero indexed
    verticesOnCell = ds.verticesOnCell

    # padded mask of vertices within the projection boundary
    vertex_contained = np.r_[
        False, shapely.contains_xy(ext_domain, xVertex, yVertex)
    ]

    # translate vertex mask to cell mask where at least one vertex is contained
    cell_mask = np.any(vertex_contained[verticesOnCell + 1], axis=1)

    # mask of cells with any nan vertices
    nan_mask = np.any(np.isnan(cell_patches), axis=(1, 2))

    # only check centroids of cells not being culled
    x_cell = ds.xCell.values[cell_mask & ~nan_mask]
    y_cell = ds.yCell.values[cell_mask & ~nan_mask]

    # calculate patch centroid, to be compared to projected cell center
    # TODO: filter our self intersecting polygons
    x_cent, y_cent = mosaic.utils.compute_cell_centroid(
        cell_patches[cell_mask & ~nan_mask],
        verticesOnCell[cell_mask & ~nan_mask],
    )

    # TODO: can we use projection.threshold instead?
    thresh = mosaic.utils.get_radius(projection) * 1.5e-2

    # mask where cell centroids and projected cell center are too far apart
    centroid_mask = np.hypot(x_cell - x_cent, y_cell - y_cent) > thresh

    # cull cells that are nan OR with no vertices within projection domain
    cull_mask |= nan_mask | ~cell_mask
    # centroid mask was only calculated for cells not being culled
    cull_mask[cell_mask & ~nan_mask] |= centroid_mask

    # TODO: add option to return component mask for debugging
    return cull_mask
