from functools import cached_property
from typing import Literal, Tuple

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from cartopy.crs import CRS
from numpy import ndarray
from xarray.core.dataset import Dataset

renaming_dict = {"lonCell": "xCell",
                 "latCell": "yCell",
                 "lonEdge": "xEdge",
                 "latEdge": "yEdge",
                 "lonVertex": "xVertex",
                 "latVertex": "yVertex"}

connectivity_arrays = ["cellsOnEdge",
                       "cellsOnVertex",
                       "verticesOnEdge",
                       "verticesOnCell",
                       "edgesOnVertex"]

SUPPORTED_SPHERICAL_PROJECTIONS = (ccrs._RectangularProjection,
                                   ccrs._WarpedRectangularProjection,
                                   ccrs.Stereographic,
                                   ccrs.Mercator,
                                   ccrs._CylindricalProjection,
                                   ccrs.InterruptedGoodeHomolosine)


def attr_to_bool(attr: str):
    """ Format attribute strings and return a boolean value """
    match attr.strip().upper():
        case "YES":
            return True
        case "NO":
            return False
        case _:
            raise ValueError("f{attr} was unable to be parsed as YES/NO")


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
        use_latlon: bool = False
    ) -> None:
        #: The coordinate system in which patch coordinates are defined.
        #:
        #: This *could* have a different value than ``transform`` parameter
        #: passed to the constructor, because we transform the one-dimensional
        #: coordinate arrays at initialization, if both the ``transform`` and
        #: ``projection`` kwargs are provided.
        self.transform = transform

        #: Boolean whether parent mesh is (planar) periodic in at least one dim
        self.is_periodic = attr_to_bool(mesh_ds.is_periodic)
        #: Boolean whether parent mesh is spherical
        self.is_spherical = attr_to_bool(mesh_ds.on_a_sphere)

        # calls attribute setter method
        self.latlon = use_latlon

        #: :py:class:`~xarray.Dataset` that contains the minimal subset of
        #: coordinate and connectivity arrays from the parent mesh needed to
        #: create patches arrays.
        self.ds = self._create_minimal_dataset(mesh_ds)

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
                minimal_ds[f"lon{loc}"] = np.rad2deg(minimal_ds[f"lon{loc}"])
                minimal_ds[f"lat{loc}"] = np.rad2deg(minimal_ds[f"lat{loc}"])

            # rename the coordinate arrays to be named x.../y... irrespective
            # of whether spherical or Cartesian coordinates are used
            minimal_ds = minimal_ds.rename(renaming_dict)

        return minimal_ds

    @property
    def projection(self) -> CRS:
        """ The target projection for plotting. """
        return self._projection

    @projection.setter
    def projection(self, projection: CRS) -> None:
        # We don't support all map projections for spherical meshes, yet...
        if (projection is not None and self.is_spherical and
                not isinstance(projection, SUPPORTED_SPHERICAL_PROJECTIONS)):

            raise ValueError(f"Invalid projection: {type(projection).__name__}"
                             f" is not supported - consider using "
                             f"a rectangular projection.")

        reprojecting = False
        # Issue warning if changing the projection after initialization
        # TODO: Add heuristic size (i.e. ``self.ds.nbytes``) above which the
        #       warning is raised
        if hasattr(self, "_projection"):
            reprojecting = True
            print(("Reprojecting the descriptor can be inefficient "
                   "for large meshes"))

        # If both a projection and a transform are provided then
        if projection and self.transform:
            # reproject coordinate arrays in the minimal dataset
            self._transform_coordinates(projection, self.transform)
            # update the transform attribute following the reprojection
            self.transform = projection
            # Then loop over patch attributes
            for loc in ["cell", "edge", "vertex"]:
                attr = f"{loc}_patches"
                # and only delete attributes that have previously been cached
                if attr in self.__dict__:
                    del self.__dict__[attr]

        self._projection = projection

        # if periods are None (i.e. projection was not set at instantiation) or
        # the descriptor is being reprojected; update the periods
        if (hasattr(self, "_x_period") and hasattr(self, "_x_period")):
            if (not self.x_period and not self.y_period) or reprojecting:
                # dummy value b/c `_projection` attr will be used by setters
                self.x_period = None
                self.y_period = None

    @property
    def latlon(self) -> bool:
        """
        Boolean whether the lat/lon coordinate arrays should be used for
        patch construction.
        """
        return self._latlon

    @latlon.setter
    def latlon(self, value) -> None:
        """ TODO: check that the passed value is consistent with transform """
        if self.is_spherical:
            value = True

        self._latlon = value

    def _parse_period(self, ds, dim: Literal["x", "y"]):
        """ Parse period attribute, return None for non-periodic meshes """

        attr = f"{dim}_period"
        try:
            period = float(ds.attrs[attr])
        except KeyError:
            period = None

        # in the off chance mesh is periodic but does not have period attribute
        if self.is_periodic and attr not in ds.attrs:
            raise AttributeError((f"Mesh file: \"{ds.encoding['source']}\""
                                  f"does not have attribute `{attr}` despite"
                                  f"being a planar periodic mesh."))
        if period == 0.0:
            return None
        else:
            return period

    @property
    def x_period(self) -> float | None:
        """ Period along x-dimension, is ``None`` for non-periodic meshes """
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
        """ Period along y-dimension, is ``None`` for non-periodic meshes """
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
    def origin(self) -> Tuple[float, float]:
        """Coordinates of bottom left corner of plot"""

        def get_axis_min(self, coord: Literal["x", "y"]) -> float:
            """ """
            edge_min = float(self.ds[f"{coord}Edge"].min())
            vertex_min = float(self.ds[f"{coord}Vertex"].min())

            # an edge connects two vertices, so a vertices most extreme
            # position should always be more extended than an edge's
            if vertex_min > edge_min:
                max = float(self.ds[f"{coord}Vertex"].max())
                min = max - self.__getattribute__(f"{coord}_period")
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
        patches = self._wrap_patches(patches, "Cell")

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            mirrored, mirrored_idxs = self._mirror_patches(patches, "Cell")

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._cell_mirrored = mirrored
                self._cell_mirrored_idxs = mirrored_idxs

        # cartopy doesn't handle nans in patches, so store a mask of the
        # invalid patches to set the dataarray at those locations to nan.
        if self.projection:
            self._cell_pole_mask = self._compute_pole_mask("Cell")

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
        patches = self._wrap_patches(patches, "Edge")

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            mirrored, mirrored_idxs = self._mirror_patches(patches, "Edge")

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._edge_mirrored = mirrored
                self._edge_mirrored_idxs = mirrored_idxs

        # cartopy doesn't handle nans in patches, so store a mask of the
        # invalid patches to set the dataarray at those locations to nan.
        if self.projection:
            self._edge_pole_mask = self._compute_pole_mask("Edge")

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
        patches = self._wrap_patches(patches, "Vertex")

        # do not try to mirror patches for spherical meshes (yet...)
        if not self.is_spherical:
            mirrored, mirrored_idxs = self._mirror_patches(patches, "Vertex")

            # if mirrored patches were returned above store as attributes
            if mirrored is not None:
                self._vertex_mirrored = mirrored
                self._vertex_mirrored_idxs = mirrored_idxs

        # cartopy doesn't handle nans in patches, so store a mask of the
        # invalid patches to set the dataarray at those locations to nan.
        if self.projection:
            self._vertex_pole_mask = self._compute_pole_mask("Vertex")

        return patches

    def _transform_coordinates(self, projection, transform):

        for loc in ["Cell", "Edge", "Vertex"]:

            transformed_coords = projection.transform_points(
                transform, self.ds[f"x{loc}"], self.ds[f"y{loc}"])

            # ``transformed_coords`` is a numpy array so needs to assigned to
            # the values of the dataarray
            self.ds[f"x{loc}"].values = transformed_coords[:, 0]
            self.ds[f"y{loc}"].values = transformed_coords[:, 1]

    def _wrap_patches(self, patches, loc):
        """Wrap patches for spherical and planar-periodic meshes
        """

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

            mask = np.abs(diff) > np.abs(period) / (2. * np.sqrt(2.))
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

    def _compute_pole_mask(self, loc) -> ndarray:
        """ """
        limits = self.projection.y_limits
        centers = self.ds[f"y{loc.title()}"].values

        # TODO: determine threshold for ``isclose`` computation
        at_pole = np.any(
            np.isclose(centers.reshape(-1, 1), limits, rtol=1e-2), axis=1
        )
        past_pole = np.abs(centers) > np.abs(limits[1])

        return (at_pole | past_pole)

    def _mirror_patches(self, patches, loc):
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
            else:
                return np.unique(array[array != 0])

        def _find_mirrored(pathces, coord, period):
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
                error_str = ("A patch is periodic across multiple periods "
                             "(i.e. a patch has a non-unqiue sign). "
                             "Therefore patches cannot bed mirrored")
                raise ValueError(error_str) from exc

            # make a 1D array so broadcasting below will work
            mirror_sign = mirror_sign.squeeze()
            # convert sign into boolean mask
            mirror_mask = mirror_sign.astype(bool)

            return mirror_mask, mirror_sign

        def _mirror_1D(pathces, mask, sign, coord, period):
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

    nodes = np.stack((x_nodes, y_nodes), axis=-1)

    return nodes


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

    x_nodes = np.stack((xCell[:, 0], xVertex[:, 0],
                        xCell[:, 1], xVertex[:, 1]), axis=-1)

    y_nodes = np.stack((yCell[:, 0], yVertex[:, 0],
                        yCell[:, 1], yVertex[:, 1]), axis=-1)

    nodes = np.stack((x_nodes, y_nodes), axis=-1)

    return nodes


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
