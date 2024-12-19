from functools import cached_property

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
        # Issue warning if changing the projection after initialization
        # TODO: Add heuristic size (i.e. ``self.ds.nbytes``) above which the
        #       warning is raised
        if hasattr(self, "_projection"):
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
        patches = self._fix_antimeridian(patches, "Cell")
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
        patches = self._fix_antimeridian(patches, "Edge")
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
        patches = self._fix_antimeridian(patches, "Vertex")
        return patches

    def _transform_coordinates(self, projection, transform):

        for loc in ["Cell", "Edge", "Vertex"]:

            transformed_coords = projection.transform_points(
                transform, self.ds[f"x{loc}"], self.ds[f"y{loc}"])

            # ``transformed_coords`` is a numpy array so needs to assigned to
            # the values of the dataarray
            self.ds[f"x{loc}"].values = transformed_coords[:, 0]
            self.ds[f"y{loc}"].values = transformed_coords[:, 1]

    def _fix_antimeridian(self, patches, loc, projection=None):
        """Correct vertices of patches that cross the antimeridian.

        NOTE: Can this be a decorator?
        """
        # coordinate arrays are transformed at initalization, so using the
        # transform size limit, not the projection
        if not projection:
            projection = self.projection

        # should be able to come up with a default size limit here, or maybe
        # it's already an attribute(?) Should also factor in a precomputed
        # axis period, as set in the attributes of the input dataset
        if projection:
            # convert to numpy array to that broadcasting below will work
            x_center = np.array(self.ds[f"x{loc}"])

            # get distance b/w the center and vertices of the patches
            # NOTE: using data from masked patches array so that we compute
            #       mask only corresponds to patches that cross the boundary,
            #       (i.e. NOT a mask of all invalid cells). May need to be
            #       carefull about the fillvalue depending on the transform
            half_distance = x_center[:, np.newaxis] - patches[..., 0].data

            # get the size limit of the projection;
            size_limit = np.abs(projection.x_limits[1] -
                                projection.x_limits[0]) / (2 * np.sqrt(2))

            # left and right mask, with same number of dims as the patches
            l_mask = (half_distance > size_limit)[..., np.newaxis]
            r_mask = (half_distance < -size_limit)[..., np.newaxis]

            """
            # Old approach masks out all patches that cross the antimeridian.
            # This is unnessarily restrictive. New approach corrects
            # the x-coordinates of vertices that lie outside the projections
            # bounds, which isn't perfect either

            patches.mask |= l_mask
            patches.mask |= r_mask
            """

            l_boundary_mask = ~np.any(l_mask, axis=1) | l_mask[..., 0]
            r_boundary_mask = ~np.any(r_mask, axis=1) | r_mask[..., 0]
            # get valid half distances for the patches that cross boundary
            l_offset = np.ma.MaskedArray(half_distance, l_boundary_mask)
            r_offset = np.ma.MaskedArray(half_distance, r_boundary_mask)

            # For vertices that cross the antimeridian reset the x-coordinate
            # of invalid vertex to be the center of the patch plus the
            # mean valid half distance.
            #
            # NOTE: this only fixes patches on the side of plot where they
            # cross the antimeridian, leaving an empty zipper like pattern
            # mirrored over the y-axis.
            patches[..., 0] = np.ma.where(
                ~l_mask[..., 0], patches[..., 0],
                x_center[:, np.newaxis] + l_offset.mean(1)[..., np.newaxis])
            patches[..., 0] = np.ma.where(
                ~r_mask[..., 0], patches[..., 0],
                x_center[:, np.newaxis] + r_offset.mean(1)[..., np.newaxis])

        return patches


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
    unionMask = cellMask & edgeMask

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
    # NOTE: While the condition below probably only applies to the final edge
    #       node we apply it to all, since the conditions above ensure the
    #       patches will still be created correctly
    # -------------------------------------------------------------------------
    # if cell and edge missing collapse edge node to the first edge.
    # Because edges lead the vertices this ensures the patch encompasses
    # the full kite area and is properly closed.
    nodes[:, ::2, 0] = np.where(unionMask, nodes[:, 0:1, 0], nodes[:, ::2, 0])
    nodes[:, ::2, 1] = np.where(unionMask, nodes[:, 0:1, 1], nodes[:, ::2, 1])

    return nodes
