from functools import cached_property

import numpy as np
import xarray as xr

renaming_dict = {"lonCell": "xCell",
                 "latCell": "yCell",
                 "lonEdge": "xEdge",
                 "latEdge": "yEdge",
                 "lonVertex": "xVertex",
                 "latVertex": "yVertex"}

connectivity_arrays = ["cellsOnEdge",
                       "cellsOnVertex",
                       "verticesOnEdge",
                       "verticesOnCell"]


class Descriptor:
    """
    Class describing unstructured MPAS meshes in order to support plotting
    within ``matplotlib``. The class contains various methods to create
    :py:class:`matplotlib.collections.PolyCollection` objects for
    variables defined at cell centers, vertices, and edges.


    Attributes
    ----------
    latlon : boolean
        Whethere to use the lat/lon coordinates in patch construction

        NOTE: I don't think this is needed if the projection arg is
              properly used at initilaization

    projection : :py:class:`cartopy.crs.CRS`

    transform : :py:class:`cartopy.crs.CRS`

    cell_patches : :py:class:`numpy.ndarray`

    edge_patches : :py:class:`numpy.ndarray`

    vertex_patches : :py:class:`numpy.ndarray`
    """
    def __init__(self, ds, projection=None, transform=None, use_latlon=False):
        """
        """
        self.latlon = use_latlon
        self.projection = projection
        self.transform = transform

        # if mesh is on a sphere, force the use of lat lon coords
        if ds.attrs["on_a_sphere"].strip().upper() == 'YES':
            self.latlon = True
        # also check if projection requires lat/lon coords

        # create a minimal dataset, stored as an attr, for patch creation
        self.ds = self.create_minimal_dataset(ds)

        # reproject the minimal dataset, even for non-spherical meshes
        if projection and transform:
            self._transform_coordinates(projection, transform)

    def create_minimal_dataset(self, ds):
        """
        Create a xarray.Dataset that contains the minimal subset of
        coordinate / connectivity arrays needed to create pathces for plotting
        """

        def fix_outofbounds_indices(ds, array_name):
            """
            Some meshes (e.g. QU240km) don't do masking of ragged indices
            with 0. Instead they use `nInidices+1` as the invalid value,
            which can produce out of bounds errors. Check if that the case,
            and if so set the out of bounds indices to (-1)

            NOTE: Assumes connectivity arrays are zero indexed
            """
            # progamatically create the appropriate dimension name
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

        # get the subset of arrays from the mesh dataset
        minimal_ds = ds[mesh_arrays]

        # delete the attributes in the minimal dataset to avoid confusion
        minimal_ds.attrs.clear()

        for array in connectivity_arrays:
            # zero index all the connectivty arrays
            minimal_ds[array] = minimal_ds[array] - 1
            # fix any out of bounds indicies
            minimal_ds = fix_outofbounds_indices(minimal_ds, array)

        if self.latlon:

            # convert lat/lon coordinates from radian to degrees
            for loc in ["Cell", "Edge", "Vertex"]:
                minimal_ds[f"lon{loc}"] = np.rad2deg(minimal_ds[f"lon{loc}"])
                minimal_ds[f"lat{loc}"] = np.rad2deg(minimal_ds[f"lat{loc}"])

            # rename the coordinate arrays to all be named x.../y...
            # irrespective of whether spherical or cartesian coords are used
            minimal_ds = minimal_ds.rename(renaming_dict)

        return minimal_ds

    @cached_property
    def cell_patches(self):
        patches = _compute_cell_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Cell")
        return patches

    @cached_property
    def edge_patches(self):
        patches = _compute_edge_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Edge")
        return patches

    @cached_property
    def vertex_patches(self):
        patches = _compute_vertex_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Vertex")
        return patches

    def _transform_coordinates(self, projection, transform):
        """
        """

        for loc in ["Cell", "Edge", "Vertex"]:

            transformed_coords = projection.transform_points(
                transform, self.ds[f"x{loc}"], self.ds[f"y{loc}"])

            # transformed_coords is a np array so need to assign to the values
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

    def transform_patches(self, patches, projection, transform):
        """
        """

        raise NotImplementedError("This is a place holder. Do not use.")

        transformed_patches = projection.transform_points(
            transform, patches[..., 0], patches[..., 1])

        # transformation will return x,y,z values. Only need x and y
        patches.data[...] = transformed_patches[..., 0:2]

        return patches


def _compute_cell_patches(ds):

    # connectivity arrays have already been zero indexed
    verticesOnCell = ds.verticesOnCell
    # get a mask of the active vertices
    mask = verticesOnCell == -1

    # get the coordinates needed to patch construction
    xVertex = ds.xVertex
    yVertex = ds.yVertex

    # reshape/expand the vertices coordinate arrays
    x_vert = np.ma.MaskedArray(xVertex[verticesOnCell], mask=mask)
    y_vert = np.ma.MaskedArray(yVertex[verticesOnCell], mask=mask)

    verts = np.ma.stack((x_vert, y_vert), axis=-1)

    return verts


def _compute_edge_patches(ds, latlon=False):

    # connectivity arrays have already been zero indexed
    cellsOnEdge = ds.cellsOnEdge
    verticesOnEdge = ds.verticesOnEdge

    # this mask will never be true along the first column
    # otherwise it's not an edge it's just a point
    # that isn't possible (right?)
    cellMask = cellsOnEdge[:, 1] < 0

    # get subset of cell coordinate arrays corresponding to edge patches
    xCell = ds.xCell.values[cellsOnEdge]
    yCell = ds.yCell.values[cellsOnEdge]
    # get subset of vertex coordinate arrays corresponding to edge patches
    xVertex = ds.xVertex.values[verticesOnEdge]
    yVertex = ds.yVertex.values[verticesOnEdge]

    # if only one cell on edge, then collapse patch vertex point
    # from what would be the missing cell center to the edge location
    if np.any(cellMask):
        xCell[:, 1] = np.where(cellMask, ds.xEdge.values, xCell[:, 1])
        yCell[:, 1] = np.where(cellMask, ds.yEdge.values, yCell[:, 1])

    x_vert = np.stack((xCell[:, 0], xVertex[:, 0],
                       xCell[:, 1], xVertex[:, 1]), axis=-1)

    y_vert = np.stack((yCell[:, 0], yVertex[:, 0],
                       yCell[:, 1], yVertex[:, 1]), axis=-1)

    verts = np.stack((x_vert, y_vert), axis=-1)

    return verts


def _compute_vertex_patches(ds, latlon=False):

    # connectivity arrays have already been zero indexed
    cellsOnVertex = ds.cellsOnVertex
    # get a mask of the active vertices
    mask = ds.cellsOnVertex == -1

    # get the coordinates needed to patch construction
    xCell = ds.xCell
    yCell = ds.yCell

    # reshape/expand the vertices coordinate arrays
    x_vert = np.ma.MaskedArray(xCell[cellsOnVertex], mask=mask)
    y_vert = np.ma.MaskedArray(yCell[cellsOnVertex], mask=mask)

    verts = np.ma.stack((x_vert, y_vert), axis=-1)

    return verts
