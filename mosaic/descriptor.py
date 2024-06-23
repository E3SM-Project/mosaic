import numpy as np 

from xarray.core.dataset import Dataset  

class Descriptor:
    """
    MPAS mesh descriptor
    """
    def __init__(self, ds, projection=None, transform=None, use_latlon=False): 

        self.latlon = use_latlon

        # if mesh is on a sphere, force the use of lat lon coords
        if ds.attrs["on_a_sphere"].strip().upper() == 'YES':
            self.latlon = True
        
        # if we have spherical data convert from radians to degrees
        if self.latlon:
            ds = self._rad2deg_coordinates(ds)
        
        # reproject the data, even for non-spherical meshes
        if projection and transform: 
            ds = self._transform_coordinates(ds, projection, transform)

        # create the patches for each location a variable can be defined
        self.cell_patches = _compute_cell_patches(ds, latlon=self.latlon)
        self.edge_patches = _compute_edge_patches(ds, latlon=self.latlon) 
        self.vertex_patches = _compute_vertex_patches(ds, latlon=self.latlon)
    
        # temporary antimeridian fix
        if projection: 
            self.cell_patches = self._mask_antimeridian(ds, "Cell", projection)
            self.edge_pathces = self._mask_antimeridian(ds, "Edge", projection)
            self.vertex_patches = self._mask_antimeridian(ds, "Vertex", projection)

    def _rad2deg_coordinates(self, ds): 
        """If using the lat/lon coords convert them all from radian to degrees
        """
        for loc in ["Cell", "Edge", "Vertex"]:
            ds[f"lon{loc}"] = np.rad2deg(ds[f"lon{loc}"])
            ds[f"lat{loc}"] = np.rad2deg(ds[f"lat{loc}"])

        return ds

    def _transform_coordinates(self, ds, projection, transform):
        """
        """

        for loc in ["Cell", "Edge", "Vertex"]:

            transformed_coords = projection.transform_points(transform,
                *_get_coordinates(ds, loc, self.latlon))
           
            # transformed_coords is a np array so need to assign to the values
            if self.latlon:
                ds[f"lon{loc}"].values = transformed_coords[:, 0]
                ds[f"lat{loc}"].values = transformed_coords[:, 1]
            else:
                ds[f"x{loc}"].values = transformed_coords[:, 0]
                ds[f"y{loc}"].values = transformed_coords[:, 1]

        return ds
    
    def _mask_antimeridian(self, ds, loc, projection): 
        
        # convert to numpy array to that broadcasting below will work
        x_center = np.array(_get_coordinates(ds, loc, self.latlon)[0])
        
        patches = self.__getattribute__(f"{loc.lower()}_patches")
        
        #
        half_distance = x_center[:, np.newaxis] - patches[...,0]
        # get the size limit of the projection; 
        size_limit = np.abs(projection.x_limits[1] -
                            projection.x_limits[0]) / (2 * np.sqrt(2))
    
        # left and right mask, with same number of dimensions as the patches
        l_mask = (half_distance > size_limit)[..., np.newaxis]
        r_mask = (half_distance < -size_limit)[..., np.newaxis]
        
        patches.mask |= l_mask
        patches.mask |= r_mask

        return patches

    def transform_patches(self, patches, projection, transform):
        """
        """

        raise NotImplementedError("This is a place holder. Do not use.")
         
        transformed_patches = projection.transform_points(transform,
            patches[..., 0], patches[..., 1])
    
        # transformation will return x,y,z values. Only need x and y
        patches.data[...] = transformed_patches[..., 0:2] 

        return patches

def _get_coordinates(ds, location, latlon=False): 
    
    if latlon:
        x = ds[f"lon{location}"]
        y = ds[f"lat{location}"]
    else:
        x = ds[f"x{location}"]
        y = ds[f"y{location}"]

    return (x, y)

def _compute_cell_patches(ds, latlon=False):
    
    # get a mask of the active vertices
    mask = ds.verticesOnCell == 0
    
    # get the coordinates needed to patch construction
    xVertex, yVertex = _get_coordinates(ds, "Vertex", latlon)
    
    # account for zero indexing
    verticesOnCell = ds.verticesOnCell - 1

    # reshape/expand the vertices coordinate arrays
    x_vert = np.ma.MaskedArray(xVertex[verticesOnCell], mask=mask)
    y_vert = np.ma.MaskedArray(yVertex[verticesOnCell], mask=mask)

    verts = np.ma.stack((x_vert, y_vert), axis=-1)

    return verts

def _compute_edge_patches(ds, latlon=False):
    
    # account for zeros indexing
    cellsOnEdge = ds.cellsOnEdge - 1
    verticesOnEdge = ds.verticesOnEdge - 1
    
    # is this masking sufficent ?
    cellMask = cellsOnEdge <= 0
    vertexMask = verticesOnEdge <= 0

    # get the coordinates needed to patch construction
    xVertex, yVertex = _get_coordinates(ds, "Vertex", latlon)
    # get the coordinates needed to patch construction
    xCell, yCell = _get_coordinates(ds, "Cell", latlon)

    # get subset of cell coordinate arrays corresponding to edge patches
    xCell = np.ma.MaskedArray(xCell[cellsOnEdge], mask=cellMask)
    yCell = np.ma.MaskedArray(yCell[cellsOnEdge], mask=cellMask)
    # get subset of vertex coordinate arrays corresponding to edge patches
    xVertex = np.ma.MaskedArray(xVertex[verticesOnEdge], mask=vertexMask)
    yVertex = np.ma.MaskedArray(yVertex[verticesOnEdge], mask=vertexMask)

    x_vert = np.ma.stack((xCell[:,0], xVertex[:,0],
                          xCell[:,1], xVertex[:,1]), axis=-1)
    
    y_vert = np.ma.stack((yCell[:,0], yVertex[:,0],
                          yCell[:,1], yVertex[:,1]), axis=-1)

    
    verts = np.ma.stack((x_vert, y_vert), axis=-1)

    return verts

def _compute_vertex_patches(ds, latlon=False):
    
    # get a mask of the active vertices
    mask = ds.cellsOnVertex == 0
    
    # get the coordinates needed to patch construction
    xCell, yCell = _get_coordinates(ds, "Cell", latlon)
    
    # account for zero indexing
    cellsOnVertex = ds.cellsOnVertex - 1

    # reshape/expand the vertices coordinate arrays
    x_vert = np.ma.MaskedArray(xCell[cellsOnVertex], mask=mask)
    y_vert = np.ma.MaskedArray(yCell[cellsOnVertex], mask=mask)

    verts = np.ma.stack((x_vert, y_vert), axis=-1)

    return verts
