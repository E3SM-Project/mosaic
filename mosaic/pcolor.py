from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, Colormap
from mosaic.descriptor import Descriptor

from numpy.typing import ArrayLike

from xarray.core.dataarray import DataArray


def pcolor(
    ax: Axes,
    descriptor: Descriptor,
    c: ArrayLike,
    alpha: float = 1.0,
    norm: str | Normalize | None = None, 
    cmap: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    facecolors: ArrayLike | None = None,
    **kwargs
    ) -> PolyCollection:
    """
    Create a pseudocolor plot of a unstructured MPAS grid.

    Call signatures::
        
        pcolor(ax, descriptor, c, *, ...)
        pcolor(as, mesh_dataset, c, *, ....)

    The unstructued grid can be specified either by passing a `.Descriptor`
    object as the second parameter, or by passing the mesh datatset. See 
    `.Descriptor` for an explination of what the mesh_dataset has to be. 
    """

    if "nCells" in c.dims:
        verts = descriptor.cell_patches

    elif "nEdges" in c.dims:
        verts = descriptor.edge_patches

    elif "nVertices" in c.dims:
        verts = descriptor.vertex_patches

    collection = PolyCollection(verts, alpha=alpha, array=c,
                                cmap=cmap, norm=norm, **kwargs)

    collection._scale_norm(norm, vmin, vmax)
    
    # get the limits of the **data**, which could exceed the valid
    # axis limits of the transform
    minx = verts[..., 0].min()
    maxx = verts[..., 0].max()
    miny = verts[..., 1].min()
    maxy = verts[..., 1].max()
  
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax._request_autoscale_view()
    ax.add_collection(collection)
    
    # if data has been transformed use the transforms x-limits.
    # patches have vertices that exceed the transfors x-limits to visually
    # correct the antimeridian problem
    if descriptor.transform:
        minx = descriptor.transform.x_limits[0]
        maxx = descriptor.transform.x_limits[1]

        miny = descriptor.transform.y_limits[0]
        maxy = descriptor.transform.y_limits[1]
     
        ax.set_xbound(minx, maxx)
        ax.set_ybound(miny, maxy)

    return collection
