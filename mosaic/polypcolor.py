from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike
from xarray.core.dataarray import DataArray

from mosaic.descriptor import Descriptor


def polypcolor(
    ax: Axes,
    descriptor: Descriptor,
    c: DataArray,
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

        polypcolor(ax, descriptor, c, *, ...)

    The unstructued grid can be specified either by passing a
    :py:class:`mosaic.Descriptor` object as the second parameter, or by
    passing the mesh datatset. See  :py:class:`mosaic.Descriptor` for an
    explanation of what the ``mesh_dataset`` has to be.

    Parameters:
        ax :
            An Axes or GeoAxes where the pseduocolor plot will be added

        descriptor : :py:class:`Descriptor`
            An already created ``Descriptor`` object

        c : :py:class:`xarray.DataArray`
            The color values to plot. Must have a dimension named either
            ``nCells``, ``nEdges``, or ``nVertices``.

        other_parameters
            All other parameters including the ``kwargs`` are the same as
            for :py:func:`matplotlib.pyplot.pcolor`.
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
    if descriptor.projection:
        minx = descriptor.projection.x_limits[0]
        maxx = descriptor.projection.x_limits[1]

        miny = descriptor.projection.y_limits[0]
        maxy = descriptor.projection.y_limits[1]

        ax.set_xbound(minx, maxx)
        ax.set_ybound(miny, maxy)

    return collection