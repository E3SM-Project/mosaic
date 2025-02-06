import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
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

    The unstructured grid is specified by passing a
    :py:class:`~mosaic.Descriptor` object as the second parameter.
    See :py:class:`mosaic.Descriptor` for an explanation of what the
    ``Descriptor`` is and how to construct it.

    Parameters
    ----------

    ax : matplotlib axes object
        Axes, or GeoAxes, on which to plot

    descriptor : :py:class:`Descriptor`
        An already created ``Descriptor`` object

    c : DataArray
        The color values to plot. Must have a dimension named either
        ``nCells``, ``nEdges``, or ``nVertices``.

    other_parameters
        All other parameters are the same as for
        :py:func:`~matplotlib.pyplot.pcolor`.
    """

    if "nCells" in c.dims:
        verts = descriptor.cell_patches
        if descriptor.projection and np.any(descriptor._cell_pole_mask):
            c = c.where(~descriptor._cell_pole_mask, np.nan)

    elif "nEdges" in c.dims:
        verts = descriptor.edge_patches
        if descriptor.projection and np.any(descriptor._edge_pole_mask):
            c = c.where(~descriptor._edge_pole_mask, np.nan)

    elif "nVertices" in c.dims:
        verts = descriptor.vertex_patches
        if descriptor.projection and np.any(descriptor._vertex_pole_mask):
            c = c.where(~descriptor._vertex_pole_mask, np.nan)

    transform = descriptor.transform

    collection = PolyCollection(verts, alpha=alpha, array=c,
                                cmap=cmap, norm=norm, **kwargs)

    # only set the transform if GeoAxes
    if isinstance(ax, GeoAxes):
        collection.set_transform(transform)

    collection._scale_norm(norm, vmin, vmax)
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)
    ax.autoscale_view()

    # for planar periodic plot explicity set the axis limit
    if not descriptor.is_spherical and descriptor.x_period:
        xmin, xmax = _find_planar_periodic_axis_limits(descriptor, "x")
        ax.set_xlim(xmin, xmax)

    if not descriptor.is_spherical and descriptor.y_period:
        ymin, ymax = _find_planar_periodic_axis_limits(descriptor, "y")
        ax.set_ylim(ymin, ymax)

    return collection


def _find_planar_periodic_axis_limits(descriptor, coord):
    """Find the correct (tight) axis limits for planar periodic meshes.
    """

    # get the axis period
    period = descriptor.__getattribute__(f"{coord}_period")
    # get axis index we are inquiring over
    axis = 0 if coord == "x" else 1

    min = descriptor.origin[axis]
    max = min + period

    return min, max
