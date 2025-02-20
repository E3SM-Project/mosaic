import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from xarray.core.dataarray import DataArray

from mosaic.descriptor import Descriptor


def _get_array_location(descriptor, array):
    """Helper function to find at what mesh location the dataarray is defined
    """

    if "nCells" in array.dims:
        return "cell"
    elif "nEdges" in array.dims:
        return "edge"
    elif "nVertices" in array.dims:
        return "vertex"


def _parse_args(descriptor, array):
    """Helper function to get patch array corresponding to dataarray
    """

    loc = _get_array_location(descriptor, array)

    verts = getattr(descriptor, f"{loc}_patches")
    pole_mask = getattr(descriptor, f"_{loc}_pole_mask", None)

    if descriptor.projection and pole_mask is not None:
        array = array.where(~pole_mask, np.nan)

    return verts, array


def _mirror_polycollection(ax, collection, descriptor, array, **kwargs):
    """Handle patches that need to be mirrored.

    Following ``cartopy.mpl.geoaxes._wrap_quadmesh``
    """

    loc = _get_array_location(descriptor, array)

    mirrored_verts = getattr(descriptor, f"_{loc}_mirrored", None)
    mirrored_idxs = getattr(descriptor, f"_{loc}_mirrored_idxs", None)

    # if no patches to mirror then break here
    if mirrored_verts is None:
        return collection

    zorder = collection.zorder - .1
    kwargs.pop('zorder', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    norm = kwargs.pop('norm', None)
    cmap = kwargs.pop('cmap', None)

    # create a second PolyCollection for the mirrored patches
    mirrored_collection = PolyCollection(
        mirrored_verts, array=array[mirrored_idxs], zorder=zorder, **kwargs
    )

    mirrored_collection.set_cmap(cmap)
    mirrored_collection.set_norm(norm)
    mirrored_collection.set_clim(vmin, vmax)
    # if vmin or vmax is None, use min/max from *orig* data to scale mirrored
    mirrored_collection.norm.autoscale_None(array)

    # TODO: support wrapping for spherical meshes
    if isinstance(ax, GeoAxes):
        mirrored_collection.set_transform(descriptor.transform)

    # add the mirrored collection to the axes
    ax.add_collection(mirrored_collection)

    # store the mirrored_collection and associated indices
    collection._mirrored_idxs = mirrored_idxs
    collection._mirrored_collection_fix = mirrored_collection

    return collection


def polypcolor(
    ax: Axes,
    descriptor: Descriptor,
    array: DataArray,
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

    array : DataArray
        The color values to plot. Must have a dimension named either
        ``nCells``, ``nEdges``, or ``nVertices``.

    other_parameters
        All other parameters are the same as for
        :py:func:`~matplotlib.pyplot.pcolor`.
    """

    verts, array = _parse_args(descriptor, array)

    collection = PolyCollection(verts, array=array, **kwargs)

    # only set the transform if GeoAxes
    if isinstance(ax, GeoAxes):
        collection.set_transform(descriptor.transform)

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    norm = kwargs.pop('norm', None)

    collection._scale_norm(norm, vmin, vmax)
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)

    # Mirror patches that cross periodic boundaries
    collection = _mirror_polycollection(
        ax, collection, descriptor, array, **kwargs
    )

    # for planar periodic plot explicity set the axis limit
    if not descriptor.is_spherical and descriptor.x_period:
        xmin, xmax = _find_planar_periodic_axis_limits(descriptor, "x")
        ax.set_xlim(xmin, xmax)

    if not descriptor.is_spherical and descriptor.y_period:
        ymin, ymax = _find_planar_periodic_axis_limits(descriptor, "y")
        ax.set_ylim(ymin, ymax)

    ax.autoscale_view()
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
