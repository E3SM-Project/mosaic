from __future__ import annotations

import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from xarray.core.dataarray import DataArray

from mosaic.descriptor import Descriptor
from mosaic.mpas_collection import MPASCollection


def _get_array_location(array):
    """Helper function to find mesh location where dataarray is defined"""

    if "nCells" in array.dims:
        return "cell"
    if "nEdges" in array.dims:
        return "edge"
    if "nVertices" in array.dims:
        return "vertex"

    return None


def _parse_args(descriptor, array):
    """Helper function to get patch array corresponding to dataarray"""

    loc = _get_array_location(array)

    verts = getattr(descriptor, f"{loc}_patches")
    pole_mask = getattr(descriptor, f"_{loc}_pole_mask", None)

    if descriptor.projection and pole_mask is not None:
        array = array.where(~pole_mask, np.nan)

    return verts, array


def _mirror_polycollection(ax, collection, descriptor, array, **kwargs):
    """Handle patches that need to be mirrored.

    Following ``cartopy.mpl.geoaxes._wrap_quadmesh``
    """

    loc = _get_array_location(array)

    mirrored_verts = getattr(descriptor, f"_{loc}_mirrored", None)
    mirrored_idxs = getattr(descriptor, f"_{loc}_mirrored_idxs", None)

    # if no patches to mirror then break here
    if mirrored_verts is None:
        return collection

    zorder = collection.zorder - 0.1
    kwargs.pop("zorder", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    norm = kwargs.pop("norm", None)
    cmap = kwargs.pop("cmap", None)

    # create a second PolyCollection for the mirrored patches
    mirrored_collection = PolyCollection(
        mirrored_verts, array=array[mirrored_idxs], zorder=zorder, **kwargs
    )

    if np.allclose(array, array[0], rtol=1e-8, atol=1e-10):
        norm = collection.norm

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
    ax: Axes, descriptor: Descriptor, array: DataArray, **kwargs
) -> MPASCollection:
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
        All other parameters are the forwarded to the
        :py:class:`~matplotlib.collections.PolyCollection` creation.
    """

    verts, array = _parse_args(descriptor, array)

    # need to pop b/c PolyCollection does not accept these
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    norm = kwargs.pop("norm", None)

    collection = PolyCollection(verts, array=array, norm=norm, **kwargs)

    # only set the transform if GeoAxes
    if isinstance(ax, GeoAxes):
        collection.set_transform(descriptor.transform)

    collection._scale_norm(norm, vmin, vmax)
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)

    # repack so mirrored collection has consistent color limits
    kwargs.update({"vmin": vmin, "vmax": vmax, "norm": norm})

    # Mirror patches that cross periodic boundaries
    collection = _mirror_polycollection(
        ax, collection, descriptor, array, **kwargs
    )

    # Re-cast the PolyCollection as MPASCollection for mirrored patch handling
    collection.__class__ = MPASCollection

    # for planar periodic plot explicitly set the axis limit
    # TODO: use ``ax.update_datalims`` instead of explicitly setting axis limits
    if not descriptor.is_spherical and descriptor.x_period:
        xmin, xmax = _find_planar_periodic_axis_limits(descriptor, "x")
        ax.set_xlim(xmin, xmax)

    if not descriptor.is_spherical and descriptor.y_period:
        ymin, ymax = _find_planar_periodic_axis_limits(descriptor, "y")
        ax.set_ylim(ymin, ymax)

    ax.autoscale_view()
    return collection


def _find_planar_periodic_axis_limits(descriptor, coord):
    """Find the correct (tight) axis limits for planar periodic meshes."""

    # get the axis period
    period = descriptor.__getattribute__(f"{coord}_period")
    # get axis index we are inquiring over
    axis = 0 if coord == "x" else 1

    min = descriptor.origin[axis]
    max = min + period

    return min, max
