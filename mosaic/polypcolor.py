from __future__ import annotations

import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from numpy.typing import ArrayLike
from xarray.core.dataarray import DataArray

from mosaic.descriptor import Descriptor
from mosaic.mpas_collection import MPASCollection


def _get_array_location(
    descriptor: Descriptor, array: ArrayLike
) -> tuple[str, ArrayLike]:
    """Helper function to find mesh location where array is defined"""

    dimension_to_location = {
        "nCells": "cell",
        "nEdges": "edge",
        "nVertices": "vertex",
    }

    if array.ndim != 1:
        msg = f"Array should be one dimensional, instead has {array.ndim} dims"
        if sum([size > 1 for size in array.shape]) > 1:
            # more than one dimensions is greater than length one
            raise ValueError(msg)
        array = array.squeeze()

    # dict of dim lengths of the original dataset
    origin_rev = {v: k for k, v in descriptor.sizes.items()}
    # dict of dim lengths of the culled dataset
    culled_rev = {descriptor.ds.sizes[d]: d for d in descriptor.sizes}

    # start by trying to find match in original dataset
    dim = origin_rev.get(array.size)
    if dim is not None:
        loc = dimension_to_location[dim]
        array_name = f"indexTo{loc.capitalize()}ID"
        if array_name in descriptor.ds:
            lut = descriptor.ds[array_name]
            return loc, array[lut]
        return loc, array

    # fallback to checking culled dataset
    dim = culled_rev.get(array.size)
    if dim is not None:
        # no need to look up table with culled mesh array
        return dimension_to_location[dim], array

    msg = (
        f"Size of array {array.size} is incompatible with mesh dimensions "
        f"{descriptor.sizes}"
    )
    raise ValueError(msg)


def _parse_args(
    descriptor: Descriptor, array: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Helper function to get patch array corresponding to dataarray"""

    loc, array = _get_array_location(descriptor, array)

    verts = getattr(descriptor, f"{loc}_patches")

    return verts, array


def _mirror_polycollection(ax, collection, descriptor, array, **kwargs):
    """Handle patches that need to be mirrored.

    Following ``cartopy.mpl.geoaxes._wrap_quadmesh``
    """

    loc, array = _get_array_location(descriptor, array)

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
