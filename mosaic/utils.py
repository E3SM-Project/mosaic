from typing import Literal

import numpy as np
from shapely import LineString, is_valid

from mosaic.descriptor import Descriptor


def get_invalid_patches(
    descriptor: Descriptor, location: Literal["cell", "edge", "vertex"]
) -> None | np.ndarray:
    """Helper function to identify problematic patches.

    Returns the indices of the problematic patches as determined by
    :py:func:`shapely.is_valid`.

    Parameters
    ----------
    descriptor : :py:class:`Descriptor`
        The ``Descriptor`` object you want to check the patches of
    location : string
        The patch location to check: "cell" or "edge" or "vertex"

    Returns
    -------
    :py:class:`~numpy.ndarray` or None
        Indices of invalid patches. If no patches are invalid then returns None
    """
    # extract the patches
    patches = descriptor.__getattribute__(f"{location}_patches")
    # get the pole mask b/c we know patches will be invalid there
    pole_mask = descriptor.__getattribute__(f"_{location}_pole_mask")
    # convert the patches to a list of shapely geometries
    geoms = [LineString(patch) for patch in patches[~pole_mask]]
    # check if the shapely geometries are valid
    valid = is_valid(geoms)

    if np.all(valid):
        # no invalid patches, so return None
        return None
    else:
        # return the indices of the invalid patches
        index_array = np.arange(patches.shape[0])
        return index_array[~pole_mask][~valid]
