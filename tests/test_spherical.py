from __future__ import annotations

import inspect

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import mosaic
import mosaic.utils

mpl.use("Agg", force=True)

# get the names, as strings, of unsupported projections for spherical meshes
unsupported = [
    p.__name__ for p in mosaic.descriptor.UNSUPPORTED_SPHERICAL_PROJECTIONS
]

PROJECTIONS = [
    obj()
    for name, obj in inspect.getmembers(ccrs)
    if inspect.isclass(obj)
    and issubclass(obj, ccrs.Projection)
    and not name.startswith("_")  # skip internal classes
    and obj is not ccrs.Projection  # skip base Projection class
    and name not in unsupported  # skip unsupported projections
]


def id_func(projection):
    return type(projection).__name__


class TestSphericalWrapping:
    ds = mosaic.datasets.open_dataset("QU.240km")

    @pytest.fixture(scope="module", params=PROJECTIONS, ids=id_func)
    def setup_descriptor(self, request):
        return mosaic.Descriptor(self.ds, request.param, ccrs.Geodetic())

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    @pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
    def test_timeout(self, setup_descriptor, tmp_path, patch):
        # do the test setup with the parameterized projection
        descriptor = setup_descriptor

        projection = descriptor.projection

        # get the projection name
        proj_name = type(projection).__name__

        # setup with figure with the parameterized projection
        fig, ax = plt.subplots(subplot_kw={"projection": projection})

        # get the appropriate dataarray for the parameterized patch location
        da = self.ds[f"indexTo{patch}ID"]

        # just testing that this doesn't hang, not for correctness
        mosaic.polypcolor(ax, descriptor, da, antialiaseds=True)

        # save the figure so that patches are rendered
        fig.savefig(f"{tmp_path}/{proj_name}-{patch}.png")
        plt.close()

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    def test_valid_patches(self, setup_descriptor, patch):
        # do the test setup with the parameterized projection
        descriptor = setup_descriptor

        # extract the patches
        patches = descriptor.__getattribute__(f"{patch.lower()}_patches")

        # get list of invalid patches
        invalid = mosaic.utils.get_invalid_patches(patches)

        # assert that all the patches are valid
        assert invalid is None
