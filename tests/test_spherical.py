from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely import LinearRing, is_valid

import mosaic

mpl.use("Agg", force=True)

SUPPORTED_PROJECTIONS = [
    ccrs.PlateCarree(),
    ccrs.LambertCylindrical(),
    ccrs.Mercator(),
    ccrs.Miller(),
    ccrs.Robinson(),
    ccrs.Stereographic(),
    ccrs.RotatedPole(),
    ccrs.InterruptedGoodeHomolosine(),
    ccrs.EckertI(),
    ccrs.EckertII(),
    ccrs.EckertIII(),
    ccrs.EckertIV(),
    ccrs.EckertV(),
    ccrs.EckertVI(),
    ccrs.EqualEarth(),
    ccrs.NorthPolarStereo(),
    ccrs.SouthPolarStereo(),
]


def id_func(projection):
    return type(projection).__name__


class TestSphericalWrapping:
    ds = mosaic.datasets.open_dataset("QU.240km")

    @pytest.fixture(scope="module", params=SUPPORTED_PROJECTIONS, ids=id_func)
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
        # get the pole mask b/c we know patches will be invalid there
        pole_mask = descriptor.__getattribute__(f"_{patch.lower()}_pole_mask")

        # convert the patches to a list of shapely geometries
        geoms = [LinearRing(patch) for patch in patches[~pole_mask]]

        # assert that all the patches are valid
        assert np.all(is_valid(geoms))
