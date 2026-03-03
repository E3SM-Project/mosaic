from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import mosaic
import mosaic.utils

mpl.use("Agg", force=True)


class TestDuckTyping:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, ccrs.Orthographic(), ccrs.Geodetic())

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    @pytest.mark.parametrize("length", ["unculled", "culled"])
    @pytest.mark.parametrize("array_type", ["xr.DataArray", "np.ndarray"])
    def test_valid_array(self, tmp_path, patch, length, array_type):
        descriptor = self.descriptor
        projection = descriptor.projection

        # setup with figure with the parameterized projection
        fig, ax = plt.subplots(subplot_kw={"projection": projection})

        if length == "unculled":
            array = self.ds[f"indexTo{patch}ID"]
        else:
            array = self.descriptor.ds[f"indexTo{patch}ID"]

        if array_type == "np.ndarray":
            array = array.values
            assert isinstance(array, np.ndarray)
        else:
            assert isinstance(array, xr.DataArray)

        # just testing that this doesn't hang, not for correctness
        mosaic.polypcolor(ax, descriptor, array, antialiaseds=True)

        # save the figure so that patches are rendered
        fig.savefig(f"{tmp_path}/unculled-{patch}.png")
        plt.close()

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    @pytest.mark.parametrize("length", ["unculled", "culled"])
    @pytest.mark.parametrize("array_type", ["xr.DataArray", "np.ndarray"])
    def test_invalid_array(self, patch, length, array_type):
        descriptor = self.descriptor
        projection = descriptor.projection

        # setup with figure with the parameterized projection
        _fig, ax = plt.subplots(subplot_kw={"projection": projection})

        if length == "unculled":
            array = self.ds[f"indexTo{patch}ID"]
        else:
            array = self.descriptor.ds[f"indexTo{patch}ID"]

        if array_type == "np.ndarray":
            array = array.values
            assert isinstance(array, np.ndarray)
        else:
            assert isinstance(array, xr.DataArray)

        try:
            with pytest.raises(ValueError, match="Size of array"):
                mosaic.polypcolor(ax, descriptor, array[:10])
        finally:
            plt.close()

    @pytest.mark.parametrize("dim", ["domo", "domi", "domg"])
    def test_coupler_dims(self, tmp_path, dim):
        descriptor = self.descriptor
        projection = descriptor.projection

        fig, ax = plt.subplots(subplot_kw={"projection": projection})

        # make array dimension match coupler history files
        # tests: squeezing dimensions and arbitrarily names dimensions
        array = self.ds.nCells.expand_dims(dim={f"{dim}_ny": 1}).rename(
            nCells=f"{dim}_nx"
        )

        mosaic.polypcolor(ax, descriptor, array, antialiaseds=True)

        # save the figure so that patches are rendered
        fig.savefig(f"{tmp_path}/unculled-{dim}.png")
        plt.close()
