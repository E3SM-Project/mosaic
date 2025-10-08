from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely import LinearRing, is_valid

import mosaic

mpl.use("Agg", force=True)

rng = np.random.default_rng()


def set_clim(case, dim_size):
    match case:
        case None:
            return None
        case "mid":
            return dim_size // 2
        case "over":
            return dim_size + dim_size // 2
        case "under":
            return -dim_size // 2


class TestPlanarPeriodicMirroring:
    ds = mosaic.datasets.open_dataset("doubly_periodic_4x4")
    descriptor = mosaic.Descriptor(ds)

    def get_dim_size(self, patch):
        match patch:
            case "Cell":
                plural = "Cells"
            case "Edge":
                plural = "Edges"
            case "Vertex":
                plural = "Vertices"

        return self.ds.sizes[f"n{plural}"]

    def test_offset_coordinates(self):
        """Only testing for cell patches currently"""

        # create a copy of the dataset for offsetting the coordinates
        ds = self.ds.copy()

        # create random offsets, for x and y direction independently
        delta_x = rng.integers(1, 100) * rng.standard_normal()
        delta_y = rng.integers(1, 100) * rng.standard_normal()

        for loc in ["Cell", "Edge", "Vertex"]:
            ds[f"x{loc}"] = ds[f"x{loc}"] + delta_x
            ds[f"y{loc}"] = ds[f"y{loc}"] + delta_y

        descriptor = mosaic.Descriptor(ds)

        x_period = descriptor.x_period
        y_period = descriptor.y_period

        # extract the interior (i.e. original and un-mirrored) patches
        interior_patches = descriptor.cell_patches
        # extract the mirrored patches
        mirrored_patches = descriptor._cell_mirrored

        # get the extent of the patches, including the mirrored patches
        minx = min(
            np.min(interior_patches[..., 0]), np.min(mirrored_patches[..., 0])
        )
        maxx = max(
            np.max(interior_patches[..., 0]), np.max(mirrored_patches[..., 0])
        )
        miny = min(
            np.min(interior_patches[..., 1]), np.min(mirrored_patches[..., 1])
        )
        maxy = max(
            np.max(interior_patches[..., 1]), np.max(mirrored_patches[..., 1])
        )

        range_x = maxx - minx
        range_y = maxy - miny

        # ensure the dcEdge is constant for all cells
        assert np.all(np.isclose(self.ds.dcEdge, self.ds.dcEdge[0]))
        assert np.all(np.isclose(self.ds.dvEdge, self.ds.dvEdge[0]))

        dc = float(self.ds.dcEdge[0])
        dv = float(self.ds.dvEdge[0])

        # need to add an extra half cell spacing because of ragged rows
        true_range_x = x_period + 1.5 * dc
        # circumradius of hexagon is equal edge length
        true_range_y = y_period + 2.0 * dv

        assert np.isclose(range_x, true_range_x, atol=dc * 0.05, rtol=0), (
            f"failed for a x-offset of {delta_x: .3f}"
        )

        assert np.isclose(range_y, true_range_y, atol=dv * 0.05, rtol=0), (
            f"failed for a y-offset of {delta_y: .3f}"
        )

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    def test_valid_patches(self, patch):
        descriptor = mosaic.Descriptor(self.ds)

        # extract the interior (i.e. original and un-mirrored) patches
        interior_patches = getattr(descriptor, f"{patch.lower()}_patches")
        # extract the mirrored patches
        mirrored_patches = getattr(descriptor, f"_{patch.lower()}_mirrored")
        # stack the patch arrays for complete testing
        patches = np.vstack([interior_patches, mirrored_patches])

        # convert the patches to a list of shapely geometries
        geoms = [LinearRing(patch) for patch in patches]

        # assert that all the patches are valid
        assert np.all(is_valid(geoms))

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    @pytest.mark.parametrize("vmin", [None, "mid", "over", "under"])
    @pytest.mark.parametrize("vmax", [None, "mid", "over", "under"])
    def test_clims(self, patch, vmin, vmax):
        dim_size = self.get_dim_size(patch)

        kwargs = {
            "vmin": set_clim(vmin, dim_size),
            "vmax": set_clim(vmax, dim_size),
        }

        _fig, ax = plt.subplots()

        collection = mosaic.polypcolor(
            ax, self.descriptor, self.ds[f"indexTo{patch}ID"], **kwargs
        )

        original_clim = collection.get_clim()
        mirrored_clim = collection._mirrored_collection_fix.get_clim()

        plt.close()

        assert original_clim == mirrored_clim
