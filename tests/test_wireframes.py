from __future__ import annotations

from unittest.mock import patch

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import lines as mlines

import mosaic
from mosaic import _geometries

mpl.use("Agg", force=True)


class TestWireframeRaises:
    def test_periodic_mesh_raises(self):
        ds = mosaic.datasets.open_dataset("doubly_periodic_4x4")
        descriptor = mosaic.Descriptor(ds)

        fig, ax = plt.subplots()
        try:
            with pytest.raises(NotImplementedError, match="periodic"):
                mosaic.wireframe(ax, descriptor)
        finally:
            plt.close(fig)

    def test_invalid_location_raises(self):
        ds = mosaic.datasets.open_dataset("QU.240km")
        descriptor = mosaic.Descriptor(
            ds, ccrs.Orthographic(), ccrs.Geodetic()
        )

        fig, ax = plt.subplots(
            subplot_kw={"projection": descriptor.projection}
        )
        try:
            with pytest.raises(ValueError, match="Invalid location"):
                mosaic.wireframe(ax, descriptor, location="invalid")
        finally:
            plt.close(fig)


class TestWireframeKwargs:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, ccrs.Orthographic(), ccrs.Geodetic())

    def _plot_calls(self, location="Cell", **kwargs):
        """Render a wireframe with `ax.plot` mocked out, returning the
        (lines_call, markers_call) arguments it was invoked with."""
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            with patch.object(ax, "plot") as mock_plot:
                mosaic.wireframe(ax, self.descriptor, location, **kwargs)
                lines_call, markers_call = mock_plot.call_args_list
                return lines_call, markers_call
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("location", ["Cell", "Edge", "Vertex"])
    def test_location_dispatch(self, location):
        lines_call, _markers_call = self._plot_calls(location)

        expected_x, expected_y = getattr(
            self.descriptor, f"{location.lower()}_wireframe"
        )
        np.testing.assert_array_equal(lines_call.args[0], expected_x.ravel())
        np.testing.assert_array_equal(lines_call.args[1], expected_y.ravel())

    @pytest.mark.parametrize("location", ["cell", "CELL", "cells", "vertices"])
    def test_location_normalization(self, location):
        lines_call, _markers_call = self._plot_calls(location, marker="o")
        norm_location = mosaic.utils._normalize_point_type(location)

        expected_x, expected_y = getattr(
            self.descriptor, f"{norm_location.lower()}_wireframe"
        )
        np.testing.assert_array_equal(lines_call.args[0], expected_x.ravel())
        np.testing.assert_array_equal(lines_call.args[1], expected_y.ravel())

    def test_no_marker_draws_no_markers(self):
        _lines_call, markers_call = self._plot_calls("Cell")
        assert markers_call.args == ([], [])

    def test_marker_draws_node_coords(self):
        _lines_call, markers_call = self._plot_calls("Cell", marker="o")

        expected_x, expected_y = _geometries.compute_node_coords(
            self.descriptor.ds, "Cell"
        )
        np.testing.assert_array_equal(markers_call.args[0], expected_x)
        np.testing.assert_array_equal(markers_call.args[1], expected_y)

    @pytest.mark.parametrize("linestyle", [None, "None", "", " "])
    def test_linestyle_none_draws_no_lines(self, linestyle):
        lines_call, _markers_call = self._plot_calls(
            "Cell", linestyle=linestyle
        )
        assert lines_call.args == ([], [])


class TestWireframeReturnValue:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, ccrs.Orthographic(), ccrs.Geodetic())

    def test_returns_line_and_marker_artists(self):
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            artists = mosaic.wireframe(ax, self.descriptor, "Cell", marker="o")

            # one Line2D artist for the wireframe lines, one for the markers
            assert len(artists) == 2
            assert all(isinstance(a, mlines.Line2D) for a in artists)
        finally:
            plt.close(fig)


class TestWireframeSmoke:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, ccrs.Orthographic(), ccrs.Geodetic())

    @pytest.mark.parametrize("location", ["Cell", "Edge", "Vertex"])
    def test_renders_without_error(self, tmp_path, location):
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            mosaic.wireframe(ax, self.descriptor, location, marker="o")
            fig.savefig(f"{tmp_path}/wireframe-{location}.png")
        finally:
            plt.close(fig)
