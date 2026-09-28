from __future__ import annotations

from typing import Literal

from matplotlib import cbook
from matplotlib import lines as mlines
from matplotlib.axes import Axes

from mosaic import _geometries
from mosaic.descriptor import Descriptor

point_types = Literal["Cell", "Edge", "Vertex"]


def wireframe(
    ax: Axes,
    descriptor: Descriptor,
    location: point_types = "Cell",
    *args,
    **kwargs,
) -> None:
    """Draw a wireframe (an outline of the mesh topology) on the given axes.

    If the "marker" keyword argument is provided, markers are drawn at the nodes
    that define the wireframe lines rather than the coordinates of the location
    itself (e.g., for location="Cell", markers are drawn at vertices rather than
    cell centers).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes, or GeoAxes, on which to plot
    descriptor : Descriptor
        The descriptor that defines the wireframe.
    location : Cell, Edge, or Vertex
        Location to plot wireframe of. Default is "Cell".
    other_parameters
        All other args and kwargs are forwarded to `~matplotlib.Axes.plot`.

    Returns
    -------
    lines : `~matplotlib.lines.Line2D`
        The drawn wireframe lines.

    marker : `~matplotlib.lines.Line2D`
        The drawn wireframe markers.
    """

    # TODO: Periodic mesh extension (inverse of spherical meshes culling),
    #       will enable wireframe plotting for periodic meshes.
    if descriptor.ds.is_periodic:
        err = "mosaic.wireframe does not supported for periodic meshes, yet."
        raise NotImplementedError(err)

    # insert plot format string into copy of kwargs (kwarg values prevail).
    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)

    # Plot lines and markers separately to avoid duplicate markers on shared vertices
    linestyle = kw.get("linestyle", "-")
    kw_lines = kw | {
        **kw,
        "marker": "None",  # No marker to draw.
        "zorder": kw.get("zorder", 1),  # Path default zorder is used.
    }

    if linestyle not in [None, "None", "", " "]:
        wireframe_lines_x, wireframe_lines_y = getattr(
            descriptor, f"{location.lower()}_wireframe"
        )
        wireframe_lines = ax.plot(
            wireframe_lines_x.ravel(), wireframe_lines_y.ravel(), **kw_lines
        )
    else:
        wireframe_lines = ax.plot([], [], **kw_lines)

    marker = kw.get("marker", None)
    kw_markers = {
        **kw,
        "linestyle": "None",  # No line to draw.
    }
    kw_markers.pop("label", None)

    if marker not in [None, "None", "", " "]:
        wireframe_nodes_x, wireframe_nodes_y = _geometries.compute_node_coords(
            descriptor.ds, location
        )
        wireframe_markers = ax.plot(
            wireframe_nodes_x, wireframe_nodes_y, **kw_markers
        )

    else:
        wireframe_markers = ax.plot([], [], **kw_markers)

    return wireframe_lines + wireframe_markers
