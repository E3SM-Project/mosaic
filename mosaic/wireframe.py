from __future__ import annotations

from matplotlib import cbook
from matplotlib import lines as mlines
from matplotlib.axes import Axes

from mosaic import _geometries
from mosaic._typing import PointType
from mosaic.descriptor import Descriptor
from mosaic.utils import _normalize_point_type


def wireframe(
    ax: Axes,
    descriptor: Descriptor,
    location: PointType = "Cell",
    **kwargs,
) -> list[mlines.Line2D]:
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
        All other kwargs are forwarded to `~matplotlib.Axes.plot`.

    Returns
    -------
    artists : list[~matplotlib.lines.Line2D]
        The wireframe line and marker artists.
    """

    # TODO: Periodic mesh extension (inverse of spherical meshes culling),
    #       will enable wireframe plotting for periodic meshes.
    if descriptor.is_periodic:
        err = "mosaic.wireframe does not support periodic meshes, yet."
        raise NotImplementedError(err)

    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    location = _normalize_point_type(location)

    # Plot lines and markers separately to avoid duplicate markers on shared vertices
    linestyle = kw.get("linestyle", "-")
    kw_lines = kw | {
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
    kw_markers = (
        {"color": wireframe_lines[0].get_color()}
        | kw
        | {
            "linestyle": "None",  # No line to draw.
        }
    )
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
