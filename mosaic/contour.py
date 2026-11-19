from __future__ import annotations

from collections.abc import Iterator

import matplotlib._docstring
import matplotlib.path as mpath
import networkx as nx
import numpy as np
import shapely
from matplotlib.contour import ContourSet
from numpy.typing import ArrayLike
from shapely import Polygon, STRtree, prepare

from mosaic.descriptor import Descriptor

# Single shared docstring for both contour and contourf, mirroring the
# structure of matplotlib's contour_doc.  %(cmap_doc)s etc. are expanded
# eagerly at definition time (same trick as matplotlib/contour.py L1702).
_contour_doc_template = """
:py:func:`contour` and :py:func:`contourf` draw contour lines and filled
contours respectively on an unstructured MPAS mesh.  Except as noted,
function signatures and return values are the same for both versions.

Parameters
----------
descriptor : ~mosaic.Descriptor
    MPAS mesh descriptor containing the unstructured grid connectivity and
    coordinate arrays

Z : array-like
    Scalar field defined at cell centers over which the contour is
    drawn.  Color-mapping is controlled by *cmap*, *norm*, *vmin*, and
    *vmax*.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use :py:class:`~matplotlib.ticker.MaxNLocator`, which tries
    to automatically choose no more than *n+2* "nice" contour level
    boundaries between the minimum and maximum values of *Z*.

    If array-like, draw contour lines at the specified levels.  The
    values must be in increasing order.

    If not given, a reasonable default is chosen automatically.

Returns
-------
:py:class:`~mosaic.contour.MPASContourSet`

Other Parameters
----------------
colors : :mpltype:`color` or list of :mpltype:`color`, optional
    The colors of the levels, i.e. the lines for ``mosaic.contour`` and
    the areas for ``mosaic.contourf``.

    The sequence is cycled for the levels in ascending order. If the
    sequence is shorter than the number of levels, it's repeated.

    As a shortcut, a single color may be used in place of one-element
    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels
    with the same color.

    By default (value *None*), the colormap specified by *cmap* will be
    used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

%(cmap_doc)s

    This parameter is ignored if *colors* is set.

%(norm_doc)s

    This parameter is ignored if *colors* is set.

%(vmin_vmax_doc)s

    If *vmin* or *vmax* are not given, the default color scaling is
    based on *levels*.

    This parameter is ignored if *colors* is set.

locator : :py:class:`~matplotlib.ticker.Locator` subclass, optional
    The locator is used to determine the contour levels if they are not
    given explicitly via *levels*.  Defaults to
    `~matplotlib.ticker.MaxNLocator`.

extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    Determines the coloring of values that are outside the *levels*
    range.

    If 'neither', values outside the *levels* range are not colored.
    If 'min', 'max' or 'both', color the values below, above or below
    and above the *levels* range.

    Values below ``min(levels)`` and above ``max(levels)`` are mapped
    to the under/over values of the `.Colormap`.  Note that most
    colormaps do not have dedicated colors for these by default, so
    that the over and under values are the edge values of the colormap.
    You may want to set these values explicitly using
    `.Colormap.set_under` and `.Colormap.set_over`.

linewidths : float or array-like, default: :rc:`contour.linewidth`
    *Only applies to* :py:func:`contour`.

    The line width of the contour lines.

    If a number, all levels will be plotted with this linewidth.

    If a sequence, the levels in ascending order will be plotted with
    the linewidths in the order specified.

    If *None*, this falls back to :rc:`lines.linewidth`.

linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
    *Only applies to* :py:func:`contour`.

    If *linestyles* is *None*, the default is 'solid' unless the lines
    are monochrome.  In that case, negative contours will instead take
    their linestyle from the *negative_linestyles* argument.

    *linestyles* can also be an iterable of the above strings specifying
    a set of linestyles to be used. If this iterable is shorter than the
    number of contour levels it will be repeated as necessary.

negative_linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, \
optional
    *Only applies to* :py:func:`contour`.

    If *linestyles* is *None* and the lines are monochrome, this
    argument specifies the line style for negative contours.

    If *negative_linestyles* is *None*, the default is taken from
    :rc:`contour.negative_linestyle`.

hatches : list[str], optional
    *Only applies to* :py:func:`contourf`.

    A list of cross hatch patterns to use on the filled areas.  If
    *None*, no hatching will be added to the contour.  See
    `.Patch.set_hatch` for pattern syntax.

antialiased : bool, optional
    Enable antialiasing, overriding the defaults.  For filled contours,
    the default is *False*.  For line contours, it is taken from
    :rc:`lines.antialiased`.

Notes
-----
1. ``mosaic.contourf`` does not draw polygon edges.  To draw edges, add
   line contours with calls to `mosaic.contour`.

2. ``mosaic.contourf`` fills intervals that are closed at the top; that
   is, for boundaries *z1* and *z2*, the filled region is::

      z1 < Z <= z2

   except for the lowest interval, which is closed on both sides (i.e.
   it includes the lowest value).

3. Contouring is performed directly on the MPAS unstructured mesh by
   traversing the dual graph of cell edges, without any intermediate
   grid interpolation.
"""
_contour_doc = _contour_doc_template % matplotlib._docstring.interpd.params

matplotlib._docstring.interpd.register(mosaic_contour_doc=_contour_doc)


@matplotlib._docstring.interpd
def contour(ax, *args, **kwargs):
    """
    Plot contour lines on an unstructured MPAS mesh.

    Call signature::

        contour(ax, descriptor, Z, [levels], **kwargs)

    %(mosaic_contour_doc)s
    """
    kwargs["filled"] = False
    contours = MPASContourSet(ax, *args, **kwargs)
    ax._request_autoscale_view()
    return contours


@matplotlib._docstring.interpd
def contourf(ax, *args, **kwargs):
    """
    Plot filled contours on an unstructured MPAS mesh.

    Call signature::

        contourf(ax, descriptor, Z, [levels], **kwargs)

    %(mosaic_contour_doc)s
    """
    kwargs["filled"] = True
    contours = MPASContourSet(ax, *args, **kwargs)
    ax._request_autoscale_view()
    return contours


class MPASContourSet(ContourSet):
    """
    A :class:`matplotlib.contour.ContourSet` subclass for MPAS meshes.

    This contour set is created by :func:`mosaic.contour` and
    :func:`mosaic.contourf` to draw contour lines and filled contours on an
    unstructured MPAS mesh.

    Compared to :class:`matplotlib.contour.ContourSet`, the first
    positional argument is expected to be a
    :class:`~mosaic.descriptor.Descriptor` describing the MPAS mesh,
    and the contour generator is provided by
    :class:`MPASContourGenerator`.

    Users normally do not instantiate this class directly; instead,
    call :func:`mosaic.contour` or :func:`mosaic.contourf`.
    """

    def _process_args(self, *args, **kwargs):
        """ """
        descriptor, z, *args = args
        z = np.asarray(z)

        self.zmax = z.max().astype(float)
        self.zmin = z.min().astype(float)

        self._process_contour_level_args(args, z.dtype)

        self._contour_generator = MPASContourGenerator(descriptor, z)

        x_vertex = np.asarray(descriptor.ds.xVertex)
        y_vertex = np.asarray(descriptor.ds.yVertex)

        self._mins = [x_vertex.min(), y_vertex.min()]
        self._maxs = [x_vertex.max(), y_vertex.max()]

        return kwargs


class MPASContourGenerator:
    def __init__(self, descriptor: Descriptor, z: ArrayLike):
        loc, array = descriptor._get_array_location(z)
        if loc == "edge":
            msg = "Contour levels can not be defined on edges"
            raise ValueError(msg)

        self.ds = descriptor.ds
        self._z = np.asarray(array)

        self.boundary_edge_mask = (self.ds.cellsOnEdge < 0).any("TWO").values
        self.boundary_vertices = np.unique(
            self.ds.verticesOnEdge[self.boundary_edge_mask]
        )

    def create_filled_contour(
        self, lower_level: float, upper_level: float
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ """

        lower_level, upper_level = self.check_levels(lower_level, upper_level)

        mask = (self._z > lower_level) & (self._z <= upper_level)

        graph = self._create_contour_graph(mask, filled=True)
        polys = self._split_and_order_graph(graph)
        codes = self._assemble_contour_codes(polys)

        polys, codes = self._sort_filled_contours(polys, codes)

        return polys, codes

    def create_contour(
        self, level: float
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ """
        mask = self._z > level

        graph = self._create_contour_graph(mask, filled=False)
        lines = self._split_and_order_graph(graph)
        codes = self._assemble_contour_codes(lines)

        return lines, codes

    def _create_contour_graph(
        self, mask: np.ndarray, filled: bool
    ) -> ContourGraph:
        """ """
        ds = self.ds

        padded_mask = np.r_[False, False, mask]
        # mark mask as False for all cells outside domain
        cells_on_edge_mask = np.asarray(padded_mask[ds.cellsOnEdge + 2])

        # boolean mask for edges along contour
        edge_mask = np.logical_xor.reduce(cells_on_edge_mask, axis=1)

        if not filled:
            # unfilled contours should not follow mesh boundaries
            edge_mask = edge_mask & ~self.boundary_edge_mask

        # get the vertices
        vertex_1 = ds.verticesOnEdge[edge_mask].isel(TWO=0).values
        vertex_2 = ds.verticesOnEdge[edge_mask].isel(TWO=1).values

        return ContourGraph(vertex_1, vertex_2)

    def _create_vertex_contour_graph(
        self, mask: np.ndarray, filled: bool
    ) -> ContourGraph:
        """ """
        ds = self.ds

        padded_mask = np.r_[False, False, mask]
        # mark mask as False for all cells outside domain
        vertices_on_edge_mask = np.asarray(padded_mask[ds.verticesOnEdge + 2])

        # boolean mask for edges along contour
        edge_mask = np.logical_xor.reduce(vertices_on_edge_mask, axis=1)

        if filled:
            # filled contours should not follow mesh boundaries
            msg = "Not there yet"
            raise ValueError(msg)

        # get the vertices
        cell_1 = ds.cellsOnEdge[edge_mask].isel(TWO=0).values
        cell_2 = ds.cellsOnEdge[edge_mask].isel(TWO=1).values

        # create a graph from the boundary edges
        graph = nx.Graph()
        graph.add_edges_from(
            zip(cell_1[cell_2 != -1], cell_2[cell_2 != -1], strict=True)
        )

        self.boundary_edges = ds.nEdges[edge_mask][cell_2 == -1]
        self.boundary_cells = cell_1[cell_2 == -1]

        return graph

    def _split_and_order_graph(self, graph: ContourGraph) -> list[np.ndarray]:
        """ """

        if not graph:
            return []

        x_vertex = self.ds.xVertex.values
        y_vertex = self.ds.yVertex.values

        lines = []

        for component in graph.components():
            if len(component) == 1:
                node = next(iter(component))
                msg = f"Invalid contour component: singleton node {node}"
                raise ValueError(msg)

            # With max degree <= 2, endpoints are exactly degree-1 nodes
            endpoints = [v for v in component if graph.degree(v) == 1]

            # cycle (i.e. closed loop)
            if len(endpoints) == 0:
                contour_nodes = graph.walk(next(iter(component)))
                contour_nodes.append(contour_nodes[0])

            # path (i.e. unclosed loop)
            elif len(endpoints) == 2:
                boundary_nodes = [
                    v for v in endpoints if v in self.boundary_vertices
                ]

                if len(boundary_nodes) != 2:
                    msg = (
                        "Couldn't find start/end of contour that intersects"
                        "boundary"
                    )
                    raise ValueError(msg)

                start, _ = boundary_nodes
                contour_nodes = graph.walk(start)
            else:
                node = next(iter(component))
                msg = (
                    f"Invalid contour component: node ({node}) degree is not"
                    f"1 or 2. Instead is {len(endpoints)}"
                )
                raise ValueError(msg)

            _lines = np.stack(
                [x_vertex[contour_nodes], y_vertex[contour_nodes]], -1
            )

            lines.append(_lines)

        return lines

    def _assemble_contour_codes(
        self, contours: list[np.ndarray]
    ) -> list[np.ndarray]:
        """ """

        if len(contours) == 0:
            return []

        codes = []

        line_to = mpath.Path.LINETO
        move_to = mpath.Path.MOVETO
        code_dtype = mpath.Path.code_type

        for contour in contours:
            _codes = np.ones(len(contour), dtype=code_dtype) * line_to
            _codes[0] = move_to

            codes.append(_codes)

        return codes

    def _sort_filled_contours(
        self, polys: list[np.ndarray], codes: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ """
        polygons = list(map(Polygon, polys))
        # prepare returns None, so do not assign
        [prepare(p) for p in polygons]

        n_polygons = len(polygons)

        if n_polygons == 0:
            return polys, codes

        tree = STRtree(polygons)

        idx = tree.query(polygons, predicate="contains_properly")

        if idx.size == 0:
            return polys, codes

        all_parents, all_children = idx

        # Nesting depth: number of polygons that contain polygon i.
        depth = np.bincount(all_children, minlength=n_polygons)

        # For each child, find its direct parent: the containing polygon with
        # the greatest depth (i.e., the closest/tightest enclosing polygon).
        direct_parent = np.full(n_polygons, -1, dtype=int)
        for p, c in zip(all_parents, all_children, strict=False):
            if direct_parent[c] == -1 or depth[p] > depth[direct_parent[c]]:
                direct_parent[c] = p

        _polys = []
        _codes = []
        processed = set()

        # Even-depth polygons are exterior rings; their direct (odd-depth)
        # children are interior holes. Odd-depth polygons inside even-depth
        # holes are new exterior rings at depth+1, handled in a later iteration
        for i in range(n_polygons):
            if depth[i] % 2 == 0:
                p_ccw = _is_ccw(polygons[i])
                c_list = np.where(direct_parent == i)[0]

                strides = [
                    _stride(p_ccw, _is_ccw(polygons[j])) for j in c_list
                ]

                ext_poly = [polys[i]]
                int_polys = [
                    polys[j][::s]
                    for j, s in zip(c_list, strides, strict=False)
                ]

                ext_codes = [codes[i]]
                int_codes = [codes[j] for j in c_list]

                _polys.append(np.vstack(ext_poly + int_polys))
                _codes.append(np.hstack(ext_codes + int_codes))

                # adds exterior ring
                processed.add(i)
                # adds interior holes
                processed.update(c_list.tolist())

        # Catch any polygons not handled above
        for i in range(n_polygons):
            if i not in processed:
                _polys.append(polys[i])
                _codes.append(codes[i])

        return _polys, _codes

    def check_levels(
        self, lower_level: float, upper_level: float
    ) -> tuple[float, float]:
        if not lower_level < upper_level:
            msg = "Contour levels must be increasing"
            raise ValueError(msg)

        return lower_level, upper_level


class ContourGraph:
    """Lightweight undirected graph for MPAS contour traversal.

    Represents the set of mesh line segments that form a contour level as an
    adjacency-list graph. Each connected component is guaranteed to be either
    a path graph (an open arc whose endpoints lie on the domain boundary) or a
    cycle graph (a closed loop entirely within the domain interior). Both
    topologies have maximum node degree two, which makes full graph-library
    machinery unnecessary.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Parallel arrays of vertex IDs defining the contour edges.  Each pair
        ``(v1[i], v2[i])`` is an undirected edge.
    """

    def __init__(self, v1: np.ndarray, v2: np.ndarray) -> None:
        adj: dict[int, list[int]] = {}
        for u, v in zip(v1, v2, strict=True):
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        self._adj = adj

    def __bool__(self) -> bool:
        return bool(self._adj)

    def __iter__(self) -> Iterator[int]:
        return iter(self._adj)

    def degree(self, node: int) -> int:
        """Return the degree (number of neighbors) of *node*."""
        return len(self._adj[node])

    def components(self) -> Iterator[set[int]]:
        """Yield each connected component as a set of node IDs.

        Uses an iterative depth-first flood fill so that the call stack is
        never deeper than O(1) regardless of component size.

        Yields
        ------
        set[int]
            Node IDs belonging to one connected component.  Components are
            yielded in the order their seed node is first encountered during
            iteration over the adjacency dict.

        References
        ----------
        .. [1] "Component (graph theory)", Wikipedia,
               https://en.wikipedia.org/wiki/Component_(graph_theory)
        .. [2] "Depth-first search", Wikipedia,
               https://en.wikipedia.org/wiki/Depth-first_search
        """
        visited: set[int] = set()
        for seed in self._adj:
            if seed in visited:
                continue
            component: set[int] = set()
            stack = [seed]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(self._adj[node])
            yield component

    def walk(self, start: int) -> list[int]:
        """Return a ordered list of node IDs by traversing from *start*.

        Because every node has degree <= 2, there is at most one unvisited
        neighbor at each step, reducing traversal to a simple linear chain
        walk.  The method handles both path graphs (open chains) and cycle
        graphs (closed loops); for cycles the caller is responsible for
        appending ``path[0]`` to close the loop.

        Parameters
        ----------
        start : int
            The node ID from which to begin the walk.  For path components
            this should be one of the two degree-1 endpoints; for cycle
            components any node may be used.

        Returns
        -------
        list[int]
            Node IDs in traversal order, beginning with *start*.

        References
        ----------
        .. [1] "Path graph", Wikipedia,
               https://en.wikipedia.org/wiki/Path_graph
        .. [2] "Cycle graph", Wikipedia,
               https://en.wikipedia.org/wiki/Cycle_graph
        """
        path, seen, cur = [start], {start}, start
        while nxt := [n for n in self._adj[cur] if n not in seen]:
            cur = nxt[0]
            path.append(cur)
            seen.add(cur)
        return path

    def to_networkx(self):
        """Convert to a :class:`networkx.Graph` for testing and inspection.

        networkx is an testing dependency and is not required for normal use

        Returns
        -------
        networkx.Graph
            An undirected graph with the same nodes and edges.

        Raises
        ------
        ImportError
            If networkx is not installed.
        """
        try:
            import networkx as nx  # noqa: PLC0415
        except ImportError as e:
            msg = (
                "networkx is required to call to_networkx(). "
                "Install it with: pip install networkx"
            )
            raise ImportError(msg) from e

        g = nx.Graph()
        g.add_edges_from(
            (u, v)
            for u, neighbors in self._adj.items()
            for v in neighbors
            if u < v
        )
        return g


def _is_ccw(polygon: Polygon) -> bool:
    return shapely.is_ccw(polygon.exterior)


def _stride(is_parent_ccw: bool, is_child_ccw: bool) -> int:
    return 2 * int(is_parent_ccw != is_child_ccw) - 1
