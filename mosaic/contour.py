from __future__ import annotations

from collections.abc import Iterator

import matplotlib.path as mpath
import numpy as np
import shapely
from matplotlib.contour import ContourSet
from numpy.typing import ArrayLike
from shapely import Polygon, STRtree, prepare

from mosaic.descriptor import Descriptor


def contour(ax, *args, **kwargs):
    """
    Plot contour lines.

    Call signature::
        contour(ax, descriptor, Z, [levels], **kwargs)
    """
    kwargs["filled"] = False
    contours = MPASContourSet(ax, *args, **kwargs)
    ax._request_autoscale_view()
    return contours


def contourf(ax, *args, **kwargs):
    """
    Plot contour lines.

    Call signature::
        contour(ax, descriptor, Z, [levels], **kwargs)
    """
    kwargs["filled"] = True
    contours = MPASContourSet(ax, *args, **kwargs)
    ax._request_autoscale_view()
    return contours


class MPASContourSet(ContourSet):
    """ """

    def _process_args(self, *args, **kwargs):
        """ """
        descriptor, z, *args = args

        self.zmax = z.max().astype(float)
        self.zmin = z.min().astype(float)

        self._process_contour_level_args(args, z.dtype)

        self._contour_generator = MPASContourGenerator(descriptor, z)

        x_vertex = descriptor.ds.xVertex
        y_vertex = descriptor.ds.yVertex

        self._mins = [x_vertex.min(), y_vertex.min()]
        self._maxs = [x_vertex.max(), y_vertex.max()]

        return kwargs


class MPASContourGenerator:
    def __init__(self, descriptor: Descriptor, z: ArrayLike):
        loc, array = descriptor._get_array_location(z)
        if loc != "cell":
            msg = f"Contour levels must be defined on cell centers, not {loc}"
            raise ValueError(msg)

        self.ds = descriptor.ds
        self._z = np.asarray(array)

        self.boundary_edge_mask = (self.ds.cellsOnEdge == -1).any("TWO").values
        self.boundary_vertices = np.unique(
            self.ds.verticesOnEdge[self.boundary_edge_mask]
        )

    def create_filled_contour(
        self, lower_level: float, upper_level: float
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ """

        lower_level, upper_level = self.check_levels(lower_level, upper_level)

        mask = (self._z > lower_level) & (self._z < upper_level)

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

        padded_mask = np.r_[False, mask]
        # mark mask as False for all cells outside domain
        cells_on_edge_mask = np.asarray(padded_mask[ds.cellsOnEdge + 1])

        # boolean mask for edges along contour
        edge_mask = np.logical_xor.reduce(cells_on_edge_mask, axis=1)

        if not filled:
            # unfilled contours should not follow mesh boundaries
            edge_mask = edge_mask & ~self.boundary_edge_mask

        # get the vertices
        vertex_1 = ds.verticesOnEdge[edge_mask].isel(TWO=0).values
        vertex_2 = ds.verticesOnEdge[edge_mask].isel(TWO=1).values

        return ContourGraph(vertex_1, vertex_2)

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
                        "Couldn't find start/end of contour that intersects "
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
        self, polys: list[np.ndarray], codes: list[int]
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
        # holes are new exterior rings at depth+1, handled in a later iteration.
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
