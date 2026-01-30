from __future__ import annotations

import matplotlib.path as mpath
import networkx as nx
import numpy as np
import shapely
from matplotlib.contour import ContourSet
from shapely import Polygon, STRtree, prepare


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

        xCell = descriptor.ds.xCell
        yCell = descriptor.ds.yCell

        self._mins = [xCell.min(), yCell.min()]
        self._maxs = [xCell.max(), yCell.max()]

        return kwargs


class MPASContourGenerator:
    def __init__(self, descriptor, z):
        self.ds = descriptor.ds
        self._z = z

        self.boundary_edge_mask = (self.ds.cellsOnEdge == -1).any("TWO").values
        self.boundary_vertices = np.unique(
            self.ds.verticesOnEdge[self.boundary_edge_mask]
        )

    def create_filled_contour(self, lower_level: float, upper_level: float):
        """ """

        lower_level, upper_level = self.check_levels(lower_level, upper_level)

        mask = (self._z > lower_level) & (self._z < upper_level)

        graph = self._create_contour_graph(mask, filled=True)
        polys = self._split_and_order_graph(graph)
        codes = self._assemble_contour_codes(polys)

        polys, codes = self._sort_filled_contours(polys, codes)

        return polys, codes

    def create_contour(self, level: float):
        """ """
        mask = self._z > level

        graph = self._create_contour_graph(mask, filled=False)
        lines = self._split_and_order_graph(graph)
        codes = self._assemble_contour_codes(lines)

        return lines, codes

    def _create_contour_graph(self, mask, filled: bool):
        """ """
        ds = self.ds

        # mark mask as False for all cells outside domain
        cells_on_edge_mask = np.where(
            ds.cellsOnEdge == -1, False, mask[ds.cellsOnEdge]
        )

        # boolean mask for edges along contour
        edge_mask = np.logical_xor.reduce(cells_on_edge_mask, axis=1)

        if not filled:
            # unfilled contours should not follow mesh boundaries
            edge_mask = edge_mask & ~self.boundary_edge_mask

        # get the vertices
        vertex_1 = ds.verticesOnEdge[edge_mask].isel(TWO=0).values
        vertex_2 = ds.verticesOnEdge[edge_mask].isel(TWO=1).values

        # create a graph from the boundary edges
        graph = nx.Graph()
        graph.add_edges_from(zip(vertex_1, vertex_2, strict=False))

        return graph

    def _split_and_order_graph(self, graph):
        """ """
        ds = self.ds

        # empty lists where we'll store output to be returned
        lines = []

        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)

            # What about try nx.eulerian_path
            # pad nodes only if closed (nx.is_tree is False)
            # also accept NetworkX Error is euler path does not exists
            if nx.is_eulerian(subgraph):
                contour_nodes = np.array(
                    [u for u, v in nx.eulerian_circuit(subgraph)]
                )

                # need to close the paths
                contour_nodes = np.pad(contour_nodes, (0, 1), mode="wrap")

            else:
                nodes = list(subgraph.nodes)

                (idxs,) = np.where(np.isin(nodes, self.boundary_vertices))

                if len(idxs) != 2:
                    msg = (
                        "Couldn't find start/end of contour that intersects "
                        "boundary"
                    )
                    raise ValueError(msg)

                start, end = idxs

                paths = list(
                    nx.all_simple_paths(subgraph, nodes[start], nodes[end])
                )

                if len(list(paths)) != 1:
                    msg = "More than one path through boundary graph found"
                    raise ValueError(msg)

                contour_nodes = paths[0]

            _lines = np.stack(
                [ds.xVertex[contour_nodes], ds.yVertex[contour_nodes]], -1
            )

            lines.append(_lines)

        return lines

    def _assemble_contour_codes(self, contours: list):
        """ """
        codes = []

        line_to = mpath.Path.LINETO
        move_to = mpath.Path.MOVETO
        code_dtype = mpath.Path.code_type

        for contour in contours:
            _codes = np.ones(len(contour), dtype=code_dtype) * line_to
            _codes[0] = move_to

            codes.append(_codes)

        return codes

    def _sort_filled_contours(self, polys, codes):
        """ """
        polygons = list(map(Polygon, polys))
        # prepare returns None, so do not assign
        [prepare(p) for p in polygons]

        n_polygons = len(polygons)

        # create a tree...
        tree = STRtree(polygons)

        idx = tree.query(polygons, predicate="contains_properly")

        # TODO: Add condition for empty idx
        parents, children = idx

        # how many polygons contain i-th polygon
        depth = np.bincount(children, minlength=n_polygons)

        if any(depth > 1):
            msg = "Cannot Properly Handle Nested Interior Boundaries Yet"
            raise ValueError(msg)

        _polys = []
        _codes = []

        for p in set(parents):
            c = children[parents == p]

            p_ccw = _is_ccw(polygons[p])
            strides = [_stride(p_ccw, _is_ccw(polygons[i])) for i in c]

            ext_poly = [polys[p]]
            int_polys = [
                polys[i][::s] for i, s in zip(c, strides, strict=False)
            ]

            ext_codes = [codes[p]]
            int_codes = [codes[i] for i in c]

            _polys.append(np.vstack(ext_poly + int_polys))
            _codes.append(np.hstack(ext_codes + int_codes))

        others_set = set(np.concat([parents, children]))
        others_idx = others_set.symmetric_difference(set(range(n_polygons)))

        _polys.extend([polys[i] for i in others_idx])
        _codes.extend([codes[i] for i in others_idx])

        return _polys, _codes

    def check_levels(self, upper_level, lower_level):
        """ """
        # TODO: checked filled are monotonic
        # TODO: check no nan's

        return upper_level, lower_level


def _is_ccw(polygon) -> bool:
    return shapely.is_ccw(polygon.exterior)


def _stride(is_parent_ccw: bool, is_child_ccw: bool) -> int:
    return 2 * int(is_parent_ccw != is_child_ccw) - 1
