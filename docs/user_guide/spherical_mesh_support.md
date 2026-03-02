# Spherical Mesh Support

We support all
[cartopy projections](https://cartopy.readthedocs.io/stable/reference/projections.html),
expect UTM, for spherical meshes.
To be able to support this wide range of projections, with very different
projection boundaries, we cull problem cells from the mesh stored internally
by the `mosaic.Descriptor`.
We determine the cells that need to be culled based on **any** of the
following three conditions being met for a given cell, after transforming
the coordinates to the requested projection:

1. All `verticesOnCell` lie outside the projection boundary
1. Any `verticesOnCell` is NaN after projection
1. If the centroid of the projected cell patch does not match the
   projected cell center position

Unless you are using a *very* coarse mesh (e.g. `QU.960`), you will most likely
not notice the results of the culling.

```{Note}
This subset of supported projections only applies to spherical meshes.
Planar (non-periodic) meshes are not limited to this list and can choose from
the full range of `cartopy`
[projections](https://scitools.org.uk/cartopy/docs/latest/reference/projections.html).
```
