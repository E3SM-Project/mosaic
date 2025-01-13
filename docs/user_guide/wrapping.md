---
file_format: mystnb
kernelspec:
  name: python3
---

# Periodic Mesh Support

We currently make the simplifying assumption that spherical meshes are a
special instance of periodic meshes, which are periodic across the antimeridian.
We support both planar periodic and map projected spherical meshes, using the
same methods, by assuming that the period (in the x and y directions
respectively) is constant. This assumption is valid for planar periodic meshes
and for some map projections, but falls apart for certain map projection where
the antimeridian does not follow a strait line. Therefore we only support a
subset of `cartopy` projections, which are listed below. Future work will
develop a more elaborate method of dealing with spherical mesh periodicity,
which in turn will expand the list supported map projections.

For patches that cross a periodic boundary we simply correct the coordinates to
remove the periodicity, which enables plotting. Future work will mirror the
patches across the periodic boundary, so as to more accurately demonstrate the
periodic nature of the mesh.

<!-- 
## Planar Periodic Meshes

```{code-cell} ipython3
---
mystnb:
    remove_code_source: true
---
import mosaic
import matplotlib.pyplot as plt

# download and read the mesh from lcrc
ds = mosaic.datasets.open_dataset("doubly_periodic_4x4")

# create the figure and a GeoAxis 
fig, ax = plt.subplots(constrained_layout=True,)

descriptor = mosaic.Descriptor(ds)

pc = mosaic.polypcolor(
    ax, descriptor, ds.indexToCellID, alpha=0.8, antialiaseds=True, ec="k"
)

ax.scatter(descriptor.ds.xCell, descriptor.ds.yCell, c='k')
ax.scatter(*descriptor.cell_patches.T, c='tab:blue', marker='^')
ax.scatter(ds.xVertex, ds.yVertex, ec='tab:orange', fc='none', marker='o', s=5.)
ax.set_aspect('equal')
```
-->

## Supported Map Projections for Spherical Meshes

Currently, the only support map projection are:
- <inv:#*.PlateCarree>
- <inv:#*.LambertCylindrical>
- <inv:#*.Mercator>
- <inv:#*.Miller>
- <inv:#*.Robinson>
- <inv:#*.Stereographic>
- <inv:#*.RotatedPole>
- <inv:#*.InterruptedGoodeHomolosine>
- <inv:#*.EckertI>
- <inv:#*.EckertII>
- <inv:#*.EckertIII>
- <inv:#*.EckertIV>
- <inv:#*.EckertV>
- <inv:#*.EckertVI>
- <inv:#*.EqualEarth>
- <inv:#*.NorthPolarStereo>
- <inv:#*.SouthPolarStereo>

It is important to note that planer (non-periodic) meshes are not limited to
this list of map projections and can choose from the full list of `cartopy`
[projections](https://scitools.org.uk/cartopy/docs/latest/reference/projections.html).
