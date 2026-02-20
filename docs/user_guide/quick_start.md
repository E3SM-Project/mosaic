---
file_format: mystnb
kernelspec:
  name: python3
---

# Quick Start for Users

## Installing `mosaic`

You can install the latest version of `mosaic` from conda-forge by running:

```
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install -y mosaic
```

For a developers installation guide, see the [section](Dev_install_guide) in the developers quickstart.

## Example Usage

### Global Mesh

Then we can use `mosaic` to plot on the native mesh using `matplotlib`. For example:

```{code-cell} ipython3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt

import mosaic

# download and read the mesh from lcrc
ds = mosaic.datasets.open_dataset("QU.240km")

# define a map projection for our figure
projection = ccrs.InterruptedGoodeHomolosine()
# define the transform that describes our dataset
transform = ccrs.Geodetic()

# create the figure and a GeoAxis
fig, ax = plt.subplots(
    figsize=(9, 7),
    constrained_layout=True,
    subplot_kw={"projection": projection},
)

cmap = cmocean.tools.crop(cmocean.cm.topo, -5e3, 0, 0.0)

# create a `Descriptor` object which takes the mesh information and creates
# the polygon coordinate arrays needed for `matplotlib.collections.PolyCollection`.
descriptor = mosaic.Descriptor(ds, projection, transform)

# using the `Descriptor` object we just created, make a pseudocolor plot of
# the "indexToCellID" variable, which is defined at cell centers.
collection = mosaic.polypcolor(
    ax, descriptor, -ds.bottomDepth, antialiaseds=True, cmap=cmap
)

ax.coastlines(lw=0.5)
ax.add_feature(cfeature.LAND, fc="grey", zorder=-1, alpha=0.8)
fig.colorbar(
    collection,
    fraction=0.1,
    shrink=0.4,
    extend="both",
    label="Topography [m a.s.l.]",
);
```

For more information about how spherical meshes are handled and a list of supported
map projections, see: <project:#spherical_mesh_support>

### Planar Non-Periodic

In this case the underlying coordinate arrays (i.e. `xCell/yCell`)
correspond to a South Polar Stereographic projection, which is also the map projection we
want to us. Therefore, the `projection` and the `transform` will be equivalent
for this example. When instantiating the `mosaic.Descriptor` object we have to
be careful to set `use_latlon=False` to ensure the `xCell`/`yCell` coordinate
arrays are parsed (c.f. `lonCell`/`latCell`).

```{code-cell} ipython3
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from matplotlib import colors

import mosaic

# seconds per year
spy = 60 * 60 * 24 * 365

# download and read the mesh from lcrc
ds = mosaic.datasets.open_dataset("mpasli.AIS8to30")

# define a map projection for our figure
projection = ccrs.SouthPolarStereo()
# define the transform that describes our dataset
transform = ccrs.SouthPolarStereo()

# create the figure and a GeoAxis
fig, ax = plt.subplots(
    figsize=(7, 7),
    constrained_layout=True,
    subplot_kw={"projection": projection},
)

# create a `Descriptor` object which takes the mesh information and creates
# the polygon coordinate arrays needed for `matplotlib.collections.PolyCollection`.
descriptor = mosaic.Descriptor(ds, projection, transform, use_latlon=False)

# calculate reconstructed speed in m/s
speed_reconstruct = np.sqrt(ds.uReconstructX**2 + ds.uReconstructY**2)
# extract surface speed and convert for m/yr
speed_reconstruct = speed_reconstruct.isel(nVertInterfaces=0) * spy

# using the `Descriptor` object we just created, make a pseudocolor plot of
# the surface speed, which is defined at cell centers.
collection = mosaic.polypcolor(
    ax,
    descriptor,
    speed_reconstruct,
    antialiaseds=False,
    norm=colors.LogNorm(vmin=1, vmax=3e3),
    cmap="plasma",
)

# Because this is not a global mesh, it's necessary to explicitly set it's extent.
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

# Below is needed for a circular boundary
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)


ax.gridlines()
ax.coastlines()
fig.colorbar(collection, fraction=0.1, shrink=0.5, label="Surface Speed [m/yr]");
```

In the case where we do not know what projection the coordinate arrays of the
mesh correspond to we can use the `lonCell`/`latCell` coordinates and `mosaic`
will handle the transformation to the requested map projection under the hood.
In this scenario the `transform` should correspond to `ccrs.Geodetic()`
and `use_latlon=True` must be set in the `mosaic.Descriptor` object
instantiation. Nearly all the lines would be the same as the above example,
with the exception of the transform definition:

```python
# define the transform that describes our dataset
transform = ccrs.Geodetic()
```

and the `mosaic.Descriptor` instantiation:

```python
# ensure the `Descriptor` object use the lat/lon arrays
descriptor = mosaic.Descriptor(ds, projection, transform, use_latlon=True)
```
