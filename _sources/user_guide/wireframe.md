---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell} ipython3
---
tags: [remove-cell]
---
%config InlineBackend.figure_formats = ['svg']
```

# Wireframes

`mosaic` can draw the connectivity of an MPAS mesh using
{func}`mosaic.wireframe`. This is useful for visualizing the mesh topology
on its own, or for layering it on top of plotted data, as shown in
[Overlaying Wireframes on Data](#overlaying-wireframes-on-data) below.

## Plotting a Wireframe

```{code-cell} ipython3
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import mosaic

# download and read a coarse, global mesh from lcrc
ds = mosaic.datasets.open_dataset("QU.960km")

projection = ccrs.Geostationary()
descriptor = mosaic.Descriptor(ds, projection)

fig, ax = plt.subplots(
    subplot_kw={"projection": projection},
)

mosaic.wireframe(ax, descriptor, color="black", linewidth=0.5);

```

## Wireframe Locations

The `location` keyword selects which part of the mesh is drawn. By default,
`location="Cell"`, which draws the boundary of each cell (i.e. the primal
mesh). Setting `location="Edge"` instead draws the boundary of the
quadrilateral region associated with each edge, and `location="Vertex"`
draws the boundary of the triangular dual mesh. Passing the `marker`
keyword additionally draws the points that define each of these lines.

```{code-cell} ipython3
fig, axs = plt.subplots(
    1, 3,
    figsize=(12, 4),
    constrained_layout=True,
    subplot_kw={"projection": projection},
)

for ax, location in zip(axs, ["Cell", "Edge", "Vertex"]):
    mosaic.wireframe(
        ax,
        descriptor,
        location,
        marker="o",
        color="black",
        linewidth=0.6,
        markersize=3,
    )
    ax.set_title(location)
```

## Overlaying Wireframes on Data

Because `mosaic.wireframe` only draws lines and markers, it can be layered on
top of an existing plot, such as one created with {func}`mosaic.polypcolor`,
to show the mesh structure alongside the data:

```{code-cell} ipython3
fig, ax = plt.subplots(
    subplot_kw={"projection": projection},
)

mosaic.polypcolor(ax, descriptor, ds.indexToCellID, cmap='tab20b')

mosaic.wireframe(
    ax,
    descriptor,
    location="Vertex",
    color="black",
    linewidth=0.6,
);
```

## Current Limitations

`mosaic.wireframe` does not yet support planar periodic meshes; calling it
with a periodic `Descriptor` raises `NotImplementedError`.
