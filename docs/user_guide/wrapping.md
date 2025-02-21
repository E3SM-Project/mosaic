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

## Planar Periodic Meshes

For patches of a planar periodic mesh that cross a periodic boundary we correct
the patch coordinates to remove the periodicity **and** mirror the patches
across the periodic boundary. The end product of both correcting and mirroring
periodic patches is a fully periodic plot as demonstrated below:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Source code to generate figure below"
:  code_prompt_hide: "Source code to generate figure below"
:  figure: {figure : center}

import mosaic
import matplotlib.pyplot as plt

# download and read the mesh from lcrc
ds = mosaic.datasets.open_dataset("doubly_periodic_4x4")

# create the figure 
fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True,)

descriptor = mosaic.Descriptor(ds)

pc = mosaic.polypcolor(
    ax, descriptor, ds.indexToCellID, alpha=0.6, antialiaseds=True, ec="k"
)

ax.scatter(descriptor.ds.xCell, descriptor.ds.yCell, c='k', marker='x')
ax.set_aspect('equal')
```
Periodic plotting (i.e. correcting and mirroring) of `Edge` and `Vertex` fields
is also supported. All planar periodic patches will have the same "tight" axis
limits, as defined by periods of the underlying mesh.


Future work will extend the patch mirroring functionality to spherical meshes.

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

It is important to note that planar (non-periodic) meshes are not limited to
this list of map projections and can choose from the full list of `cartopy`
[projections](https://scitools.org.uk/cartopy/docs/latest/reference/projections.html).
