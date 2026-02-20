---
file_format: mystnb
kernelspec:
  name: python3
---

# Planar Periodic Meshes

For patches of a planar periodic mesh that cross a periodic boundary we correct
the patch coordinates to remove the periodicity **and** mirror the patches
across the periodic boundary. The end product of both correcting and mirroring
periodic patches is a fully periodic plot as demonstrated below:

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  code_prompt_show: Source code to generate figure below
  code_prompt_hide: Source code to generate figure below
  figure: {figure: center}
---
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
