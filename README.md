# Mosaic 

> :warning: **This package in under early development and not ready for production use, yet.**

`mosaic` provides the functionality to visualize unstructured mesh data on it's native grid within `matplotlib`. 
Currently `mosaic` only supports MPAS meshes, but future work will add support for other unstructured meshes used in `E3SM`.

## (Developer) Installation 

Assuming you have a working `conda` installation, you can install the latest development version of `mosaic` by running: 
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -n mosaic-dev --file dev-environment.txt
conda activate mosaic-dev
python -m pip install -e .
```

If you have an existing `conda` environment you'd like install the development version of `mosaic` in, you can run: 
```
conda install --file dev-environment.txt

python -m pip install -e .
```

## Example Usage

First we need to download a valid MPAS mesh. To do so run:
```
curl https://web.lcrc.anl.gov/public/e3sm/inputdata/ocn/mpas-o/EC30to60E2r3/mpaso.EC30to60E2r3.230313.nc -o mpaso.EC30to60E2r3.230313.nc
```

Then we can use `mosaic` to plot on the native mesh using `matplotlib`. For example:
```python

import cartopy.crs as ccrs
import mosaic
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("mpaso.EC30to60E2r3.230313.nc")

# define a map projection for our figure
projection = ccrs.InterruptedGoodeHomolosine()
# define the transform that describes our dataset
transform = ccrs.PlateCarree()

# create the figure and a GeoAxis 
fig, ax = plt.subplots(1, 1, figsize=(9,7), facecolor="w",
                       constrained_layout=True,
                       subplot_kw=dict(projection=projection))

# create a `Descriptor` object which takes the mesh information and creates 
# the polygon coordinate arrays needed for `matplotlib.collections.PolyCollection`.
descriptor = mosaic.Descriptor(ds, projection, transform)

# using the `Descriptor` object we just created, make a pseudocolor plot of
# the "indexToCellID" variable, which is defined at cell centers.
collection = mosaic.polypcolor(ax, descriptor, ds.indexToCellID, antialiaseds=False)

ax.gridlines()
ax.coastlines()
fig.colorbar(collection, fraction=0.1, label="Cell Index")

plt.show()
```
Which should produce: 
![readme](https://github.com/andrewdnolan/mosaic/assets/32367657/5716e8b5-0ee0-4a03-9c48-9cdec5a650fa)
