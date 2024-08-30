import os
import pathlib
import pooch
import xarray as xr

from mosaic.version import __version__
from xarray import Dataset

# this dictionary uses the short handed mesh names for convience, 
# but can not be parsed by pooch
registry = {
        "QU.960km": {
            "lcrc_path": "mpas_standalonedata/mpas-ocean/mesh_database/mesh.QU.960km.151026.nc", 
            "sha256_hash": "sha256:524d2d5a93851395a3bdfafb30a5acb2a3dbbecd0db07cd29e5e3f87da6eb82f"
            },

        "QU.240km": {
            "lcrc_path": "inputdata/ocn/mpas-o/oQU240/ocean.QU.240km.151209.nc",
            "sha256_hash": "sha256:a3758f88ceff3d91e86dba7922f6dd7d5672157b4793ef78214624ab8b2724ae"
            },

        "mpaso.EC30to60E2r3": {
            "lcrc_path": "inputdata/ocn/mpas-o/EC30to60E2r3/mpaso.EC30to60E2r3.230313.nc",
            "sha256_hash": "sha256:55e7cc33c890f7b9f1188bcc07fd8218b57cb1cd5ba32ee66fe3162a36995a7c"
            },

        "mpasli.AIS8to30": {
            "lcrc_path": "inputdata/glc/mpasli/mpas.ais8to30km/ais_8to30km.20221027.nc", 
            "sha256_hash": "sha256:932a1989ff8e51223413ef3ff0056d6737a1fc7f53e440359884a567a93413d2"
            }
        }

# create a parsable registry for pooch from human friendly one 
_registry = {registry[m]["lcrc_path"] : registry[m]["sha256_hash"] for m in registry} 

mesh_db = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("mosaic"),
    # The remote data is from LCRC
    base_url="https://web.lcrc.anl.gov/public/e3sm",
    version=__version__, 
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry=_registry)

# idea borrowed/copied from xarray
def open_dataset(
    name: str,
    cache: bool = True,
    **kwargs,
) -> Dataset:
    """
    Open a dataset from the lcrc database (requires internet), unless a local 
    copy is found. 
    
    Available datasets:

    * ``"QU.960km"`` : Quasi-uniform spherical mesh, with approximately 960km horizontal resolution
    * ``"QU.240km"`` : Quasi-uniform spherical mesh, with approximately 240km horizontal resolution
    * ``"mpaso.EC30to60E2r3"`` : (E)ddy-(C)losure 30 to 60 km MPAS-Ocean mesh
    * ``"mpasli.AIS8to30"`` : 8-30 km resolution planar non-periodic MALI mesh of Antarctica

    Parameters
    ----------
    name : str
        Name of the file containing the dataset. (e.g. ``"QU.960km"``)
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    kwargs : dict, optional
        Passed to xarray.open_dataset
    """

    # invalid dataset name requested
    if name not in registry:
        raise FileNotFoundError(f"Requsted dataset \"{name}\" cannot be found")
    
    # use human readable registry to find filepath on lcrc
    lcrc_path = registry[name]["lcrc_path"]
    # retrive the file using pooch
    filepath = mesh_db.fetch(lcrc_path, progressbar=True)
    
    # open dataset, and squeeze time dimensions if present
    ds = xr.open_dataset(filepath, **kwargs).squeeze()

    # if `cache==False` persist file in memory and delete the downloaded file
    if not cache: 
        ds = ds.load()
        pathlib.Path(filepath).unlink()

    return ds
