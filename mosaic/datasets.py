from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

import pooch
import xarray as xr
from xarray import Dataset

# this dictionary uses the short handed mesh names for convenience,
# but can not be parsed by pooch
registry = {
    "QU.960km": {
        "lcrc_path": "mpas_standalonedata/mpas-ocean/mesh_database/mesh.QU.960km.151026.nc",
        "sha256_hash": "sha256:524d2d5a93851395a3bdfafb30a5acb2a3dbbecd0db07cd29e5e3f87da6eb82f",
    },
    "QU.240km": {
        "lcrc_path": "inputdata/ocn/mpas-o/oQU240/ocean.QU.240km.151209.nc",
        "sha256_hash": "sha256:a3758f88ceff3d91e86dba7922f6dd7d5672157b4793ef78214624ab8b2724ae",
    },
    "mpaso.IcoswISC30E3r5": {
        "lcrc_path": "inputdata/share/meshes/mpas/ocean/IcoswISC30E3r5.20231120.nc",
        "sha256_hash": "sha256:970afa546aa1e219b45234ae0bc8f5d4f41849cb42e6292abfd75ac89a561faf",
    },
    "mpasli.AIS8to30": {
        "lcrc_path": "inputdata/glc/mpasli/mpas.ais8to30km/ais_8to30km.20221027.nc",
        "sha256_hash": "sha256:932a1989ff8e51223413ef3ff0056d6737a1fc7f53e440359884a567a93413d2",
    },
    "doubly_periodic_4x4": {
        "lcrc_path": "mpas_standalonedata/mpas-ocean/mesh_database/doubly_periodic_1920km_7680x7680km.151124.nc",
        "sha256_hash": "sha256:5409d760845fb682ec56e30d9c6aa6dfe16b5d0e0e74f5da989cdaddbf4303c7",
    },
}

# create a parsable registry for pooch from human friendly one
_registry = {
    registry[m]["lcrc_path"]: registry[m]["sha256_hash"] for m in registry
}

mesh_db = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("mosaic"),
    # The remote data is from LCRC
    base_url="https://web.lcrc.anl.gov/public/e3sm",
    registry=_registry,
)


# idea borrowed/copied from xarray
def open_dataset(
    name: str,
    cache_dir: str | os.PathLike | None = None,
    cache: bool = True,
    **kwargs,
) -> Dataset:
    """
    Open a dataset from the lcrc database (requires internet), unless a local
    copy is found.

    Available datasets:

    * ``"QU.960km"`` : Quasi-uniform spherical mesh, with approximately 960km horizontal resolution
    * ``"QU.240km"`` : Quasi-uniform spherical mesh, with approximately 240km horizontal resolution
    * ``"mpaso.IcoswISC30E3r5"`` : Icosahedral 30 km MPAS-Ocean mesh with ice shelf cavities
    * ``"mpasli.AIS8to30"`` : 8-30 km resolution planar non-periodic MALI mesh of Antarctica
    * ``"doubly_periodic_4x4"``: Doubly periodic planar mesh that is four cells wide in both the x and y dimensions.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset. (e.g. ``"QU.960km"``)
    cache_dir : path-like, optional
        The directory in which to search for and write cached data
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    """

    # invalid dataset name requested
    if name not in registry:
        msg = f'Requsted dataset "{name}" cannot be found'
        raise FileNotFoundError(msg)

    # use human readable registry to find filepath on lcrc
    lcrc_path = registry[name]["lcrc_path"]

    if cache_dir:
        # join the url paths together for the requested file
        url = urljoin(mesh_db.base_url, lcrc_path)
        # get the known hash of the file
        known_hash = registry[name]["sha256_hash"]
        # retrieve the file using pooch, and cache in custom location
        filepath = pooch.retrieve(
            url=url, known_hash=known_hash, path=cache_dir, progressbar=True
        )

    else:
        # retrieve the file using pooch
        filepath = mesh_db.fetch(lcrc_path, progressbar=True)

    # open dataset, and squeeze time dimensions if present
    ds = xr.open_dataset(filepath, **kwargs).squeeze()

    # if `cache==False` persist file in memory and delete the downloaded file
    if not cache:
        ds = ds.load()
        Path(filepath).unlink()

    return ds
