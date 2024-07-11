# Mosaic 

> :warning: **This package in under early development and not ready for production use, yet.**

`mosaic` provides the functionality to visualize unstructured mesh data on it's native grid within `matplotlib`. 
Currently `mosaic` only supports MPAS meshes, but future work will add support for other unstructured meshes used in `E3SM`.

## (Developer) Installation 

Assuming you have a working `conda` installation, you can install the latest development version of `mosaic` by running: 
```
conda env create --file dev-environment.yml
conda activate mosaic-dev

python -m pip install -e .
```

If you have an existing `conda` environment you'd like install the development version of `mosaic` in, you can run: 
```bash
conda install --file dev-environment.yml

python -m pip install -e .
```
