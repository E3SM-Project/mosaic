# Quick Start


## User Installation

You can install the latest version of `mosaic` from conda-forge by running:
```
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install -y mosaic
```

## Developer Installation

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

