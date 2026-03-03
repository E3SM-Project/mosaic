# Quick Start for Developers

(Dev_install_guide)=

## Developer Installation

Assuming you have a working `conda` installation, you can install the latest development version of `mosaic` by running:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -n mosaic-dev --file dev-environment.txt
conda activate mosaic-dev
python -m pip install --no-deps --no-build-isolation -e .
```

If you have an existing `conda` environment you'd like install the development version of `mosaic` in, you can run:

```
conda install --file dev-environment.txt

python -m pip install -e .
```

(dev-code-styling)=

## Code Styling and Linting

`mosaic` uses [`pre-commit`](https://pre-commit.com/) to enforce code style and
quality. After setting up your environment, run:

```bash
pre-commit install
```

This only needs to be done once per environment. `pre-commit` will
automatically check and format your code on each commit. If it makes changes,
review and re-commit as needed. Some issues (like inconsistent variable types)
may require manual fixes.

Internally, `pre-commit` uses:

- [ruff](https://docs.astral.sh/ruff/) for PEP8 compliance and import formatting,
- [flynt](https://github.com/ikamensh/flynt) to convert format strings to
  f-strings,
- [mypy](https://mypy-lang.org/) for type checking.

## Running Tests

To run the test suite:

```bash
pytest
```

Make sure all tests pass before submitting a pull request.
