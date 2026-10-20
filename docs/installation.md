# Installation

## Requirements

| Component | Version |
|---|---|
| Python | 3.9 – 3.11 |
| PyTorch | ≥ 2.1 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| pint | ≥ 0.22 |
| PyYAML | ≥ 6.0 |
| click | ≥ 8.1 |
| CUDA (optional) | 11.8+ |

## From PyPI

```bash
pip install tornado-gcm
```

Install the optional data and GPU extras as needed:

```bash
pip install "tornado-gcm[data]"   # xarray, zarr, netCDF4
pip install "tornado-gcm[gpu]"    # triton
```

## From source

The repository root **is** the `tornado_gcm` package, and `pyproject.toml`
maps the package to the parent directory. Clone into a directory named
exactly `tornado_gcm`:

```bash
git clone https://github.com/yongquanf/tornado-gcm.git tornado_gcm
cd tornado_gcm
pip install -e ".[dev,data,gpu]"
pre-commit install
```

!!! warning "Directory name matters"
    An editable install resolves the `tornado_gcm` package relative to the
    checkout's parent directory. If the directory is named `tornado-gcm`
    (with a hyphen), the install will not find the package. Rename it to
    `tornado_gcm`.

## Verify the installation

```bash
python -c "import tornado_gcm; print(tornado_gcm.__version__)"
python cli.py --help
```

## Build the documentation locally

```bash
pip install -r docs/requirements.txt
mkdocs serve        # http://127.0.0.1:8000
mkdocs build --strict
```
