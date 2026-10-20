# Development

## Setup

```bash
git clone https://github.com/yongquanf/tornado-gcm.git tornado_gcm
cd tornado_gcm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,data,gpu]"
pre-commit install
```

## Format and lint

```bash
pre-commit run --all-files
black . && isort . && ruff check .
```

Line length is 100. Ruff selects `E`, `F`, `W`, `I`, and `UP`; `E501` and
`F401` are intentionally ignored because black owns line length and package
`__init__` files re-export symbols.

## Smoke checks

```bash
python -m compileall -q .
python -c "import tornado_gcm; print(tornado_gcm.__version__)"
```

## Tests

The repository does not currently include a `tests/` directory. `pytest` is
configured with `testpaths = ["tests"]`; once tests are added:

```bash
pytest
pytest tests/test_file.py::test_name -v
pytest --cov=tornado_gcm --cov-report=term-missing
```

## Documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve                     # live preview at 127.0.0.1:8000
mkdocs build --strict            # exactly what Read the Docs runs
```

Read the Docs reads `.readthedocs.yaml`, which installs
`docs/requirements.txt` and builds `mkdocs.yml`.

## Build a release

```bash
python -m build --wheel --no-isolation
```

The repository root is the `tornado_gcm` package and `pyproject.toml` maps
it through `package-dir = { "" = ".." }`. A wheel builds directly from the
root. A source distribution cannot be produced directly because setuptools
cannot emit files outside the project root; build the sdist from a
standard-layout staging copy containing a `tornado_gcm/` directory.

## Upload to PyPI

```bash
python -m twine check dist/*
python -m twine upload --repository pypi dist/*
```

On Windows, twine's progress renderer can raise `UnicodeEncodeError` under
a GBK console. Run the upload with UTF-8 mode:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m twine upload --repository pypi dist/*
```

## Known limitations in 0.1.0

- `tornado-gcm train` and `tornado-gcm infer` reference library APIs that
  are not present; use the Python APIs in the [API Reference](api.md).
- `tornado_gcm.scripts.*` is referenced by the JAX-aligned builder and the
  evaluation pipeline but is not shipped in this release.
- Time units differ between paths: the JAX-aligned model uses a
  nondimensional step of about one hour, while evaluation helpers assume
  six-hour steps.
