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

## Automated release

`.github/workflows/release.yml` implements the release loop.

| Trigger | Build + validate | PyPI | Read the Docs |
|---|---|---|---|
| push to `main` | yes | no | `latest` |
| push tag `vX.Y.Z` | yes | yes | tag version (falls back to `latest`) |
| manual dispatch | yes | only if `publish` is checked | no |

PyPI rejects duplicate versions, so publishing is tag-driven. Every release
needs a new version.

### One-time setup

1. **PyPI Trusted Publishing** (no API token stored in GitHub):
   - Open <https://pypi.org/manage/project/tornado-gcm/settings/publishing/>
   - Add a GitHub publisher with owner `yongquanf`, repository
     `tornado-gcm`, workflow `release.yml`, and environment `pypi`.
2. **Read the Docs** (choose one):
   - Connect the GitHub integration in the RTD project dashboard, which
     builds on every push; or
   - Create an RTD API token and add it as the GitHub Actions secret
     `RTD_API_TOKEN`. The workflow then triggers the matching version.
3. **GitHub environment** `pypi` (optional): add protection rules or
   required reviewers for the publish job.

### Release a new version

```bash
# 1. Bump the version in pyproject.toml, e.g. 0.1.0 -> 0.1.1
# 2. Commit and tag
git add pyproject.toml
git commit -m "release: 0.1.1"
git tag v0.1.1
git push origin main --tags
```

The workflow then:

1. compiles the package and builds the wheel;
2. builds the sdist through `.github/scripts/build_sdist.py`;
3. runs `twine check`;
4. verifies the tag matches `pyproject.toml` and publishes through
   Trusted Publishing;
5. triggers the Read the Docs build.

## Manual build and upload

```bash
python -m build --wheel --no-isolation --outdir dist
python .github/scripts/build_sdist.py --output dist
python -m twine check dist/*
python -m twine upload --repository pypi dist/*
```

The repository root is the `tornado_gcm` package and `pyproject.toml` maps
it through `package-dir = { "" = ".." }`. A wheel builds directly from the
root; the sdist script stages a standard layout because setuptools cannot
emit files outside the project root.

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
