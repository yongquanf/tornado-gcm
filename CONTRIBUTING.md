# Contributing to TornadoGCM

Thank you for your interest in contributing! This document describes how to set up the development environment, code standards, and the pull-request process.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Code Style](#code-style)
3. [Testing](#testing)
4. [Pull Request Process](#pull-request-process)
5. [Issue Reporting](#issue-reporting)

---

## Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/yongquanf/tornado-gcm.git
cd tornado-gcm

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev,data,gpu]"

# 4. Install pre-commit hooks
pre-commit install
```

**Python:** 3.9–3.11. **PyTorch:** ≥ 2.1. **CUDA:** 11.8+ recommended.

---

## Code Style

We use **black** (formatter), **isort** (import ordering), and **ruff** (linter), all configured in `pyproject.toml`.

```bash
# Format a file
black tornado_gcm/my_file.py

# Check all files
ruff check tornado_gcm/

# Run all pre-commit hooks
pre-commit run --all-files
```

Key conventions:
- Line length: **100** characters.
- All public functions and classes must have docstrings (Google style).
- Type hints required on all public APIs.
- Prefer `logging` over `print` in library code.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tornado_gcm --cov-report=term-missing

# Run a specific test file
pytest tests/test_spherical_harmonic.py -v
```

New features must be accompanied by tests in `tests/`. Bug fixes should add a regression test.

---

## Pull Request Process

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, following the code-style guidelines above.
3. Ensure all tests pass and pre-commit checks are clean.
4. Write a clear PR description explaining the *what* and *why*.
5. Submit the PR to the `main` branch.
6. A maintainer will review within 5–10 business days.

**Commit message format** (conventional commits):
```
feat: add FP8 support for Z3 zone
fix: correct SHT padding for odd wavenumber truncation
docs: update quick-start example in README
```

---

## Issue Reporting

- **Bugs:** Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md).
- **Feature requests:** Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md).
- **Questions:** Open a [Discussion](https://github.com/yongquanf/tornado-gcm/discussions).

Please include your PyTorch version, CUDA version, and a minimal reproducible example when reporting bugs.

---

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
