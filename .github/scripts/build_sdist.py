#!/usr/bin/env python3
"""Build a source distribution for TornadoGCM.

The repository root is the ``tornado_gcm`` package, and ``pyproject.toml``
maps the package through ``package-dir = {"" = ".."}``. Setuptools cannot
emit files that live outside the project root, so a direct
``python -m build --sdist`` produces an archive without package code.

This script stages a standard layout (``tornado_gcm/`` inside a temporary
project root) and builds the sdist there.

Usage:
    python .github/scripts/build_sdist.py --output dist
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist", help="Output directory for the sdist.")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        # Path.resolve() on Python 3.9/Windows can leave a missing relative
        # path relative; anchor it to the invocation directory first.
        output = Path.cwd() / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Repository root: {ROOT}")
    print(f"Output directory: {output}")

    with tempfile.TemporaryDirectory(prefix="tornado_gcm_sdist_") as tmp:
        stage = Path(tmp)
        package = stage / "tornado_gcm"
        package.mkdir()

        # Top-level modules of the package live at the repository root.
        for path in ROOT.glob("*.py"):
            shutil.copy2(path, package / path.name)

        # Subpackages are directories containing __init__.py.
        for path in ROOT.iterdir():
            if path.is_dir() and (path / "__init__.py").exists():
                shutil.copytree(path, package / path.name, ignore=IGNORE)

        for name in ("README.md", "LICENSE", "LICENSE-WEIGHTS"):
            shutil.copy2(ROOT / name, stage / name)

        # Replace the repo-specific [tool.setuptools] table with standard
        # discovery relative to the staging root.
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        find_block = (
            '[tool.setuptools.packages.find]\n'
            'where = ["."]\n'
            'include = ["tornado_gcm*"]\n'
        )
        pyproject, replaced = re.subn(
            r"\[tool\.setuptools\][\s\S]*?(?=\n\[tool\.)",
            find_block,
            pyproject,
            count=1,
        )
        if replaced != 1:
            raise SystemExit("could not rewrite [tool.setuptools] in pyproject.toml")
        (stage / "pyproject.toml").write_text(pyproject, encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(output),
            ],
            cwd=stage,
            check=True,
        )
        print(f"Output files: {sorted(p.name for p in output.iterdir())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
