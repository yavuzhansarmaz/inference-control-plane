#!/usr/bin/env python3
"""Fail if a release tag does not match the package version.

Usage: python scripts/validate_tag_version.py <git-ref>

- Non-tag refs (branch pushes, workflow_dispatch) pass immediately:
  the tag/version contract only applies to tag pushes.
- Tag refs must look like ``vX.Y.Z`` and ``X.Y.Z`` must equal both
  ``project.version`` in ``pyproject.toml`` and the runtime
  ``inference_control_plane.__version__``. The package must be
  installed first (``pip install -e .``) so ``__version__`` resolves
  from package metadata instead of the ``0.0.0+unknown`` fallback.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

TAG_RE = re.compile(r"v\d+\.\d+\.\d+")


def main(argv: list[str]) -> int:
    ref = argv[1] if len(argv) > 1 else ""
    if not ref.startswith("refs/tags/"):
        print(f"Not a tag push (ref={ref!r}); skipping tag/version check.")
        return 0

    tag = ref.removeprefix("refs/tags/")
    if not TAG_RE.fullmatch(tag):
        print(f"::error::Tag {tag!r} does not match the expected vX.Y.Z format.")
        return 1
    expected = tag[1:]

    pyproject_version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]

    from inference_control_plane import __version__ as package_version

    problems = []
    if pyproject_version != expected:
        problems.append(
            f"pyproject.toml version is {pyproject_version!r}, tag requires {expected!r}"
        )
    if package_version != expected:
        problems.append(
            f"inference_control_plane.__version__ is {package_version!r}, "
            f"tag requires {expected!r}"
        )
    if problems:
        for problem in problems:
            print(f"::error::{problem}. Refusing to publish.")
        return 1

    print(f"Tag {tag} matches pyproject.toml and __version__ ({expected}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
