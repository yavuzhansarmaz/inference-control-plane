#!/usr/bin/env python3
"""Set the package version for a release.

Usage: python scripts/set_version.py X.Y.Z [--date YYYY-MM-DD]

- Rewrites only the ``version = "..."`` line of the ``[project]`` table
  in ``pyproject.toml``; nothing else in the file is touched.
- Prepends a ``## X.Y.Z - <date>`` skeleton section to ``CHANGELOG.md``
  (right after the ``# Changelog`` title) with a ``### Changed``
  placeholder, so the publish workflow's release-notes extraction has
  a section to find.

The installed package needs no manual sync: ``__version__`` in
``inference_control_plane/__init__.py`` is read from package metadata
via ``importlib.metadata.version("inference-control-plane")``, so it
follows ``pyproject.toml`` on the next install/build automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

VERSION_LINE_RE = re.compile(r'(?m)^(version\s*=\s*")[^"]*(")')
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def set_version_in_text(text: str, new_version: str) -> str:
    """Replace the first ``version = "..."`` value in *text*."""
    replaced, count = VERSION_LINE_RE.subn(rf"\g<1>{new_version}\g<2>", text, count=1)
    if count == 0:
        raise ValueError("no 'version = \"...\"' line found")
    return replaced


def changelog_section(version: str, day: str) -> str:
    return f"## {version} - {day}\n\n### Changed\n\n- TODO: describe changes in this release.\n"


def prepend_changelog_section(text: str, version: str, day: str) -> str:
    """Insert the new version section after the ``# Changelog`` title."""
    section = changelog_section(version, day)
    if f"## {version} " in text or f"## {version}\n" in text:
        return text  # section already present; stay idempotent
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("# "):
            insert_at = index + 1
            updated = lines[:insert_at] + ["\n", section, "\n"] + lines[insert_at:]
            return "".join(updated)
    return section + "\n" + text


def set_version(new_version: str, pyproject: Path, changelog: Path, day: str) -> None:
    if not SEMVER_RE.match(new_version):
        raise ValueError(f"not a major.minor.patch version: {new_version!r}")
    pyproject.write_text(
        set_version_in_text(pyproject.read_text(encoding="utf-8"), new_version),
        encoding="utf-8",
    )
    if changelog.exists():
        changelog.write_text(
            prepend_changelog_section(changelog.read_text(encoding="utf-8"), new_version, day),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    parser.add_argument("--date", default=date.today().isoformat(), help="release date")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--changelog", default="CHANGELOG.md")
    args = parser.parse_args(argv)
    try:
        set_version(args.version, Path(args.pyproject), Path(args.changelog), args.date)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Version set to {args.version} ({args.date}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
