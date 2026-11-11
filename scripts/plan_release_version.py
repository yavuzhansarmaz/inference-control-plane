#!/usr/bin/env python3
"""Plan the next release version from Conventional Commits.

Inputs (gathered from the checkout by default, overridable for tests):

- the current ``project.version`` in ``pyproject.toml``,
- the latest ``v*`` git tag, if any,
- the commit messages since that tag.

Output is a single line ``"<version> <bump> <release>"``, for example
``"0.2.0 minor true"`` or ``"0.1.0 none false"``. When the
``GITHUB_OUTPUT`` environment variable is set, ``version=``, ``bump=``
and ``release=`` lines are appended to that file as well.

Bump rules (Conventional Commits):

- ``BREAKING CHANGE`` / ``BREAKING-CHANGE`` anywhere in the message, or
  a ``!`` before the colon (``feat!:``, ``fix(scope)!:``) -> major
- ``feat`` -> minor
- ``fix``, ``perf``, ``refactor`` -> patch
- only ``docs``/``chore``/``test``/``ci``/other types, or messages with
  no conventional type at all -> no release

Pairproof discipline, adapted: with no tag yet, the version already in
``pyproject.toml`` is the first release. If ``pyproject.toml`` is ahead
of the latest tag (a manual bump landed on main), that version ships
as-is. If it lags the tag, the bump is computed from the tag version so
the next release is always ahead of everything already published.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
HEADER_RE = re.compile(r"^(?P<type>[A-Za-z]+)(?:\([^)]*\))?(?P<bang>!)?:")
BREAKING_RE = re.compile(r"^BREAKING[ -]CHANGE(?:\s|:|$)", re.MULTILINE)

PATCH_TYPES = frozenset({"fix", "perf", "refactor"})
LEVEL_ORDER = {"patch": 1, "minor": 2, "major": 3}


@dataclass(frozen=True)
class Plan:
    version: str
    bump: str  # "major" | "minor" | "patch" | "none"
    release: bool


def parse_version(text: str) -> tuple[int, int, int]:
    match = SEMVER_RE.match(text.strip())
    if not match:
        raise ValueError(f"not a major.minor.patch version: {text!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def bump_version(version: str, level: str) -> str:
    major, minor, patch = parse_version(version)
    if level == "major":
        return format_version((major + 1, 0, 0))
    if level == "minor":
        return format_version((major, minor + 1, 0))
    if level == "patch":
        return format_version((major, minor, patch + 1))
    raise ValueError(f"unknown bump level: {level!r}")


def classify_commit(message: str) -> str | None:
    """Return the bump level one commit message implies, or None."""
    if BREAKING_RE.search(message):
        return "major"
    first_line = message.splitlines()[0] if message.strip() else ""
    header = HEADER_RE.match(first_line.strip())
    if not header:
        return None
    if header.group("bang"):
        return "major"
    commit_type = header.group("type").lower()
    if commit_type == "feat":
        return "minor"
    if commit_type in PATCH_TYPES:
        return "patch"
    return None


def determine_bump(messages: list[str]) -> str | None:
    """Highest bump level implied by *messages*, or None for no release."""
    best: str | None = None
    for message in messages:
        level = classify_commit(message)
        if level is not None and (best is None or LEVEL_ORDER[level] > LEVEL_ORDER[best]):
            best = level
    return best


def normalize_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    cleaned = tag.strip().removeprefix("v")
    return cleaned or None


def plan_release(current: str, latest_tag: str | None, messages: list[str]) -> Plan:
    """Decide the version to release, if any. Pure and testable."""
    current_v = parse_version(current)
    latest = normalize_tag(latest_tag)
    if latest is None:
        # No release yet: the version in pyproject.toml is the first one.
        return Plan(version=format_version(current_v), bump="none", release=True)
    latest_v = parse_version(latest)
    if current_v > latest_v:
        # Manual bump already landed on main: ship it unchanged.
        return Plan(version=format_version(current_v), bump="none", release=True)
    base = format_version(latest_v)
    level = determine_bump(messages)
    if level is None:
        return Plan(version=base, bump="none", release=False)
    return Plan(version=bump_version(base, level), bump=level, release=True)


def current_version_from_pyproject(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def latest_tag_from_git() -> str | None:
    tags = _git("tag", "--list", "v*", "--sort=-v:refname")
    for line in tags.splitlines():
        if line.strip():
            return line.strip()
    return None


def commit_messages_from_git(since_tag: str | None) -> list[str]:
    rev_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
    raw = _git("log", rev_range, "--format=%B%x1e")
    return [chunk.strip() for chunk in raw.split("\x1e") if chunk.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", help="current version (default: pyproject.toml)")
    parser.add_argument("--latest-tag", help="latest v* tag (default: from git)")
    parser.add_argument(
        "--messages-file",
        help="file with commit messages separated by blank lines "
        "(default: git log since the latest tag)",
    )
    args = parser.parse_args(argv)

    try:
        current = args.current or current_version_from_pyproject(Path("pyproject.toml"))
        if args.latest_tag is not None:
            latest_tag: str | None = args.latest_tag or None
        else:
            latest_tag = latest_tag_from_git()
        if args.messages_file:
            text = Path(args.messages_file).read_text(encoding="utf-8")
            messages = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        else:
            messages = commit_messages_from_git(latest_tag)
        plan = plan_release(current, latest_tag, messages)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    release = str(plan.release).lower()
    print(f"{plan.version} {plan.bump} {release}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"version={plan.version}\n")
            handle.write(f"bump={plan.bump}\n")
            handle.write(f"release={release}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
