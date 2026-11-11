"""Unit tests for the automatic release version tooling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load("plan_release_version")
set_mod = _load("set_version")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("feat: add optimizer endpoint", "minor"),
        ("feat(api): add optimizer endpoint", "minor"),
        ("fix: handle empty metrics", "patch"),
        ("perf: cache snapshots", "patch"),
        ("refactor: split vllm client", "patch"),
        ("feat!: drop python 3.10", "major"),
        ("fix(parser)!: change output shape", "major"),
        ("feat: new thing\n\nBREAKING CHANGE: api removed", "major"),
        ("docs: update readme", None),
        ("chore: bump deps", None),
        ("test: add cases", None),
        ("ci: tweak workflow", None),
        ("Harden publish pipeline", None),
        ("", None),
    ],
)
def test_classify_commit(message: str, expected: str | None) -> None:
    assert plan_mod.classify_commit(message) == expected


def test_determine_bump_picks_highest() -> None:
    messages = ["fix: a", "feat: b", "docs: c"]
    assert plan_mod.determine_bump(messages) == "minor"
    assert plan_mod.determine_bump(["fix: a", "feat!: b"]) == "major"
    assert plan_mod.determine_bump(["docs: a", "chore: b"]) is None
    assert plan_mod.determine_bump([]) is None


def test_plan_major_bump() -> None:
    plan = plan_mod.plan_release("0.1.0", "v0.1.0", ["feat!: drop old api"])
    assert (plan.version, plan.bump, plan.release) == ("1.0.0", "major", True)


def test_plan_minor_bump() -> None:
    plan = plan_mod.plan_release("0.1.0", "v0.1.0", ["feat: add endpoint"])
    assert (plan.version, plan.bump, plan.release) == ("0.2.0", "minor", True)


def test_plan_patch_bump() -> None:
    plan = plan_mod.plan_release("0.1.0", "v0.1.0", ["fix: correct rates"])
    assert (plan.version, plan.bump, plan.release) == ("0.1.1", "patch", True)


def test_plan_no_release_for_docs_only() -> None:
    plan = plan_mod.plan_release("0.1.0", "v0.1.0", ["docs: readme", "ci: cache"])
    assert (plan.version, plan.bump, plan.release) == ("0.1.0", "none", False)


def test_plan_no_tag_ships_current_version() -> None:
    plan = plan_mod.plan_release("0.3.0", None, ["feat: anything"])
    assert (plan.version, plan.bump, plan.release) == ("0.3.0", "none", True)


def test_plan_manual_bump_ahead_of_tag_ships_as_is() -> None:
    plan = plan_mod.plan_release("0.2.0", "v0.1.0", ["docs: notes"])
    assert (plan.version, plan.bump, plan.release) == ("0.2.0", "none", True)


def test_plan_bumps_from_tag_when_pyproject_lags() -> None:
    plan = plan_mod.plan_release("0.1.0", "v0.2.3", ["fix: correct rates"])
    assert (plan.version, plan.bump, plan.release) == ("0.2.4", "patch", True)


def test_plan_rejects_bad_versions() -> None:
    with pytest.raises(ValueError, match="major.minor.patch"):
        plan_mod.plan_release("0.1", "v0.1.0", ["fix: a"])


def test_set_version_in_text_touches_only_version_line() -> None:
    text = '[project]\nname = "x"\nversion = "0.1.0"\ndescription = "version = \\"9.9.9\\""\n'
    updated = set_mod.set_version_in_text(text, "0.2.0")
    assert 'version = "0.2.0"' in updated
    assert 'name = "x"' in updated
    assert updated.count("0.2.0") == 1


def test_set_version_files(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## 0.1.0\n\n- First.\n", encoding="utf-8")

    set_mod.set_version("0.2.0", pyproject, changelog, "2026-01-02")

    assert 'version = "0.2.0"' in pyproject.read_text(encoding="utf-8")
    text = changelog.read_text(encoding="utf-8")
    assert "## 0.2.0 - 2026-01-02" in text
    assert "### Changed" in text
    assert text.index("## 0.2.0") < text.index("## 0.1.0")

    # Idempotent: a second run does not duplicate the section.
    set_mod.set_version("0.2.0", pyproject, changelog, "2026-01-02")
    assert changelog.read_text(encoding="utf-8").count("## 0.2.0") == 1


def test_set_version_rejects_bad_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="major.minor.patch"):
        set_mod.set_version("0.2", pyproject, tmp_path / "CHANGELOG.md", "2026-01-02")
