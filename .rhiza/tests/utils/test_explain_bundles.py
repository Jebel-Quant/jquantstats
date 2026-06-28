"""Unit tests for explain_bundles.py."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from test_utils import strip_ansi


def _load_module(root: Path, monkeypatch, tmp_path: Path, yaml_text: str):
    """Load explain_bundles.py against a temp project whose template-bundles.yml holds yaml_text."""
    module_path = root / ".rhiza" / "utils" / "explain_bundles.py"
    config_dir = tmp_path / ".rhiza"
    config_dir.mkdir()
    (config_dir / "template-bundles.yml").write_text(dedent(yaml_text), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    spec = importlib.util.spec_from_file_location(f"explain_bundles_{tmp_path.name}", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("github", "github"),
        ("github-tests", "github"),
        ("gitlab", "gitlab"),
        ("gitlab-book", "gitlab"),
        ("core", "base"),
    ],
)
def test_bundle_group_classifies_bundles(root, monkeypatch, tmp_path, name, expected):
    """Bundle groups should be assigned by platform prefix or default base."""
    module = _load_module(
        root,
        monkeypatch,
        tmp_path,
        """
        bundles: {}
        profiles: {}
        """,
    )

    assert module._bundle_group(name) == expected


def test_print_bundle_renders_dependencies_and_standalone_tag(root, monkeypatch, tmp_path, capsys):
    """Bundle rendering should show only the first description line and optional metadata."""
    module = _load_module(
        root,
        monkeypatch,
        tmp_path,
        """
        bundles: {}
        profiles: {}
        """,
    )
    capsys.readouterr()

    module._print_bundle(
        "github-tests",
        {
            "description": "GitHub workflows\nAdditional detail",
            "requires": ["tests"],
            "recommends": ["book"],
            "standalone": False,
        },
    )

    output = strip_ansi(capsys.readouterr().out)
    assert "github-tests            GitHub workflows  [not standalone]" in output
    assert "requires:   tests" in output
    assert "recommends: book" in output
    assert "Additional detail" not in output


def test_import_prints_grouped_bundle_and_profile_sections(root, monkeypatch, tmp_path, capsys):
    """Importing the script should render bundle and profile summaries from YAML."""
    module = _load_module(
        root,
        monkeypatch,
        tmp_path,
        """
        bundles:
          core:
            description: Core bundle
          github-tests:
            description: GitHub test workflows
          gitlab-book:
            description: GitLab docs workflow
            standalone: false
        profiles:
          local:
            description: Local development profile
            bundles: [core, github-tests]
        """,
    )

    output = strip_ansi(capsys.readouterr().out)
    assert module.groups["base"] == {"core": {"description": "Core bundle"}}
    assert "Bundles  (3 total)" in output
    assert "Core & Feature  (1)" in output
    assert "GitHub  (1)" in output
    assert "GitLab  (1)" in output
    assert "Profiles  (1 total)" in output
    assert "expands to: core, github-tests" in output


def test_import_exits_with_install_hint_when_pyyaml_is_missing(root, monkeypatch, tmp_path):
    """A missing PyYAML dependency should surface the install guidance."""
    module_path = root / ".rhiza" / "utils" / "explain_bundles.py"
    monkeypatch.chdir(tmp_path)
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):
        """Stand in for __import__, raising ImportError for ``yaml`` to simulate the missing dep."""
        if name == "yaml":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    spec = importlib.util.spec_from_file_location("explain_bundles_missing_yaml", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(SystemExit, match="pyyaml is not installed — run: make install"):
        spec.loader.exec_module(module)
