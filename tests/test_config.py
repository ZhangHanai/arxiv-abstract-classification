"""Tests for the project configuration loader."""

import os
import subprocess

import pytest
import yaml

import src.config as config_module
from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, resolve_relative_path


def test_config_loads_expected_values():
    config = load_config()

    assert config["seed"] == 42
    assert len(config["classes"]) == 8
    assert config["data"]["max_seq_length"] == 256


def test_resolved_paths_are_absolute():
    config = load_config()

    for resolved_path in config["paths"].values():
        assert resolved_path.is_absolute()


def test_resolved_paths_stay_inside_project():
    config = load_config()

    for resolved_path in config["paths"].values():
        assert PROJECT_ROOT in resolved_path.parents


def test_resolved_paths_point_to_expected_subdirs():
    config = load_config()

    assert config["paths"]["raw_data"] == (PROJECT_ROOT / "data" / "raw").resolve()
    assert config["paths"]["processed_data"] == (PROJECT_ROOT / "data" / "processed").resolve()
    assert config["paths"]["figures"] == (PROJECT_ROOT / "results" / "figures").resolve()
    assert config["paths"]["metrics"] == (PROJECT_ROOT / "results" / "metrics").resolve()


def _raw_config():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "unsafe_path",
    ["/outside", "Z:/outside", "Z:outside", r"\\server\share\raw", r"\outside",
     "../outside", r"..\outside", ""],
)
def test_config_rejects_absolute_drive_unc_and_escaping_paths(tmp_path, unsafe_path):
    config = _raw_config()
    config["paths"]["raw_data"] = unsafe_path
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"paths.raw_data.*(relative|escapes|path)"):
        load_config(path)


@pytest.mark.parametrize("relative_config", ["config/custom.yaml", r"config\custom.yaml"])
def test_custom_config_and_paths_are_portable_and_independent_of_cwd(
    tmp_path, monkeypatch, relative_config,
):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    config = _raw_config()
    config["paths"]["raw_data"] = r"data\raw"
    config["paths"]["processed_data"] = "data/processed"
    config["data"]["raw_filename"] = r"snapshot\metadata.jsonl"
    (root / "config" / "custom.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.chdir(external)

    loaded = load_config(relative_config)

    assert loaded["paths"]["raw_data"] == root / "data" / "raw"
    assert loaded["paths"]["processed_data"] == root / "data" / "processed"
    assert resolve_relative_path(loaded["data"]["raw_filename"], loaded["paths"]["raw_data"], "raw") == (
        root / "data" / "raw" / "snapshot" / "metadata.jsonl"
    )


def test_yaml_accepts_utf8_bom_and_preserves_unicode(tmp_path):
    config = _raw_config()
    config["paths"]["raw_data"] = "data/论文"
    path = tmp_path / "unicode.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8-sig")

    assert load_config(path)["paths"]["raw_data"] == PROJECT_ROOT / "data" / "论文"


@pytest.mark.parametrize("contents", ["", "[]", "paths: []"])
def test_malformed_config_structure_fails_clearly(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="(Configuration|paths) must be a mapping"):
        load_config(path)


def test_resolved_symlink_or_junction_cannot_escape_path_root(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("Host does not permit creating directory symlinks")
        # Windows directory junctions need no symlink privilege. Pass literal
        # paths through environment variables, never interpolate shell code.
        environment = dict(os.environ, INTEGRITY_LINK_PATH=str(link), INTEGRITY_LINK_TARGET=str(outside))
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "New-Item -ItemType Junction -Path $env:INTEGRITY_LINK_PATH "
             "-Target $env:INTEGRITY_LINK_TARGET -ErrorAction Stop | Out-Null"],
            env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        assert result.returncode == 0, result.stderr

    with pytest.raises(ValueError, match="escapes"):
        resolve_relative_path("linked/data", root, "paths.raw_data")
