"""Tests for the project configuration loader."""

from src.config import PROJECT_ROOT, load_config


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
