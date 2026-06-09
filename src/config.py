"""Project configuration loader."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path=None):
    """Load project config and resolve configured paths from the repo root."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["paths"] = {
        name: (PROJECT_ROOT / relative_path).resolve()
        for name, relative_path in config["paths"].items()
    }

    return config
