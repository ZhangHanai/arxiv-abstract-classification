"""Project configuration loader."""

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

import yaml

from src.data.integrity import validate_classes, validate_split_fractions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_relative_path(value, root, name):
    """Resolve either relative separator style and enforce root containment."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty relative path")
    raw = str(value)
    # PureWindowsPath recognizes drives, drive-relative paths, UNC, and roots
    # even when this code runs on a non-Windows host.
    if Path(raw).is_absolute() or PureWindowsPath(raw).anchor:
        raise ValueError(f"{name} must be relative; absolute or drive-qualified paths are unsupported")
    root = Path(root).resolve()
    resolved = (root / raw.replace("\\", "/")).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its root directory")
    return resolved


def resolve_runtime_path(value, name):
    """Accept explicit native absolute paths, or resolve relative paths at root.

    Absolute Path values are the programmatic injection interface used by
    temporary/off-repository data callers. YAML paths use the stricter resolver.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return resolve_relative_path(value, PROJECT_ROOT, name)


def validate_preprocessing_config(config):
    """Check settings also when callers inject configuration directly."""
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping")
    validate_classes(config.get("classes"))
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    cap = data.get("samples_per_class")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("samples_per_class must be a positive integer")
    validate_split_fractions(data.get("splits"))
    if data.get("decode_errors", "strict") not in ("strict", "skip"):
        raise ValueError("data.decode_errors must be 'strict' or 'skip'")
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("paths must be a mapping")
    for name in ("raw_data", "processed_data"):
        if name not in paths:
            raise ValueError(f"paths is missing {name!r}")
        resolve_runtime_path(paths[name], f"paths.{name}")
    raw_dir = resolve_runtime_path(paths["raw_data"], "paths.raw_data")
    resolve_relative_path(data.get("raw_filename"), raw_dir, "data.raw_filename")


def load_config(config_path=None):
    """Load UTF-8 YAML; anchor relative config/data paths at the repository."""
    path = DEFAULT_CONFIG_PATH if config_path is None else resolve_runtime_path(config_path, "config_path")

    with path.open("r", encoding="utf-8-sig", errors="strict") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping")
    if not isinstance(config.get("paths"), Mapping):
        raise ValueError("paths must be a mapping")
    config["paths"] = {
        name: resolve_relative_path(relative_path, PROJECT_ROOT, f"paths.{name}")
        for name, relative_path in config["paths"].items()
    }
    validate_preprocessing_config(config)
    return config
