"""Utilities for loading processed dataset splits."""

import pandas as pd

from src.config import load_config, resolve_relative_path, resolve_runtime_path
from src.data.integrity import SPLIT_NAMES, validate_classes, validate_labels, validate_splits

VALID_SPLITS = set(SPLIT_NAMES)


def build_label_maps(classes):
    """Return label_to_id and id_to_label from the configured class order."""
    validate_classes(classes)
    label_to_id = {label: index for index, label in enumerate(classes)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def load_split(split, config=None):
    """Validate a split and check ID/text overlap with any existing siblings.

    This retains the single-file API. Training should use load_splits(), which
    also requires a complete train/val/test bundle and every configured class.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}")

    config = load_config() if config is None else config
    processed_dir = resolve_runtime_path(config["paths"]["processed_data"], "paths.processed_data")
    frames = {split: _read_split(processed_dir, split)}
    for name in SPLIT_NAMES:
        if name != split and (processed_dir / f"{name}.parquet").exists():
            frames[name] = _read_split(processed_dir, name)
    validate_splits(frames, config["classes"], require_all=False)
    return frames[split]


def _read_split(processed_dir, name):
    path = resolve_relative_path(f"{name}.parquet", processed_dir, f"Split {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed split {name!r}: {path.name}")
    return pd.read_parquet(path)


def load_splits(config=None):
    """Require and validate all three splits together before baseline training."""
    config = load_config() if config is None else config
    processed_dir = resolve_runtime_path(config["paths"]["processed_data"], "paths.processed_data")
    frames = {name: _read_split(processed_dir, name) for name in SPLIT_NAMES}
    validate_splits(frames, config["classes"])
    return frames


def get_texts_labels(split, config=None):
    """Return texts and string labels as aligned lists."""
    config = load_config() if config is None else config
    dataframe = load_split(split, config)
    validate_labels(dataframe, config["classes"])
    return dataframe["text"].tolist(), dataframe["label"].tolist()


def get_texts_label_ids(split, config=None):
    """Return texts and integer-encoded labels as aligned lists."""
    config = load_config() if config is None else config
    dataframe = load_split(split, config)
    validate_labels(dataframe, config["classes"])
    label_to_id, _ = build_label_maps(config["classes"])
    label_ids = [label_to_id[label] for label in dataframe["label"]]
    return dataframe["text"].tolist(), label_ids
