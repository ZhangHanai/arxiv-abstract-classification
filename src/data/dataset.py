"""Utilities for loading processed dataset splits."""

from pathlib import Path

import pandas as pd

from src.config import load_config

VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"id", "text", "label"}


def build_label_maps(classes):
    """Return label_to_id and id_to_label from the configured class order."""
    label_to_id = {label: index for index, label in enumerate(classes)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def load_split(split, config=None):
    """Load processed_data/{split}.parquet as a dataframe."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}")

    config = load_config() if config is None else config
    split_path = Path(config["paths"]["processed_data"]) / f"{split}.parquet"
    dataframe = pd.read_parquet(split_path)

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Split {split!r} is missing required columns: {missing}")

    return dataframe


def validate_labels(df, classes):
    """Raise ValueError if any label is outside the configured classes."""
    valid_labels = set(classes)
    invalid_labels = [label for label in df["label"].unique() if label not in valid_labels]
    if invalid_labels:
        raise ValueError(f"Labels outside configured classes: {invalid_labels}")


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
