"""Streaming preprocessing utilities for the arXiv metadata dataset."""

import json
import random
from pathlib import Path

import pandas as pd

from src.config import load_config


def iter_records(raw_path):
    """Yield one parsed JSON object per line, skipping blank or malformed lines."""
    path = Path(raw_path)

    with path.open("r", encoding="utf-8") as raw_file:
        for line in raw_file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if isinstance(record, dict):
                yield record


def primary_category(record):
    """Return the first category token, or None if categories is missing or empty."""
    categories = record.get("categories")
    if not isinstance(categories, str):
        return None

    tokens = categories.split()
    return tokens[0] if tokens else None


def clean_text(text):
    """Collapse whitespace and strip the text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def collect_samples(records, classes, cap, seed):
    """Collect a balanced per-class sample using reservoir sampling."""
    target_classes = set(classes)
    samples_by_class = {label: [] for label in classes}
    seen_by_class = {label: 0 for label in classes}
    rng = random.Random(seed)

    for record in records:
        label = primary_category(record)
        if label not in target_classes:
            continue

        record_id = record.get("id")
        abstract = record.get("abstract")
        if not isinstance(record_id, str) or not record_id.strip():
            continue
        if not isinstance(abstract, str):
            continue

        text = clean_text(abstract)
        if not text:
            continue

        sample = {"id": record_id, "text": text, "label": label}
        seen_by_class[label] += 1
        seen_count = seen_by_class[label]
        reservoir = samples_by_class[label]

        if len(reservoir) < cap:
            reservoir.append(sample)
            continue

        replacement_index = rng.randrange(seen_count)
        if replacement_index < cap:
            reservoir[replacement_index] = sample

    return samples_by_class


def stratified_split(samples_by_class, splits, seed):
    """Create seeded train/val/test splits while preserving class balance."""
    rng = random.Random(seed)
    split_data = {"train": [], "val": [], "test": []}

    for class_samples in samples_by_class.values():
        shuffled_samples = list(class_samples)
        rng.shuffle(shuffled_samples)

        sample_count = len(shuffled_samples)
        train_end = int(sample_count * splits["train"])
        val_end = train_end + int(sample_count * splits["val"])

        split_data["train"].extend(shuffled_samples[:train_end])
        split_data["val"].extend(shuffled_samples[train_end:val_end])
        split_data["test"].extend(shuffled_samples[val_end:])

    for records in split_data.values():
        rng.shuffle(records)

    return split_data


def write_splits(split_data, processed_dir):
    """Write train, val, and test parquet files and return their paths."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "text", "label"]
    written_paths = []

    for split_name in ("train", "val", "test"):
        output_path = output_dir / f"{split_name}.parquet"
        dataframe = pd.DataFrame(split_data[split_name], columns=columns)
        dataframe.to_parquet(output_path, index=False)
        written_paths.append(output_path)

    return written_paths


def run_preprocessing(config=None):
    """Run the full preprocessing pipeline."""
    if config is None:
        config = load_config()

    raw_path = Path(config["paths"]["raw_data"]) / config["data"]["raw_filename"]
    processed_dir = Path(config["paths"]["processed_data"])
    samples_by_class = collect_samples(
        iter_records(raw_path),
        config["classes"],
        config["data"]["samples_per_class"],
        config["seed"],
    )
    split_data = stratified_split(
        samples_by_class,
        config["data"]["splits"],
        config["seed"],
    )
    return write_splits(split_data, processed_dir)
