"""Streaming preprocessing utilities for the arXiv metadata dataset."""

import argparse
import hashlib
import json
import random
import warnings

import pandas as pd

from src.config import (
    load_config,
    resolve_relative_path,
    resolve_runtime_path,
    validate_preprocessing_config,
)
from src.data.integrity import (
    SPLIT_NAMES,
    TEXT_IDENTITY_RULE,
    validate_classes,
    validate_samples,
    validate_split_fractions,
    validate_splits,
)


def iter_records(raw_path, *, decode_errors="strict", stats=None):
    """Stream UTF-8 JSONL, accepting a BOM only at the start of the file.

    Invalid UTF-8 fails with a line number by default. Explicit 'skip' discards
    the entire malformed line, counts it, and warns; bytes are never replaced.
    Blank, invalid JSON, and non-object lines are filtered and counted.
    """
    if decode_errors not in ("strict", "skip"):
        raise ValueError("decode_errors must be 'strict' or 'skip'")
    path = resolve_runtime_path(raw_path, "raw_path")
    counts = {
        "lines": 0, "blank_lines": 0, "invalid_utf8_lines": 0,
        "malformed_json_lines": 0, "non_object_lines": 0, "records": 0,
    }
    if stats is not None:
        stats.update(counts)
        counts = stats
    source_hash = hashlib.sha256()
    with path.open("rb") as raw_file:
        for line_number, raw_line in enumerate(raw_file, start=1):
            source_hash.update(raw_line)
            counts["lines"] += 1
            try:
                line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8", errors="strict")
            except UnicodeDecodeError:
                counts["invalid_utf8_lines"] += 1
                if decode_errors == "strict":
                    raise ValueError(
                        f"Invalid UTF-8 in {path.name!r} at line {line_number}; "
                        "use data.decode_errors='skip' to discard and count malformed lines"
                    ) from None
                continue
            if not line.strip():
                counts["blank_lines"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_json_lines"] += 1
                continue

            if isinstance(record, dict):
                counts["records"] += 1
                yield record
            else:
                counts["non_object_lines"] += 1
    counts["source_sha256"] = source_hash.hexdigest()
    if counts["invalid_utf8_lines"]:
        warnings.warn(
            f"Skipped {counts['invalid_utf8_lines']} invalid UTF-8 line(s) under explicit 'skip' policy",
            UserWarning, stacklevel=2,
        )


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


def collect_samples(records, classes, cap, seed, *, stats=None):
    """Seeded per-class reservoirs; a cap is a ceiling, not a balance guarantee.

    Duplicate rejection happens on the entire selected pool before splitting.
    Reservoir sampling itself neither deduplicates nor audits discarded rows.
    """
    validate_classes(classes)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("cap must be a positive integer")
    target_classes = set(classes)
    samples_by_class = {label: [] for label in classes}
    seen_by_class = {label: 0 for label in classes}
    rng = random.Random(seed)
    counts = {"non_target_records": 0, "invalid_id_records": 0, "invalid_abstract_records": 0}

    for record in records:
        label = primary_category(record)
        if label not in target_classes:
            counts["non_target_records"] += 1
            continue

        record_id = record.get("id")
        abstract = record.get("abstract")
        if not isinstance(record_id, str) or not record_id.strip():
            counts["invalid_id_records"] += 1
            continue
        if not isinstance(abstract, str):
            counts["invalid_abstract_records"] += 1
            continue

        text = clean_text(abstract)
        if not text:
            counts["invalid_abstract_records"] += 1
            continue

        sample = {"id": record_id.strip(), "text": text, "label": label}
        seen_by_class[label] += 1
        seen_count = seen_by_class[label]
        reservoir = samples_by_class[label]

        if len(reservoir) < cap:
            reservoir.append(sample)
            continue

        replacement_index = rng.randrange(seen_count)
        if replacement_index < cap:
            reservoir[replacement_index] = sample

    if stats is not None:
        stats.update(counts)
        stats["eligible_per_class"] = seen_by_class
        stats["retained_per_class"] = {label: len(rows) for label, rows in samples_by_class.items()}
    return samples_by_class


def stratified_split(samples_by_class, splits, seed):
    """Validate identities/fractions, then shuffle within each configured class.

    Train and validation counts are floors; test receives the remainder. Each
    configured class must have at least one row in every split after rounding.
    Repeated selected IDs or normalized texts fail; no deduplication is applied.
    """
    validate_split_fractions(splits)
    classes = list(samples_by_class)
    validate_classes(classes)
    validate_samples([row for rows in samples_by_class.values() for row in rows], classes)
    rng = random.Random(seed)
    split_data = {"train": [], "val": [], "test": []}

    for label, class_samples in samples_by_class.items():
        if any(row["label"] != label for row in class_samples):
            raise ValueError(f"Sample labels do not match class bucket {label!r}")
        shuffled_samples = list(class_samples)
        rng.shuffle(shuffled_samples)

        sample_count = len(shuffled_samples)
        train_end = int(sample_count * splits["train"])
        val_end = train_end + int(sample_count * splits["val"])
        counts = {"train": train_end, "val": val_end - train_end, "test": sample_count - val_end}
        if any(count <= 0 for count in counts.values()):
            raise ValueError(
                f"Class {label!r} has {sample_count} selected samples: {counts}; "
                "each class needs at least one example in train, val, and test after rounding"
            )

        split_data["train"].extend(shuffled_samples[:train_end])
        split_data["val"].extend(shuffled_samples[train_end:val_end])
        split_data["test"].extend(shuffled_samples[val_end:])

    for records in split_data.values():
        rng.shuffle(records)

    validate_splits(split_data, classes)
    return split_data


def _output_paths(processed_dir, *, include_report=False):
    filenames = {name: f"{name}.parquet" for name in SPLIT_NAMES}
    if include_report:
        filenames["report"] = "preprocessing_report.json"
    paths = {
        name: resolve_relative_path(filename, processed_dir, f"Output {name}")
        for name, filename in filenames.items()
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("Preprocessing output paths must be distinct; output links alias each other")
    return paths


def write_splits(split_data, processed_dir, classes=None):
    """Validate the full bundle before creating or overwriting any parquet file."""
    validate_splits(split_data, classes)
    output_dir = resolve_runtime_path(processed_dir, "processed_dir")
    output_paths = _output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "text", "label"]
    written_paths = []

    for split_name in SPLIT_NAMES:
        output_path = output_paths[split_name]
        dataframe = pd.DataFrame(split_data[split_name], columns=columns)
        dataframe.to_parquet(output_path, index=False)
        written_paths.append(output_path)

    return written_paths


def run_preprocessing(config=None):
    """Run the integrity-checked pipeline and save a counted provenance report."""
    if config is None:
        config = load_config()

    validate_preprocessing_config(config)
    raw_dir = resolve_runtime_path(config["paths"]["raw_data"], "paths.raw_data")
    raw_path = resolve_relative_path(config["data"]["raw_filename"], raw_dir, "data.raw_filename")
    processed_dir = resolve_runtime_path(config["paths"]["processed_data"], "paths.processed_data")
    output_paths = _output_paths(processed_dir, include_report=True)
    if raw_path in output_paths.values():
        raise ValueError("Raw source must not be a preprocessing output file")
    decode_errors = config["data"].get("decode_errors", "strict")
    input_stats = {}
    sampling_stats = {}
    samples_by_class = collect_samples(
        iter_records(raw_path, decode_errors=decode_errors, stats=input_stats),
        config["classes"],
        config["data"]["samples_per_class"],
        config["seed"],
        stats=sampling_stats,
    )
    split_data = stratified_split(
        samples_by_class,
        config["data"]["splits"],
        config["seed"],
    )
    shortfalls = {
        label: len(rows) for label, rows in samples_by_class.items()
        if len(rows) < config["data"]["samples_per_class"]
    }
    if shortfalls:
        warnings.warn(
            f"Class shortfall relative to sampling cap {config['data']['samples_per_class']}: "
            f"{shortfalls}; class counts are not guaranteed balanced",
            UserWarning, stacklevel=2,
        )
    paths = write_splits(split_data, processed_dir, config["classes"])
    report = {
        "schema_version": 1,
        "source": {
            "filename": raw_path.relative_to(raw_dir).as_posix(),
            "sha256": input_stats.pop("source_sha256"),
            "encoding": "UTF-8 (optional initial BOM)", "decode_errors": decode_errors,
        },
        "seed": config["seed"], "classes": list(config["classes"]),
        "samples_per_class_cap": config["data"]["samples_per_class"],
        "fractions": dict(config["data"]["splits"]),
        "policies": {
            "id_identity": "strip leading/trailing whitespace",
            "text_identity": TEXT_IDENTITY_RULE,
            "duplicate_policy": "reject repeated IDs or normalized text in selected pool; never drop duplicates",
            "duplicate_scope": "selected samples; unsampled source records are not deduplicated or audited",
            "rounding": "floor train and val per class; test gets remainder; every class required in every split",
            "shortfall": "warn and report below-cap counts; reject missing classes or empty class splits",
        },
        "input_counts": input_stats, "sampling_counts": sampling_stats,
        "class_shortfalls": shortfalls,
        "split_counts": {
            name: {
                "total": len(rows),
                "per_class": {
                    label: sum(row["label"] == label for row in rows)
                    for label in config["classes"]
                },
            }
            for name, rows in split_data.items()
        },
    }
    output_paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", errors="strict",
    )
    return paths


def main():
    """Run from the repository with python -m src.data.preprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="UTF-8 YAML; relative config paths are repository-relative")
    args = parser.parse_args()
    paths = run_preprocessing(load_config(args.config))
    print(f"Wrote {len(paths)} validated splits and preprocessing_report.json")


if __name__ == "__main__":
    main()
