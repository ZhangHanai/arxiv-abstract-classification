"""Tests for the streaming arXiv metadata preprocessing pipeline."""

import json

import pandas as pd

from src.data.preprocess import (
    clean_text,
    collect_samples,
    iter_records,
    primary_category,
    run_preprocessing,
    stratified_split,
    write_splits,
)


def test_iter_records_yields_valid_json_and_skips_invalid_lines(tmp_path):
    raw_path = tmp_path / "metadata.json"
    raw_path.write_text(
        '{"id": "1", "abstract": "First", "categories": "cs.AI"}\n'
        "\n"
        "not valid json\n"
        '{"id": "2", "abstract": "Second", "categories": "cs.LG"}\n',
        encoding="utf-8",
    )

    records = list(iter_records(raw_path))

    assert [record["id"] for record in records] == ["1", "2"]


def test_primary_category_returns_first_token():
    assert primary_category({"categories": "cs.AI cs.LG"}) == "cs.AI"


def test_primary_category_returns_none_for_missing_or_empty_categories():
    assert primary_category({}) is None
    assert primary_category({"categories": ""}) is None
    assert primary_category({"categories": "   \n  "}) is None
    assert primary_category({"categories": ["cs.AI"]}) is None


def test_clean_text_collapses_and_strips_whitespace():
    assert clean_text("  First line\n\n second   line\tend  ") == (
        "First line second line end"
    )


def test_collect_samples_filters_invalid_and_non_target_records():
    records = [
        {"id": "ai-1", "abstract": " AI abstract ", "categories": "cs.AI cs.LG"},
        {"id": "lg-1", "abstract": "LG abstract", "categories": "cs.LG"},
        {"id": "math-1", "abstract": "Math", "categories": "math.CO"},
        {"abstract": "Missing ID", "categories": "cs.AI"},
        {"id": "missing-abstract", "categories": "cs.AI"},
        {"id": "blank-abstract", "abstract": " \n ", "categories": "cs.AI"},
        {"id": "missing-category", "abstract": "Text"},
    ]

    samples = collect_samples(records, ["cs.AI", "cs.LG"], cap=5, seed=42)

    assert samples == {
        "cs.AI": [{"id": "ai-1", "text": "AI abstract", "label": "cs.AI"}],
        "cs.LG": [{"id": "lg-1", "text": "LG abstract", "label": "cs.LG"}],
    }


def test_collect_samples_respects_cap_and_is_deterministic():
    records = [
        {"id": f"ai-{index}", "abstract": f"Abstract {index}", "categories": "cs.AI"}
        for index in range(20)
    ]

    first = collect_samples(iter(records), ["cs.AI"], cap=4, seed=7)
    second = collect_samples(iter(records), ["cs.AI"], cap=4, seed=7)

    assert len(first["cs.AI"]) == 4
    assert first == second


def _balanced_samples(samples_per_class=10):
    return {
        label: [
            {"id": f"{label}-{index}", "text": f"Text {index}", "label": label}
            for index in range(samples_per_class)
        ]
        for label in ("cs.AI", "cs.LG")
    }


def test_stratified_split_has_expected_sizes_no_overlap_and_is_deterministic():
    samples = _balanced_samples()
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}

    first = stratified_split(samples, splits, seed=11)
    second = stratified_split(samples, splits, seed=11)

    assert set(first) == {"train", "val", "test"}
    assert {name: len(records) for name, records in first.items()} == {
        "train": 16,
        "val": 2,
        "test": 2,
    }
    split_ids = {
        name: {record["id"] for record in records}
        for name, records in first.items()
    }
    assert split_ids["train"].isdisjoint(split_ids["val"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])
    assert first == second


def test_write_splits_creates_readable_parquet_files_with_expected_columns(tmp_path):
    split_data = {
        "train": [{"id": "1", "text": "Train", "label": "cs.AI"}],
        "val": [{"id": "2", "text": "Validation", "label": "cs.LG"}],
        "test": [{"id": "3", "text": "Test", "label": "cs.AI"}],
    }

    paths = write_splits(split_data, tmp_path / "processed")

    assert [path.name for path in paths] == [
        "train.parquet",
        "val.parquet",
        "test.parquet",
    ]
    for path in paths:
        assert path.exists()
        dataframe = pd.read_parquet(path)
        assert list(dataframe.columns) == ["id", "text", "label"]


def test_run_preprocessing_uses_injected_config_and_writes_non_empty_splits(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    raw_filename = "fake-arxiv.json"
    records = []
    for label in ("cs.AI", "cs.LG"):
        records.extend(
            {
                "id": f"{label}-{index}",
                "abstract": f"Abstract\nfor {label} paper {index}",
                "categories": f"{label} cs.CL",
            }
            for index in range(10)
        )
    raw_path = raw_dir / raw_filename
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for record in records:
            raw_file.write(json.dumps(record) + "\n")

    config = {
        "seed": 42,
        "classes": ["cs.AI", "cs.LG"],
        "data": {
            "raw_filename": raw_filename,
            "samples_per_class": 10,
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "paths": {"raw_data": raw_dir, "processed_data": processed_dir},
    }

    paths = run_preprocessing(config)

    assert paths == [
        processed_dir / "train.parquet",
        processed_dir / "val.parquet",
        processed_dir / "test.parquet",
    ]
    for path in paths:
        assert path.exists()
        assert not pd.read_parquet(path).empty
