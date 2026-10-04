"""Tests for the streaming arXiv metadata preprocessing pipeline."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

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
            {"id": f"{label}-{index}", "text": f"Text {label} {index}", "label": label}
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
    split_texts = {name: {row["text"] for row in rows} for name, rows in first.items()}
    assert split_texts["train"].isdisjoint(split_texts["val"])
    assert split_texts["train"].isdisjoint(split_texts["test"])
    assert split_texts["val"].isdisjoint(split_texts["test"])
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


def _pipeline_fixture(tmp_path, cap=10):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = {
        "seed": 42, "classes": ["cs.AI", "cs.LG"],
        "data": {"raw_filename": "synthetic.jsonl", "samples_per_class": cap,
                 "splits": {"train": 0.8, "val": 0.1, "test": 0.1}},
        "paths": {"raw_data": raw_dir, "processed_data": tmp_path / "processed"},
    }
    records = [
        {"id": f"{label}-{i}", "abstract": f"论文 {label} abstract {i}", "categories": label}
        for label in config["classes"] for i in range(10)
    ]
    raw_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    (raw_dir / "synthetic.jsonl").write_bytes(raw_bytes)
    return config, records, raw_bytes


def test_iter_records_handles_bom_and_explicit_skip_without_replacing_text(tmp_path):
    path = tmp_path / "unicode.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + '{"id":"before", "abstract":"论文 ∇"}\r\n'.encode("utf-8")
        + b'{"id":"corrupt", "abstract":"bad\xfftext"}\n'
        + '{"id":"after", "abstract":"café"}\n'.encode("utf-8")
    )
    stats = {}
    with pytest.warns(UserWarning, match="Skipped 1 invalid UTF-8"):
        records = list(iter_records(path, decode_errors="skip", stats=stats))

    assert [row["id"] for row in records] == ["before", "after"]
    assert [row["abstract"] for row in records] == ["论文 ∇", "café"]
    assert stats["invalid_utf8_lines"] == 1
    assert stats["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_utf8_failure_has_line_number_and_creates_no_outputs(tmp_path):
    config, _, raw_bytes = _pipeline_fixture(tmp_path)
    path = config["paths"]["raw_data"] / config["data"]["raw_filename"]
    first, remaining = raw_bytes.split(b"\n", 1)
    path.write_bytes(first + b"\n\xff\n" + remaining)

    with pytest.raises(ValueError, match="Invalid UTF-8.*line 2"):
        run_preprocessing(config)

    assert not config["paths"]["processed_data"].exists()


@pytest.mark.parametrize("unsafe_filename", ["../outside.jsonl", r"..\outside.jsonl", "Z:/outside.jsonl"])
def test_injected_raw_filename_cannot_escape_raw_directory(tmp_path, unsafe_filename):
    config, _, _ = _pipeline_fixture(tmp_path)
    config["data"]["raw_filename"] = unsafe_filename

    with pytest.raises(ValueError, match=r"data.raw_filename.*(relative|escapes)"):
        run_preprocessing(config)

    assert not config["paths"]["processed_data"].exists()


@pytest.mark.parametrize("column", ["id", "abstract"])
def test_preprocessing_rejects_selected_duplicates_without_writing_data(tmp_path, column):
    config, records, _ = _pipeline_fixture(tmp_path)
    records[-1][column] = records[0][column]
    raw_path = config["paths"]["raw_data"] / config["data"]["raw_filename"]
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    message = "Duplicate ID" if column == "id" else "Duplicate normalized text"

    with pytest.raises(ValueError, match=message):
        run_preprocessing(config)

    assert not config["paths"]["processed_data"].exists()
    assert len(records) == 20


def test_write_splits_validates_before_overwriting_existing_files(tmp_path):
    rows = {
        name: [{"id": name, "text": f"{name} abstract", "label": "cs.AI"}]
        for name in ("train", "val", "test")
    }
    output = tmp_path / "processed"
    paths = write_splits(rows, output)
    before = {path: path.read_bytes() for path in paths}
    rows["test"][0]["text"] = rows["val"][0]["text"]

    with pytest.raises(ValueError, match="Duplicate normalized text"):
        write_splits(rows, output)

    assert all(path.read_bytes() == content for path, content in before.items())


def test_preprocessing_cannot_overwrite_its_own_raw_source(tmp_path):
    config, _, raw_bytes = _pipeline_fixture(tmp_path)
    raw_dir = config["paths"]["raw_data"]
    (raw_dir / config["data"]["raw_filename"]).rename(raw_dir / "train.parquet")
    config["data"]["raw_filename"] = "train.parquet"
    config["paths"]["processed_data"] = raw_dir

    with pytest.raises(ValueError, match="Raw source must not be a preprocessing output file"):
        run_preprocessing(config)

    assert (raw_dir / "train.parquet").read_bytes() == raw_bytes
    assert not (raw_dir / "val.parquet").exists()
    assert not (raw_dir / "preprocessing_report.json").exists()


def test_preprocessing_report_counts_filtering_and_is_reproducible(tmp_path):
    config, _, raw_bytes = _pipeline_fixture(tmp_path)
    config["data"]["decode_errors"] = "skip"
    extras = (
        b"\xff\nnot-json\n[]\n\n"
        + json.dumps({"id": "other", "abstract": "Other", "categories": "math.CO"}).encode() + b"\n"
        + json.dumps({"abstract": "No ID", "categories": "cs.AI"}).encode() + b"\n"
        + json.dumps({"id": "blank", "abstract": " ", "categories": "cs.AI"}).encode() + b"\n"
    )
    source_bytes = b"\xef\xbb\xbf" + raw_bytes + extras
    path = config["paths"]["raw_data"] / config["data"]["raw_filename"]
    path.write_bytes(source_bytes)
    with pytest.warns(UserWarning, match="Skipped 1 invalid UTF-8"):
        run_preprocessing(config)
    report_path = config["paths"]["processed_data"] / "preprocessing_report.json"
    first = report_path.read_bytes()
    report = json.loads(first)

    assert report["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert report["source"]["filename"] == "synthetic.jsonl"
    assert report["input_counts"] == {
        "lines": 27, "blank_lines": 1, "invalid_utf8_lines": 1,
        "malformed_json_lines": 1, "non_object_lines": 1, "records": 23,
    }
    assert report["sampling_counts"] == {
        "non_target_records": 1, "invalid_id_records": 1, "invalid_abstract_records": 1,
        "eligible_per_class": {"cs.AI": 10, "cs.LG": 10},
        "retained_per_class": {"cs.AI": 10, "cs.LG": 10},
    }
    assert report["split_counts"] == {
        "train": {"total": 16, "per_class": {"cs.AI": 8, "cs.LG": 8}},
        "val": {"total": 2, "per_class": {"cs.AI": 1, "cs.LG": 1}},
        "test": {"total": 2, "per_class": {"cs.AI": 1, "cs.LG": 1}},
    }
    assert report["class_shortfalls"] == {}
    assert "never drop duplicates" in report["policies"]["duplicate_policy"]
    with pytest.warns(UserWarning, match="Skipped 1 invalid UTF-8"):
        run_preprocessing(config)
    assert report_path.read_bytes() == first


def test_preprocessing_reports_class_shortfalls_without_claiming_balance(tmp_path):
    config, _, _ = _pipeline_fixture(tmp_path, cap=20)
    with pytest.warns(UserWarning, match="Class shortfall.*not guaranteed balanced"):
        run_preprocessing(config)
    path = config["paths"]["processed_data"] / "preprocessing_report.json"

    assert json.loads(path.read_text(encoding="utf-8"))["class_shortfalls"] == {"cs.AI": 10, "cs.LG": 10}


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_sampling_rejects_unusable_caps(cap):
    with pytest.raises(ValueError, match="cap must be a positive integer"):
        collect_samples([], ["cs.AI"], cap=cap, seed=42)


def test_preprocessing_cli_runs_with_portable_custom_config(tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(Path(__file__).resolve().parents[1] / "src", root / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "config").mkdir()
    config, _, _ = _pipeline_fixture(root)
    config["paths"] = {"raw_data": r"raw", "processed_data": r"data\processed"}
    config_path = root / "config" / "custom.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    result = subprocess.run(
        [sys.executable, "-B", "-m", "src.data.preprocess", "--config", r"config\custom.yaml"],
        cwd=root, env=environment, text=True, encoding="utf-8", errors="strict",
        capture_output=True, timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote 3 validated splits" in result.stdout
    report_path = root / "data" / "processed" / "preprocessing_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["split_counts"]["train"]["total"] == 16
