"""Evidence reporting must distinguish valid outputs from rejected source pools."""

import json

import pytest

from src.data.profile import length_statistics, profile_snapshot


def fixture_config(tmp_path, *, duplicate=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for label in ("cs.AI", "cs.LG"):
        for index in range(10):
            records.append({
                "id": f"{label}-{index}", "title": f"Title {label} {index}",
                "categories": f"{label} math.ST", "abstract": f"Abstract {label} unique {index}",
            })
    if duplicate:
        records[-1]["abstract"] = records[0]["abstract"].upper()
    # These are source defects/exclusions with independently known counts.
    records.extend([
        {"id": "outside", "categories": "math.ST cs.AI", "abstract": "Excluded"},
        {"id": None, "categories": "cs.AI", "abstract": "Invalid ID"},
        {"id": "invalid-text", "categories": "cs.LG", "abstract": None},
    ])
    (raw / "snapshot.json").write_text(
        "\n".join(json.dumps(row) for row in records) + "\nnot-json\n[]\n\n", encoding="utf-8",
    )
    return {
        "seed": 42, "classes": ["cs.AI", "cs.LG"],
        "data": {"raw_filename": "snapshot.json", "samples_per_class": 10, "decode_errors": "strict", "splits": {"train": .8, "val": .1, "test": .1}},
        "paths": {"raw_data": raw, "processed_data": tmp_path / "processed"},
    }


def test_profile_counts_source_defects_validates_public_bundle_and_repeats(tmp_path):
    config = fixture_config(tmp_path)
    output = tmp_path / "summaries"
    report = profile_snapshot(config, {"dataset_version": "synthetic-test"}, output)
    assert report["preprocessing"] == {"status": "passed", "public_load_splits": "passed"}
    assert report["input_counts"] == {
        "lines": 26, "blank_lines": 1, "invalid_utf8_lines": 0,
        "malformed_json_lines": 1, "non_object_lines": 1, "records": 23,
    }
    assert report["sampling_counts"]["non_target_records"] == 1
    assert report["sampling_counts"]["invalid_id_records"] == 1
    assert report["sampling_counts"]["invalid_abstract_records"] == 1
    assert report["integrity"]["total_rows"] == 20
    assert [report["integrity"]["splits"][name]["rows"] for name in ("train", "val", "test")] == [16, 2, 2]
    assert all(all(count == 0 for count in counts.values()) for counts in report["integrity"]["cross_split_overlaps"].values())
    assert report["source_audit"]["selected_first_token_label_mismatches"] == 0
    assert report["source_audit"]["selected_ids_not_found_in_source"] == 0
    assert report["reproducibility"]["preprocessing_report_identical"]
    assert all(report["reproducibility"]["ordered_rows_identical"].values())
    assert report["ready_for_baseline_data_input"] is True
    assert json.loads((output / "profile.json").read_text(encoding="utf-8")) == report
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"train", "val", "test"}
    assert all(seconds >= 0 for seconds in manifest["timings"].values())
    assert not list(config["paths"]["processed_data"].glob("repeat-*"))


def test_profile_records_duplicate_rejection_without_claiming_produced_splits(tmp_path):
    config = fixture_config(tmp_path, duplicate=True)
    report = profile_snapshot(config, {}, tmp_path / "summaries")
    assert report["preprocessing"]["status"] == "failed"
    assert "Duplicate normalized text" in report["preprocessing"]["error"]
    assert report["sampling_counts"]["retained_per_class"] == {"cs.AI": 10, "cs.LG": 10}
    assert report["selected_pool_duplicates"]["normalized_text"] == {"repeated_groups": 1, "extra_rows": 1}
    assert report["selected_pool_duplicates"]["exact_text"]["extra_rows"] == 0
    assert report["produced_split_rows"] is None
    assert report["ready_for_baseline_data_input"] is False
    assert "integrity" not in report
    assert report["reproducibility"]["status"] == "not_run"
    assert not list(tmp_path.rglob("*.parquet"))


def test_length_statistics_define_unicode_characters_and_whitespace_tokens():
    stats = length_statistics(["a b", "世界"])
    assert stats["characters"]["min"] == 2
    assert stats["characters"]["mean"] == 2.5
    assert stats["whitespace_tokens"]["mean"] == 1.5
    assert length_statistics([]) == {"characters": {"count": 0}, "whitespace_tokens": {"count": 0}}


def test_profile_missing_snapshot_does_not_manufacture_a_report(tmp_path):
    config = fixture_config(tmp_path)
    (config["paths"]["raw_data"] / "snapshot.json").unlink()
    output = tmp_path / "summaries"
    with pytest.raises(FileNotFoundError, match="Supply the documented raw snapshot"):
        profile_snapshot(config, {}, output)
    assert not output.exists()
