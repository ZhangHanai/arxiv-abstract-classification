"""Opt-in curation must exclude complete cross-class groups, not pick a winner."""

import hashlib
import json

import pytest

from src.data.curate import curate_snapshot


def source_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {"id": "keep-ai", "categories": "cs.AI cs.LG", "abstract": "Unique AI text", "title": "Preserve fields"},
        {"id": "keep-lg", "categories": "cs.LG", "abstract": "Unique LG text"},
        {"id": "text-ai", "categories": "cs.AI", "abstract": "Same\u200b text"},
        {"id": "text-lg", "categories": "cs.LG", "abstract": "SAME TEXT"},
        {"id": " repeated-id ", "categories": "cs.AI", "abstract": "First ID variant"},
        {"id": "repeated-id", "categories": "cs.LG", "abstract": "Second ID variant"},
        {"id": "non-target", "categories": "math.ST cs.AI", "abstract": "Unique AI text"},
        {"id": "invalid", "categories": "cs.LG", "abstract": None},
    ]
    source = raw / "source.json"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\nmalformed\n", encoding="utf-8")
    return {
        "seed": 42, "classes": ["cs.AI", "cs.LG"],
        "data": {"raw_filename": source.name, "samples_per_class": 7000, "splits": {"train": .8, "val": .1, "test": .1}},
        "paths": {"raw_data": raw, "processed_data": tmp_path / "processed"},
    }, rows, source


def test_curation_excludes_all_group_members_preserves_source_and_repeats(tmp_path):
    config, original_rows, source = source_fixture(tmp_path)
    before = source.read_bytes()
    report = curate_snapshot(config, tmp_path / "summaries", {})
    derived = config["paths"]["raw_data"] / report["derived_source"]["filename"]
    assert [json.loads(line) for line in derived.read_text(encoding="utf-8").splitlines()] == original_rows[:2]
    assert source.read_bytes() == before
    assert report["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert report["input_counts"]["records"] == 8
    assert report["input_counts"]["malformed_json_lines"] == 1
    assert report["duplicate_groups"] == {"ids": 1, "normalized_text": 1}
    assert report["excluded_per_class"] == {"cs.AI": 2, "cs.LG": 2}
    assert report["exclusion_reasons"] == {"id_only": 2, "text_only": 2, "id_and_text": 0}
    assert report["eligible_per_class_before_curation"] == {"cs.AI": 3, "cs.LG": 3}
    assert report["eligible_per_class_after_curation"] == {"cs.AI": 1, "cs.LG": 1}
    assert report["labels_changed"] == 0
    audit = [json.loads(line) for line in (config["paths"]["processed_data"] / "curation_exclusions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in audit} == {"text-ai", "text-lg", "repeated-id"}
    assert len(audit) == 4
    repeated = curate_snapshot(config, tmp_path / "repeated-summaries", {})
    assert report == repeated
    assert not list(tmp_path.rglob("*.partial"))


def test_curation_rejects_a_different_raw_snapshot_before_output(tmp_path):
    config, _, source = source_fixture(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        curate_snapshot(config, tmp_path / "summaries", {"raw_sha256_observed": "0" * 64})
    assert source.read_bytes() == before
    assert not (config["paths"]["raw_data"] / "arxiv-metadata-oai-snapshot.curated.jsonl").exists()


def test_curation_cannot_overwrite_source(tmp_path):
    config, _, source = source_fixture(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must be distinct"):
        curate_snapshot(config, tmp_path / "summaries", {}, output_filename=source.name)
    assert source.read_bytes() == before


def test_overlapping_id_and_text_groups_are_counted_as_a_union(tmp_path):
    config, _, source = source_fixture(tmp_path)
    rows = [
        {"id": "same-id", "categories": "cs.AI", "abstract": "same text"},
        {"id": "same-id", "categories": "cs.LG", "abstract": "different text"},
        {"id": "other-id", "categories": "cs.LG", "abstract": "SAME TEXT"},
        {"id": "retained", "categories": "cs.AI", "abstract": "unique"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = curate_snapshot(config, tmp_path / "summaries", {})
    assert report["excluded_total"] == 3
    assert report["exclusion_reasons"] == {"id_only": 1, "text_only": 1, "id_and_text": 1}
    assert report["duplicate_group_member_occurrences"] == {"ids": 2, "normalized_text": 2}
    derived = config["paths"]["raw_data"] / report["derived_source"]["filename"]
    assert [json.loads(line) for line in derived.read_text(encoding="utf-8").splitlines()] == [rows[-1]]
