"""Opt-in source curation: exclude all eligible duplicate-group members."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from src.config import PROJECT_ROOT, load_config, resolve_relative_path, validate_preprocessing_config
from src.data.integrity import text_identity
from src.data.preprocess import clean_text, iter_records, primary_category
from src.data.profile import file_sha256


POLICY = "exclude every eligible member of any repeated stripped-ID or normalized-text group; no keep-first, relabeling, seed changes or cap changes"


def eligible_record(record, classes, counts):
    """Use the existing pipeline's category/ID/abstract eligibility rules."""
    label = primary_category(record)
    if label not in classes:
        counts["non_target_records"] += 1
        return None
    record_id, abstract = record.get("id"), record.get("abstract")
    if not isinstance(record_id, str) or not record_id.strip():
        counts["invalid_id_records"] += 1
        return None
    if not isinstance(abstract, str) or not clean_text(abstract):
        counts["invalid_abstract_records"] += 1
        return None
    return record_id.strip(), text_identity(clean_text(abstract)), label


def curate_snapshot(config, output_dir, source_manifest, *, output_filename="arxiv-metadata-oai-snapshot.curated.jsonl"):
    """Two full passes, exact equality keys, counted union exclusion, immutable source."""
    validate_preprocessing_config(config)
    raw_dir = Path(config["paths"]["raw_data"])
    source = resolve_relative_path(config["data"]["raw_filename"], raw_dir, "raw source")
    derived = resolve_relative_path(output_filename, raw_dir, "curated output")
    temporary = resolve_relative_path(output_filename + ".partial", raw_dir, "temporary curated output")
    if len({source, derived, temporary}) != 3:
        raise ValueError("Curation source, output and temporary paths must be distinct")
    output_dir = Path(output_dir)
    report_path = resolve_relative_path("curation_report.json", output_dir, "curation report")
    runtime_path = resolve_relative_path("curation_runtime_manifest.json", output_dir, "curation runtime manifest")
    if source in {report_path, runtime_path} or derived in {report_path, runtime_path}:
        raise ValueError("Curation summaries must not overwrite source or derived data")
    exclusions_path = resolve_relative_path("curation_exclusions.jsonl", config["paths"]["processed_data"], "exclusion audit")
    if exclusions_path in {source, derived, temporary, report_path, runtime_path}:
        raise ValueError("Curation exclusion audit must be distinct from all data/summary paths")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    code_hashes = {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in (
        PROJECT_ROOT / "src/data/curate.py", PROJECT_ROOT / "src/data/integrity.py",
        PROJECT_ROOT / "src/data/preprocess.py", PROJECT_ROOT / "src/config.py",
    )}
    classes = config["classes"]
    input_counts, filtered = {}, Counter()
    eligible_per_class = {label: 0 for label in classes}
    id_counts, text_counts = Counter(), Counter()
    timings = {}
    started = perf_counter()
    for record in iter_records(source, decode_errors=config["data"].get("decode_errors", "strict"), stats=input_counts):
        identity = eligible_record(record, classes, filtered)
        if identity is None:
            continue
        record_id, key, label = identity
        id_counts[record_id] += 1
        text_counts[key] += 1
        eligible_per_class[label] += 1
    timings["identity_scan_seconds"] = perf_counter() - started
    source_sha256 = input_counts.pop("source_sha256")
    if source_manifest.get("raw_sha256_observed") is not None and source_manifest["raw_sha256_observed"] != source_sha256:
        raise ValueError("Raw source SHA-256 does not match the version-pinned source manifest")
    repeated_ids = {key for key, count in id_counts.items() if count > 1}
    repeated_text = {key for key, count in text_counts.items() if count > 1}
    id_group_count = len(repeated_ids)
    text_group_count = len(repeated_text)
    id_member_count = sum(id_counts[key] for key in repeated_ids)
    text_member_count = sum(text_counts[key] for key in repeated_text)
    # Only repeated equality keys are needed during the second source pass.
    del id_counts, text_counts
    examples = {
        "ids": {key: [] for key in sorted(repeated_ids)[:5]},
        "normalized_text": {key: [] for key in sorted(repeated_text)[:5]},
    }
    second_counts, second_filtered = {}, Counter()
    retained = {label: 0 for label in classes}
    excluded = {label: 0 for label in classes}
    exclusion_reasons = Counter({"id_only": 0, "text_only": 0, "id_and_text": 0})
    output_hash = hashlib.sha256()
    derived.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        with temporary.open("wb") as target, exclusions_path.open("w", encoding="utf-8") as exclusion_audit:
            for record in iter_records(source, decode_errors=config["data"].get("decode_errors", "strict"), stats=second_counts):
                identity = eligible_record(record, classes, second_filtered)
                if identity is None:
                    continue
                record_id, key, label = identity
                id_duplicate, text_duplicate = record_id in repeated_ids, key in repeated_text
                if id_duplicate or text_duplicate:
                    excluded[label] += 1
                    reason = "id_and_text" if id_duplicate and text_duplicate else "id_only" if id_duplicate else "text_only"
                    exclusion_reasons[reason] += 1
                    entry = {"id": record_id, "first_token_label": label, "reason": reason, "normalized_text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()}
                    exclusion_audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    for group, group_key in (("ids", record_id), ("normalized_text", key)):
                        if group_key in examples[group] and len(examples[group][group_key]) < 5:
                            examples[group][group_key].append({"id": record_id, "label": label})
                    continue
                payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                target.write(payload)
                output_hash.update(payload)
                retained[label] += 1
        if second_counts.pop("source_sha256") != source_sha256 or second_counts != input_counts or second_filtered != filtered:
            raise ValueError("Raw source changed between the two curation passes")
        temporary.replace(derived)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    timings["curation_write_seconds"] = perf_counter() - started
    report = {
        "schema_version": 1, "source_provider": source_manifest,
        "source": {"filename": source.name, "bytes": source.stat().st_size, "sha256": source_sha256},
        "derived_source": {"filename": derived.name, "bytes": derived.stat().st_size, "sha256": output_hash.hexdigest()},
        "classes": classes, "policy": POLICY,
        "scope": "all records eligible for configured first-token classes under existing ID/abstract filtering, before reservoir sampling",
        "equality": "stripped ID; full normalized string equality using src.data.integrity.text_identity (not hash-based grouping)",
        "input_counts": input_counts,
        "filtering_counts": {key: filtered[key] for key in ("non_target_records", "invalid_id_records", "invalid_abstract_records")},
        "eligible_per_class_before_curation": eligible_per_class,
        "duplicate_groups": {"ids": id_group_count, "normalized_text": text_group_count},
        "duplicate_group_member_occurrences": {"ids": id_member_count, "normalized_text": text_member_count},
        "excluded_per_class": excluded, "excluded_total": sum(excluded.values()),
        "exclusion_reasons": dict(exclusion_reasons),
        "eligible_per_class_after_curation": retained, "derived_records": sum(retained.values()),
        "duplicate_group_examples": {
            "ids": [{"key": key, "up_to_five_members": members} for key, members in examples["ids"].items()],
            "normalized_text": [{"normalized_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(), "normalized_excerpt": key[:160], "up_to_five_members": members} for key, members in examples["normalized_text"].items()],
        },
        "exclusions_audit": {"filename": exclusions_path.name, "bytes": exclusions_path.stat().st_size, "sha256": file_sha256(exclusions_path), "rows": sum(excluded.values()), "committed": False},
        "raw_source_preserved": True, "labels_changed": 0,
    }
    assert all(eligible_per_class[label] == retained[label] + excluded[label] for label in classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {"git_head_at_run": git_head, "code_sha256": code_hashes, "timings": timings,
               "timing_note": "time.perf_counter measurements for two complete source passes; no extrapolation"}
    for path, value in ((report_path, report), (runtime_path, runtime)):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-filename", default="arxiv-metadata-oai-snapshot.curated.jsonl")
    args = parser.parse_args()
    report = curate_snapshot(load_config(args.config), args.output_dir,
                             json.loads(args.source_manifest.read_text(encoding="utf-8")), output_filename=args.output_filename)
    print(json.dumps({"raw_records": report["input_counts"]["records"], "duplicate_groups": report["duplicate_groups"], "excluded_rows": report["excluded_total"], "derived_records": report["derived_records"]}, indent=2))


if __name__ == "__main__":
    main()
