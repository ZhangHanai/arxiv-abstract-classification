"""Profile a real snapshot through public preprocessing/loading APIs; no training."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_relative_path
from src.data.dataset import load_splits
from src.data.integrity import SPLIT_NAMES, text_identity, validate_samples
from src.data.preprocess import clean_text, collect_samples, iter_records, primary_category, run_preprocessing


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_statistics(texts):
    """Observed cleaned-text lengths; words mean whitespace-delimited tokens."""
    def describe(values):
        series = pd.Series(values, dtype="int64")
        if series.empty:
            return {"count": 0}
        return {
            "count": len(series), "min": int(series.min()), "max": int(series.max()),
            "mean": float(series.mean()), "std_population": float(series.std(ddof=0)),
            "p05": float(series.quantile(.05)), "median": float(series.quantile(.5)),
            "p95": float(series.quantile(.95)), "p99": float(series.quantile(.99)),
        }
    return {
        "characters": describe([len(text) for text in texts]),
        "whitespace_tokens": describe([len(text.split()) for text in texts]),
    }


def duplicate_counts(rows):
    """Count repeated groups and rows beyond the first; do not remove rows."""
    def count(values):
        groups = Counter(values)
        return {
            "repeated_groups": sum(n > 1 for n in groups.values()),
            "extra_rows": sum(n - 1 for n in groups.values() if n > 1),
        }
    return {
        "ids": count(row["id"].strip() for row in rows),
        "exact_text": count(row["text"] for row in rows),
        "normalized_text": count(text_identity(row["text"]) for row in rows),
    }


def audit_frames(frames, config):
    """Independently count defects and verify the documented per-class floors."""
    classes = config["classes"]
    per_split = {}
    identities = {}
    for name, frame in frames.items():
        rows = frame.to_dict("records")
        per_split[name] = {
            "rows": len(rows),
            "per_class": {label: int((frame["label"] == label).sum()) for label in classes},
            "unknown_labels": int((~frame["label"].isin(classes)).sum()),
            "null_required_values": int(frame[["id", "text", "label"]].isna().sum().sum()),
            "invalid_required_values": sum(
                not isinstance(row[column], str) or not row[column].strip()
                for row in rows for column in ("id", "text", "label")
            ),
            "duplicates": duplicate_counts(rows),
        }
        identities[name] = {
            "ids": {row["id"].strip() for row in rows},
            "exact_text": {row["text"] for row in rows},
            "normalized_text": {text_identity(row["text"]) for row in rows},
        }
    overlaps = {
        f"{left}/{right}": {
            key: len(identities[left][key] & identities[right][key])
            for key in ("ids", "exact_text", "normalized_text")
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    totals = {
        label: sum(per_split[name]["per_class"][label] for name in SPLIT_NAMES)
        for label in classes
    }
    expected = {
        label: {
            "train": int(totals[label] * config["data"]["splits"]["train"]),
            "val": int(totals[label] * config["data"]["splits"]["val"]),
        }
        for label in classes
    }
    for label in classes:
        expected[label]["test"] = totals[label] - sum(expected[label].values())
    checks_passed = all(
        values["unknown_labels"] == values["null_required_values"] == values["invalid_required_values"] == 0
        and all(group["extra_rows"] == 0 for group in values["duplicates"].values())
        for values in per_split.values()
    ) and all(count == 0 for values in overlaps.values() for count in values.values())
    return {
        "splits": per_split, "cross_split_overlaps": overlaps,
        "independent_row_and_overlap_checks_passed": checks_passed,
        "per_class_total": totals, "total_rows": sum(len(frame) for frame in frames.values()),
        "expected_per_class_split_counts": expected,
        "split_rounding_matches": all(
            expected[label][name] == per_split[name]["per_class"][label]
            for label in classes for name in SPLIT_NAMES
        ),
        "every_class_in_every_split": all(
            per_split[name]["per_class"][label] > 0 for label in classes for name in SPLIT_NAMES
        ),
    }


def logical_digest(frame):
    """Hash ordered rows independently of parquet encoders and metadata."""
    digest = hashlib.sha256()
    for row in frame[["id", "text", "label"]].itertuples(index=False, name=None):
        digest.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def profile_snapshot(config, source_manifest, output_dir, *, repeat=True):
    """Persist observed evidence even when the real-data integrity gate rejects it."""
    output_dir = Path(output_dir)
    raw_path = resolve_relative_path(config["data"]["raw_filename"], config["paths"]["raw_data"], "raw filename")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Supply the documented raw snapshot first: {raw_path.name}")
    summary_paths = [resolve_relative_path(name, output_dir, "summary output") for name in ("profile.json", "run_manifest.json")]
    if raw_path in summary_paths or len(set(summary_paths)) != len(summary_paths):
        raise ValueError("Summary outputs must be distinct and must not overwrite the raw source")
    timings = {}
    preprocessing = {"status": "failed"}
    frames = None
    started = perf_counter()
    try:
        paths = run_preprocessing(config)
    except (ValueError, UnicodeError) as error:
        preprocessing["error"] = str(error)
    else:
        preprocessing["status"] = "passed"
    timings["preprocessing_seconds"] = perf_counter() - started
    if preprocessing["status"] == "passed":
        started = perf_counter()
        frames = load_splits(config)
        timings["public_loading_seconds"] = perf_counter() - started
        preprocessing["public_load_splits"] = "passed"

    # A second source pass counts the same selection and checks selected rows
    # against original metadata. No full-corpus deduplication is claimed.
    input_counts, sampling_counts = {}, {}
    source_audit = Counter()
    selected = {
        row["id"]: row
        for frame in (frames or {}).values() for row in frame.to_dict("records")
    }
    representatives = {}
    if frames is not None:
        for label in config["classes"]:
            rows = [row for row in selected.values() if row["label"] == label]
            rows.sort(key=lambda row: (len(row["text"].split()), row["id"]))
            for kind, row in (("shortest", rows[0]), ("median_length", rows[len(rows) // 2]), ("longest", rows[-1])):
                representatives.setdefault(row["id"], []).append(kind)
    examples = []
    matched_ids = set()
    def audited_records():
        for record in iter_records(raw_path, decode_errors=config["data"].get("decode_errors", "strict"), stats=input_counts):
            categories = record.get("categories")
            tokens = categories.split() if isinstance(categories, str) else []
            source_audit["invalid_or_missing_categories"] += not bool(tokens)
            source_audit["multiple_category_tokens"] += len(tokens) > 1
            source_audit["category_tokens_not_lexicographically_sorted"] += tokens != sorted(tokens)
            record_id = record.get("id")
            record_id = record_id.strip() if isinstance(record_id, str) else None
            if record_id in selected:
                row = selected[record_id]
                source_audit["selected_source_occurrences"] += 1
                source_audit["selected_first_token_label_mismatches"] += primary_category(record) != row["label"]
                source_audit["selected_cleaned_text_mismatches"] += clean_text(record.get("abstract")) != row["text"]
                source_audit["selected_multiple_category_tokens"] += len(tokens) > 1
                source_audit["selected_multiple_configured_category_tokens"] += sum(t in config["classes"] for t in tokens) > 1
                words = len(row["text"].split())
                source_audit["selected_at_most_30_whitespace_tokens"] += words <= 30
                source_audit["selected_more_than_1000_whitespace_tokens"] += words > 1000
                metadata_text = " ".join(str(record.get(key) or "") for key in ("title", "abstract", "comments")).casefold()
                source_audit["selected_with_withdrawal_marker"] += "withdrawn" in metadata_text
                matched_ids.add(record_id)
                if record_id in representatives:
                    examples.append({
                        "id": record_id, "label": row["label"], "selection": representatives[record_id],
                        "title": record.get("title"), "source_categories": categories,
                        "whitespace_tokens": words, "characters": len(row["text"]),
                        "text_excerpt": row["text"][:240],
                        "withdrawal_marker": "withdrawn" in metadata_text,
                    })
            yield record
    started = perf_counter()
    samples = collect_samples(audited_records(), config["classes"], config["data"]["samples_per_class"], config["seed"], stats=sampling_counts)
    timings["source_audit_seconds"] = perf_counter() - started
    rows = [row for bucket in samples.values() for row in bucket]
    pool_validation = "passed"
    try:
        validate_samples(rows, config["classes"])
    except ValueError as error:
        pool_validation = str(error)
    source_audit["selected_ids_not_found_in_source"] = len(set(selected) - matched_ids)
    report = {
        "schema_version": 1, "preprocessing": preprocessing,
        "source": {"filename": raw_path.name, "bytes": raw_path.stat().st_size, "sha256": input_counts.pop("source_sha256")},
        "configuration": {"seed": config["seed"], "classes": config["classes"], "data": config["data"]},
        "input_counts": input_counts, "sampling_counts": sampling_counts,
        "shortfalls_from_cap": {label: config["data"]["samples_per_class"] - len(bucket) for label, bucket in samples.items()},
        "selected_pool_validation": pool_validation, "selected_pool_duplicates": duplicate_counts(rows),
        "abstract_lengths_scope": "selected cleaned model text; whitespace tokens are not model tokenizer tokens",
        "abstract_lengths": {"overall": length_statistics([row["text"] for row in rows]), "per_class": {
            label: length_statistics([row["text"] for row in bucket]) for label, bucket in samples.items()
        }},
        "source_audit": dict(source_audit),
        "source_audit_thresholds": {"very_short_max_whitespace_tokens": 30, "very_long_min_whitespace_tokens_exclusive": 1000, "withdrawal_marker": "case-insensitive substring 'withdrawn' in title/abstract/comments; heuristic, not confirmed withdrawal"},
        "representative_examples": sorted(examples, key=lambda row: (config["classes"].index(row["label"]), row["whitespace_tokens"], row["id"])),
        "label_rule": "first whitespace-separated category token; authoritative primary-category semantics not established by the Kaggle data card",
        "reproducibility": {"repeat_requested": repeat, "status": "not_run"},
    }
    artifacts = {}
    if frames is not None:
        report["integrity"] = audit_frames(frames, config)
        report["abstract_lengths"]["per_split"] = {name: length_statistics(frame["text"].tolist()) for name, frame in frames.items()}
        first_report = json.loads((Path(config["paths"]["processed_data"]) / "preprocessing_report.json").read_text(encoding="utf-8"))
        assert first_report["source"]["sha256"] == report["source"]["sha256"], "Source changed during profiling"
        assert first_report["input_counts"] == input_counts
        assert first_report["sampling_counts"] == sampling_counts
        for name, path in zip(SPLIT_NAMES, paths):
            artifacts[name] = {"filename": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path), "ordered_rows_sha256": logical_digest(frames[name])}
        if repeat:
            with TemporaryDirectory(prefix="repeat-", dir=config["paths"]["processed_data"]) as temporary_dir:
                repeated_config = deepcopy(config)
                repeated_config["paths"]["processed_data"] = Path(temporary_dir)
                started = perf_counter()
                repeated_paths = run_preprocessing(repeated_config)
                timings["repeat_preprocessing_seconds"] = perf_counter() - started
                repeated_frames = load_splits(repeated_config)
                repeated_report = json.loads((Path(temporary_dir) / "preprocessing_report.json").read_text(encoding="utf-8"))
                report["reproducibility"] = {
                    "repeat_requested": True, "status": "passed",
                    "preprocessing_report_identical": first_report == repeated_report,
                    "ordered_rows_identical": {name: frames[name].equals(repeated_frames[name]) for name in SPLIT_NAMES},
                    "parquet_bytes_identical_in_this_environment": {
                        name: artifacts[name]["sha256"] == file_sha256(path) for name, path in zip(SPLIT_NAMES, repeated_paths)
                    },
                }
                assert report["reproducibility"]["preprocessing_report_identical"]
                assert all(report["reproducibility"]["ordered_rows_identical"].values())
        report["ready_for_baseline_data_input"] = (
            report["integrity"]["independent_row_and_overlap_checks_passed"] and
            report["integrity"]["split_rounding_matches"] and
            report["integrity"]["every_class_in_every_split"] and
            not source_audit["selected_first_token_label_mismatches"] and
            not source_audit["selected_cleaned_text_mismatches"] and
            not source_audit["selected_ids_not_found_in_source"]
        )
    else:
        report["ready_for_baseline_data_input"] = False
        report["produced_split_rows"] = None
    manifest = {
        "schema_version": 1, "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_provider": source_manifest, "raw_file": report["source"],
        "configuration": report["configuration"], "timings": timings,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": {
            package: version(package) for package in ("pandas", "pyarrow", "pyyaml", "scikit-learn", "pytest")
        }},
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "code_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
            for path in [PROJECT_ROOT / "config/config.yaml", PROJECT_ROOT / "src/config.py", *sorted((PROJECT_ROOT / "src/data").glob("*.py"))]
        },
        "artifacts": artifacts,
        "timing_note": "Measured with time.perf_counter(); source audit, repeat and loading are separate; not an extrapolated benchmark.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (("profile.json", report), ("run_manifest.json", manifest)):
        (output_dir / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-repeat", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    report = profile_snapshot(config, source, args.output_dir, repeat=not args.no_repeat)
    print(json.dumps({"preprocessing": report["preprocessing"], "selected_per_class": report["sampling_counts"]["retained_per_class"], "ready_for_baseline_data_input": report["ready_for_baseline_data_input"]}, indent=2))
    raise SystemExit(0 if report["ready_for_baseline_data_input"] else 1)


if __name__ == "__main__":
    main()
