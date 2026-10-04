# Data integrity increment

Verified on 2026-10-04 (Asia/Shanghai), on `codex/data-integrity`, based on
`6d58b38a0eb5bfbe7b0c110d26937a88f32c742f`. The archaeology/evaluation increment
is preserved in [PR #5](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/5)
into `main`. That PR remains open; no remote CI checks were present. Its three
files are unchanged by this increment.

## Bugs found and fixed

| Bug | Previous behavior | Resulting behavior |
|---|---|---|
| Configured paths escaped their root | Absolute or `..` paths were accepted; Windows path syntax was host-dependent | Reject absolute/drive/UNC/root-qualified YAML paths; resolve either separator style and enforce containment, including resolved links |
| Relative custom config paths depended on CWD | An external working directory changed which custom YAML was read | Anchor relative custom config paths at the repository; allow explicit native absolute config-file paths |
| Raw filename could escape its directory or collide with output | Joining an absolute/traversing filename bypassed the raw directory; matching a split filename could overwrite source data | Resolve raw filenames under the raw directory; reject input/output collisions before writing; keep resolved split/output paths inside the processed directory |
| UTF-8 error handling was ineffective | Text-stream decoding failed before the parser's exception handler; an initial BOM was not handled | Decode binary lines explicitly; strict errors include filename/line; an explicit skip policy counts whole skipped lines and warns; accept only an initial BOM |
| Escaped lone surrogates failed during table/parquet encoding | JSON may decode invalid Unicode scalar values that fail during pandas string inference or writing | Build validation tables without premature string encoding, then reject unencodable selected row values with row/column context |
| Test fractions were ignored | Missing `test`, inconsistent totals, and invalid ranges/types were accepted | Require exactly three finite positive fractions summing to one; document train/val floors and test remainder |
| Class shortfalls and empty class splits were silent | A class could be missing or vanish from validation/test despite apparently balanced sampling | Warn/report below-cap counts; reject classes that cannot occur in every split after rounding |
| Selected duplicate IDs/text leaked across splits | Row-level shuffling could place a paper or identical abstract into multiple splits, even under different IDs/labels | Reject repeated stripped IDs and normalized text in the selected pool before assignment; validate every split pair before writing and loading |
| Loader validation was incomplete | Only required columns and helper-level label membership were checked; duplicate class definitions were silently collapsed | Validate nonempty string/UTF-8 row values, labels, repeated IDs/text, and class definitions; add complete-bundle loading with class coverage checks |
| Filtering and source identity were unrecorded | A seeded run still had no record of malformed/excluded rows, shortfalls, or exact source bytes | Save a deterministic UTF-8 report containing source SHA-256, settings/policies, exclusions, per-class samples, and actual split counts; expose a tested preprocessing CLI |

Duplicate resolution uses **rejection without row deletion**. The text equality
key is NFKC, case folding, removal of U+200B/U+FEFF/U+00AD, and whitespace
collapse. Model text is preserved apart from the existing whitespace cleaning.
This catches exact and specified formatting/case variants, not paraphrases.
Checks cover the selected pool and emitted/loaded splits; unsampled source rows
are not deduplicated or audited for uniqueness. Existing filtering and reservoir
sampling rules are documented and counted in [data/README.md](data/README.md).

## Exact files changed relative to the preserved increment

| File | Change |
|---|---|
| `src/config.py` | Portable contained path resolution; explicit YAML encoding; structural/settings validation |
| `config/config.yaml` | Explicit `data.decode_errors: strict` default |
| `src/data/integrity.py` (new) | Shared class/fraction/schema/ID/text/bundle validation |
| `src/data/preprocess.py` | Per-line decoding, selected-pool rejection, split checks, protected output paths, counts/fingerprint report, CLI |
| `src/data/dataset.py` | Stronger loading validation, existing-sibling overlap checks, complete `load_splits()` API |
| `tests/test_config.py` | 5 new test functions, 15 cases |
| `tests/test_preprocess.py` | 10 new test functions, 16 cases; repair the old fixture's cross-class duplicate texts and strengthen its overlap assertions |
| `tests/test_dataset.py` | 6 new test functions, 13 cases |
| `tests/test_integrity.py` (new) | 7 new test functions, 21 cases |
| `data/README.md` | Runnable command, encoding/filtering/duplicate/split rules, provenance contract and training validation boundary |
| `README.md` | Current task/evaluation behavior and reproducible test/preprocessing commands |
| `DATA_INTEGRITY_REPORT.md` (new) | This bug, file, test, result, and readiness record |

No changes to `ARCHAEOLOGY_REPORT.md`, `src/evaluation.py`, or
`tests/test_evaluation.py`. No datasets, checkpoints, runtime preprocessing
reports, or generated evaluation outputs are committed.

## Exact tests added

Parameter expansion gives **65 new cases in 28 new functions**. Cases exercise
distinct path dialects, fraction failures, row failures, and split pairs rather
than repeating successful behavior to increase counts.

| File | Test function | Cases |
|---|---|---:|
| `tests/test_config.py` | `test_config_rejects_absolute_drive_unc_and_escaping_paths` | 8 |
| `tests/test_config.py` | `test_custom_config_and_paths_are_portable_and_independent_of_cwd` | 2 |
| `tests/test_config.py` | `test_yaml_accepts_utf8_bom_and_preserves_unicode` | 1 |
| `tests/test_config.py` | `test_malformed_config_structure_fails_clearly` | 3 |
| `tests/test_config.py` | `test_resolved_symlink_or_junction_cannot_escape_path_root` | 1 |
| `tests/test_preprocess.py` | `test_iter_records_handles_bom_and_explicit_skip_without_replacing_text` | 1 |
| `tests/test_preprocess.py` | `test_strict_utf8_failure_has_line_number_and_creates_no_outputs` | 1 |
| `tests/test_preprocess.py` | `test_injected_raw_filename_cannot_escape_raw_directory` | 3 |
| `tests/test_preprocess.py` | `test_preprocessing_rejects_selected_duplicates_without_writing_data` | 2 |
| `tests/test_preprocess.py` | `test_write_splits_validates_before_overwriting_existing_files` | 1 |
| `tests/test_preprocess.py` | `test_preprocessing_cannot_overwrite_its_own_raw_source` | 1 |
| `tests/test_preprocess.py` | `test_preprocessing_report_counts_filtering_and_is_reproducible` | 1 |
| `tests/test_preprocess.py` | `test_preprocessing_reports_class_shortfalls_without_claiming_balance` | 1 |
| `tests/test_preprocess.py` | `test_sampling_rejects_unusable_caps` | 4 |
| `tests/test_preprocess.py` | `test_preprocessing_cli_runs_with_portable_custom_config` | 1 |
| `tests/test_dataset.py` | `test_load_splits_requires_complete_bundle_and_preserves_order` | 1 |
| `tests/test_dataset.py` | `test_loaders_reject_cross_split_leakage_from_parquet` | 2 |
| `tests/test_dataset.py` | `test_load_split_rejects_invalid_row_values` | 5 |
| `tests/test_dataset.py` | `test_load_split_rejects_empty_or_repeated_rows` | 3 |
| `tests/test_dataset.py` | `test_load_splits_rejects_missing_class_coverage` | 1 |
| `tests/test_dataset.py` | `test_build_label_maps_rejects_duplicate_class_definitions` | 1 |
| `tests/test_integrity.py` | `test_split_validation_rejects_id_and_text_leakage_between_every_pair` | 6 |
| `tests/test_integrity.py` | `test_effectively_identical_text_is_rejected_without_mutating_originals` | 1 |
| `tests/test_integrity.py` | `test_selected_pool_rejects_duplicates_before_any_split_assignment` | 2 |
| `tests/test_integrity.py` | `test_invalid_split_fractions_fail_before_assignment` | 8 |
| `tests/test_integrity.py` | `test_tiny_or_missing_classes_fail_instead_of_silently_disappearing` | 1 |
| `tests/test_integrity.py` | `test_fraction_rounding_preserves_every_row_and_each_class` | 1 |
| `tests/test_integrity.py` | `test_empty_normalized_or_unencodable_text_fails_clearly` | 2 |

The six split-pair cases independently check IDs and text between train/val,
train/test, and val/test. The parquet integration also proves that requesting
train detects val/test leakage. Tests confirm rejection leaves input rows and
existing output bytes intact. The existing deterministic split test now also
asserts exact-text disjointness for all three pairs.

## Full test results

Preservation run: `python -B -m pytest -q -p no:cacheprovider` produced
**63 passed in 5.83s** before this increment.

Final complete run: `python -B -m pytest -q -p no:cacheprovider -ra` produced
**128 passed in 9.70s**, with **0 failures and 0 skipped cases**. This was run
with the existing audit environment's Python 3.13.5 on Windows, from the
repository root, with bytecode generation disabled. No dependency changes were
made; the preservation environment's `pip check` had no broken requirements.

| File | Existing cases retained | New cases | Passing cases |
|---|---:|---:|---:|
| `tests/test_config.py` | 4 | 15 | 19 |
| `tests/test_preprocess.py` | 9 | 16 | 25 |
| `tests/test_dataset.py` | 8 | 13 | 21 |
| `tests/test_evaluation.py` | 42 | 0 | 42 |
| `tests/test_integrity.py` | 0 | 21 | 21 |
| Total | 63 | 65 | 128 |

`git diff --check` passed. Windows execution includes POSIX/Windows relative
notation, foreign absolute/drive/UNC rejection, external-CWD configuration,
Unicode paths, and a real directory-junction containment check. Native
Linux/macOS execution and remote CI were not performed; no such result is
claimed. All input data and generated outputs used in verification are temporary
synthetic fixtures.

## Baseline readiness and remaining work

The preprocessing/loading layer is ready to support baseline training **for a
dataset that successfully passes preprocessing and `load_splits()`**. Those
entry points enforce the ID/text integrity checks; bypassing them is outside the
verified contract. Actual training still requires a supplied real snapshot and
a successful real-data integrity pass. None is present in this checkout.

No Logistic Regression, SVM, or Transformer training was started. No measured
model results exist. A real source's provenance/category-ordering semantics,
dependency pinning, CI, the training runner, and model comparisons remain
unfinished. The report fingerprints streamed source bytes, not installed
packages or emitted parquet bytes. Writes are validated before I/O, but the
three parquet files/report are not a filesystem transaction if disk I/O fails.
