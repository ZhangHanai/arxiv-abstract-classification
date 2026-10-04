# Dataset Setup

This project uses the **Cornell University arXiv metadata** dataset published
as [`Cornell-University/arxiv` on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).

## Obtain the dataset

1. Create or sign in to a Kaggle account.
2. Open the Kaggle dataset page for **Cornell University arXiv**.
3. Review and accept any terms presented by Kaggle.
4. Download the dataset archive manually from the dataset page.
5. Extract the archive on your local machine.

## Place the raw metadata file

Copy the extracted raw JSON metadata file into:

```text
data/raw/
```

The default `data.raw_filename` in `config/config.yaml` is
`arxiv-metadata-oai-snapshot.json`. Change that relative filename if your manually
obtained snapshot uses a different name. Files placed in `data/raw/` are
intentionally excluded from version control and must not be committed.

No dataset is downloaded or included in this repository.

## Run preprocessing

From the repository root, in an environment containing `requirements.txt`:

```sh
python -m src.data.preprocess
python -m src.data.preprocess --config config/config.yaml
```

The command writes `train.parquet`, `val.parquet`, `test.parquet`, and
`preprocessing_report.json` under `paths.processed_data`. All are local, ignored
data artifacts. No model is trained. The callable `run_preprocessing(config)`
continues returning the three parquet paths.

Relative config-file paths and all YAML `paths` entries are anchored at the
repository, independent of the working directory. Both `/` and `\` separators
are supported. YAML paths and `data.raw_filename` reject absolute paths,
Windows drive/UNC/root-qualified paths, and traversal or symlink/junction
resolution outside their root. Absolute native paths may be supplied explicitly
through a Python configuration dictionary for temporary or external data; the
raw filename must still remain inside its raw-data directory. No current working
directory or machine-specific path is embedded in the defaults.

Resolved output/split filenames must also remain in the processed directory.
Output links cannot alias each other, and an input filename that collides with
an output is rejected before writing, preserving the raw source.

## Encoding and filtering rules

- Raw input is line-oriented UTF-8, with an optional BOM only at the start of
  the file. YAML is explicitly UTF-8 with an optional initial BOM.
- `data.decode_errors: strict` is the default: invalid bytes fail with a filename
  and line number before any output is written. An explicit `skip` value skips
  the entire invalid line, warns with the count, and records that count. Bytes
  are never silently ignored or replaced within otherwise accepted text.
- Blank lines, malformed JSON, and JSON values that are not objects are skipped
  and counted. Parsed records whose first category token is outside the
  configured classes are excluded and counted. Records with missing/blank or
  non-string IDs/abstracts are also excluded and counted.
- Accepted IDs are stripped at their ends. Accepted model texts retain their
  case and Unicode characters while whitespace is collapsed. Escaped lone
  Unicode surrogates in selected rows fail validation before parquet encoding.

The operational label is the **first whitespace-separated token in
`categories`**. This implementation rule is not independent verification of the
meaning of category ordering in a real source snapshot.
The [Kaggle data card](https://www.kaggle.com/datasets/Cornell-University/arxiv)
describes this field as category tags; it does not promise that the first token
is the authoritative primary category. Multi-category records retain that first
token rule; this project does not manually relabel them.

## Sampling, duplicates, and splits

Sampling is a seeded, per-class reservoir with a positive integer cap. A cap is
a ceiling, not a guarantee of equal class counts. Below-cap classes are reported
and warned about. A configured class with no samples, or too few samples for
every split to contain that class, fails clearly.

**Duplicate policy: reject, without deleting rows.** Before splitting, the
entire selected pool is checked for repeated IDs and repeated normalized text,
including collisions across category buckets. A duplicate error identifies both
rows/IDs. The normalization key uses Unicode NFKC, case folding, removal of
U+200B (zero-width space), U+FEFF (invisible BOM), and U+00AD (soft hyphen), and
whitespace collapse. The original text is retained for models; normalization is
used only to detect identity. Exact matching texts necessarily share this key.
This deliberately conservative rule also rejects case/format variants, but is
not semantic or fuzzy paraphrase detection. Resolve such errors through a
separately documented source-curation decision; preprocessing provides no
automatic deletion or keep-first option.

Checks apply to **selected samples and emitted/loaded splits**. Unsampled raw
records are not deduplicated or audited for uniqueness. This preserves streaming
input and capped sample storage; it is not a full-corpus duplicate analysis.

All three fractions (`train`, `val`, `test`) must be finite numbers strictly
between zero and one and sum to one within `1e-9`. Train and validation sizes
are rounded down per class; test receives the remainder. Every configured class
must appear at least once in each split after rounding. This policy can give a
larger test fraction for small classes. Splits use local seeded RNGs and never
modify the input lists or global random state.

The writer validates schemas, nonempty UTF-8 string values, labels, and ID/text
uniqueness within and across all three splits **before creating or overwriting
parquet files**. This validation does not promise an atomic three-file commit
in the event of a subsequent disk/I/O failure.

## Validate data before training

```python
from src.data.dataset import load_splits

splits = load_splits()  # Complete bundle; schema, classes, IDs, and text checked.
train = splits["train"]
validation = splits["val"]
test = splits["test"]
```

`load_splits(config)` requires all three parquet files, checks class coverage,
and rejects repeated IDs/text both within and across splits. `load_split()` and
the existing text/label helpers validate the requested file and all existing
sibling split files, but retain compatibility with a directory containing only
one split. Use the complete-bundle API at a future training boundary.

`preprocessing_report.json` records the source's relative filename and SHA-256
of the exact streamed bytes, seed, ordered classes, cap, fractions, encoding and
identity policies, exclusion counts, eligible/selected class counts, shortfalls,
and actual split counts. It has no machine-local absolute paths or timestamps;
the same source/settings produce the same report in the tested environment.
It does not pin dependencies or guarantee cross-version parquet byte identity.

The integrity pass is verified with synthetic fixtures. A real snapshot still
needs to be supplied, successfully preprocessed, and loaded with `load_splits()`
before any measured baseline run. Source provenance/category interpretation and
the training workflow remain outstanding.

## Reproduce the real-data profile

The first validation uses Kaggle version **306**, released
`2026-10-03T23:53:00.087Z`. Its identity and exact observed archive/raw hashes are
in `results/real_data_validation/source_manifest.json`. The raw file is
5,589,459,042 bytes. Obtain that exact official archive from Kaggle, extract its
single `arxiv-metadata-oai-snapshot.json` member into `data/raw/`, and compare
SHA-256 before reproducing this particular run. A newer version is a different
run and requires a new source manifest.

The archive was downloaded manually from the official version-pinned download
endpoint recorded in the manifest. The public endpoint did not present login or
additional dataset terms. No third-party mirror or automatic repository
downloader was used. If Kaggle asks for account access or terms, follow its
dataset-page process above.

After acquiring the snapshot, from the repository root:

```sh
python -B -m src.data.curate --source-manifest results/real_data_validation/source_manifest.json --output-dir results/real_data_validation
python -B -m src.data.preprocess --config config/real_data.yaml
python -B -m src.data.profile --config config/real_data.yaml --source-manifest results/real_data_validation/curation_report.json --output-dir results/real_data_validation
python -B -m pytest -q -p no:cacheprovider -ra
```

The profile command runs the existing public preprocessing/complete-bundle
loading APIs, counts a second streamed source pass, and repeats preprocessing
in a temporary ignored directory. It compares ordered dataset rows and the
deterministic preprocessing report, and separately records whether parquet bytes
match in the current environment. The temporary repeat directory is removed.
No model is fitted. It records selected cleaned-text character and whitespace
token lengths; these are not Transformer tokenizer lengths.

Only the small profile, source manifest, runtime manifest and written report are
saved in Git. Raw archives, extracted metadata, processed parquet, runtime
preprocessing reports and local acquisition observations remain ignored. The
runtime manifest records measured durations, exact package versions, code
fingerprints, output hashes and the Git head at execution. Timings are observed
on one machine, not extrapolated benchmarks.

A rejected selected pool is reported as a failed preprocessing gate, with
observed selection/duplicate counts and no fabricated split statistics. The
profile command exits nonzero and marks the data unready; it does not delete,
deduplicate, relabel, or quietly change the seed/cap to evade the integrity gate.

### Opt-in source-curation decision

The unmodified version-306 snapshot failed the selected-pool rejection guard.
Two selected normalized-text groups repeated; the first detected pair,
`1206.6899` and `1201.6078`, contains the same withdrawal notice under different
IDs/labels. The original rejected run is preserved in
`results/real_data_validation/unmodified_snapshot_rejection/`.

For the real-data validation, the user authorized this separate curation rule:
**exclude every member of any duplicate stripped-ID or normalized-text group
among all eligible records for the eight configured first-token classes before
reservoir sampling.** No representative is chosen from a conflicting group.
ID/text conditions are combined as a union, so a record is excluded once even
when both match. Normalized grouping uses full string equality under the existing
`text_identity()` rule, not hash equality or fuzzy similarity.

`src.data.curate` makes two complete source passes. The first counts eligible
identities, and the second writes eligible records outside every repeated group
to an ignored derived JSONL file, retaining all fields and category labels. The
original snapshot is read-only and hashed on both passes; a source change fails
before the derived file is replaced. The small `curation_report.json` records
original/derived hashes, raw filtering, every class's eligible/excluded/retained
counts, union exclusion reasons and representative duplicate groups. A complete
per-record exclusion audit remains in ignored `data/processed/`; its count and
hash are recorded in the report. Timings/code identity are recorded separately
in `curation_runtime_manifest.json` so the curation report is deterministic.

This is opt-in: the default `config/config.yaml` and the original preprocessing
duplicate-rejection contract remain unchanged. The explicit
`config/real_data.yaml` points at the derived JSONL with the same ordered classes,
seed 42, cap 7,000 and 80/10/10 fractions. Both the writer and public loader still
reject any duplicate surviving the curated source.

This rule removes duplicate-group records, not all withdrawals, short abstracts,
multi-category papers or semantic paraphrases. Such cases are flagged for
inspection; no heuristic relabeling or unapproved extra filtering is performed.
