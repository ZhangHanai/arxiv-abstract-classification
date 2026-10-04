# Dataset Setup

This project is designed to use the **Cornell University arXiv metadata** dataset published on Kaggle.

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
