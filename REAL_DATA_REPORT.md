# First real-data pipeline validation

Observed on 2026-10-04 (Asia/Shanghai). **The real Cornell/Kaggle version-306
snapshot produced 56,000 validated rows after the explicitly authorized,
documented opt-in duplicate-group curation.** The original snapshot alone failed
the reject-only guard. No TF-IDF, Logistic Regression, SVM or Transformer model
was fitted, and no model-performance results are claimed.

## Git preservation

PR #5 was merged with a merge commit at
`cfccf369c54f4d04ed507bbe2d2dadb8b55d2223`. Local main was fetched and
fast-forwarded. The single integrity implementation commit was rebased onto that
main without conflicts, preserving its patch. The complete suite then passed
**128 tests in 9.48s** with zero failures/skips. The branch was pushed and is
preserved in [PR #6](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/6).
Only the integrity implementation and its verification note are in that PR;
PR #5's archaeology/evaluation files and unrelated historical work are not
replayed or squashed into it. PR #6 remains open for review.

This validation is on the separate `codex/real-data-validation` branch, based on
the preserved integrity branch. `DATA_INTEGRITY_REPORT.md` and PR #6 enumerate the
path/encoding/fraction/shortfall/schema/identity/leakage bugs fixed and the exact
integrity tests. Default preprocessing's duplicate-rejection policy is unchanged.

## Source identity and acquisition

Source: [Cornell University arXiv on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv),
`Cornell-University/arxiv`, dataset ID 612177, **version 306**.
Kaggle reports version creation at `2026-10-03T23:53:00.087Z` and metadata
licensing as CC0. Its data card describes categories as tags; its historical
headline is not used as an observed record count.

The dataset was absent locally. The official archive was downloaded manually
from the version-pinned Kaggle archive endpoint recorded in
`results/real_data_validation/source_manifest.json`, then extracted locally.
The public endpoint presented no login or additional dataset-terms prompt. No
third-party mirror, substitute dataset or repository downloader was used.
The archive contains exactly one expected raw metadata member; ZIP extraction
verified CRC `69fd6fa2`. The original source remains intact.

| File | Actual bytes | Observed SHA-256 |
| --- | ---: | --- |
| `arxiv-v306.zip` (local archive) | 1,852,304,107 | `b078129ac784a5a67901e06b8bed934180810a36c72eb9dacce78464d33143fa` |
| `arxiv-metadata-oai-snapshot.json` (original raw) | 5,589,459,042 | `b5c73be5958b20f4c07ea30a9352941bddfe59fed943dde6614e6d3415347c88` |
| `arxiv-metadata-oai-snapshot.curated.jsonl` (derived source) | 1,108,749,510 | `d0100471cc2da0ec4d998e5754c79fef0f9bef2271c69fe8efd861a90c66ae68` |

These are locally measured fingerprints, not invented publisher checksums.
The raw size matches Kaggle's reported 5,589,459,042 bytes.
The archive member timestamp is recorded as ZIP-local with an unspecified
zone; it is not presented as an authoritative UTC snapshot date.

## Configuration and observed population

Original input: `config/config.yaml`. Successful derived input:
`config/real_data.yaml`. Both use seed **42**, the same ordered eight classes,
cap **7,000 per class**, strict UTF-8 and **0.8 / 0.1 / 0.1** fractions. Train
and validation counts are floored per class; test receives the remainder. Every
class must appear in every split. Model text is whitespace-cleaned abstract text;
identity normalization is only an equality check, not replacement model text.
`max_seq_length: 256` is retained as a future model setting; this task does not
truncate abstracts or tokenize/train a Transformer.

Original raw scan: **3,195,172 objects on
3,195,172 lines**. Eligible under configured first-token
labels and existing ID/abstract filters: **568,998**.
Excluded as non-target first-token categories: **2,626,174**.
Malformed JSON, invalid UTF-8, blank lines and non-object JSON counts were all
**0**. Target-class records with invalid IDs or invalid/blank abstracts were
both **0**; this does not claim that every field of non-target rows was audited.

### Required opt-in curation

The unmodified run exited **1** before creating parquet files. Its selected pool
had **2 repeated normalized-text groups**,
**2 rows beyond the first**, and
**0 repeated ID groups**.
The first detected pair was `1206.6899` (cs.DS) and `1201.6078` (cs.SE), both
containing the identical withdrawal notice. The rejected run's observed profile
is retained under `results/real_data_validation/unmodified_snapshot_rejection/`.

The user authorized a separate rule: **exclude every member of any repeated
stripped-ID or normalized-text group among all eligible records before reservoir
sampling**. No keep-first choice, manual relabeling, seed/cap change, paraphrase
filter or blanket withdrawal/short-text filter was added. Grouping uses full
normalized-string equality (NFKC, case folding, removal of U+200B/U+FEFF/U+00AD,
whitespace collapse). Exclusions use the union of ID and text conditions.

Two original-source passes found **163 text groups** and
**0 ID groups**; all **331 group-member records** were excluded.
**568,667 records** remain in the derived eligible source.
The complete local exclusion audit has 331 rows,
55,850 bytes and SHA-256
`282ec0b388fac520fbdd1dd98344a086468018757baf38753ebf425e0415128c`; it remains ignored in `data/processed/`.
The small committed curation report records all counts and sample group identities.

| Class | Eligible original | Excluded duplicate members | Eligible curated | Selected | Cap shortfall | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cs.AI | 47611 | 54 | 47557 | 7000 | 0 | 5600 | 700 | 700 |
| cs.LG | 146755 | 97 | 146658 | 7000 | 0 | 5600 | 700 | 700 |
| cs.CL | 89123 | 32 | 89091 | 7000 | 0 | 5600 | 700 | 700 |
| cs.CV | 162351 | 66 | 162285 | 7000 | 0 | 5600 | 700 | 700 |
| cs.CR | 36233 | 31 | 36202 | 7000 | 0 | 5600 | 700 | 700 |
| cs.RO | 44779 | 23 | 44756 | 7000 | 0 | 5600 | 700 | 700 |
| cs.DS | 19220 | 10 | 19210 | 7000 | 0 | 5600 | 700 | 700 |
| cs.SE | 22926 | 18 | 22908 | 7000 | 0 | 5600 | 700 | 700 |
| **Total** | **568,998** | **331** | **568,667** | **56,000** | **0** | **44,800** | **5,600** | **5,600** |

## Integrity and reproducibility outcomes

- Public complete-bundle loading passed with both `load_splits()` and
  `load_splits(load_config("config/real_data.yaml"))`. The public integer-label
  helper returned 44,800 aligned training rows and configured IDs 0 through 7.
- All rows belong to the configured eight classes. Each required ID/text/label
  value is non-null, nonempty string data that passes UTF-8 validation.
- Within each split and the entire selected pool: **0 duplicate IDs,
  0 exact-text duplicate groups, 0 normalized-text duplicate groups**.
- For **train/val**, **train/test** and **val/test**, each ID, exact-text and
  normalized-text intersection is **0** (all nine pair/key checks).
- Per-class floor/remainder rules, class presence in all three splits and caps
  match the observed counts. There are no class shortfalls.
- Every selected row was found in the source. First-token label mismatches and
  cleaned-abstract mismatches were both **0** across **56,000 selected rows**.
  Independently reconstructed selected rows match the written bundle.
- The derived source hash matches its curation manifest. Original-source hashes
  agree across both curation passes and a second independent curation execution.
- A second complete curation execution produced **identical derived bytes and
  exclusion results**. A second preprocessing run produced **identical ordered
  rows, preprocessing report and all three parquet bytes** in this environment.
  Temporary repeat datasets were removed.

| Local output (ignored) | Actual bytes | SHA-256 |
| --- | ---: | --- |
| train.parquet | 31,718,125 | `57a86d4b1eefe02fde53f98666d152170b9a18bd6b610d75332344dc1bf8c1a0` |
| val.parquet | 3,955,377 | `bb86a1bf21c90e65afb52344bf4e2d6fe08bc1654deac66b3c38fe0b6f82ac34` |
| test.parquet | 3,976,152 | `539302fa2ae5f1e925b3ea8cbcee57731134ecbcb462bef8f0bd1556013e06e8` |

Parquet byte identity is an observed result in this runtime, not a guarantee
across different dependency versions. Ordered-row hashes are also in the run
manifest, independent of parquet encoding.

## Abstract lengths and sanity inspection

These statistics describe **selected cleaned model text**, not the full corpus
or Transformer tokenizer tokens. Whitespace tokens are counted with Python
`str.split()`. Percentiles use pandas' default linear quantiles; means and
population standard deviations are saved at full precision in `profile.json`.
The display below rounds only means to six decimals. Per-class and per-split
length statistics are in that same machine-readable profile.

| Length unit | Minimum | Mean | Median | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| characters | 25 | 1241.109125 | 1249.0 | 1845.0 | 1912.0 | 2914 |
| whitespace_tokens | 3 | 174.791571 | 174.0 | 260.0 | 287.0 | 416 |

For each class, shortest, middle-by-word-length and longest representatives were
inspected. All eight middle representatives were read in full through the
public loader; the profile retains 24 short excerpts with IDs, titles, source
category strings and observed lengths. Review interpretations below do not
change the source labels or claim human-validated ground truth.

| Operational class | Inspected middle representative | Source category string | Review observation |
| --- | --- | --- | --- |
| cs.AI | `2010.07533` | `cs.AI` | Relation extraction with tensor decomposition. Close to language processing; the source first token is cs.AI, so no relabeling. |
| cs.LG | `2406.01411` | `cs.LG` | Constrained subgroup discovery and interpretable optimization; consistent with the operational machine-learning label. |
| cs.CL | `2208.09770` | `cs.CL cs.AI` | Abstractive text summarization; cross-listed cs.AI but retained as first-token cs.CL. |
| cs.CV | `2203.07740` | `cs.CV` | Style transfer and visual domain generalization; consistent with the operational vision label. |
| cs.CR | `2401.00148` | `cs.CR cs.CV` | Physical adversarial patches for perception; overlaps vision, with cs.CR first and cs.CV later. |
| cs.RO | `2609.27154` | `cs.RO` | Multimodal human intent inference and control for a robotic blimp; consistent with robotics. |
| cs.DS | `2607.24827` | `cs.DS cs.CC cs.LG math.OC` | Oracle-complexity and memory lower bounds for convex optimization; overlaps learning/optimization, with cs.DS first. |
| cs.SE | `2301.13615` | `cs.SE` | Property-based mutation testing for safety-critical software; consistent with software engineering. |

Actually observed quality flags in the final selected dataset:

- **127 abstracts have at most 30 whitespace tokens**.
  For example, cs.AI record `0906.2824` is the three-word "Short philosophical
  essay"; cs.SE `1301.5878` is a short spreadsheet-style description.
- **116 selected source records contain the case-insensitive substring
  `withdrawn` in title/abstract/comments**. This is a flag, not 116
  independently confirmed withdrawals. The shortest cs.DS example `1102.1124`
  contains a withdrawal notice and remains because its normalized text is unique
  within the eligible corpus. No extra withdrawal rule was authorized or applied.
- **28,131 selected records have multiple category tokens**;
  **17,777 include multiple configured classes**.
  Relation extraction under cs.AI, adversarial vision under cs.CR, and optimization
  theory under cs.DS illustrate category overlap. They are inspection flags,
  not proven label mistakes or permission to relabel.
- **0 selected abstracts exceeded 1,000 whitespace tokens** under the explicit
  diagnostic threshold. The observed maximum is 416.

### First-token rule versus authoritative primary category

The implementation takes `categories.split()[0]`. This exact rule is confirmed
against every selected source record. It is the label definition for this task.
The [Kaggle data card](https://www.kaggle.com/datasets/Cornell-University/arxiv)
describes category tags without an ordering guarantee. The
[official arXiv Atom API manual](https://info.arxiv.org/help/api/user-manual.html)
documents a separate `arxiv:primary_category` element. That does not establish
snapshot-wide equality between that explicit field and this JSON string's first
token. No such equality, or authoritative human topic assignment, is claimed.
The historic helper name `primary_category()` is interpreted operationally here.

## Measured processing time and environment

- Unmodified preprocessing CLI: **125.531038s**, exit 1.
- Original-source curation: identity scan **121.116094s**,
  derived-source write **182.201097s**.
- Curated preprocessing CLI: **48.139237s**, exit 0.
- Profiling invocation: preprocessing **36.075681s**,
  public loading **4.288033s**,
  streamed source audit **44.748838s**,
  repeat preprocessing **60.071340s**.
- Repeat curation durations are recorded separately in
  `curation_reproducibility.json`. These are independently measured runs, not
  extrapolated benchmarks. No reliable end-to-end download duration is claimed
  because the transfer was interrupted and resumed through the existing proxy.

Runtime: Python **3.13.5**, `Windows-11-10.0.26200-SP0`;
packages `{"pandas": "3.0.6", "pyarrow": "25.0.1", "pytest": "9.1.1", "pyyaml": "6.0.3", "scikit-learn": "1.9.1"}`.
Git heads and exact code/config hashes are recorded in runtime manifests. The
final real-data application code is committed at
`cf5424cf92f63616dd48c90fa491e86130b528a4`; later commits add these observations
and documentation. Dependencies remain unpinned in `requirements.txt`.

## Commands and exact tests

All Python commands used the existing external audit environment's Python
3.13.5; `python` below denotes that interpreter. Commands run from the repository
root; no package changes or local model downloads were performed.

Git preservation, with the existing system proxy supplied per Git call after
direct fetch attempts failed (no global Git settings changed):

```sh
gh pr merge 5 --repo ZhangHanai/arxiv-abstract-classification --merge --match-head-commit 6d58b38a0eb5bfbe7b0c110d26937a88f32c742f
git -c http.proxy=http://127.0.0.1:7897 -c http.version=HTTP/1.1 fetch --no-tags origin main
git switch main
git merge --ff-only origin/main
git switch codex/data-integrity
git rebase main
git diff main...HEAD --check
git -c http.proxy=http://127.0.0.1:7897 -c http.version=HTTP/1.1 push -u origin codex/data-integrity
```

Core real-data commands (the first two preserve the initial rejected run;
that expected rejection exits 1):

```sh
python -B -m src.data.preprocess
python -B -m src.data.profile --source-manifest results/real_data_validation/source_manifest.json --output-dir results/real_data_validation/unmodified_snapshot_rejection
python -B -m src.data.curate --source-manifest results/real_data_validation/source_manifest.json --output-dir results/real_data_validation
python -B -m src.data.preprocess --config config/real_data.yaml
python -B -m src.data.profile --config config/real_data.yaml --source-manifest results/real_data_validation/curation_report.json --output-dir results/real_data_validation
python -B -m pytest -q -p no:cacheprovider -ra
python -m pip check
git diff --check
```

Manual archive acquisition used the version-pinned official endpoint in the
source manifest. The initial direct `curl` transfer was interrupted; the resumed
transfer used `--proxy http://127.0.0.1:7897 --continue-at -`, `--fail --location`
and wrote only `data/raw/arxiv-v306.zip`. A Python inline ZIP extraction checked
the single member, expected size and CRC and measured raw SHA-256 while writing
it to a temporary raw file before replacement. Archive SHA-256 was separately
measured. All such local acquisition observations remain ignored.

An additional `python -B -` inline verification called `curate_snapshot()` again
with an isolated temporary output filename, compared the two derived hashes,
bytes, class/filter/exclusion counts and exclusion-audit hash, and deleted its
temporary dataset. Another inline verification called default/configured
`load_splits()` and `get_texts_label_ids('train', config)` and inspected all eight
middle abstracts. These measured outcomes are saved in the small manifests;
they are verification operations, not hidden model experiments.

| Complete suite boundary | Exact result |
| --- | --- |
| PR #5 exact head before merge | **63 passed in 5.15s** |
| Integrity before rebase | **128 passed in 10.52s** |
| Integrity after merge/rebase | **128 passed in 9.48s** |
| Final real-data validation | **137 passed in 17.74s** |

All complete runs used `python -B -m pytest -q -p no:cacheprovider -ra` and had
**0 failures and 0 skips**. Final per-file totals: config 19, preprocessing 25,
dataset 21, evaluation 42, integrity 21, curation 4, profile 5.

Focused runs while adding the harness:
`python -B -m pytest -q -p no:cacheprovider tests/test_profile.py -ra` produced
**4 passed in 4.55s**, **4 passed in 2.41s**, and **4 passed in 2.61s**.
`python -B -m pytest -q -p no:cacheprovider tests/test_curate.py tests/test_profile.py -ra`
produced **7 passed in 3.64s**, **8 passed in 4.24s**, and **9 passed in 6.61s**.
These successive results reflect added functionality/tests, not a summed pass
count. Final `python -m pip check`: **No broken requirements found.** Diff checks
passed. No remote CI, native Linux/macOS execution or model fitting is claimed.

## Exact tracked file inventory

Integrity PR #6, relative to the updated main:

- `DATA_INTEGRITY_REPORT.md`
- `README.md`
- `config/config.yaml`
- `data/README.md`
- `src/config.py`
- `src/data/dataset.py`
- `src/data/integrity.py`
- `src/data/preprocess.py`
- `tests/test_config.py`
- `tests/test_dataset.py`
- `tests/test_integrity.py`
- `tests/test_preprocess.py`

Real-data validation, relative to the preserved integrity branch:

- `README.md`
- `PROJECT_STATUS.md`
- `data/README.md`
- `REAL_DATA_REPORT.md`
- `config/real_data.yaml`
- `src/data/curate.py`
- `src/data/profile.py`
- `tests/test_curate.py`
- `tests/test_profile.py`
- `results/real_data_validation/source_manifest.json`
- `results/real_data_validation/curation_report.json`
- `results/real_data_validation/curation_runtime_manifest.json`
- `results/real_data_validation/curation_reproducibility.json`
- `results/real_data_validation/profile.json`
- `results/real_data_validation/run_manifest.json`
- `results/real_data_validation/verification.json`
- `results/real_data_validation/unmodified_snapshot_rejection/profile.json`
- `results/real_data_validation/unmodified_snapshot_rejection/run_manifest.json`

Raw archives, original/derived JSONL, exclusion audit, processed parquet,
`data/processed/preprocessing_report.json`, local acquisition/timer observations,
repeat directories and checkpoints are not committed. The committed JSON files
are small observed summaries/provenance only.

## Baseline readiness and remaining limitations

**Yes: the curated real-data bundle is ready as input for initial TF-IDF +
Logistic Regression and Linear SVM experiments under the documented first-token
classification task.** Reproduction requires the exact source plus the explicit
curation and `config/real_data.yaml`; the original default reject-only run still
fails by design. Use the complete public loader at the training boundary.

The flagged unique withdrawal notices, short texts and category overlap remain
potential noise for interpretation of future results. Their removal or relabeling
would require a further documented decision; this run does not assert a clean
human topic ground truth or authoritative-primary equivalence. Identity checks
are exact/specified normalization, not semantic paraphrase detection. Curation
covers all eligible records for these eight first-token classes, not all other
arXiv categories, and its memory grows with eligible identity keys. Splits are
seeded random stratified samples, not temporal holdouts. Multi-file parquet/report
writes are validated before I/O but are not atomic filesystem transactions.
Dependencies/other operating systems and cross-version parquet identity remain
unverified beyond the recorded runtime. The model training runner, fitting,
metrics and Transformer work remain future tasks.
