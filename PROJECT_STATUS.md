# Project Status

## Verified milestone on 2026-10-04

[PR #5](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/5)
was merged into `main` at `cfccf369c54f4d04ed507bbe2d2dadb8b55d2223`.
The integrity increment was rebased without conflicts and passed **128 tests in
9.48s**. [PR #6](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/6)
was subsequently merged with normal merge commit
`042f7ff5238a486c8670f5a7fc0153e313aafaf8`.

[PR #7](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/7)
completed the first real-data run
using Cornell/Kaggle snapshot version 306. The unmodified source was rejected
for duplicate text. The user authorized a separate opt-in exclusion of every
eligible duplicate-group member: 331 records in 163 normalized-text groups.
The curated source produced **56,000 validated rows**, **7,000 per class**,
with **44,800 / 5,600 / 5,600** train/validation/test rows. Repeated preprocessing
and public loading passed, and the source/row/artifact identities are recorded
in [`REAL_DATA_REPORT.md`](REAL_DATA_REPORT.md). Raw and processed data stay
local and ignored. PR #7 was rebased onto the merged integrity work, retargeted
to `main`, verified with **137 passed in 9.78s, zero failures/skips**, marked ready,
and merged with normal merge commit
`a2c77ea9d23c6aee780f96000fdeb1ab1ee0d4fb`. Its 18-file diff and all five new
commits contain no datasets, archives, checkpoints, secrets or large model artifacts.

Local `main` was updated before creating `codex/classical-baselines`. On that
branch, clean implementation commit `77b2c7376e5740c1064b790c0dfecc6dbe10d4f5`
ran the first TF-IDF + Logistic Regression / Linear SVM experiment using
`config/real_data.yaml`. The same verified splits and shared evaluation were
used. Two C values per model were compared only on validation; both selected
`C=1`. Each selected model was evaluated once on test, after both selections
were saved. Logistic Regression test macro F1 is **0.825156**; Linear SVM is
**0.820346**. Full metrics, confusion matrices, timings and provenance are in the
[baseline report](results/classical_baselines/real_v1/report.md). No Transformer
has been implemented or trained.

## What this repository is

`arxiv-abstract-classification` is a self-initiated NLP portfolio project created to demonstrate text classification, reproducible ML engineering, and later Transformer fine-tuning for graduate-school applications.

It is not a course assignment, competition project, or published research project.

The intended task is to classify arXiv computer-science abstracts into eight
configured categories, using the first token in the source `categories` field:

- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.CL` — Computation and Language
- `cs.CV` — Computer Vision
- `cs.CR` — Cryptography and Security
- `cs.RO` — Robotics
- `cs.DS` — Data Structures and Algorithms
- `cs.SE` — Software Engineering

The intended research/engineering comparison is between classical TF-IDF linear baselines and a fine-tuned Transformer, with emphasis on error analysis rather than simply maximizing one score.

## What is actually implemented on `main`

At main commit `a2c77ea9d23c6aee780f96000fdeb1ab1ee0d4fb`, the repository
contains the original data-engineering layer, shared evaluation, integrity
hardening and real-data validation. The classical baseline implementation and
measured results are the separate `codex/classical-baselines` increment.

Implemented:

- repository scaffold and metadata;
- MIT license, README skeleton, requirements, gitignore;
- repository-root-aware YAML configuration;
- deterministic class definitions and path resolution;
- streaming parsing of the large arXiv JSONL metadata file;
- first-category-token extraction;
- light text cleaning;
- deterministic per-class reservoir sampling;
- stratified train/validation/test splitting;
- parquet output writing;
- processed split loading;
- label validation and configured label-ID mapping;
- unit tests for configuration, preprocessing, and dataset loading.
- shared accuracy, macro/weighted F1, per-class metrics and confusion matrices;
- saved evaluation JSON/CSV and aligned prediction parquet artifacts;
- synthetic evaluation tests;
- complete ID/normalized-text integrity checks across all selected splits;
- opt-in complete duplicate-group curation, real-data profiling and source/split provenance.

Implemented in the classical baseline increment:

- configurable word TF-IDF pipelines with Logistic Regression and LinearSVC;
- bounded validation-only selection, train-only fitting and frozen selections;
- one held-out test pass per selected model, with shared metrics and aligned artifacts;
- exact verified split binding, runtime/code manifests and measured timing;
- observed confusion rankings, low-F1 classes and high-confidence LR errors;
- focused tests for fitting/selection boundaries, ordering and artifact alignment.

Not yet implemented on `main`:

- TF-IDF + Logistic Regression baseline;
- Linear SVM baseline;
- training scripts;
- error-analysis pipeline;
- DistilBERT tokenization/data module;
- DistilBERT fine-tuning;
- RoBERTa or SciBERT comparison;
- interpretability tooling;
- substantive notebooks;
- CI / GitHub Actions;
- final benchmark results;
- portfolio-ready figures and final README results section.

## Development history recovered from pull requests

### PR #1 — Scaffold repository structure and project metadata

Created the project skeleton, README structure, dependency hints, data instructions, ignore rules, and directory placeholders. No application logic was implemented.

### PR #2 — Add path-safe configuration system

Added `config/config.yaml`, `src/config.py`, root-aware path resolution, and configuration tests.

Recorded test result: `4 passed`.

### PR #3 — Implement streaming preprocessing pipeline for arXiv metadata

Added `src/data/preprocess.py` with streaming JSONL parsing, filtering, deterministic reservoir sampling, stratified splitting, and parquet writing.

Recorded new preprocessing tests: `9 passed`.
Recorded full suite at that point: `13 passed`.

### PR #4 — Add dataset loading module for processed parquet splits

Added `src/data/dataset.py` and dataset-loading tests covering configured label ordering, parquet loading, validation, row-order preservation, and integer label encoding.

Recorded new dataset tests: `8 passed`.
The three existing test files therefore contain 4 + 9 + 8 = 21 tests, and the PR records state that the full suite passed at the time.

The initial development history stopped at this data-loading layer. PR #5 then
added archaeology and shared evaluation; PR #6 preserves the next integrity
increment. The current real-data milestone is recorded above.

## Current honest resume status

Safe to claim now, after independently verifying the repository:

- built a reproducible preprocessing pipeline for a large arXiv metadata dump;
- implemented seeded capped sampling and stratified dataset splitting, with
  observed equal 7,000-per-class counts in the first curated real-data run;
- built path-safe configuration and parquet dataset-loading utilities;
- added automated unit tests around the data and model-selection boundaries;
- trained and compared conventional TF-IDF Logistic Regression and Linear SVM;
- measured held-out macro F1 of 0.825156 and 0.820346 respectively on the documented split.

Not safe to claim yet:

- fine-tuned DistilBERT/RoBERTa/SciBERT;
- completed a baseline-vs-Transformer benchmark;
- completed model error analysis or interpretability experiments.

Those are project goals, not current results.

## Recommended next milestone

Review the separate classical baseline PR. The verified data, fixed splits,
shared evaluation and measured baseline references are ready for a DistilBERT
comparison. A future increment still needs tokenizer/data adaptation, a training
loop, an explicit compute budget and validation-only configuration choices.
The observed held-out benchmark must not guide further tuning. Detailed error
interpretation and interpretability analysis also remain future work.

The repository now also contains `CODEX_ARCHAEOLOGY_PROMPT.md`, which should be used before resuming implementation so Codex verifies the current state rather than relying on old plans.
