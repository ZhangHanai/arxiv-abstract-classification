# Project Status

## Verified milestone on 2026-10-04

[PR #5](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/5)
was merged into `main` at `cfccf369c54f4d04ed507bbe2d2dadb8b55d2223`.
The integrity increment was rebased without conflicts, passed **128 tests in
9.48s**, and is preserved separately in open
[PR #6](https://github.com/ZhangHanai/arxiv-abstract-classification/pull/6).

The `codex/real-data-validation` branch has completed the first real-data run
using Cornell/Kaggle snapshot version 306. The unmodified source was rejected
for duplicate text. The user authorized a separate opt-in exclusion of every
eligible duplicate-group member: 331 records in 163 normalized-text groups.
The curated source produced **56,000 validated rows**, **7,000 per class**,
with **44,800 / 5,600 / 5,600** train/validation/test rows. Repeated preprocessing
and public loading passed, and the source/row/artifact identities are recorded
in [`REAL_DATA_REPORT.md`](REAL_DATA_REPORT.md). Raw and processed data stay
local and ignored. No model training or measured model results exist.

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

At main commit `cfccf369c54f4d04ed507bbe2d2dadb8b55d2223`, the repository
contains the original data-engineering layer plus shared evaluation. Integrity
hardening and real-data validation are on the separate review branches described
above.

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
- synthetic evaluation tests; no measured model performance.

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
- added automated unit tests around the data pipeline.

Not safe to claim yet:

- trained a TF-IDF baseline;
- fine-tuned DistilBERT/RoBERTa/SciBERT;
- achieved any accuracy/F1 score;
- completed a baseline-vs-Transformer benchmark;
- completed model error analysis or interpretability experiments.

Those are project goals, not current results.

## Recommended next milestone

The next core milestone should be a complete classical baseline and evaluation harness before adding a Transformer:

1. use the verified real-data configuration and complete-bundle loader with the
   already implemented shared evaluation utilities;
2. implement TF-IDF + Logistic Regression and Linear SVM pipelines;
3. add a baseline training script that fits only on the train split;
4. save validation/test metrics, confusion matrices, and prediction artifacts;
5. add tests for leakage prevention, label ordering, metric correctness, and artifact writing;
6. only then add the DistilBERT data/training layer.

The repository now also contains `CODEX_ARCHAEOLOGY_PROMPT.md`, which should be used before resuming implementation so Codex verifies the current state rather than relying on old plans.
