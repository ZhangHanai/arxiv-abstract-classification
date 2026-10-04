# arXiv Abstract Classification: TF-IDF and Transformer Models with Error Analysis and Interpretability

## Project Overview

This project will explore the classification of arXiv abstracts into subject categories using classical and transformer-based natural language processing methods.

## Motivation

The project is intended to demonstrate an end-to-end, reproducible text classification workflow suitable for an AI, computer science, or data science portfolio.

## Dataset

The project uses the Cornell University arXiv metadata dataset distributed through Kaggle. Dataset acquisition and the observed version-306 run are documented in [`data/README.md`](data/README.md) and [`REAL_DATA_REPORT.md`](REAL_DATA_REPORT.md).

## Classification Task

The current configuration selects eight computer-science categories and uses the
first token in each record's `categories` field as its label. Defaults request
up to 7,000 samples per class and seeded 80/10/10 splits. See
[`data/README.md`](data/README.md) for rounding, shortfall, encoding, and duplicate
rejection rules. The first real snapshot produced 56,000 validated rows after
the documented opt-in exclusion of all eligible duplicate-group members.
Use `config/real_data.yaml` to reproduce that run. The first-token label rule is
verified against all selected source records; its equivalence to an authoritative
primary category has not been established for this JSON snapshot.

## Planned Models

Planned model families include TF-IDF-based classical machine learning baselines and transformer-based text classifiers.

## Evaluation

`src/evaluation.py` provides shared accuracy, macro/support-weighted F1,
per-class scores, ordered confusion matrices, and aligned prediction artifacts.
No model has been trained and no experimental results are reported yet.

## Error Analysis

Future work will examine representative errors, category-level performance, and recurring patterns in model failures.

## Interpretability

Future work will compare suitable interpretability methods for classical and transformer-based models.

## Repository Structure

The repository is organized into directories for configuration, data documentation and local artifacts, source packages, notebooks, tests, and generated results.

## Reproducibility

Install the current dependencies and run tests from the repository root:

```sh
python -m pip install -r requirements.txt
python -B -m pytest -q -p no:cacheprovider
```

After manually obtaining the version-pinned raw snapshot, follow the curation,
preprocessing and profiling commands in [`data/README.md`](data/README.md).
The unmodified snapshot intentionally fails the default reject-only duplicate
guard; the explicit real-data configuration uses a separately derived source.
Use `src.data.dataset.load_splits()` to validate the complete dataset before
training. Configuration/data paths, explicit encoding policies, counted filtering,
source fingerprints, and failure behavior are documented in
[`data/README.md`](data/README.md). The real-data bundle passes integrity checks
and repeated runs match in the recorded environment. Dependencies remain
unpinned; baseline training and CI remain unfinished. No model has been trained.

## Limitations and Future Work

Limitations, threats to validity, and possible extensions will be documented after the experimental scope is finalized.
