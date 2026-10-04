# arXiv Abstract Classification: TF-IDF and Transformer Models with Error Analysis and Interpretability

## Project Overview

This project will explore the classification of arXiv abstracts into subject categories using classical and transformer-based natural language processing methods.

## Motivation

The project is intended to demonstrate an end-to-end, reproducible text classification workflow suitable for an AI, computer science, or data science portfolio.

## Dataset

The project will use the Cornell University arXiv metadata dataset distributed through Kaggle. Dataset acquisition instructions are documented in [`data/README.md`](data/README.md).

## Classification Task

The current configuration selects eight computer-science categories and uses the
first token in each record's `categories` field as its label. Defaults request
up to 7,000 samples per class and seeded 80/10/10 splits. See
[`data/README.md`](data/README.md) for rounding, shortfall, encoding, and duplicate
rejection rules. These defaults have been tested with synthetic data; a real
snapshot and its category-ordering semantics have not yet been audited.

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

After manually obtaining the raw snapshot, run `python -m src.data.preprocess`.
Use `src.data.dataset.load_splits()` to validate the complete dataset before
training. Configuration/data paths, explicit encoding policies, counted filtering,
source fingerprints, and failure behavior are documented in
[`data/README.md`](data/README.md). Dependencies remain unpinned; real-data
verification, baseline training, and CI remain unfinished.

## Limitations and Future Work

Limitations, threats to validity, and possible extensions will be documented after the experimental scope is finalized.
