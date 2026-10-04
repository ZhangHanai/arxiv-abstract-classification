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

## Models

The first real experiment implements TF-IDF + Logistic Regression and TF-IDF +
Linear SVM. Both use word unigrams/bigrams, `min_df=3`, at most 100,000 features,
sublinear TF and the identical verified splits. Only `C=1` and `C=4` were compared
on validation; both models selected `C=1`. Transformer training remains future
work.

## Evaluation

`src/evaluation.py` provides shared accuracy, macro/support-weighted F1,
per-class scores, ordered confusion matrices, and aligned prediction artifacts.
The first held-out evaluation uses Cornell/Kaggle snapshot version 306 and
44,800 / 5,600 / 5,600 train/validation/test rows. Each selected train-only
pipeline was evaluated once on the test set after both choices were frozen.

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.824098 | 0.826607 | 0.825156 | 0.825156 |
| Linear SVM | 0.821041 | 0.822500 | 0.820346 | 0.820346 |

See the [baseline report](results/classical_baselines/real_v1/report.md) for
candidate results, full per-class precision/recall/F1/support, confusion matrices,
measured timings, exact split identities and limitations. The
[run manifest](results/classical_baselines/real_v1/run_manifest.json) records the
clean implementation commit, code/configuration hashes and runtime versions.

## Error Analysis

Aligned validation/test predictions are saved through the shared artifact
interface and remain local. Confusion rankings and low-F1 classes are prepared
for later inspection; 17 Logistic Regression errors have predicted-class
probability at least 0.9. These probabilities are not separately calibrated.
Linear SVM stores decision scores, with no invented probabilities. The observed
`cs.AI`/`cs.LG`, `cs.LG`/`cs.CV`, and `cs.LG`/`cs.CL` confusions are quantified
in the report without causal explanations.

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
mostly unpinned; exact experimental versions are in the run manifest, and remote
CI remains unfinished.

After reproducing the verified data bundle, run the classical experiment with
a fresh output directory:

```sh
python -B -m src.models.baselines --config config/real_data.yaml --experiment-config config/classical_baselines.yaml --output-dir results/classical_baselines/my_run
```

The runner requires the saved real-data readiness profile and matches all three
loaded splits to the verified manifest. It fits every candidate only on training,
selects by validation macro F1 (first listed C breaks exact ties), freezes both
choices, retains the selected fitted pipelines and performs one final test pass
per model. It refuses to overwrite an existing run. Training times include
TF-IDF fitting; inference includes transformation and label prediction. Score
diagnostics are timed separately. Full prediction parquet and score CSV files
are ignored, and no model binaries or sparse matrices are serialized.

## Limitations and Future Work

The balanced random split is not a temporal or external evaluation. Operational
first-token labels, semantic overlap, source-quality flags and unaudited near
duplicates remain limitations. Only two regularization values per model and one
seed were evaluated; no statistical significance claim is made. The data and
baseline references are ready for a separately scoped DistilBERT comparison,
with future choices made on validation. No Transformer was implemented or trained.
