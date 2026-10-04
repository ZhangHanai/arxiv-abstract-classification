# First classical baseline experiment

Completed 2026-10-04T13:38:54.218839+00:00. Dataset: Cornell-University/arxiv version 306. The 8-class results below use the first category token as the operational label. The real-data curation policy and exact exclusion counts are recorded in `REAL_DATA_REPORT.md` and the verified source manifest.

**Data and exact split identity**

Data configuration: seed 42, cap 7,000 per class, seeded stratified fractions `{"train": 0.8, "val": 0.1, "test": 0.1}`. Both models use the same 44800 / 5600 / 5600 train / validation / test rows (5600 / 700 / 700 per class). The complete public loader passed its integrity checks, and every split matched the previously verified parquet and ordered-row hashes.

Original source SHA-256: `b5c73be5958b20f4c07ea30a9352941bddfe59fed943dde6614e6d3415347c88`.
Curated source SHA-256: `d0100471cc2da0ec4d998e5754c79fef0f9bef2271c69fe8efd861a90c66ae68`.

| Split | Rows | Parquet SHA-256 | Ordered id/text/label SHA-256 |
| --- | --- | --- | --- |
| train | 44800 | `57a86d4b1eefe02fde53f98666d152170b9a18bd6b610d75332344dc1bf8c1a0` | `151bafcaa005992c08db9927a997497b0d5cebaaeb1c7339e07fc55f20f16773` |
| val | 5600 | `bb86a1bf21c90e65afb52344bf4e2d6fe08bc1654deac66b3c38fe0b6f82ac34` | `f057f724d1714b8c495baf41339a88586a06388eba766a9f3bc36565e67fc6ab` |
| test | 5600 | `539302fa2ae5f1e925b3ea8cbcee57731134ecbcb462bef8f0bd1556013e06e8` | `383ca28352286783a28a4c3ccf63cf9db39a544ba9903c18a7071a9e9e39776a` |

**Configuration and selection discipline**

Identical word TF-IDF for both models: `{"lowercase": true, "max_df": 1.0, "max_features": 100000, "min_df": 3, "ngram_range": [1, 2], "norm": "l2", "smooth_idf": true, "stop_words": null, "strip_accents": null, "sublinear_tf": true, "token_pattern": "(?u)\\b\\w\\w+\\b", "use_idf": true}`; float64, no text truncation. Bigrams retain short phrases; minimum document frequency and the vocabulary cap limit rare features; sublinear TF reduces the influence of repeated terms within an abstract. These choices were fixed before test evaluation, not selected using held-out results.

Logistic Regression: L2 multinomial loss, `solver=lbfgs`, `l1_ratio=0`, intercept enabled. Linear SVM: `LinearSVC`, L2, squared hinge, one-vs-rest, `dual=auto`, intercept enabled. No class weights, no calibration, no feature selection outside the training pipeline. Seed comes from the data configuration. Both compare only the listed C candidates. Full effective estimator defaults are recorded in `comparison.json`.

Selection: maximum validation macro_f1; exact ties use first listed C. Both choices were saved in `validation_selection.json` before either test prediction. Selected pipelines were retained without refitting on validation. Each selected model predicted the test rows once and was evaluated once through `src.evaluation.save_evaluation`.

| Model | C | max_iter | tol | Val accuracy | Val macro F1 | Val weighted F1 | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0 | 1000 | 0.0001 | 0.824821 | 0.824098 | 0.824098 | 41.534 |
| Logistic Regression | 4.0 | 1000 | 0.0001 | 0.823393 | 0.822943 | 0.822943 | 53.299 |
| Linear SVM | 1.0 | 5000 | 0.0001 | 0.823214 | 0.821041 | 0.821041 | 28.186 |
| Linear SVM | 4.0 | 5000 | 0.0001 | 0.811429 | 0.809876 | 0.809876 | 34.085 |

**Selected model results**

| Model | Selected C | Split | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0 | validation | 0.824821 | 0.824098 | 0.824098 |
| Logistic Regression | 1.0 | test | 0.826607 | 0.825156 | 0.825156 |
| Linear SVM | 1.0 | validation | 0.823214 | 0.821041 | 0.821041 |
| Linear SVM | 1.0 | test | 0.822500 | 0.820346 | 0.820346 |

Highest held-out macro F1: **Logistic Regression**. Linear SVM minus Logistic Regression test macro F1: -0.004810. This is an observed comparison on one fixed split; no significance or repeated-run claim is made.

| Model | Selected fit s | All candidate fits s | Val inference s | Test inference s | Test diagnostics s | Features |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 41.534 | 94.833 | 1.293 | 1.372 | 0.015 | 100000 |
| Linear SVM | 28.186 | 62.270 | 1.626 | 1.392 | 0.007 | 100000 |

Times are wall-clock measurements on the recorded CPU runtime with 1 numerical thread(s). Fit includes TF-IDF fitting and transformation; inference includes TF-IDF transformation plus label prediction for all 5,600 test rows. Diagnostic probability/margin computation is timed separately on the same transformed matrix. Artifact I/O and data loading are excluded. Candidate warning/iteration records are in `validation_candidates.json`; a convergence warning aborts before test evaluation.

**Logistic Regression: held-out per-class metrics**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cs.AI | 0.675591 | 0.612857 | 0.642697 | 700 |
| cs.LG | 0.733906 | 0.732857 | 0.733381 | 700 |
| cs.CL | 0.782369 | 0.811429 | 0.796634 | 700 |
| cs.CV | 0.847887 | 0.860000 | 0.853901 | 700 |
| cs.CR | 0.868228 | 0.847143 | 0.857556 | 700 |
| cs.RO | 0.915612 | 0.930000 | 0.922750 | 700 |
| cs.DS | 0.903096 | 0.958571 | 0.930007 | 700 |
| cs.SE | 0.868687 | 0.860000 | 0.864322 | 700 |

Confusion matrix: rows are true labels, columns predicted labels; configured order is retained.

| True / predicted | cs.AI | cs.LG | cs.CL | cs.CV | cs.CR | cs.RO | cs.DS | cs.SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cs.AI | 429 | 82 | 75 | 28 | 10 | 28 | 27 | 21 |
| cs.LG | 52 | 513 | 40 | 36 | 20 | 8 | 20 | 11 |
| cs.CL | 74 | 26 | 568 | 13 | 6 | 1 | 1 | 11 |
| cs.CV | 12 | 41 | 13 | 602 | 10 | 17 | 3 | 2 |
| cs.CR | 20 | 13 | 12 | 14 | 593 | 0 | 13 | 35 |
| cs.RO | 13 | 6 | 0 | 15 | 0 | 651 | 6 | 9 |
| cs.DS | 7 | 11 | 0 | 0 | 8 | 1 | 671 | 2 |
| cs.SE | 28 | 7 | 18 | 2 | 36 | 5 | 2 | 602 |

| True label | Predicted label | Count |
| --- | --- | --- |
| cs.AI | cs.LG | 82 |
| cs.AI | cs.CL | 75 |
| cs.CL | cs.AI | 74 |
| cs.LG | cs.AI | 52 |
| cs.CV | cs.LG | 41 |

Lowest F1 classes: cs.AI (0.642697); cs.LG (0.733381); cs.CL (0.796634).

- cs.AI / cs.LG: cs.AI -> cs.LG = 82 (directed rank 1); reverse = 52 (rank 4). At least one direction is a major observed pair.
- cs.LG / cs.CV: cs.LG -> cs.CV = 36 (directed rank 7); reverse = 41 (rank 5). At least one direction is a major observed pair.
- cs.LG / cs.CL: cs.LG -> cs.CL = 40 (directed rank 6); reverse = 26 (rank 14). At least one direction is a major observed pair.

**Linear SVM: held-out per-class metrics**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cs.AI | 0.656911 | 0.577143 | 0.614449 | 700 |
| cs.LG | 0.712230 | 0.707143 | 0.709677 | 700 |
| cs.CL | 0.785515 | 0.805714 | 0.795487 | 700 |
| cs.CV | 0.840170 | 0.848571 | 0.844350 | 700 |
| cs.CR | 0.862360 | 0.877143 | 0.869688 | 700 |
| cs.RO | 0.906944 | 0.932857 | 0.919718 | 700 |
| cs.DS | 0.921918 | 0.961429 | 0.941259 | 700 |
| cs.SE | 0.866287 | 0.870000 | 0.868140 | 700 |

Confusion matrix: rows are true labels, columns predicted labels; configured order is retained.

| True / predicted | cs.AI | cs.LG | cs.CL | cs.CV | cs.CR | cs.RO | cs.DS | cs.SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cs.AI | 404 | 89 | 76 | 35 | 12 | 29 | 22 | 33 |
| cs.LG | 66 | 495 | 38 | 35 | 22 | 10 | 20 | 14 |
| cs.CL | 73 | 28 | 564 | 15 | 5 | 3 | 0 | 12 |
| cs.CV | 14 | 46 | 11 | 594 | 10 | 20 | 2 | 3 |
| cs.CR | 14 | 14 | 11 | 14 | 614 | 1 | 9 | 23 |
| cs.RO | 13 | 8 | 2 | 13 | 1 | 653 | 3 | 7 |
| cs.DS | 8 | 8 | 0 | 0 | 8 | 1 | 673 | 2 |
| cs.SE | 23 | 7 | 16 | 1 | 40 | 3 | 1 | 609 |

| True label | Predicted label | Count |
| --- | --- | --- |
| cs.AI | cs.LG | 89 |
| cs.AI | cs.CL | 76 |
| cs.CL | cs.AI | 73 |
| cs.LG | cs.AI | 66 |
| cs.CV | cs.LG | 46 |

Lowest F1 classes: cs.AI (0.614449); cs.LG (0.709677); cs.CL (0.795487).

- cs.AI / cs.LG: cs.AI -> cs.LG = 89 (directed rank 1); reverse = 66 (rank 4). At least one direction is a major observed pair.
- cs.LG / cs.CV: cs.LG -> cs.CV = 35 (directed rank 9); reverse = 46 (rank 5). At least one direction is a major observed pair.
- cs.LG / cs.CL: cs.LG -> cs.CL = 38 (directed rank 7); reverse = 28 (rank 12). At least one direction is a major observed pair.

**Error-analysis preparation and artifacts**

Logistic Regression has 17 incorrect predictions with predicted-class probability >= 0.9; the top 17 IDs and probabilities are in `logistic_regression/test/high_confidence_errors.csv`. These probabilities are not separately calibrated. No causal explanation of these errors is claimed. Linear SVM stores decision scores and has no `predict_proba`; its margins are not probabilities.

Each model/split directory contains the shared `metrics.json`, labeled `confusion_matrix.csv`, and locally retained `predictions.parquet` (ordered IDs, original text, true/predicted label, correct flag). Test `prediction_scores.csv` retains IDs and all score columns in configured label order. Prediction rows and score tables are ignored by Git. Only small metrics, matrices, summaries, configuration and provenance are committed. No trained model, dataset, sparse matrix or cache is committed.

Validation per-class metrics and matrices are included in each `validation/metrics.json` and `validation/confusion_matrix.csv`. The frozen selection, code/configuration hashes, exact package versions, split hashes, timings and output hashes are recorded in `run_manifest.json`, `validation_selection.json`, `validation_candidates.json` and `comparison.json`.

**Limitations and next comparison**

The balanced eight-class cap does not reflect natural class prevalence. The split is random, not temporal or an external generalization test. First-token labels are operational and have not been verified as an authoritative primary category; multi-category papers and semantic overlap remain. Exact/normalized duplicates are checked, but paraphrases and near duplicates are not audited. The source-quality flags (short abstracts and withdrawal markers) are retained. This small validation search evaluates only two C values for each model, with one seed and no confidence intervals. Word TF-IDF represents lexical patterns without contextual embeddings. Observed confusion pairs identify candidates for later inspection and do not establish a cause.

The verified dataset, fixed splits, shared metrics and reproducible baseline references are ready for a separately scoped DistilBERT comparison. Its tokenizer, training loop, compute budget and validation-only configuration choices still need implementation. The test set is now an observed benchmark; future configuration choices must continue to use validation only. No Transformer was implemented or trained in this increment.

Reproduce in the recorded package environment after reproducing the verified data bundle, using a fresh output directory:

```sh
python -B -m src.models.baselines --config config/real_data.yaml --experiment-config config/classical_baselines.yaml --verified-manifest results/real_data_validation/run_manifest.json --output-dir results/classical_baselines/real_v1
```

Estimator semantics: [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

**Repository verification**

PR #6 and PR #7 were merged with normal merge commits, and local main was updated before the model branch. PR #7 passed 137 tests after rebasing; the final baseline suite passed 150 tests in 10.51s with zero failures or skips. The 13 focused behavioral tests passed separately. Dependency and artifact checks passed. See [verification.json](verification.json) for exact commands, named tests, artifact checks and the complete 24-file change inventory.

The command above records the original run. For reproduction, replace its output path with a fresh directory such as `results/classical_baselines/reproduction_v1`; the committed `real_v1` directory is deliberately protected from overwrite.

**Portable provenance correction before merge**

The original Windows run hashed CRLF working-tree text. Git stored LF text, so those identities were not portable. The manifest now records LF artifact identities and source hashes from the original clean training commit; original observed working-tree hashes are retained separately. Text writers and Git attributes enforce LF for future runs. No model was retrained, no predictions were regenerated, and no metrics or model choices changed.
