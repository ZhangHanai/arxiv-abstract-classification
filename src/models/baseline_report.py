"""Render a compact, measured baseline comparison from recorded artifacts."""

import json
from pathlib import Path


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def write_report(path, manifest, candidates, comparison, errors):
    names = ("logistic_regression", "linear_svm")
    display = {"logistic_regression": "Logistic Regression", "linear_svm": "Linear SVM"}
    classes = manifest["data_configuration"]["classes"]
    source = manifest["source_identity"]
    provider = source["provider"]
    data = manifest["data_configuration"]
    counts = manifest["split_identity"]
    count_text = " / ".join(str(counts[name]["rows"]) for name in ("train", "val", "test"))
    per_class_text = " / ".join(str(counts[name]["per_class"][classes[0]]) for name in ("train", "val", "test"))
    lines = [
        "# First classical baseline experiment", "",
        f"Completed {manifest['completed_utc']}. Dataset: {provider.get('dataset_ref', 'fixture')} "
        f"version {provider.get('dataset_version', 'fixture')}. The {len(classes)}-class results below use the "
        "first category token as the operational label. The real-data curation policy and exact "
        "exclusion counts are recorded in `REAL_DATA_REPORT.md` and the verified source manifest.", "",
        "**Data and exact split identity**", "",
        f"Data configuration: seed {data['seed']}, cap {data['data']['samples_per_class']:,} per class, "
        f"seeded stratified fractions `{json.dumps(data['data']['splits'])}`. "
        f"Both models use the same {count_text} train / validation / test rows "
        f"({per_class_text} per class). The complete public loader passed its integrity checks, "
        "and every split matched the previously verified parquet and ordered-row hashes.", "",
        f"Original source SHA-256: `{source['original_source'].get('sha256', 'fixture')}`.  ",
        f"Curated source SHA-256: `{source['curated_source'].get('sha256', 'fixture')}`.", "",
        _table(["Split", "Rows", "Parquet SHA-256", "Ordered id/text/label SHA-256"], [
            [name, identity["rows"], f"`{identity['sha256']}`", f"`{identity['ordered_rows_sha256']}`"]
            for name, identity in manifest["split_identity"].items()
        ]), "",
        "**Configuration and selection discipline**", "",
        "Identical word TF-IDF for both models: "
        f"`{json.dumps(manifest['experiment']['tfidf'], sort_keys=True)}`; float64, no text truncation. "
        "Bigrams retain short phrases; minimum document frequency and the vocabulary cap limit rare "
        "features; sublinear TF reduces the influence of repeated terms within an abstract. "
        "These choices were fixed before test evaluation, not selected using held-out results.", "",
        "Logistic Regression: L2 multinomial loss, `solver=lbfgs`, `l1_ratio=0`, intercept enabled. "
        "Linear SVM: `LinearSVC`, L2, squared hinge, one-vs-rest, `dual=auto`, intercept enabled. "
        "No class weights, no calibration, no feature selection outside the training pipeline. "
        "Seed comes from the data configuration. Both compare only the listed C candidates. "
        "Full effective estimator defaults are recorded in `comparison.json`.", "",
        f"Selection: {manifest['selection_rule']}. Both choices were saved in "
        "`validation_selection.json` before either test prediction. Selected pipelines were retained "
        "without refitting on validation. Each selected model predicted the test rows once and "
        "was evaluated once through `src.evaluation.save_evaluation`.", "",
        _table(["Model", "C", "max_iter", "tol", "Val accuracy", "Val macro F1", "Val weighted F1", "Fit seconds"], [
            [display[name], row["configuration"]["C"], row["configuration"]["max_iter"], row["configuration"]["tol"],
             f"{row['validation_metrics']['accuracy']:.6f}", f"{row['validation_metrics']['macro_f1']:.6f}",
             f"{row['validation_metrics']['weighted_f1']:.6f}", f"{row['training_seconds']:.3f}"]
            for name in names for row in candidates[name]
        ]), "",
        "**Selected model results**", "",
        _table(["Model", "Selected C", "Split", "Accuracy", "Macro F1", "Weighted F1"], [
            [display[name], comparison[name]["configuration"]["C"], split,
             f"{comparison[name][split]['accuracy']:.6f}", f"{comparison[name][split]['macro_f1']:.6f}",
             f"{comparison[name][split]['weighted_f1']:.6f}"]
            for name in names for split in ("validation", "test")
        ]), "",
        f"Highest held-out macro F1: **{display[comparison['best_by_held_out_macro_f1']]}**. "
        f"Linear SVM minus Logistic Regression test macro F1: "
        f"{comparison['linear_svm']['test']['macro_f1'] - comparison['logistic_regression']['test']['macro_f1']:+.6f}. "
        "This is an observed comparison on one fixed split; no significance or repeated-run claim is made.", "",
        _table(["Model", "Selected fit s", "All candidate fits s", "Val inference s", "Test inference s", "Test diagnostics s", "Features"], [
            [display[name], *[f"{comparison[name][key]:.3f}" for key in (
                "selected_training_seconds", "all_candidate_training_seconds", "validation_prediction_seconds",
                "test_prediction_seconds", "test_score_seconds",
            )], comparison[name]["vocabulary_size"]] for name in names
        ]), "",
        "Times are wall-clock measurements on the recorded CPU runtime with "
        f"{manifest['experiment']['threads']} numerical thread(s). Fit includes TF-IDF fitting and transformation; "
        f"inference includes TF-IDF transformation plus label prediction for all {counts['test']['rows']:,} test rows. "
        "Diagnostic probability/margin computation is timed separately on the same transformed matrix. "
        "Artifact I/O and data loading are excluded. Candidate warning/iteration records are in "
        "`validation_candidates.json`; a convergence warning aborts before test evaluation.", "",
    ]
    for name in names:
        metrics = comparison[name]["test"]
        lines.extend([
            f"**{display[name]}: held-out per-class metrics**", "",
            _table(["Class", "Precision", "Recall", "F1", "Support"], [
                [label, *[f"{metrics['per_class'][label][key]:.6f}" for key in ("precision", "recall", "f1")],
                 metrics["per_class"][label]["support"]] for label in classes
            ]), "",
            "Confusion matrix: rows are true labels, columns predicted labels; configured order is retained.", "",
            _table(["True / predicted", *classes], [
                [label, *metrics["confusion_matrix"][index]] for index, label in enumerate(classes)
            ]), "",
            _table(["True label", "Predicted label", "Count"], [
                [pair["true_label"], pair["predicted_label"], pair["count"]]
                for pair in errors[name]["top_confusion_pairs"][:5]
            ]), "",
            "Lowest F1 classes: " + "; ".join(f"{row['label']} ({row['f1']:.6f})" for row in errors[name]["lowest_f1_classes"]) + ".", "",
        ])
        for pair in errors[name]["expected_overlap_pairs"]:
            left, right = pair["classes"]
            major = any(rank is not None and rank <= 10 for rank in (pair["left_to_right_rank"], pair["right_to_left_rank"]))
            lines.append(
                f"- {left} / {right}: {left} -> {right} = {pair['left_to_right']} "
                f"(directed rank {pair['left_to_right_rank']}); reverse = {pair['right_to_left']} "
                f"(rank {pair['right_to_left_rank']}). "
                + ("At least one direction is a major observed pair." if major else "Neither direction is in the top ten observed pairs.")
            )
        lines.append("")
    confidence = errors["logistic_regression"]["high_confidence_errors"]
    lines.extend([
        "**Error-analysis preparation and artifacts**", "",
        f"Logistic Regression has {confidence['count']} incorrect predictions with predicted-class "
        f"probability >= {confidence['threshold']}; the top {confidence['saved_count']} IDs and probabilities "
        "are in `logistic_regression/test/high_confidence_errors.csv`. These probabilities are not "
        "separately calibrated. No causal explanation of these errors is claimed. Linear SVM stores "
        "decision scores and has no `predict_proba`; its margins are not probabilities.", "",
        "Each model/split directory contains the shared `metrics.json`, labeled `confusion_matrix.csv`, "
        "and locally retained `predictions.parquet` (ordered IDs, original text, true/predicted label, correct flag). "
        "Test `prediction_scores.csv` retains IDs and all score columns in configured label order. "
        "Prediction rows and score tables are ignored by Git. Only small metrics, matrices, summaries, "
        "configuration and provenance are committed. No trained model, dataset, sparse matrix or cache is committed.", "",
        "Validation per-class metrics and matrices are included in each `validation/metrics.json` and "
        "`validation/confusion_matrix.csv`. The frozen selection, code/configuration hashes, exact package "
        "versions, split hashes, timings and output hashes are recorded in `run_manifest.json`, "
        "`validation_selection.json`, `validation_candidates.json` and `comparison.json`.", "",
        "**Limitations and next comparison**", "",
        "The balanced eight-class cap does not reflect natural class prevalence. The split is random, "
        "not temporal or an external generalization test. First-token labels are operational and have "
        "not been verified as an authoritative primary category; multi-category papers and semantic "
        "overlap remain. Exact/normalized duplicates are checked, but paraphrases and near duplicates "
        "are not audited. The source-quality flags (short abstracts and withdrawal markers) are retained. "
        "This small validation search evaluates only two C values for each model, with one seed and no "
        "confidence intervals. Word TF-IDF represents lexical patterns without contextual embeddings. "
        "Observed confusion pairs identify candidates for later inspection and do not establish a cause.", "",
        "The verified dataset, fixed splits, shared metrics and reproducible baseline references are "
        "ready for a separately scoped DistilBERT comparison. Its tokenizer, training loop, compute "
        "budget and validation-only configuration choices still need implementation. The test set is "
        "now an observed benchmark; future configuration choices must continue to use validation only. "
        "No Transformer was implemented or trained in this increment.", "",
        "Reproduce in the recorded package environment after reproducing the verified data bundle, "
        "using a fresh output directory:", "",
        "```sh", manifest["provenance"].get("command", "python -m src.models.baselines --help"), "```", "",
        "Estimator semantics: [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), "
        "[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), "
        "[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).", "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
