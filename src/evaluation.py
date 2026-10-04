"""Shared classification metrics and prediction artifacts, without model fitting."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _label_list(values, name):
    """Require an ordered collection rather than a scalar or unordered labels."""
    if isinstance(values, (str, bytes, set, frozenset, dict)):
        raise ValueError(f"{name} must be an ordered collection of labels")
    try:
        return list(values)
    except TypeError:
        raise ValueError(f"{name} must be an ordered collection of labels") from None


def evaluate_predictions(y_true, y_pred, classes):
    """Return JSON-ready metrics with every axis in configured class order.

    Macro F1 includes every configured class, including classes with no true or
    predicted examples. Undefined precision/recall/F1 values are zero. Weighted
    F1 uses true-label support. Confusion-matrix rows are true labels and columns
    are predicted labels. Inputs must be nonempty, aligned string-label sequences.
    """
    label_order = _label_list(classes, "classes")
    if not label_order or any(
        not isinstance(label, str) or not label.strip() for label in label_order
    ):
        raise ValueError("classes must contain nonempty string labels")
    if len(set(label_order)) != len(label_order):
        raise ValueError("classes must contain unique labels")

    true_labels = _label_list(y_true, "y_true")
    predicted_labels = _label_list(y_pred, "y_pred")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_labels:
        raise ValueError("Evaluation requires at least one example")

    allowed_labels = set(label_order)
    for name, labels in (("y_true", true_labels), ("y_pred", predicted_labels)):
        invalid_labels = [
            label
            for label in labels
            if not isinstance(label, str) or label not in allowed_labels
        ]
        if invalid_labels:
            raise ValueError(
                f"{name} contains labels outside configured classes: {invalid_labels!r}"
            )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_order, zero_division=0
    )
    return {
        "classes": label_order,
        "n_samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels, predicted_labels,
                labels=label_order, average="macro", zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels, predicted_labels,
                labels=label_order, average="weighted", zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(label_order)
        },
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=label_order
        ).tolist(),
    }


def save_evaluation(dataframe, predictions, classes, output_dir):
    """Save metrics, a labeled matrix CSV, and aligned prediction parquet rows.

    Consume the dataset loader's id/text/label schema without mutating it. A
    caller chooses a separate output directory for each run/split, preferably
    beneath a resolved configuration path. This function does not fit models,
    choose hyperparameters, or check overlap with other dataset splits.
    """
    missing_columns = {"id", "text", "label"}.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Evaluation data is missing required columns: {missing}")

    for column in ("id", "text"):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in dataframe[column]
        ):
            raise ValueError(f"Evaluation {column} values must be nonempty strings")
    if dataframe["id"].duplicated().any():
        raise ValueError("Evaluation paper IDs must be unique")

    predicted_labels = _label_list(predictions, "predictions")
    metrics = evaluate_predictions(dataframe["label"], predicted_labels, classes)
    prediction_rows = dataframe[["id", "text", "label"]].copy()
    prediction_rows = prediction_rows.rename(columns={"label": "true_label"})
    prediction_rows["predicted_label"] = predicted_labels
    prediction_rows["correct"] = (
        prediction_rows["true_label"] == prediction_rows["predicted_label"]
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_path / "metrics.json",
        "confusion_matrix": output_path / "confusion_matrix.csv",
        "predictions": output_path / "predictions.parquet",
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=pd.Index(metrics["classes"], name="true_label"),
        columns=metrics["classes"],
    )
    matrix.to_csv(paths["confusion_matrix"], encoding="utf-8")
    prediction_rows.to_parquet(paths["predictions"], index=False)
    return paths
