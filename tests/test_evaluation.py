"""Metric correctness and traceable artifact tests using synthetic predictions."""

import json

import pandas as pd
import pytest

from src.data.dataset import load_split
from src.data.preprocess import run_preprocessing
from src.evaluation import evaluate_predictions, save_evaluation


@pytest.fixture
def classes():
    return ["cs.LG", "cs.AI", "cs.CL"]


@pytest.fixture
def evaluation_data():
    return pd.DataFrame(
        {
            "id": ["paper-2", "paper-1", "paper-3", "paper-4"],
            "text": ["First abstract ∇", "Second abstract", "Third abstract", "Fourth"],
            "label": ["cs.LG", "cs.LG", "cs.AI", "cs.AI"],
        },
        index=[20, 10, 40, 30],
    )


def test_metrics_match_hand_calculated_multiclass_example(classes):
    metrics = evaluate_predictions(
        ["cs.LG", "cs.LG", "cs.AI", "cs.AI"],
        ["cs.LG", "cs.AI", "cs.AI", "cs.CL"],
        classes,
    )

    assert metrics["classes"] == classes
    assert metrics["n_samples"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["weighted_f1"] == pytest.approx(7 / 12)
    assert list(metrics["per_class"]) == classes
    assert metrics["per_class"]["cs.LG"] == pytest.approx(
        {"precision": 1, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )
    assert metrics["per_class"]["cs.AI"] == {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2,
    }
    assert metrics["per_class"]["cs.CL"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_absent_classes_keep_axes_and_count_in_macro_f1(classes):
    metrics = evaluate_predictions(["cs.AI"], ["cs.AI"], classes)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["weighted_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert metrics["per_class"]["cs.LG"]["support"] == 0


def test_completely_wrong_predictions_have_zero_scores(classes):
    metrics = evaluate_predictions(
        ["cs.LG", "cs.AI", "cs.CL"], ["cs.AI", "cs.CL", "cs.LG"], classes,
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["weighted_f1"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


def test_series_labels_are_compared_by_position(classes):
    metrics = evaluate_predictions(
        pd.Series(["cs.AI", "cs.LG"], index=[5, 3]),
        pd.Series(["cs.AI", "cs.LG"], index=[100, 200]),
        classes,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["n_samples"] == 2


def test_weighted_f1_uses_true_class_support(classes):
    metrics = evaluate_predictions(
        ["cs.AI", "cs.AI", "cs.AI", "cs.LG"],
        ["cs.AI", "cs.AI", "cs.LG", "cs.LG"],
        classes,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(22 / 45)
    assert metrics["weighted_f1"] == pytest.approx(23 / 30)


def test_evaluation_integrates_with_existing_preprocessing_and_loader(tmp_path, classes):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_records = [
        {"id": f"{label}-{index}", "abstract": f"Abstract\n{label} {index}", "categories": label}
        for label in classes
        for index in range(10)
    ]
    (raw_dir / "synthetic.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records), encoding="utf-8",
    )
    config = {
        "seed": 42,
        "classes": classes,
        "data": {
            "raw_filename": "synthetic.jsonl", "samples_per_class": 10,
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "paths": {"raw_data": raw_dir, "processed_data": tmp_path / "processed"},
    }
    run_preprocessing(config)
    validation = load_split("val", config)
    predictions = validation["label"].tolist()
    predictions[0] = next(label for label in classes if label != predictions[0])

    paths = save_evaluation(validation, predictions, classes, tmp_path / "results" / "val")

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    rows = pd.read_parquet(paths["predictions"])
    assert metrics["n_samples"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert rows["id"].tolist() == validation["id"].tolist()
    assert rows["predicted_label"].tolist() == predictions
    assert rows["correct"].sum() == 2


@pytest.mark.parametrize(
    ("true_labels", "predictions", "message"),
    [
        ([], [], "at least one"),
        (["cs.AI"], [], "same length"),
        ([], ["cs.AI"], "same length"),
        (["math.CO"], ["cs.AI"], "y_true.*outside"),
        (["cs.AI"], ["math.CO"], "y_pred.*outside"),
        ([None], ["cs.AI"], "y_true.*outside"),
        (["cs.AI"], [0], "y_pred.*outside"),
        ("cs.AI", ["cs.AI"], "y_true.*ordered"),
        (["cs.AI"], "cs.AI", "y_pred.*ordered"),
        (["cs.AI"], {"cs.AI"}, "y_pred.*ordered"),
    ],
)
def test_invalid_predictions_fail_clearly(true_labels, predictions, message, classes):
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(true_labels, predictions, classes)


@pytest.mark.parametrize(
    ("invalid_classes", "message"),
    [
        ([], "nonempty string"),
        (["cs.AI", "cs.AI"], "unique"),
        ([None], "nonempty string"),
        ([""], "nonempty string"),
        ([" "], "nonempty string"),
        ("cs.AI", "ordered"),
        ({"cs.AI", "cs.LG"}, "ordered"),
        (None, "ordered"),
        ({"cs.AI": 0}, "ordered"),
    ],
)
def test_invalid_class_definitions_fail_clearly(invalid_classes, message):
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(["cs.AI"], ["cs.AI"], invalid_classes)


def test_saved_artifacts_round_trip_with_class_and_row_order(
    tmp_path, evaluation_data, classes,
):
    before = evaluation_data.copy(deep=True)
    predictions = pd.Series(["cs.LG", "cs.AI", "cs.AI", "cs.CL"], index=[1, 2, 3, 4])
    output_dir = tmp_path / "run" / "val"

    paths = save_evaluation(evaluation_data, predictions, classes, str(output_dir))

    assert paths == {
        "metrics": output_dir / "metrics.json",
        "confusion_matrix": output_dir / "confusion_matrix.csv",
        "predictions": output_dir / "predictions.parquet",
    }
    saved_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert saved_metrics == evaluate_predictions(evaluation_data["label"], predictions, classes)
    matrix = pd.read_csv(paths["confusion_matrix"], index_col=0)
    assert matrix.index.name == "true_label"
    assert list(matrix.index) == classes
    assert list(matrix.columns) == classes
    assert matrix.values.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    rows = pd.read_parquet(paths["predictions"])
    assert list(rows.columns) == ["id", "text", "true_label", "predicted_label", "correct"]
    assert rows["id"].tolist() == evaluation_data["id"].tolist()
    assert rows["text"].tolist() == evaluation_data["text"].tolist()
    assert rows["true_label"].tolist() == evaluation_data["label"].tolist()
    assert rows["predicted_label"].tolist() == predictions.tolist()
    assert rows["correct"].tolist() == [True, False, True, False]
    pd.testing.assert_frame_equal(evaluation_data, before)


@pytest.mark.parametrize("missing_column", ["id", "text", "label"])
def test_artifact_saving_rejects_missing_columns_before_writing(
    tmp_path, evaluation_data, classes, missing_column,
):
    output_dir = tmp_path / "invalid"

    with pytest.raises(ValueError, match=f"missing required columns: {missing_column}"):
        save_evaluation(
            evaluation_data.drop(columns=[missing_column]),
            evaluation_data["label"], classes, output_dir,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("column", "value"),
    [(column, value) for column in ("id", "text") for value in (None, "", " ", 17)],
)
def test_artifact_saving_requires_traceable_ids_and_nonempty_text(
    tmp_path, evaluation_data, classes, column, value,
):
    invalid = evaluation_data.astype({column: "object"})
    invalid.loc[20, column] = value
    output_dir = tmp_path / "invalid"

    with pytest.raises(ValueError, match=f"{column} values must be nonempty strings"):
        save_evaluation(invalid, invalid["label"], classes, output_dir)

    assert not output_dir.exists()


def test_artifact_saving_rejects_duplicate_paper_ids(tmp_path, evaluation_data, classes):
    evaluation_data.loc[10, "id"] = evaluation_data.loc[20, "id"]
    output_dir = tmp_path / "invalid"

    with pytest.raises(ValueError, match="paper IDs must be unique"):
        save_evaluation(evaluation_data, evaluation_data["label"], classes, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("bad_input", ["true_label", "prediction", "length", "empty"])
def test_artifact_saving_validates_labels_before_creating_outputs(
    tmp_path, evaluation_data, classes, bad_input,
):
    predictions = evaluation_data["label"].tolist()
    if bad_input == "true_label":
        evaluation_data.loc[20, "label"] = "math.CO"
    elif bad_input == "prediction":
        predictions[0] = "math.CO"
    elif bad_input == "length":
        predictions.pop()
    else:
        evaluation_data = evaluation_data.iloc[:0]
        predictions = []
    output_dir = tmp_path / "invalid"

    with pytest.raises(ValueError):
        save_evaluation(evaluation_data, predictions, classes, output_dir)

    assert not output_dir.exists()
