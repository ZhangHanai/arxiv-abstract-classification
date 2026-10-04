"""Behavioral boundaries for real baseline fitting, selection, and artifacts."""

from copy import deepcopy
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.dataset import load_splits
from src.data.profile import file_sha256, logical_digest
from src.evaluation import evaluate_predictions
from src.models import baselines


@pytest.fixture
def bundle(tmp_path):
    classes = ["cs.LG", "cs.AI", "cs.CL"]  # Deliberately not alphabetical.
    topics = {"cs.LG": "learning regression gradient", "cs.AI": "planning search reasoning", "cs.CL": "language translation parsing"}
    frames = {}
    for split, count in (("train", 5), ("val", 2), ("test", 2)):
        rows = [
            {"id": f"{split}-{label}-{index}", "label": label,
             "text": f"{topics[label]} {split}onlytoken unique{split}{label.replace('.', '')}{index}"}
            for label in classes for index in range(count)
        ]
        frames[split] = pd.DataFrame(rows[::-1])
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, frame in frames.items():
        frame.to_parquet(processed / f"{name}.parquet", index=False)
    config = {
        "seed": 42, "classes": classes,
        "data": {"raw_filename": "fixture.jsonl", "samples_per_class": 9, "decode_errors": "strict",
                 "splits": {"train": .6, "val": .2, "test": .2}},
        "paths": {"processed_data": processed, "raw_data": tmp_path / "raw"},
    }
    experiment = {
        "experiment_name": "synthetic_boundary_test", "threads": 1,
        "tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 1000, "sublinear_tf": True},
        "models": {name: {"C": [1.0, 4.0], "max_iter": 1000, "tol": 1e-4} for name in baselines.MODEL_NAMES},
    }
    return config, experiment, frames


def verified_manifest(config, frames):
    processed = config["paths"]["processed_data"]
    return {
        "configuration": {key: config[key] for key in ("seed", "classes", "data")},
        "artifacts": {name: {
            "filename": f"{name}.parquet", "bytes": (processed / f"{name}.parquet").stat().st_size,
            "sha256": file_sha256(processed / f"{name}.parquet"), "ordered_rows_sha256": logical_digest(frame),
        } for name, frame in frames.items()},
    }


def test_pipeline_construction_supports_configurable_word_features(bundle):
    config, experiment, _ = bundle
    baselines.validate_experiment(experiment)
    for name, expected in (("logistic_regression", LogisticRegression), ("linear_svm", LinearSVC)):
        settings = {**experiment["models"][name], "C": 4.0}
        pipeline = baselines.build_pipeline(name, experiment["tfidf"], settings, config["seed"])
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["classifier"], expected)
        assert pipeline.named_steps["classifier"].random_state == 42
        assert pipeline.named_steps["classifier"].C == 4.0
        vectorizer = pipeline.named_steps["tfidf"]
        assert vectorizer.analyzer == "word"
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 1 and vectorizer.max_features == 1000 and vectorizer.sublinear_tf
        assert not hasattr(vectorizer, "vocabulary_")
    assert not hasattr(pipeline, "predict_proba")  # The final pipeline is LinearSVC.


def test_selection_fits_only_training_rows_and_preserves_training_idf(bundle, monkeypatch):
    config, experiment, frames = bundle
    calls = []
    original_fit = Pipeline.fit

    def recording_fit(self, texts, labels, **kwargs):
        calls.append((list(texts), list(labels)))
        return original_fit(self, texts, labels, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    winner, rows = baselines.select_on_validation(
        "logistic_regression", frames["train"], frames["val"], config["classes"], experiment, 42,
    )
    assert len(calls) == 2
    assert all(texts == frames["train"]["text"].tolist() and labels == frames["train"]["label"].tolist() for texts, labels in calls)
    tfidf = winner["pipeline"].named_steps["tfidf"]
    assert "trainonlytoken" in tfidf.vocabulary_
    assert "valonlytoken" not in tfidf.vocabulary_ and "testonlytoken" not in tfidf.vocabulary_
    training_token = tfidf.vocabulary_["trainonlytoken"]
    assert tfidf.idf_[training_token] == pytest.approx(1.0)
    # Every validation text gets an extra training-known token. It cannot change learned IDF.
    changed_val = frames["val"].copy()
    changed_val["text"] += " regression regression"
    other, _ = baselines.select_on_validation("logistic_regression", frames["train"], changed_val, config["classes"], experiment, 42)
    assert tfidf.vocabulary_ == other["pipeline"].named_steps["tfidf"].vocabulary_
    np.testing.assert_array_equal(tfidf.idf_, other["pipeline"].named_steps["tfidf"].idf_)
    assert [row["configuration"]["C"] for row in rows] == [1.0, 4.0]


def test_fixed_seed_and_tie_rule_are_deterministic(bundle):
    config, experiment, frames = bundle
    for name in baselines.MODEL_NAMES:
        first, _ = baselines.select_on_validation(name, frames["train"], frames["val"], config["classes"], experiment, 42)
        second, _ = baselines.select_on_validation(name, frames["train"], frames["val"], config["classes"], experiment, 42)
        assert first["record"]["configuration"] == second["record"]["configuration"]
        assert first["record"]["configuration"]["C"] == 1.0  # Both fixture candidates score 1.0.
        assert first["predictions"] == second["predictions"]
        np.testing.assert_allclose(first["pipeline"].named_steps["classifier"].coef_, second["pipeline"].named_steps["classifier"].coef_, rtol=0, atol=0)


def test_probabilities_and_svm_scores_use_configured_label_order(bundle):
    config, experiment, frames = bundle
    for name in baselines.MODEL_NAMES:
        winner, _ = baselines.select_on_validation(name, frames["train"], frames["val"], config["classes"], experiment, 42)
        pipeline = winner["pipeline"]
        predictions, scores, kind, seconds, score_seconds = baselines.predict_with_scores(pipeline, frames["test"]["text"], config["classes"])
        assert scores.shape == (6, 3)
        assert predictions == [config["classes"][index] for index in scores.argmax(axis=1)]
        assert seconds >= 0 and score_seconds >= 0
        if name == "logistic_regression":
            assert kind == "probabilities"
            np.testing.assert_allclose(scores.sum(axis=1), 1)
        else:
            assert kind == "decision_scores"
            assert not hasattr(pipeline, "predict_proba")
            raw = pipeline.decision_function(frames["test"]["text"].tolist())
            columns = [pipeline.classes_.tolist().index(label) for label in config["classes"]]
            np.testing.assert_allclose(scores, raw[:, columns])


def test_binary_svm_margin_order_is_explicit(bundle):
    config, experiment, frames = bundle
    classes = config["classes"][:2]
    train = frames["train"][frames["train"]["label"].isin(classes)]
    pipeline = baselines.build_pipeline("linear_svm", experiment["tfidf"], {"C": 1, "tol": 1e-4, "max_iter": 1000}, 42)
    pipeline.fit(train["text"].tolist(), train["label"].tolist())
    predictions, scores, kind, _, _ = baselines.predict_with_scores(pipeline, train["text"], classes)
    assert kind == "decision_scores" and scores.shape == (10, 2)
    np.testing.assert_allclose(scores[:, 0], -scores[:, 1])
    assert predictions == [classes[index] for index in scores.argmax(axis=1)]


def test_runner_freezes_both_choices_before_test_and_writes_aligned_artifacts(bundle, tmp_path, monkeypatch):
    config, experiment, frames = bundle
    output = tmp_path / "run"
    calls = {"load": 0, "fit": [], "test": [], "evaluate": []}
    real_fit = Pipeline.fit
    real_predict = baselines.predict_with_scores
    real_save = baselines.save_evaluation

    def public_load(settings):
        calls["load"] += 1
        return load_splits(settings)

    def train_fit(self, texts, labels, **kwargs):
        assert not (output / "validation_selection.json").exists()
        assert list(texts) == frames["train"]["text"].tolist()
        assert list(labels) == frames["train"]["label"].tolist()
        calls["fit"].append(self.named_steps["classifier"].__class__.__name__)
        return real_fit(self, texts, labels, **kwargs)

    def once_test(pipeline, texts, classes):
        frozen = json.loads((output / "validation_selection.json").read_text())
        assert set(frozen["models"]) == set(baselines.MODEL_NAMES)
        assert frozen["test_predictions_started"] is False
        assert list(texts) == frames["test"]["text"].tolist()
        calls["test"].append(pipeline.named_steps["classifier"].__class__.__name__)
        return real_predict(pipeline, texts, classes)

    def once_evaluation(frame, predictions, classes, directory):
        calls["evaluate"].append((directory.parent.name, directory.name))
        return real_save(frame, predictions, classes, directory)

    monkeypatch.setattr(baselines, "load_splits", public_load)
    monkeypatch.setattr(Pipeline, "fit", train_fit)
    monkeypatch.setattr(baselines, "predict_with_scores", once_test)
    monkeypatch.setattr(baselines, "save_evaluation", once_evaluation)
    result = baselines.run_baselines(config, experiment, verified_manifest(config, frames), output)
    assert calls["load"] == 1
    assert calls["fit"] == ["LogisticRegression", "LogisticRegression", "LinearSVC", "LinearSVC"]
    assert calls["test"] == ["LogisticRegression", "LinearSVC"]
    assert calls["evaluate"] == [(name, split) for name in baselines.MODEL_NAMES for split in ("validation", "test")]
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["test_prediction_calls"] == {name: 1 for name in baselines.MODEL_NAMES}
    assert manifest["selection_sha256_before_test"] == file_sha256(output / "validation_selection.json")
    assert manifest["label_to_id"] == {"cs.LG": 0, "cs.AI": 1, "cs.CL": 2}
    for name in baselines.MODEL_NAMES:
        for relative, identity in manifest["artifacts"][name].items():
            artifact = output / relative
            assert identity == {"bytes": artifact.stat().st_size, "sha256": file_sha256(artifact)}
            if artifact.suffix in {".json", ".csv"}:
                assert b"\r\n" not in artifact.read_bytes()
        for split, source in (("validation", frames["val"]), ("test", frames["test"])):
            directory = output / name / split
            predictions = pd.read_parquet(directory / "predictions.parquet")
            assert predictions["id"].tolist() == source["id"].tolist()
            assert predictions["text"].tolist() == source["text"].tolist()
            assert predictions["true_label"].tolist() == source["label"].tolist()
            matrix = pd.read_csv(directory / "confusion_matrix.csv", index_col=0)
            assert matrix.index.tolist() == config["classes"] and matrix.columns.tolist() == config["classes"]
            assert json.loads((directory / "metrics.json").read_text())["classes"] == config["classes"]
        scores = pd.read_csv(output / name / "test" / "prediction_scores.csv")
        assert scores.columns.tolist() == ["id", *config["classes"]]
        assert scores["id"].tolist() == frames["test"]["id"].tolist()
    assert (output / "report.md").is_file()
    assert result["linear_svm"]["score_kind"] == "decision_scores"
    assert not (output / "linear_svm" / "test" / "high_confidence_errors.csv").exists()


def test_text_provenance_survives_git_line_ending_normalization(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"first\r\nsecond\r\n")
    expected = baselines.text_sha256(path)
    path.write_bytes(b"first\nsecond\n")
    assert baselines.text_sha256(path) == expected == file_sha256(path)
    output = tmp_path / "manifest.json"
    baselines.write_json(output, {"source_sha256": expected})
    assert b"\r\n" not in output.read_bytes()


def test_changed_test_labels_do_not_change_validation_selection(bundle, tmp_path):
    config, experiment, frames = bundle
    first = baselines.run_baselines(config, experiment, verified_manifest(config, frames), tmp_path / "first")
    classes = config["classes"]
    changed = frames["test"].copy()
    changed["label"] = [classes[(classes.index(label) + 1) % len(classes)] for label in changed["label"]]
    changed.to_parquet(config["paths"]["processed_data"] / "test.parquet", index=False)
    second_frames = {**frames, "test": changed}
    second = baselines.run_baselines(config, experiment, verified_manifest(config, second_frames), tmp_path / "second")
    for name in baselines.MODEL_NAMES:
        assert first[name]["configuration"] == second[name]["configuration"]
        assert first[name]["validation"] == second[name]["validation"]
        assert first[name]["test"]["macro_f1"] != second[name]["test"]["macro_f1"]


def test_public_integrity_gate_rejects_leakage_before_fitting(bundle, tmp_path, monkeypatch):
    config, experiment, frames = bundle
    leaked = frames["test"].copy()
    leaked.loc[0, "id"] = frames["train"].iloc[0]["id"]
    leaked.to_parquet(config["paths"]["processed_data"] / "test.parquet", index=False)
    monkeypatch.setattr(baselines, "build_pipeline", lambda *args: pytest.fail("Integrity rejection must precede fitting"))
    with pytest.raises(ValueError, match="Duplicate ID"):
        baselines.run_baselines(config, experiment, verified_manifest(config, frames), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_valid_but_changed_split_is_rejected_by_exact_identity(bundle, tmp_path):
    config, experiment, frames = bundle
    expected = verified_manifest(config, frames)
    changed = frames["test"].copy()
    changed.loc[0, "text"] += " extra_valid_text"
    changed.to_parquet(config["paths"]["processed_data"] / "test.parquet", index=False)
    with pytest.raises(ValueError, match="test split identity differs"):
        baselines.run_baselines(config, experiment, expected, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_existing_output_is_preserved_without_loading_or_training(bundle, tmp_path, monkeypatch):
    config, experiment, frames = bundle
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    monkeypatch.setattr(baselines, "load_splits", lambda *args: pytest.fail("Existing runs must not be rerun"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        baselines.run_baselines(config, experiment, verified_manifest(config, frames), output)
    assert (output / "sentinel").read_text() == "preserve"


def test_convergence_failure_stops_before_any_test_evaluation(bundle, tmp_path, monkeypatch):
    config, experiment, frames = bundle
    real_fit = Pipeline.fit

    def unconverged(self, *args, **kwargs):
        result = real_fit(self, *args, **kwargs)
        warnings.warn("synthetic convergence failure", ConvergenceWarning)
        return result

    monkeypatch.setattr(Pipeline, "fit", unconverged)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="did not converge"):
        baselines.run_baselines(config, experiment, verified_manifest(config, frames), output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert all(count == 0 for count in manifest["test_prediction_calls"].values())
    assert not (output / "validation_selection.json").exists()


def test_small_search_configuration_rejects_invalid_or_excessive_candidates(bundle):
    _, experiment, _ = bundle
    invalid = deepcopy(experiment)
    invalid["models"]["logistic_regression"]["C"] = [float("inf")]
    with pytest.raises(ValueError, match="finite positive"):
        baselines.validate_experiment(invalid)
    invalid["models"]["logistic_regression"]["C"] = list(range(1, 20))
    with pytest.raises(ValueError, match="one to four"):
        baselines.validate_experiment(invalid)


def test_error_ranking_and_high_confidence_filter_use_actual_errors():
    classes = ["cs.LG", "cs.AI", "cs.CL"]
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["cs.LG", "cs.LG", "cs.AI", "cs.CL"]})
    predictions = ["cs.AI", "cs.AI", "cs.AI", "cs.LG"]
    metrics = evaluate_predictions(frame["label"], predictions, classes)
    summary = baselines.error_summary(metrics)
    assert summary["top_confusion_pairs"][0] == {"true_label": "cs.LG", "predicted_label": "cs.AI", "count": 2}
    assert summary["total_errors"] == 3
    assert summary["expected_overlap_pairs"][0]["right_to_left_rank"] == 1
    probabilities = np.array([[.03, .96, .01], [.10, .89, .01], [.005, .99, .005], [.95, .03, .02]])
    confidence = baselines.high_confidence_errors(frame, predictions, probabilities, classes)
    assert confidence["count"] == 2
    assert [row["id"] for row in confidence["top_errors"]] == ["a", "d"]
    assert confidence["top_errors"][0]["true_label_probability"] == .03
