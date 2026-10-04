"""Train-only classical baselines with validation selection and one test pass.

Run with ``python -m src.models.baselines --help``. Existing verified splits
are loaded through the complete public integrity-checking interface. No models,
vocabularies, or sparse matrices are serialized.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits
import yaml

from src.config import PROJECT_ROOT, load_config, resolve_runtime_path
from src.data.dataset import build_label_maps, load_splits
from src.data.integrity import SPLIT_NAMES
from src.data.profile import file_sha256, logical_digest
from src.evaluation import evaluate_predictions, save_evaluation

MODEL_NAMES = ("logistic_regression", "linear_svm")
SELECTION_RULE = "maximum validation macro_f1; exact ties use first listed C"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8", newline="\n",
    )


def text_sha256(path):
    """Hash the LF representation Git stores, regardless of checkout settings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def normalize_evaluation_text(paths):
    """The shared writer may use native newlines; normalize small text outputs."""
    for name in ("metrics", "confusion_matrix"):
        path = paths[name]
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type) and issubclass(value, np.number):
        return np.dtype(value).name
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def validate_experiment(experiment):
    """Keep the experiment small, explicit, and limited to the two baselines."""
    if not isinstance(experiment, dict) or set(experiment) != {"experiment_name", "threads", "tfidf", "models"}:
        raise ValueError("Experiment requires experiment_name, threads, tfidf, and models")
    if not isinstance(experiment["experiment_name"], str) or not experiment["experiment_name"].strip():
        raise ValueError("experiment_name must be a nonempty string")
    threads = experiment["threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    allowed_tfidf = {
        "ngram_range", "min_df", "max_df", "max_features", "sublinear_tf", "lowercase",
        "strip_accents", "stop_words", "token_pattern", "norm", "use_idf", "smooth_idf",
    }
    if not isinstance(experiment["tfidf"], dict) or set(experiment["tfidf"]).difference(allowed_tfidf):
        raise ValueError("Unsupported TF-IDF settings; word analyzer and float64 are fixed")
    models = experiment["models"]
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("models must contain exactly logistic_regression and linear_svm")
    for name in MODEL_NAMES:
        settings = models[name]
        if not isinstance(settings, dict) or set(settings) != {"C", "max_iter", "tol"}:
            raise ValueError(f"{name} requires C, max_iter, and tol")
        values = settings["C"]
        if not isinstance(values, list) or not 1 <= len(values) <= 4:
            raise ValueError("Provide one to four C candidates per model")
        for value in [*values, settings["tol"]]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("C and tol must be finite positive numbers")
        if len(set(values)) != len(values):
            raise ValueError("C candidates must be unique")
        if isinstance(settings["max_iter"], bool) or not isinstance(settings["max_iter"], int) or settings["max_iter"] < 1:
            raise ValueError("max_iter must be a positive integer")


def build_pipeline(model_name, tfidf_settings, model_settings, seed):
    """Construct an unfitted word TF-IDF / linear classifier pipeline."""
    tfidf = deepcopy(tfidf_settings)
    if "ngram_range" in tfidf:
        tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    common = {
        "C": model_settings["C"], "max_iter": model_settings["max_iter"],
        "tol": model_settings["tol"], "random_state": seed,
    }
    if model_name == "logistic_regression":
        # scikit-learn >=1.8 spells pure L2 as l1_ratio=0 (penalty is deprecated).
        classifier = LogisticRegression(solver="lbfgs", l1_ratio=0.0, **common)
    elif model_name == "linear_svm":
        classifier = LinearSVC(penalty="l2", loss="squared_hinge", dual="auto", multi_class="ovr", **common)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", dtype=np.float64, **tfidf)),
        ("classifier", classifier),
    ])


def select_on_validation(model_name, train, validation, classes, experiment, seed):
    """This function receives no test split and fits each candidate on train only."""
    settings = experiment["models"][model_name]
    candidates = []
    winner = None
    for index, regularization in enumerate(settings["C"]):
        candidate = {**settings, "C": regularization}
        pipeline = build_pipeline(model_name, experiment["tfidf"], candidate, seed)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(train["text"].tolist(), train["label"].tolist())
        training_seconds = perf_counter() - started
        warning_rows = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{model_name} C={regularization} did not converge; no test evaluation is permitted")
        started = perf_counter()
        predictions = pipeline.predict(validation["text"].tolist()).tolist()
        prediction_seconds = perf_counter() - started
        metrics = evaluate_predictions(validation["label"], predictions, classes)
        classifier = pipeline.named_steps["classifier"]
        record = {
            "candidate_id": f"{model_name}:candidate_{index + 1}", "model": model_name,
            "configuration": candidate, "validation_metrics": metrics,
            "training_seconds": training_seconds, "validation_prediction_seconds": prediction_seconds,
            "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
            "n_iter": np.asarray(classifier.n_iter_).tolist(), "warnings": warning_rows,
        }
        candidates.append(record)
        print(json.dumps({"stage": "validation", "model": model_name, "C": regularization,
                          "macro_f1": metrics["macro_f1"], "training_seconds": training_seconds}), flush=True)
        if winner is None or metrics["macro_f1"] > winner["record"]["validation_metrics"]["macro_f1"]:
            winner = {"pipeline": pipeline, "predictions": predictions, "record": record}
    return winner, candidates


def predict_with_scores(pipeline, texts, classes):
    """Transform once, predict once, then compute separately named diagnostics.

    Prediction time includes TF-IDF transformation and classifier.predict.
    Score time is additional classifier-only work on that same sparse matrix.
    SVM scores are margins, never probabilities. Binary margins are represented
    as [-margin, +margin] before reordering to configured class order.
    """
    started = perf_counter()
    features = pipeline.named_steps["tfidf"].transform(list(texts))
    classifier = pipeline.named_steps["classifier"]
    predictions = classifier.predict(features).tolist()
    prediction_seconds = perf_counter() - started
    learned_order = classifier.classes_.tolist()
    if set(learned_order) != set(classes):
        raise ValueError("Fitted classifier classes differ from configured classes")
    columns = [learned_order.index(label) for label in classes]
    started = perf_counter()
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(features)
        kind = "probabilities"
    else:
        scores = classifier.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        kind = "decision_scores"
    score_seconds = perf_counter() - started
    return predictions, scores[:, columns], kind, prediction_seconds, score_seconds


def verify_split_identity(frames, config, verified_manifest):
    """Bind valid loaded rows to the exact previously verified real-data bundle."""
    configuration = {key: config[key] for key in ("seed", "classes", "data")}
    if configuration != verified_manifest["configuration"]:
        raise ValueError("Data configuration differs from verified real-data manifest")
    processed = resolve_runtime_path(config["paths"]["processed_data"], "paths.processed_data")
    identities = {}
    for name in SPLIT_NAMES:
        path = processed / f"{name}.parquet"
        identity = {
            "filename": path.name, "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "ordered_rows_sha256": logical_digest(frames[name]),
        }
        if identity != verified_manifest["artifacts"][name]:
            raise ValueError(f"{name} split identity differs from verified real-data manifest")
        identities[name] = {
            **identity, "rows": len(frames[name]),
            "per_class": {label: int((frames[name]["label"] == label).sum()) for label in config["classes"]},
        }
    return identities


def error_summary(metrics):
    """Rank observed directed confusions and low-F1 classes without explanations."""
    classes = metrics["classes"]
    pairs = [
        {"true_label": true, "predicted_label": predicted, "count": int(metrics["confusion_matrix"][i][j])}
        for i, true in enumerate(classes) for j, predicted in enumerate(classes)
        if i != j and metrics["confusion_matrix"][i][j] > 0
    ]
    pairs.sort(key=lambda pair: (-pair["count"], classes.index(pair["true_label"]), classes.index(pair["predicted_label"])))
    ranks = {(pair["true_label"], pair["predicted_label"]): index + 1 for index, pair in enumerate(pairs)}
    overlaps = []
    for left, right in (("cs.AI", "cs.LG"), ("cs.LG", "cs.CV"), ("cs.LG", "cs.CL")):
        if left in classes and right in classes:
            i, j = classes.index(left), classes.index(right)
            overlaps.append({
                "classes": [left, right], "left_to_right": metrics["confusion_matrix"][i][j],
                "right_to_left": metrics["confusion_matrix"][j][i],
                "left_to_right_rank": ranks.get((left, right)), "right_to_left_rank": ranks.get((right, left)),
            })
    return {
        "total_errors": sum(pair["count"] for pair in pairs), "top_confusion_pairs": pairs[:10],
        "lowest_f1_classes": sorted(
            [{"label": label, **metrics["per_class"][label]} for label in classes],
            key=lambda row: (row["f1"], classes.index(row["label"])),
        )[:3],
        "expected_overlap_pairs": overlaps,
        "major_pair_definition": "a direction ranks within the ten largest nonzero directed confusions",
    }


def high_confidence_errors(frame, predictions, probabilities, classes, *, threshold=0.9, limit=20):
    """Save IDs and model confidence for later inspection, without copying text."""
    errors = []
    for row_index, (paper_id, truth, predicted) in enumerate(zip(frame["id"], frame["label"], predictions)):
        confidence = float(probabilities[row_index, classes.index(predicted)])
        if truth != predicted and confidence >= threshold:
            errors.append({
                "id": paper_id, "true_label": truth, "predicted_label": predicted,
                "predicted_probability": confidence,
                "true_label_probability": float(probabilities[row_index, classes.index(truth)]),
            })
    errors.sort(key=lambda row: (-row["predicted_probability"], row["id"]))
    return {"threshold": threshold, "count": len(errors), "saved_count": min(limit, len(errors)),
            "top_errors": errors[:limit], "note": "Logistic Regression probabilities are not separately calibrated."}


def _runtime():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "packages": {name: version(name) for name in (
            "numpy", "scipy", "pandas", "pyarrow", "pyyaml", "scikit-learn", "threadpoolctl", "joblib", "pytest",
        )},
        "threadpools": threadpool_info(),
    }


def run_baselines(config, experiment, verified_manifest, output_dir, *, provenance=None):
    """Select both models before any test prediction; refuse output reuse."""
    validate_experiment(experiment)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Experiment output already exists; refusing to overwrite a held-out run")
    started = perf_counter()
    frames = load_splits(config)  # Complete bundle validation is mandatory.
    identities = verify_split_identity(frames, config, verified_manifest)
    loading_seconds = perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    classes = config["classes"]
    label_to_id, _ = build_label_maps(classes)
    manifest = {
        "schema_version": 1, "status": "selecting_on_validation",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": deepcopy(experiment), "data_configuration": verified_manifest["configuration"],
        "source_identity": {
            "provider": verified_manifest.get("source_provider", {}).get("source_provider", {}),
            "original_source": verified_manifest.get("source_provider", {}).get("source", {}),
            "curated_source": verified_manifest.get("raw_file", {}),
        },
        "split_identity": identities, "public_loader": "src.data.dataset.load_splits",
        "label_to_id": label_to_id, "selection_rule": SELECTION_RULE,
        "fit_split": "train", "selection_split": "val", "final_evaluation_split": "test",
        "refit_on_train_plus_validation": False, "test_prediction_calls": {name: 0 for name in MODEL_NAMES},
        "data_loading_and_identity_seconds": loading_seconds, "provenance": provenance or {},
        "timing_definition": "Wall-clock seconds; training includes TF-IDF fit/transform and classifier fit; prediction includes transform and predict; diagnostics timed separately; I/O excluded.",
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        with threadpool_limits(limits=experiment["threads"]):
            manifest["runtime"] = _runtime()
            winners, candidate_rows = {}, {}
            for name in MODEL_NAMES:
                winners[name], candidate_rows[name] = select_on_validation(
                    name, frames["train"], frames["val"], classes, experiment, config["seed"],
                )
                write_json(output / "validation_candidates.json", candidate_rows)
            selection = {
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "selection_rule": SELECTION_RULE,
                "test_predictions_started": False,
                "models": {name: winners[name]["record"] for name in MODEL_NAMES},
            }
            write_json(output / "validation_selection.json", selection)
            manifest["selection_sha256_before_test"] = file_sha256(output / "validation_selection.json")
            manifest["status"] = "evaluating_selected_models_once_on_test"
            write_json(output / "run_manifest.json", manifest)
            comparison, errors = {}, {}
            for name in MODEL_NAMES:
                winner = winners[name]
                val_paths = save_evaluation(frames["val"], winner["predictions"], classes, output / name / "validation")
                normalize_evaluation_text(val_paths)
                predictions, scores, kind, prediction_seconds, score_seconds = predict_with_scores(
                    winner["pipeline"], frames["test"]["text"], classes,
                )
                manifest["test_prediction_calls"][name] += 1
                test_dir = output / name / "test"
                test_paths = save_evaluation(frames["test"], predictions, classes, test_dir)
                normalize_evaluation_text(test_paths)
                metrics = json.loads(test_paths["metrics"].read_text(encoding="utf-8"))
                score_rows = pd.DataFrame(scores, columns=classes)
                score_rows.insert(0, "id", frames["test"]["id"].tolist())
                score_rows.to_csv(test_dir / "prediction_scores.csv", index=False, lineterminator="\n")
                errors[name] = error_summary(metrics)
                if kind == "probabilities":
                    confidence = high_confidence_errors(frames["test"], predictions, scores, classes)
                    errors[name]["high_confidence_errors"] = confidence
                    pd.DataFrame(confidence["top_errors"], columns=[
                        "id", "true_label", "predicted_label", "predicted_probability", "true_label_probability",
                    ]).to_csv(test_dir / "high_confidence_errors.csv", index=False, lineterminator="\n")
                else:
                    errors[name]["score_note"] = "Uncalibrated LinearSVC decision scores; predict_proba is unavailable."
                pipeline = winner["pipeline"]
                comparison[name] = {
                    "selected_candidate": winner["record"]["candidate_id"],
                    "configuration": winner["record"]["configuration"],
                    "effective_pipeline_parameters": {
                        step: estimator.get_params(deep=False) for step, estimator in pipeline.named_steps.items()
                    },
                    "vocabulary_size": winner["record"]["vocabulary_size"],
                    "validation": winner["record"]["validation_metrics"], "test": metrics,
                    "selected_training_seconds": winner["record"]["training_seconds"],
                    "all_candidate_training_seconds": sum(row["training_seconds"] for row in candidate_rows[name]),
                    "validation_prediction_seconds": winner["record"]["validation_prediction_seconds"],
                    "test_prediction_seconds": prediction_seconds, "test_score_seconds": score_seconds,
                    "score_kind": kind, "score_classes": classes,
                }
                artifacts = [*val_paths.values(), *test_paths.values(), test_dir / "prediction_scores.csv"]
                if kind == "probabilities":
                    artifacts.append(test_dir / "high_confidence_errors.csv")
                manifest.setdefault("artifacts", {})[name] = {
                    path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                    for path in artifacts
                }
                print(json.dumps({"stage": "test", "model": name, "macro_f1": metrics["macro_f1"]}), flush=True)
                write_json(output / "run_manifest.json", manifest)
            comparison["best_by_held_out_macro_f1"] = max(MODEL_NAMES, key=lambda name: comparison[name]["test"]["macro_f1"])
            write_json(output / "comparison.json", comparison)
            write_json(output / "error_analysis.json", errors)
            if file_sha256(output / "validation_selection.json") != manifest["selection_sha256_before_test"]:
                raise RuntimeError("Frozen selection changed during test evaluation")
            manifest["status"] = "complete"
            manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "run_manifest.json", manifest)
            from src.models.baseline_report import write_report
            write_report(output / "report.md", manifest, candidate_rows, comparison, errors)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "run_manifest.json", manifest)
        raise
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/real_data.yaml")
    parser.add_argument("--experiment-config", default="config/classical_baselines.yaml")
    parser.add_argument("--verified-manifest", default="results/real_data_validation/run_manifest.json")
    parser.add_argument("--output-dir", default="results/classical_baselines/real_v1")
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_path = resolve_runtime_path(args.experiment_config, "experiment_config")
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    verified_path = resolve_runtime_path(args.verified_manifest, "verified_manifest")
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    profile_path = verified_path.with_name("profile.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("ready_for_baseline_data_input") is not True:
        raise ValueError("Real-data profile has not passed its readiness gate")
    config_path = resolve_runtime_path(args.config, "config_path")
    code_paths = [
        config_path, experiment_path, PROJECT_ROOT / "requirements.txt",
        *[PROJECT_ROOT / relative for relative in (
            "src/config.py", "src/data/dataset.py", "src/data/integrity.py", "src/data/profile.py",
            "src/evaluation.py", "src/models/baselines.py", "src/models/baseline_report.py",
        )],
    ]
    provenance = {
        "command": f"python -B -m src.models.baselines --config {args.config} --experiment-config {args.experiment_config} --verified-manifest {args.verified_manifest} --output-dir {args.output_dir}",
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "working_tree_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        "text_hash_representation": "UTF-8 bytes with CRLF normalized to LF, matching Git blobs",
        "code_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): text_sha256(path) for path in code_paths},
        "verified_manifest_sha256": text_sha256(verified_path), "verified_profile_sha256": text_sha256(profile_path),
    }
    run_baselines(config, experiment, verified, resolve_runtime_path(args.output_dir, "output_dir"), provenance=provenance)


if __name__ == "__main__":
    main()
