"""Tests for loading processed parquet dataset splits."""

import pandas as pd
import pytest

from src.data.dataset import (
    build_label_maps,
    get_texts_label_ids,
    get_texts_labels,
    load_split,
    load_splits,
    validate_labels,
)


@pytest.fixture
def classes():
    return ["cs.LG", "cs.AI", "cs.CL"]


@pytest.fixture
def injected_config(tmp_path, classes):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    return {
        "classes": classes,
        "paths": {"processed_data": processed_dir},
    }


def write_split(config, split="train", dataframe=None):
    if dataframe is None:
        dataframe = pd.DataFrame(
            {
                "id": ["paper-2", "paper-1", "paper-3"],
                "text": ["Second text", "First text", "Third text"],
                "label": ["cs.AI", "cs.LG", "cs.CL"],
            }
        )
    path = config["paths"]["processed_data"] / f"{split}.parquet"
    dataframe.to_parquet(path, index=False)
    return dataframe


def test_build_label_maps_uses_configured_class_order(classes):
    label_to_id, id_to_label = build_label_maps(classes)

    assert label_to_id == {"cs.LG": 0, "cs.AI": 1, "cs.CL": 2}
    assert list(label_to_id.values()) == list(range(len(classes)))
    assert id_to_label == {0: "cs.LG", 1: "cs.AI", 2: "cs.CL"}
    assert {id_to_label[index]: index for index in id_to_label} == label_to_id


def test_load_split_loads_expected_parquet_file(injected_config):
    expected = write_split(injected_config, split="val")

    loaded = load_split("val", injected_config)

    assert list(loaded.columns) == ["id", "text", "label"]
    assert len(loaded) == len(expected)
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_split_rejects_invalid_split_name(injected_config):
    with pytest.raises(ValueError, match="Invalid split"):
        load_split("validation", injected_config)


def test_load_split_rejects_missing_required_columns(injected_config):
    incomplete = pd.DataFrame({"id": ["paper-1"], "text": ["Text"]})
    write_split(injected_config, dataframe=incomplete)

    with pytest.raises(ValueError, match="missing required columns: label"):
        load_split("train", injected_config)


def test_get_texts_labels_preserves_order_and_alignment(injected_config):
    write_split(injected_config)

    texts, labels = get_texts_labels("train", injected_config)

    assert texts == ["Second text", "First text", "Third text"]
    assert labels == ["cs.AI", "cs.LG", "cs.CL"]
    assert all(isinstance(label, str) for label in labels)


def test_get_texts_label_ids_uses_configured_order(injected_config, classes):
    write_split(injected_config)

    texts, label_ids = get_texts_label_ids("train", injected_config)

    assert texts == ["Second text", "First text", "Third text"]
    assert label_ids == [1, 0, 2]
    assert all(isinstance(label_id, int) for label_id in label_ids)
    assert all(0 <= label_id < len(classes) for label_id in label_ids)


def test_validate_labels_accepts_configured_labels(classes):
    dataframe = pd.DataFrame({"label": ["cs.AI", "cs.LG", "cs.CL"]})

    validate_labels(dataframe, classes)


def test_validate_labels_rejects_unknown_label(classes):
    dataframe = pd.DataFrame({"label": ["cs.AI", "math.CO"]})

    with pytest.raises(ValueError, match="math.CO"):
        validate_labels(dataframe, classes)


def _write_bundle(config):
    frames = {}
    for name in ("train", "val", "test"):
        frame = pd.DataFrame({
            "id": [f"{name}-{i}" for i in range(len(config["classes"]))],
            "text": [f"{name} abstract {i}" for i in range(len(config["classes"]))],
            "label": config["classes"],
        })
        write_split(config, name, frame)
        frames[name] = frame
    return frames


def test_load_splits_requires_complete_bundle_and_preserves_order(injected_config):
    write_split(injected_config, "train")
    with pytest.raises(FileNotFoundError, match="Missing processed split 'val'"):
        load_splits(injected_config)

    expected = _write_bundle(injected_config)
    actual = load_splits(injected_config)

    assert list(actual) == ["train", "val", "test"]
    for name in actual:
        pd.testing.assert_frame_equal(actual[name], expected[name])


@pytest.mark.parametrize("column", ["id", "text"])
def test_loaders_reject_cross_split_leakage_from_parquet(injected_config, column):
    frames = _write_bundle(injected_config)
    value = frames["val"].loc[0, column]
    if column == "text":
        value = "  " + value.upper().replace(" ", "\n\t") + " "
    else:
        value = " " + value + " "
    frames["test"].loc[0, column] = value
    write_split(injected_config, "test", frames["test"])
    message = "Duplicate ID" if column == "id" else "Duplicate normalized text"

    with pytest.raises(ValueError, match=message):
        load_splits(injected_config)
    # Even requesting train catches leakage between the existing val/test files.
    with pytest.raises(ValueError, match=message):
        load_split("train", injected_config)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("id", None, "id must be a nonempty string"),
     ("id", 17, "id must be a nonempty string"),
     ("text", None, "text must be a nonempty string"),
     ("text", " ", "text must be a nonempty string"),
     ("label", "math.CO", "Labels outside configured classes")],
)
def test_load_split_rejects_invalid_row_values(injected_config, column, value, message):
    frame = pd.DataFrame({"id": ["paper-1"], "text": ["Abstract"], "label": ["cs.AI"]})
    frame[column] = value
    write_split(injected_config, dataframe=frame)

    with pytest.raises(ValueError, match=message):
        load_split("train", injected_config)


@pytest.mark.parametrize("problem", ["empty", "duplicate_id", "duplicate_text"])
def test_load_split_rejects_empty_or_repeated_rows(injected_config, problem):
    frame = pd.DataFrame({
        "id": ["paper-1", "paper-2"], "text": ["First abstract", "Second abstract"],
        "label": ["cs.AI", "cs.LG"],
    })
    if problem == "empty":
        frame = frame.iloc[:0]
        message = "at least one example"
    else:
        column = "id" if problem == "duplicate_id" else "text"
        frame.loc[1, column] = frame.loc[0, column]
        message = "Duplicate ID" if column == "id" else "Duplicate normalized text"
    write_split(injected_config, dataframe=frame)

    with pytest.raises(ValueError, match=message):
        load_split("train", injected_config)


def test_load_splits_rejects_missing_class_coverage(injected_config):
    frames = _write_bundle(injected_config)
    write_split(injected_config, "val", frames["val"].iloc[:1])

    with pytest.raises(ValueError, match="val.*no examples for configured classes"):
        load_splits(injected_config)


def test_build_label_maps_rejects_duplicate_class_definitions():
    with pytest.raises(ValueError, match="unique labels"):
        build_label_maps(["cs.AI", "cs.AI"])
