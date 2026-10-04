"""Reject identity/text leakage and unusable split definitions."""

import pytest

from src.data.integrity import text_identity, validate_splits
from src.data.preprocess import stratified_split


def split_rows():
    return {
        name: [{"id": f"paper-{name}", "text": f"Abstract for {name}", "label": "cs.AI"}]
        for name in ("train", "val", "test")
    }


@pytest.mark.parametrize("pair", [("train", "val"), ("train", "test"), ("val", "test")])
@pytest.mark.parametrize("column", ["id", "text"])
def test_split_validation_rejects_id_and_text_leakage_between_every_pair(pair, column):
    rows = split_rows()
    left, right = pair
    rows[right][0][column] = rows[left][0][column]
    message = "Duplicate ID" if column == "id" else "Duplicate normalized text"

    with pytest.raises(ValueError, match=message) as error:
        validate_splits(rows, ["cs.AI"])

    assert left in str(error.value)
    assert right in str(error.value)
    # Validation never silently drops or alters the overlapping record.
    assert rows[left][0][column] == rows[right][0][column]


def test_effectively_identical_text_is_rejected_without_mutating_originals():
    rows = split_rows()
    first = "  CAFÉ\n model Ａ\u200b "
    second = "cafe\u0301\tMODEL a\ufeff\u00ad"
    rows["train"][0]["text"] = first
    rows["val"][0]["text"] = second

    with pytest.raises(ValueError, match="Duplicate normalized text"):
        validate_splits(rows, ["cs.AI"])

    assert rows["train"][0]["text"] == first
    assert rows["val"][0]["text"] == second
    assert text_identity("model 1") != text_identity("model 2")
    assert text_identity("not relevant") != text_identity("relevant")


@pytest.mark.parametrize("column", ["id", "text"])
def test_selected_pool_rejects_duplicates_before_any_split_assignment(column):
    samples = [
        {"id": f"paper-{index}", "text": f"Abstract {index}", "label": "cs.AI"}
        for index in range(10)
    ]
    samples[-1][column] = samples[0][column]
    if column == "id":
        samples[-1][column] = f" {samples[-1][column]} "
    message = "Duplicate ID" if column == "id" else "Duplicate normalized text"

    with pytest.raises(ValueError, match=message):
        stratified_split({"cs.AI": samples}, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)

    assert len(samples) == 10


@pytest.mark.parametrize(
    "fractions",
    [
        {"train": 0.8, "val": 0.1, "test": 0.6},
        {"train": 0.8, "val": 0.1},
        {"train": 0.8, "val": 0.1, "test": -0.1},
        {"train": 0.8, "val": 0.1, "test": float("nan")},
        {"train": 0.8, "val": 0.1, "test": float("inf")},
        {"train": 0.8, "val": 0.1, "test": "0.1"},
        {"train": True, "val": 0.1, "test": 0.1},
        {"train": 0.8, "val": 0.2, "test": 0},
    ],
)
def test_invalid_split_fractions_fail_before_assignment(fractions):
    with pytest.raises(ValueError, match="(fraction|splits)"):
        stratified_split({}, fractions, seed=42)


def test_tiny_or_missing_classes_fail_instead_of_silently_disappearing():
    samples = {
        label: [{"id": f"{label}-{i}", "text": f"{label} abstract {i}", "label": label} for i in range(10)]
        for label in ("cs.AI", "cs.LG")
    }
    samples["cs.LG"] = samples["cs.LG"][:2]
    with pytest.raises(ValueError, match=r"cs.LG.*2 selected samples"):
        stratified_split(samples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    samples["cs.LG"] = []
    with pytest.raises(ValueError, match=r"cs.LG.*0 selected samples"):
        stratified_split(samples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)


def test_fraction_rounding_preserves_every_row_and_each_class():
    samples = {
        "cs.AI": [{"id": str(i), "text": f"Abstract {i}", "label": "cs.AI"} for i in range(11)]
    }
    splits = stratified_split(samples, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=7)

    assert {name: len(rows) for name, rows in splits.items()} == {"train": 6, "val": 2, "test": 3}
    assert {row["id"] for rows in splits.values() for row in rows} == {str(i) for i in range(11)}


@pytest.mark.parametrize("bad_text", ["\u200b\ufeff\u00ad", "bad\ud800"])
def test_empty_normalized_or_unencodable_text_fails_clearly(bad_text):
    rows = split_rows()
    rows["val"][0]["text"] = bad_text

    with pytest.raises(ValueError, match="(empty after normalization|not valid UTF-8)"):
        validate_splits(rows, ["cs.AI"])
