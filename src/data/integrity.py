"""Shared validation for sample identities, text, and dataset splits."""

from collections.abc import Mapping
import math
from numbers import Real
import unicodedata

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = {"id", "text", "label"}
TEXT_IDENTITY_RULE = "NFKC, casefold, remove U+200B/U+FEFF/U+00AD, collapse whitespace"


def validate_classes(classes):
    """Require a nonempty, ordered collection of distinct string labels."""
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("classes must be a nonempty ordered list of string labels")
    if any(not isinstance(label, str) or not label.strip() for label in classes):
        raise ValueError("classes must contain nonempty string labels")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must contain unique labels")


def validate_split_fractions(splits):
    """Require all three positive fractions, summing to one within 1e-9."""
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("splits must contain exactly train, val, and test fractions")
    for name, fraction in splits.items():
        if (
            isinstance(fraction, bool) or not isinstance(fraction, Real)
            or not math.isfinite(fraction) or not 0 < fraction < 1
        ):
            raise ValueError(f"Split fraction {name!r} must be finite and between 0 and 1")
    if not math.isclose(sum(splits.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1")


def text_identity(text):
    """Canonical equality key; preserve the original text used by models."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate({ord(char): None for char in "\u200b\ufeff\u00ad"})
    return " ".join(normalized.casefold().split())


def validate_labels(dataframe, classes):
    """Reject null, non-string, and unknown labels using configured class order."""
    validate_classes(classes)
    allowed = set(classes)
    invalid = [
        label for label in dataframe["label"]
        if not isinstance(label, str) or label not in allowed
    ]
    if invalid:
        raise ValueError(f"Labels outside configured classes: {invalid[:5]!r}")


def validate_splits(split_data, classes=None, *, require_all=True):
    """Reject invalid rows and repeated IDs/text within or across named splits.

    Accept DataFrames or lists of id/text/label records. Identity checks strip
    IDs and use text_identity() for text. No records are removed or mutated.
    A training caller should keep require_all=True to require the full bundle.
    """
    if not isinstance(split_data, Mapping) or not split_data:
        raise ValueError("Split data must be a nonempty mapping")
    if set(split_data).difference(SPLIT_NAMES):
        raise ValueError("Split data may contain only train, val, and test")
    if require_all and set(split_data) != set(SPLIT_NAMES):
        raise ValueError("Split data must contain train, val, and test")
    if classes is not None:
        validate_classes(classes)

    _validate_frames(split_data, classes)
    if require_all and classes is not None:
        for name, rows in split_data.items():
            frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, dtype=object)
            missing = set(classes).difference(frame["label"])
            if missing:
                raise ValueError(f"Split {name!r} has no examples for configured classes: {sorted(missing)}")


def validate_samples(samples, classes):
    """Validate the entire selected pool before shuffling any samples."""
    validate_classes(classes)
    _validate_frames({"selected samples": pd.DataFrame(samples, dtype=object)}, classes)


def _validate_frames(frames, classes):
    id_owners = {}
    text_owners = {}
    for name, rows in frames.items():
        # Avoid pandas' string inference encoding escaped JSON surrogates before
        # our validation can report the offending row and column clearly.
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, dtype=object)
        if frame.empty:
            raise ValueError(f"Split {name!r} must contain at least one example")
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Split {name!r} is missing required columns: {missing}")
        if classes is not None:
            validate_labels(frame, classes)

        for row_number, (record_id, text, label) in enumerate(
            frame[["id", "text", "label"]].itertuples(index=False, name=None), start=1,
        ):
            for column, value in (("id", record_id), ("text", text), ("label", label)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Split {name!r} row {row_number}: {column} must be a nonempty string"
                    )
                try:
                    value.encode("utf-8", errors="strict")
                except UnicodeEncodeError:
                    raise ValueError(
                        f"Split {name!r} row {row_number}: {column} is not valid UTF-8 text"
                    ) from None

            identity = record_id.strip()
            location = f"{name} row {row_number} (ID {identity!r})"
            if identity in id_owners:
                raise ValueError(
                    f"Duplicate ID {identity!r}: {id_owners[identity]} overlaps {location}"
                )
            id_owners[identity] = location
            key = text_identity(text)
            if not key:
                raise ValueError(f"Split {name!r} row {row_number}: text is empty after normalization")
            if key in text_owners:
                raise ValueError(
                    f"Duplicate normalized text: {text_owners[key]} overlaps {location}"
                )
            text_owners[key] = location
