"""Pure utilities for controlled class-count experiments.

The functions in this module only select rows from already prepared pandas
frames.  They do not clean data, recreate dataset splits, or train models.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from numbers import Integral

import numpy as np
import pandas as pd

from .protocol import CONTROL_SAMPLING_SEED


def _label_series(frame: pd.DataFrame, label_column: str) -> pd.Series:
    """Return one complete label column after validating the input frame."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if list(frame.columns).count(label_column) != 1:
        raise ValueError(
            f"Expected exactly one label column named {label_column!r}."
        )
    if frame.empty:
        raise ValueError("Cannot derive or match class counts from an empty frame.")

    labels = frame[label_column]
    if labels.isna().any():
        raise ValueError(f"Label column {label_column!r} contains missing values.")

    for label in labels.unique():
        if not isinstance(label, Hashable):
            raise TypeError("Class labels must be hashable.")
    return labels


def derive_class_counts(
    reference_frame: pd.DataFrame,
    *,
    label_column: str = "label",
) -> dict[Hashable, int]:
    """Return exact per-class row counts from a reference frame.

    The class order follows the order in which labels first appear in the
    reference frame.  The returned dictionary can be passed directly to
    :func:`subsample_to_class_counts`.
    """

    labels = _label_series(reference_frame, label_column)
    counts = labels.value_counts(sort=False, dropna=False)
    return {label: int(count) for label, count in counts.items()}


def _validate_requested_counts(
    requested_counts: Mapping[Hashable, int],
) -> dict[Hashable, int]:
    if not isinstance(requested_counts, Mapping):
        raise TypeError("requested_counts must be a mapping of class to row count.")
    if not requested_counts:
        raise ValueError("requested_counts must contain at least one class.")

    validated: dict[Hashable, int] = {}
    for label, count in requested_counts.items():
        if pd.isna(label):
            raise ValueError("Class labels in requested_counts cannot be missing.")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
            raise TypeError(
                f"Requested count for class {label!r} must be a positive integer."
            )
        if count <= 0:
            raise ValueError(
                f"Requested count for class {label!r} must be positive; got {count}."
            )
        validated[label] = int(count)
    return validated


def _validate_random_state(random_state: int) -> int:
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, Integral
    ):
        raise TypeError("random_state must be an integer.")
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")
    return int(random_state)


def subsample_to_class_counts(
    frame: pd.DataFrame,
    requested_counts: Mapping[Hashable, int],
    *,
    label_column: str = "label",
    random_state: int = CONTROL_SAMPLING_SEED,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Select an exact, reproducible number of rows from every class.

    Parameters
    ----------
    frame:
        Prepared data from which rows will be selected.  Every column and the
        original index of selected rows are preserved.
    requested_counts:
        Positive row count for each class.  It must describe exactly the set
        of classes present in ``frame``; this prevents silent class removal.
    label_column:
        Name of the class-label column.
    random_state:
        Non-negative integer used only for row selection and output order.
    shuffle:
        If true, deterministically mix selected classes.  If false, classes
        follow mapping order and selected rows retain source-frame order
        within each class.

    Returns
    -------
    pandas.DataFrame
        A new frame with the same columns as ``frame``.  The input is never
        mutated.
    """

    labels = _label_series(frame, label_column)
    counts = _validate_requested_counts(requested_counts)
    seed = _validate_random_state(random_state)

    available_counts = derive_class_counts(frame, label_column=label_column)
    available_classes = set(available_counts)
    requested_classes = set(counts)
    if requested_classes != available_classes:
        missing = available_classes - requested_classes
        unknown = requested_classes - available_classes
        details: list[str] = []
        if missing:
            details.append(f"unrequested classes: {sorted(missing, key=repr)!r}")
        if unknown:
            details.append(f"unknown classes: {sorted(unknown, key=repr)!r}")
        raise ValueError(
            "requested_counts must describe every class in frame ("
            + "; ".join(details)
            + ")."
        )

    for label, requested in counts.items():
        available = available_counts[label]
        if requested > available:
            raise ValueError(
                f"Class {label!r} requests {requested} rows, but only "
                f"{available} are available."
            )

    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    for label, requested in counts.items():
        class_positions = np.flatnonzero(labels.eq(label).to_numpy(dtype=bool))
        chosen = rng.choice(class_positions, size=requested, replace=False)
        selected_positions.extend(np.sort(chosen).tolist())

    if shuffle:
        selected_positions = rng.permutation(selected_positions).tolist()

    return frame.iloc[selected_positions].copy()


def match_reference_class_counts(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    label_column: str = "label",
    reference_label_column: str | None = None,
    random_state: int = CONTROL_SAMPLING_SEED,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Subsample ``frame`` to the exact class counts of ``reference_frame``.

    This convenience wrapper supports experiments such as sampling the Enron
    training split to the exact ham/spam counts of the SMS training split.
    """

    reference_counts = derive_class_counts(
        reference_frame,
        label_column=reference_label_column or label_column,
    )
    return subsample_to_class_counts(
        frame,
        reference_counts,
        label_column=label_column,
        random_state=random_state,
        shuffle=shuffle,
    )
