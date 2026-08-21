import numpy as np
import pandas as pd
import pytest

from src.controls import (
    derive_class_counts,
    match_reference_class_counts,
    subsample_to_class_counts,
)


def _frame(ham: int = 8, spam: int = 6, source: str = "enron") -> pd.DataFrame:
    rows = [
        {
            "row_id": index,
            "text": f"ham message {index}",
            "label": 0,
            "source": source,
            "metadata": f"ham-{index}",
        }
        for index in range(ham)
    ]
    rows.extend(
        {
            "row_id": ham + index,
            "text": f"spam message {index}",
            "label": 1,
            "source": source,
            "metadata": f"spam-{index}",
        }
        for index in range(spam)
    )
    return pd.DataFrame(rows, index=np.arange(100, 100 + ham + spam))


def test_derive_class_counts_uses_reference_counts() -> None:
    reference = _frame(ham=5, spam=3, source="sms")

    assert derive_class_counts(reference) == {0: 5, 1: 3}


def test_subsample_has_exact_counts_and_preserves_complete_rows() -> None:
    frame = _frame()
    result = subsample_to_class_counts(
        frame, {0: 4, 1: 2}, random_state=13
    )

    assert derive_class_counts(result) == {0: 4, 1: 2}
    assert result.columns.tolist() == frame.columns.tolist()
    assert result.index.isin(frame.index).all()
    pd.testing.assert_frame_equal(result, frame.loc[result.index])


def test_subsample_is_deterministic_without_mutating_input() -> None:
    frame = _frame()
    original = frame.copy(deep=True)

    first = subsample_to_class_counts(frame, {0: 4, 1: 3}, random_state=73)
    second = subsample_to_class_counts(frame, {0: 4, 1: 3}, random_state=73)
    different_seed = subsample_to_class_counts(
        frame, {0: 4, 1: 3}, random_state=101
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.index.tolist() != different_seed.index.tolist()
    pd.testing.assert_frame_equal(frame, original)
    assert first is not frame


def test_unshuffled_result_has_mapping_and_source_order() -> None:
    frame = _frame()

    result = subsample_to_class_counts(
        frame,
        {1: 3, 0: 2},
        random_state=42,
        shuffle=False,
    )

    assert result["label"].tolist() == [1, 1, 1, 0, 0]
    for label in (0, 1):
        positions = result.loc[result["label"].eq(label)].index.tolist()
        assert positions == sorted(positions)


def test_match_reference_counts_supports_sms_to_enron_control() -> None:
    enron = _frame(ham=9, spam=7, source="enron")
    sms_reference = _frame(ham=4, spam=2, source="sms")

    matched = match_reference_class_counts(
        enron,
        sms_reference,
        random_state=137,
    )

    assert derive_class_counts(matched) == derive_class_counts(sms_reference)
    assert matched["source"].eq("enron").all()


@pytest.mark.parametrize(
    ("requested_counts", "exception"),
    [
        ({}, ValueError),
        ({0: 0, 1: 2}, ValueError),
        ({0: -1, 1: 2}, ValueError),
        ({0: 1.5, 1: 2}, TypeError),
        ({0: True, 1: 2}, TypeError),
    ],
)
def test_subsample_rejects_invalid_counts(
    requested_counts: dict[int, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        subsample_to_class_counts(_frame(), requested_counts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "requested_counts",
    [
        {0: 2},
        {0: 2, 1: 2, 2: 1},
    ],
)
def test_subsample_requires_exactly_the_available_classes(
    requested_counts: dict[int, int],
) -> None:
    with pytest.raises(ValueError, match="must describe every class"):
        subsample_to_class_counts(_frame(), requested_counts)


def test_subsample_rejects_unavailable_row_count() -> None:
    with pytest.raises(ValueError, match="only 6 are available"):
        subsample_to_class_counts(_frame(), {0: 4, 1: 7})


@pytest.mark.parametrize("random_state", [-1, 1.5, True])
def test_subsample_rejects_invalid_random_state(random_state: object) -> None:
    exception = ValueError if random_state == -1 else TypeError
    with pytest.raises(exception):
        subsample_to_class_counts(
            _frame(),
            {0: 4, 1: 2},
            random_state=random_state,  # type: ignore[arg-type]
        )


def test_controls_reject_missing_label_column_and_missing_labels() -> None:
    with pytest.raises(ValueError, match="exactly one label column"):
        derive_class_counts(pd.DataFrame({"text": ["hello"]}))

    frame = _frame()
    frame.loc[frame.index[0], "label"] = np.nan
    with pytest.raises(ValueError, match="contains missing values"):
        subsample_to_class_counts(frame, {0: 3, 1: 2})


def test_controls_reject_empty_frames() -> None:
    empty = pd.DataFrame(columns=["text", "label", "source"])

    with pytest.raises(ValueError, match="empty frame"):
        derive_class_counts(empty)
    with pytest.raises(ValueError, match="empty frame"):
        subsample_to_class_counts(empty, {0: 1, 1: 1})
