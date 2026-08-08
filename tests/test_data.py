import pandas as pd

from src.data import clean_frame, normalize_text, split_sms_frame, validate_all_splits


def _balanced_frame(rows_per_class: int = 20) -> pd.DataFrame:
    ham = [
        {"text": f"normal message {index}", "label": 0, "source": "sms"}
        for index in range(rows_per_class)
    ]
    spam = [
        {"text": f"spam offer {index}", "label": 1, "source": "sms"}
        for index in range(rows_per_class)
    ]
    return pd.DataFrame(ham + spam)


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n\tworld  ") == "hello world"
    assert normalize_text(None) == ""


def test_clean_frame_removes_empty_duplicates_and_conflicts() -> None:
    raw = pd.DataFrame(
        {
            "text": ["Hello", " hello ", "", "same", "same", "unique spam"],
            "label": [0, 0, 1, 0, 1, 1],
        }
    )
    cleaned, audit = clean_frame(raw, source="sms")

    assert cleaned["text"].tolist() == ["Hello", "unique spam"]
    assert audit["empty_removed"] == 1
    assert audit["duplicates_removed"] == 1
    assert audit["conflicting_removed"] == 2


def test_sms_split_is_deterministic_and_leak_free() -> None:
    frame = _balanced_frame()
    first = split_sms_frame(frame, random_state=42)
    second = split_sms_frame(frame, random_state=42)

    assert first["train"]["text"].tolist() == second["train"]["text"].tolist()
    validate_all_splits({"sms": first})
    assert sum(len(split) for split in first.values()) == len(frame)

