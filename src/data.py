"""Data loading, conservative cleaning, splitting, and validation."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
LABEL_NAMES = {0: "ham", 1: "spam"}
REQUIRED_COLUMNS = {"text", "label", "source"}
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Convert a value to text and collapse repeated whitespace."""

    if value is None or pd.isna(value):
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _dedup_key(text: str) -> str:
    return text.casefold()


def validate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing standardized columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prepared dataset is empty.")
    if frame["text"].eq("").any():
        raise ValueError("Prepared dataset contains empty text.")
    labels = set(frame["label"].unique())
    if not labels.issubset(LABEL_NAMES):
        raise ValueError(f"Unexpected labels: {sorted(labels)}")


def clean_frame(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardize one source and remove empty, duplicate, or ambiguous rows."""

    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = frame.loc[:, ["text", "label"]].copy()
    raw_rows = len(cleaned)
    cleaned["text"] = cleaned["text"].map(normalize_text)
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="raise").astype("int8")
    cleaned["source"] = source

    empty_mask = cleaned["text"].eq("")
    empty_removed = int(empty_mask.sum())
    cleaned = cleaned.loc[~empty_mask].copy()

    cleaned["_dedup_key"] = cleaned["text"].map(_dedup_key)
    conflicting_keys = (
        cleaned.groupby("_dedup_key")["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    conflict_mask = cleaned["_dedup_key"].isin(conflicting_keys)
    conflicting_removed = int(conflict_mask.sum())
    cleaned = cleaned.loc[~conflict_mask].copy()

    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset="_dedup_key", keep="first")
    duplicates_removed = before_deduplication - len(cleaned)
    cleaned = cleaned.drop(columns="_dedup_key").reset_index(drop=True)

    validate_frame(cleaned)
    audit = {
        "raw_rows": raw_rows,
        "clean_rows": len(cleaned),
        "empty_removed": empty_removed,
        "conflicting_removed": conflicting_removed,
        "duplicates_removed": duplicates_removed,
    }
    return cleaned, audit


def _prepare_sms(raw_frame: pd.DataFrame) -> pd.DataFrame:
    prepared = raw_frame.loc[:, ["text", "label"]].copy()
    prepared["label"] = prepared["label"].map({"ham": 0, "spam": 1})
    if prepared["label"].isna().any():
        raise ValueError("SMS dataset contains an unknown label.")
    return prepared


def _prepare_enron(raw_frame: pd.DataFrame) -> pd.DataFrame:
    prepared = raw_frame.copy()
    if "text" not in prepared.columns:
        prepared["text"] = (
            prepared["subject"].fillna("").astype(str)
            + " "
            + prepared["message"].fillna("").astype(str)
        )

    if "label_text" in prepared.columns:
        mapped = prepared["label_text"].astype(str).str.casefold().map({"ham": 0, "spam": 1})
        if mapped.notna().all():
            prepared["label"] = mapped
    return prepared.loc[:, ["text", "label"]]


def _reset_splits(**splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame.reset_index(drop=True) for name, frame in splits.items()}


def split_sms_frame(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Create a deterministic 70/15/15 stratified SMS split."""

    train, temporary = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["label"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label"],
        random_state=random_state,
    )
    return _reset_splits(train=train, validation=validation, test=test)


def split_enron_frame(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Preserve Enron test data and create validation data from training data."""

    test_keys = set(test_frame["text"].map(_dedup_key))
    overlap_mask = train_frame["text"].map(_dedup_key).isin(test_keys)
    overlap_removed = int(overlap_mask.sum())
    train_without_overlap = train_frame.loc[~overlap_mask].copy()

    train, validation = train_test_split(
        train_without_overlap,
        test_size=0.15,
        stratify=train_without_overlap["label"],
        random_state=random_state,
    )
    return (
        _reset_splits(train=train, validation=validation, test=test_frame),
        overlap_removed,
    )


def validate_all_splits(splits: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    expected = {"train", "validation", "test"}
    for dataset_name, dataset_splits in splits.items():
        if set(dataset_splits) != expected:
            raise ValueError(f"{dataset_name} does not have the expected splits.")

        keys: dict[str, set[str]] = {}
        for split_name, frame in dataset_splits.items():
            validate_frame(frame)
            keys[split_name] = set(frame["text"].map(_dedup_key))

        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = keys[left] & keys[right]
            if overlap:
                raise ValueError(
                    f"{dataset_name} has {len(overlap)} texts in both {left} and {right}."
                )


def load_prepared_splits(
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Download both datasets and return validated pandas splits and audit data."""

    sms_raw = load_dataset("jngb-labs/sms-spam", split="train").to_pandas()
    enron_raw = load_dataset("SetFit/enron_spam")

    sms, sms_audit = clean_frame(_prepare_sms(sms_raw), source="sms")
    enron_train, enron_train_audit = clean_frame(
        _prepare_enron(enron_raw["train"].to_pandas()), source="enron"
    )
    enron_test, enron_test_audit = clean_frame(
        _prepare_enron(enron_raw["test"].to_pandas()), source="enron"
    )

    sms_splits = split_sms_frame(sms, random_state=random_state)
    enron_splits, overlap_removed = split_enron_frame(
        enron_train, enron_test, random_state=random_state
    )
    splits = {"sms": sms_splits, "enron": enron_splits}
    validate_all_splits(splits)

    audit = pd.DataFrame(
        [
            {"dataset_part": "sms_all", **sms_audit, "train_test_overlap_removed": 0},
            {
                "dataset_part": "enron_train",
                **enron_train_audit,
                "train_test_overlap_removed": overlap_removed,
            },
            {
                "dataset_part": "enron_test",
                **enron_test_audit,
                "train_test_overlap_removed": 0,
            },
        ]
    )
    return splits, audit


def summarize_splits(
    splits: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, dataset_splits in splits.items():
        for split_name, frame in dataset_splits.items():
            lengths = frame["text"].str.len()
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "rows": len(frame),
                    "ham": int(frame["label"].eq(0).sum()),
                    "spam": int(frame["label"].eq(1).sum()),
                    "spam_rate": frame["label"].mean(),
                    "median_characters": lengths.median(),
                    "p95_characters": lengths.quantile(0.95),
                }
            )
    return pd.DataFrame(rows)

