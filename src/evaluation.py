"""Shared evaluation utilities for the robustness experiments.

The functions in this module operate only on labels and predicted spam
probabilities.  They do not train models and do not require a GPU or network
access.  Spam is consistently treated as the positive class (label ``1``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .protocol import BOOTSTRAP_SEED, DECISION_THRESHOLD


CLASSIFICATION_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "predicted_spam_rate",
)
CALIBRATION_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "brier_score",
    "log_loss",
    "ece",
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    *CLASSIFICATION_METRIC_COLUMNS,
    *CALIBRATION_METRIC_COLUMNS,
)
COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "tn",
    "fp",
    "fn",
    "tp",
    "support",
    "ham_support",
    "spam_support",
)
BOOTSTRAP_METRICS: Final[tuple[str, ...]] = METRIC_COLUMNS

CALIBRATION_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "bin",
    "count",
    "weight",
    "min_probability",
    "max_probability",
    "mean_probability",
    "observed_spam_rate",
    "absolute_gap",
    "ece_contribution",
)

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "example_id",
    "model",
    "training_seed",
    "train_domain",
    "test_domain",
    "setting",
    "label",
    "prediction",
    "spam_probability",
    "correct",
    "text",
    "source",
)
RESULT_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "training_seed",
    "train_domain",
    "test_domain",
    "setting",
    "test_rows",
)
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    *RESULT_METADATA_COLUMNS,
    *METRIC_COLUMNS,
    *COUNT_COLUMNS,
)


def _as_binary_labels(labels: Sequence[int]) -> np.ndarray:
    """Return validated one-dimensional labels encoded as zero and one."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("Labels must be a one-dimensional sequence.")
    if label_array.size == 0:
        raise ValueError("Labels and probabilities must not be empty.")
    try:
        numeric_labels = label_array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must contain only binary values 0 and 1.") from exc
    if not np.all(np.isfinite(numeric_labels)) or not np.all(
        np.isin(numeric_labels, (0.0, 1.0))
    ):
        raise ValueError("Labels must contain only binary values 0 and 1.")
    return numeric_labels.astype(np.int8)


def validate_probabilities(
    labels: Sequence[int],
    spam_probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary labels and probabilities in the closed interval [0, 1].

    The returned arrays have stable numeric dtypes and can be passed directly to
    the other functions in this module.
    """

    y_true = _as_binary_labels(labels)
    try:
        y_probability = np.asarray(spam_probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spam probabilities must be numeric.") from exc
    if y_probability.ndim != 1:
        raise ValueError("Spam probabilities must be a one-dimensional sequence.")
    if len(y_true) != len(y_probability):
        raise ValueError("Labels and probabilities must contain the same number of rows.")
    if not np.all(np.isfinite(y_probability)):
        raise ValueError("Spam probabilities must contain only finite values.")
    if np.any((y_probability < 0.0) | (y_probability > 1.0)):
        raise ValueError("Spam probabilities must be between 0 and 1 inclusive.")
    return y_true, y_probability


def _validate_threshold(threshold: float) -> float:
    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("The classification threshold must be between 0 and 1.") from exc
    if not np.isfinite(numeric_threshold) or not 0.0 <= numeric_threshold <= 1.0:
        raise ValueError("The classification threshold must be between 0 and 1.")
    return numeric_threshold


def _validate_bin_count(n_bins: int, row_count: int) -> int:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins must be a positive integer.")
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    return min(int(n_bins), row_count)


def calibration_table(
    labels: Sequence[int],
    spam_probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency bins for a reliability diagram.

    Rows are stably sorted by predicted spam probability and divided into bins
    whose sizes differ by at most one.  This rank-based definition remains
    deterministic even when several examples have identical probabilities.
    """

    y_true, y_probability = validate_probabilities(labels, spam_probabilities)
    bin_count = _validate_bin_count(n_bins, len(y_true))
    sorted_indices = np.argsort(y_probability, kind="stable")
    index_bins = np.array_split(sorted_indices, bin_count)

    rows: list[dict[str, float | int]] = []
    total_rows = len(y_true)
    for bin_number, indices in enumerate(index_bins, start=1):
        bin_probabilities = y_probability[indices]
        bin_labels = y_true[indices]
        mean_probability = float(bin_probabilities.mean())
        observed_spam_rate = float(bin_labels.mean())
        absolute_gap = abs(mean_probability - observed_spam_rate)
        weight = len(indices) / total_rows
        rows.append(
            {
                "bin": bin_number,
                "count": len(indices),
                "weight": weight,
                "min_probability": float(bin_probabilities.min()),
                "max_probability": float(bin_probabilities.max()),
                "mean_probability": mean_probability,
                "observed_spam_rate": observed_spam_rate,
                "absolute_gap": absolute_gap,
                "ece_contribution": weight * absolute_gap,
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_TABLE_COLUMNS)


def binary_classification_metrics(
    labels: Sequence[int],
    spam_probabilities: Sequence[float],
    *,
    threshold: float = DECISION_THRESHOLD,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """Calculate the common classification and calibration metrics.

    ``precision``, ``recall`` and ``f1`` use spam (label 1) as the positive
    class.  ``pr_auc`` is the step-weighted area represented by average
    precision.  ROC-AUC, PR-AUC and balanced accuracy are returned as ``NaN``
    when only one true class is present, because the two-class comparison is
    then undefined.  Log loss is binary negative log-likelihood.
    """

    y_true, y_probability = validate_probabilities(labels, spam_probabilities)
    numeric_threshold = _validate_threshold(threshold)
    y_pred = (y_probability >= numeric_threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size == 2

    reliability = calibration_table(y_true, y_probability, n_bins=ece_bins)
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": (
            float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            )
            if has_both_classes
            else float("nan")
        ),
        "mcc": (
            float(matthews_corrcoef(y_true, y_pred))
            if np.unique(np.concatenate([y_true, y_pred])).size > 1
            else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, y_probability))
            if has_both_classes
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_probability))
            if has_both_classes
            else float("nan")
        ),
        "predicted_spam_rate": float(y_pred.mean()),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "ece": float(reliability["ece_contribution"].sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(len(y_true)),
        "ham_support": int((y_true == 0).sum()),
        "spam_support": int((y_true == 1).sum()),
    }
    return metrics


def _stable_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Hash source, label and text without relying on process-randomized hashing."""

    identifiers: list[str] = []
    for source, label, text in frame.loc[:, ["source", "label", "text"]].itertuples(
        index=False, name=None
    ):
        payload = json.dumps(
            [str(source), int(label), str(text)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        identifiers.append(hashlib.sha256(payload).hexdigest())
    return pd.Series(identifiers, index=frame.index, dtype="string")


def build_prediction_table(
    frame: pd.DataFrame,
    spam_probabilities: Sequence[float],
    *,
    model: str,
    training_seed: int | None,
    train_domain: str,
    test_domain: str,
    threshold: float = DECISION_THRESHOLD,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Build the stable row-level schema shared by all model notebooks.

    When ``id_column`` is omitted, ``example_id`` is a deterministic SHA-256
    hash of source, label and text.  An existing identifier can instead be
    preserved by naming its column explicitly.
    """

    required_columns = {"text", "label", "source"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"The frame is missing required columns: {sorted(missing_columns)}")
    if id_column is not None and id_column not in frame.columns:
        raise ValueError(f"The frame does not contain id_column={id_column!r}.")
    y_true, y_probability = validate_probabilities(frame["label"], spam_probabilities)
    numeric_threshold = _validate_threshold(threshold)
    y_pred = (y_probability >= numeric_threshold).astype(np.int8)

    if id_column is None:
        example_ids = _stable_example_ids(frame).reset_index(drop=True)
    else:
        supplied_ids = frame[id_column]
        if supplied_ids.isna().any():
            raise ValueError("Supplied example identifiers must not contain missing values.")
        example_ids = supplied_ids.astype("string").reset_index(drop=True)

    output = pd.DataFrame(
        {
            "example_id": example_ids,
            "model": model,
            "training_seed": training_seed,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "setting": (
                "in-domain" if train_domain == test_domain else "cross-domain"
            ),
            "label": y_true,
            "prediction": y_pred,
            "spam_probability": y_probability,
            "correct": y_true == y_pred,
            "text": frame["text"].reset_index(drop=True),
            "source": frame["source"].reset_index(drop=True),
        }
    )
    return output.loc[:, PREDICTION_COLUMNS]


def aggregate_seed_metrics(
    results: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    seed_column: str = "training_seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return both individual runs and their mean/sample-SD/count summary.

    Each group must contain at most one row per training seed.  The first
    returned frame deliberately preserves the individual runs so the aggregate
    cannot hide unstable or exceptional seeds.
    """

    groups = list(group_columns)
    metrics = list(metric_columns)
    if not groups:
        raise ValueError("At least one grouping column is required.")
    if not metrics:
        raise ValueError("At least one metric column is required.")
    required_columns = {*groups, *metrics, seed_column}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"Results are missing required columns: {sorted(missing_columns)}")
    if results.empty:
        raise ValueError("Results must contain at least one run.")
    duplicate_mask = results.duplicated([*groups, seed_column], keep=False)
    if duplicate_mask.any():
        raise ValueError("Each group must contain at most one row per training seed.")

    individual_runs = results.copy().sort_values(
        [*groups, seed_column], kind="stable", ignore_index=True
    )
    for metric in metrics:
        if not pd.api.types.is_numeric_dtype(individual_runs[metric]):
            raise ValueError(f"Metric column {metric!r} must be numeric.")

    grouped = individual_runs.groupby(groups, sort=True, dropna=False)[metrics]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    return individual_runs, summary


def _validate_bootstrap_options(
    metric: str,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> tuple[int, float, int]:
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(
            f"Unsupported bootstrap metric {metric!r}; choose from {BOOTSTRAP_METRICS}."
        )
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise ValueError("n_bootstrap must be a positive integer.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    try:
        numeric_confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be strictly between 0 and 1.") from exc
    if not np.isfinite(numeric_confidence) or not 0.0 < numeric_confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be a non-negative integer.")
    if random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    return int(n_bootstrap), numeric_confidence, int(random_state)


def _metric_value(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    metric: str,
    threshold: float,
    ece_bins: int,
) -> float:
    y_pred = (y_probability >= threshold).astype(np.int8)
    has_both_classes = np.unique(y_true).size == 2
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "macro_f1":
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
    if metric == "balanced_accuracy":
        if not has_both_classes:
            return float("nan")
        return float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
    if metric == "mcc":
        if np.unique(np.concatenate([y_true, y_pred])).size == 1:
            return 0.0
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "roc_auc":
        return (
            float(roc_auc_score(y_true, y_probability))
            if has_both_classes
            else float("nan")
        )
    if metric == "pr_auc":
        return (
            float(average_precision_score(y_true, y_probability))
            if has_both_classes
            else float("nan")
        )
    if metric == "predicted_spam_rate":
        return float(y_pred.mean())
    if metric == "brier_score":
        return float(brier_score_loss(y_true, y_probability))
    if metric == "log_loss":
        return float(log_loss(y_true, y_probability, labels=[0, 1]))
    if metric == "ece":
        table = calibration_table(y_true, y_probability, n_bins=ece_bins)
        return float(table["ece_contribution"].sum())
    raise AssertionError(f"Unhandled metric: {metric}")


def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_parts = []
    for label in np.unique(y_true):
        class_indices = np.flatnonzero(y_true == label)
        sampled_parts.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    sampled_indices = np.concatenate(sampled_parts)
    rng.shuffle(sampled_indices)
    return sampled_indices


def _interval_summary(
    bootstrap_values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float]:
    if not np.all(np.isfinite(bootstrap_values)):
        raise ValueError("The selected metric is undefined for the bootstrap samples.")
    tail_probability = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(
        bootstrap_values,
        [tail_probability, 1.0 - tail_probability],
    )
    standard_error = (
        float(bootstrap_values.std(ddof=1))
        if len(bootstrap_values) > 1
        else 0.0
    )
    return float(ci_low), float(ci_high), standard_error


def stratified_bootstrap_ci(
    labels: Sequence[int],
    spam_probabilities: Sequence[float],
    *,
    metric: str = "f1",
    threshold: float = DECISION_THRESHOLD,
    ece_bins: int = 10,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | str]:
    """Estimate a percentile CI while preserving the observed class counts."""

    y_true, y_probability = validate_probabilities(labels, spam_probabilities)
    numeric_threshold = _validate_threshold(threshold)
    _validate_bin_count(ece_bins, len(y_true))
    repetitions, confidence, seed = _validate_bootstrap_options(
        metric, n_bootstrap, confidence_level, random_state
    )
    estimate = _metric_value(
        y_true, y_probability, metric, numeric_threshold, ece_bins
    )
    if not np.isfinite(estimate):
        raise ValueError("The selected metric is undefined for the supplied labels.")

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = _stratified_bootstrap_indices(y_true, rng)
        bootstrap_values[repetition] = _metric_value(
            y_true[indices],
            y_probability[indices],
            metric,
            numeric_threshold,
            ece_bins,
        )
    ci_low, ci_high, standard_error = _interval_summary(
        bootstrap_values, confidence
    )
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_standard_error": standard_error,
        "confidence_level": confidence,
        "n_bootstrap": repetitions,
        "random_state": seed,
    }


def paired_stratified_bootstrap_ci(
    labels: Sequence[int],
    reference_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
    *,
    metric: str = "f1",
    threshold: float = DECISION_THRESHOLD,
    ece_bins: int = 10,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | str]:
    """Estimate a paired CI for ``candidate - reference`` on the same rows."""

    y_true, reference = validate_probabilities(labels, reference_probabilities)
    paired_labels, candidate = validate_probabilities(labels, candidate_probabilities)
    if not np.array_equal(y_true, paired_labels):
        raise ValueError("The paired predictions must use identical labels.")
    numeric_threshold = _validate_threshold(threshold)
    _validate_bin_count(ece_bins, len(y_true))
    repetitions, confidence, seed = _validate_bootstrap_options(
        metric, n_bootstrap, confidence_level, random_state
    )

    reference_estimate = _metric_value(
        y_true, reference, metric, numeric_threshold, ece_bins
    )
    candidate_estimate = _metric_value(
        y_true, candidate, metric, numeric_threshold, ece_bins
    )
    if not np.isfinite(reference_estimate) or not np.isfinite(candidate_estimate):
        raise ValueError("The selected metric is undefined for the supplied labels.")

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = _stratified_bootstrap_indices(y_true, rng)
        reference_value = _metric_value(
            y_true[indices],
            reference[indices],
            metric,
            numeric_threshold,
            ece_bins,
        )
        candidate_value = _metric_value(
            y_true[indices],
            candidate[indices],
            metric,
            numeric_threshold,
            ece_bins,
        )
        bootstrap_differences[repetition] = candidate_value - reference_value

    ci_low, ci_high, standard_error = _interval_summary(
        bootstrap_differences, confidence
    )
    return {
        "metric": metric,
        "reference_estimate": reference_estimate,
        "candidate_estimate": candidate_estimate,
        "difference": candidate_estimate - reference_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_standard_error": standard_error,
        "confidence_level": confidence,
        "n_bootstrap": repetitions,
        "random_state": seed,
    }
