import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    CALIBRATION_TABLE_COLUMNS,
    PREDICTION_COLUMNS,
    RESULT_COLUMNS,
    aggregate_seed_metrics,
    binary_classification_metrics,
    build_prediction_table,
    calibration_table,
    paired_stratified_bootstrap_ci,
    stratified_bootstrap_ci,
    validate_probabilities,
)


def test_result_schema_contains_metadata_metrics_and_support() -> None:
    assert RESULT_COLUMNS[:6] == (
        "model",
        "training_seed",
        "train_domain",
        "test_domain",
        "setting",
        "test_rows",
    )
    assert "macro_f1" in RESULT_COLUMNS
    assert "brier_score" in RESULT_COLUMNS
    assert "spam_support" in RESULT_COLUMNS


def test_binary_metrics_cover_classification_calibration_and_support() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]

    metrics = binary_classification_metrics(labels, probabilities, ece_bins=2)

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
    ):
        assert metrics[metric] == pytest.approx(1.0)
    assert metrics["predicted_spam_rate"] == pytest.approx(0.5)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert metrics["log_loss"] > 0
    assert metrics["ece"] == pytest.approx(0.15)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
        2,
        0,
        0,
        2,
    )
    assert metrics["support"] == 4
    assert metrics["ham_support"] == 2
    assert metrics["spam_support"] == 2


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0.2], "same number"),
        ([0, 2], [0.2, 0.8], "binary values"),
        ([0, 1], [[0.2], [0.8]], "one-dimensional"),
        ([0, 1], [0.2, np.nan], "finite"),
        ([0, 1], [0.2, 1.1], "between 0 and 1"),
    ],
)
def test_probability_validation_rejects_invalid_inputs(
    labels: list[int], probabilities: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_probabilities(labels, probabilities)


def test_single_class_metrics_mark_two_class_quantities_as_undefined() -> None:
    metrics = binary_classification_metrics([0, 0], [0.1, 0.2])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert metrics["log_loss"] > 0


def test_calibration_table_uses_balanced_equal_frequency_bins() -> None:
    labels = [0, 1, 0, 1, 0, 1, 1]
    probabilities = [0.05, 0.15, 0.25, 0.45, 0.55, 0.75, 0.95]

    table = calibration_table(labels, probabilities, n_bins=3)

    assert tuple(table.columns) == CALIBRATION_TABLE_COLUMNS
    assert table["count"].tolist() == [3, 2, 2]
    assert table["count"].sum() == len(labels)
    assert table["count"].max() - table["count"].min() <= 1
    assert table["weight"].sum() == pytest.approx(1.0)
    metrics = binary_classification_metrics(labels, probabilities, ece_bins=3)
    assert metrics["ece"] == pytest.approx(table["ece_contribution"].sum())


def test_calibration_bins_cannot_exceed_the_number_of_rows() -> None:
    table = calibration_table([0, 1], [0.2, 0.8], n_bins=10)
    assert len(table) == 2
    assert table["count"].tolist() == [1, 1]


def test_prediction_table_has_stable_schema_and_ids() -> None:
    frame = pd.DataFrame(
        {
            "text": ["team meeting", "claim prize"],
            "label": [0, 1],
            "source": ["sms", "sms"],
        },
        index=[10, 20],
    )
    first = build_prediction_table(
        frame,
        [0.1, 0.8],
        model="TextCNN",
        training_seed=13,
        train_domain="sms",
        test_domain="enron",
    )
    second = build_prediction_table(
        frame.copy(),
        [0.1, 0.8],
        model="TextCNN",
        training_seed=13,
        train_domain="sms",
        test_domain="enron",
    )

    assert tuple(first.columns) == PREDICTION_COLUMNS
    assert first["example_id"].tolist() == second["example_id"].tolist()
    assert first["example_id"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert first["setting"].eq("cross-domain").all()
    assert first["prediction"].tolist() == [0, 1]
    assert first["correct"].all()
    assert first["text"].tolist() == frame["text"].tolist()


def test_prediction_table_can_preserve_supplied_ids() -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "text": ["hello", "winner"],
            "label": [0, 1],
            "source": ["sms", "sms"],
        }
    )
    table = build_prediction_table(
        frame,
        [0.2, 0.9],
        model="baseline",
        training_seed=42,
        train_domain="sms",
        test_domain="sms",
        id_column="row_id",
    )
    assert table["example_id"].tolist() == ["a", "b"]


def test_seed_aggregation_returns_runs_and_sample_statistics() -> None:
    results = pd.DataFrame(
        {
            "model": ["TextCNN", "TextCNN", "TextCNN"],
            "training_seed": [73, 13, 42],
            "f1": [0.7, 0.5, 0.6],
            "roc_auc": [0.9, 0.7, 0.8],
        }
    )

    runs, summary = aggregate_seed_metrics(
        results,
        group_columns=["model"],
        metric_columns=["f1", "roc_auc"],
    )

    assert runs["training_seed"].tolist() == [13, 42, 73]
    assert runs["f1"].tolist() == [0.5, 0.6, 0.7]
    assert summary.loc[0, "f1_mean"] == pytest.approx(0.6)
    assert summary.loc[0, "f1_std"] == pytest.approx(0.1)
    assert summary.loc[0, "f1_count"] == 3
    assert summary.loc[0, "roc_auc_mean"] == pytest.approx(0.8)
    assert results["training_seed"].tolist() == [73, 13, 42]


def test_seed_aggregation_rejects_duplicate_seed_within_group() -> None:
    results = pd.DataFrame(
        {
            "model": ["TextCNN", "TextCNN"],
            "training_seed": [42, 42],
            "f1": [0.5, 0.6],
        }
    )
    with pytest.raises(ValueError, match="at most one row"):
        aggregate_seed_metrics(
            results,
            group_columns=["model"],
            metric_columns=["f1"],
        )


def test_stratified_bootstrap_is_deterministic_and_contains_estimate() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.1, 0.3, 0.6, 0.8, 0.2, 0.55, 0.7, 0.9]
    kwargs = {
        "metric": "f1",
        "n_bootstrap": 200,
        "random_state": 17,
    }

    first = stratified_bootstrap_ci(labels, probabilities, **kwargs)
    second = stratified_bootstrap_ci(labels, probabilities, **kwargs)

    assert first == second
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
    assert first["n_bootstrap"] == 200
    assert first["random_state"] == 17


def test_paired_bootstrap_reports_candidate_minus_reference() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    reference = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    candidate = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    result = paired_stratified_bootstrap_ci(
        labels,
        reference,
        candidate,
        metric="f1",
        n_bootstrap=100,
        random_state=23,
    )

    assert result["reference_estimate"] == pytest.approx(0.0)
    assert result["candidate_estimate"] == pytest.approx(1.0)
    assert result["difference"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)
    assert result["ci_high"] == pytest.approx(1.0)


def test_bootstrap_rejects_an_undefined_metric() -> None:
    with pytest.raises(ValueError, match="undefined"):
        stratified_bootstrap_ci(
            [0, 0],
            [0.1, 0.2],
            metric="roc_auc",
            n_bootstrap=10,
        )


def test_bootstrap_rejects_negative_random_state() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        stratified_bootstrap_ci(
            [0, 1],
            [0.1, 0.9],
            n_bootstrap=10,
            random_state=-1,
        )
