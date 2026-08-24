import numpy as np
import pandas as pd
import pytest

from src.distilbert import (
    balanced_class_weights,
    classification_metrics_from_logits,
    distilbert_prediction_details,
    probabilities_from_logits,
    validate_max_lengths,
)


def test_balanced_class_weights_give_more_weight_to_minority() -> None:
    weights = balanced_class_weights([0, 0, 0, 1])
    assert weights[1] > weights[0]


def test_probabilities_from_logits_are_valid() -> None:
    probabilities = probabilities_from_logits(
        np.array([[3.0, 1.0], [0.0, 2.0]])
    )
    assert probabilities[0] < 0.5
    assert probabilities[1] > 0.5
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_probabilities_from_logits_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        probabilities_from_logits(np.array([[0.0, np.nan]]))


def test_classification_metrics_from_logits() -> None:
    metrics = classification_metrics_from_logits(
        labels=[0, 1, 0, 1],
        logits=np.array([[4.0, 1.0], [0.0, 3.0], [2.0, 0.0], [1.0, 5.0]]),
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["spam_support"] == 2
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
        2,
        0,
        0,
        2,
    )


def test_prediction_details_preserve_rows_and_add_outputs() -> None:
    frame = pd.DataFrame(
        {
            "text": ["team meeting", "claim prize"],
            "label": [0, 1],
            "source": ["sms", "sms"],
        }
    )
    details = distilbert_prediction_details(
        frame,
        np.array([[3.0, 0.0], [0.0, 3.0]]),
    )
    assert details["prediction"].tolist() == [0, 1]
    assert details["correct"].all()
    assert details["spam_probability"].between(0, 1).all()


def test_prediction_details_can_use_shared_metadata_schema() -> None:
    frame = pd.DataFrame(
        {
            "text": ["team meeting", "claim prize"],
            "label": [0, 1],
            "source": ["sms", "sms"],
        }
    )
    details = distilbert_prediction_details(
        frame,
        np.array([[3.0, 0.0], [0.0, 3.0]]),
        training_seed=42,
        train_domain="sms",
        test_domain="enron",
    )
    assert details["model"].eq("DistilBERT").all()
    assert details["training_seed"].eq(42).all()
    assert details["train_domain"].eq("sms").all()
    assert details["test_domain"].eq("enron").all()
    assert details["example_id"].str.len().eq(64).all()


def test_validate_max_lengths_preserves_predeclared_order() -> None:
    assert validate_max_lengths(
        [64, 128, 256, 512],
        maximum_length=512,
    ) == (64, 128, 256, 512)


@pytest.mark.parametrize(
    ("lengths", "error"),
    [
        ([], ValueError),
        ([0], ValueError),
        ([513], ValueError),
        ([64, 64], ValueError),
        ([64.0], TypeError),
        ([True], TypeError),
    ],
)
def test_validate_max_lengths_rejects_invalid_values(lengths, error) -> None:
    with pytest.raises(error):
        validate_max_lengths(lengths, maximum_length=512)
