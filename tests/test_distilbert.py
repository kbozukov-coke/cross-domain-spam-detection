import numpy as np
import pandas as pd
import pytest

from src.distilbert import (
    balanced_class_weights,
    classification_metrics_from_logits,
    distilbert_prediction_details,
    probabilities_from_logits,
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


def test_classification_metrics_from_logits() -> None:
    metrics = classification_metrics_from_logits(
        labels=[0, 1, 0, 1],
        logits=np.array([[4.0, 1.0], [0.0, 3.0], [2.0, 0.0], [1.0, 5.0]]),
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
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
