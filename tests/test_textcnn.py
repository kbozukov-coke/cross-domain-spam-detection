import numpy as np
import pandas as pd

from src.textcnn import (
    balanced_class_weights,
    evaluate_textcnn,
    run_textcnn_experiments,
)


def _frame(domain: str, split: str) -> pd.DataFrame:
    ham = [
        {
            "text": f"team meeting project schedule {split} {index}",
            "label": 0,
            "source": domain,
        }
        for index in range(8)
    ]
    spam = [
        {
            "text": f"win cash prize claim offer {split} {index}",
            "label": 1,
            "source": domain,
        }
        for index in range(8)
    ]
    return pd.DataFrame(ham + spam)


def test_balanced_class_weights_give_more_weight_to_minority() -> None:
    weights = balanced_class_weights([0, 0, 0, 1])
    assert weights[1] > weights[0]


class _FixedTextCNN:
    def predict(self, texts: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert verbose == 0
        assert len(texts) == 4
        return np.array([[0.1], [0.2], [0.8], [0.9]])


def test_evaluate_textcnn_uses_the_shared_metric_schema() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "source": ["sms"] * 4,
        }
    )

    metrics = evaluate_textcnn(
        _FixedTextCNN(),
        frame,
        train_domain="sms",
        test_domain="enron",
        training_info={"training_seed": 73},
    )

    assert metrics["training_seed"] == 73
    assert metrics["macro_f1"] == 1.0
    assert metrics["spam_support"] == 2
    assert 0 <= metrics["ece"] <= 1


def test_textcnn_experiments_smoke_test() -> None:
    splits = {
        domain: {split: _frame(domain, split) for split in ("train", "validation", "test")}
        for domain in ("sms", "enron")
    }

    models, histories, results = run_textcnn_experiments(
        splits,
        max_tokens=200,
        sequence_length=16,
        embedding_dim=8,
        filters=8,
        kernel_size=3,
        dense_units=8,
        dropout=0.1,
        epochs=1,
        batch_size=8,
        patience=1,
        verbose=0,
    )

    assert set(models) == {"sms", "enron"}
    assert set(histories) == {"sms", "enron"}
    assert len(results) == 4
    assert results["f1"].between(0, 1).all()
    assert results["roc_auc"].between(0, 1).all()
