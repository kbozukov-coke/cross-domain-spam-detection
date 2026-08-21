import numpy as np
import pandas as pd

from src.modeling import evaluate_model, prediction_details, run_transfer_experiments


def _frame(domain: str, split: str) -> pd.DataFrame:
    suffix = [f"{split} {index}" for index in range(8)]
    ham = [
        {"text": f"team meeting project schedule {item}", "label": 0, "source": domain}
        for item in suffix
    ]
    spam = [
        {"text": f"win cash prize claim offer {item}", "label": 1, "source": domain}
        for item in suffix
    ]
    return pd.DataFrame(ham + spam)


class _FixedProbabilityModel:
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        assert len(texts) == len(probabilities)
        return np.column_stack([1 - probabilities, probabilities])


def test_evaluate_model_uses_the_shared_metric_schema() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "source": ["sms"] * 4,
        }
    )

    metrics = evaluate_model(
        _FixedProbabilityModel(),
        frame,
        train_domain="sms",
        test_domain="sms",
        training_seed=13,
    )

    assert metrics["training_seed"] == 13
    assert metrics["macro_f1"] == 1.0
    assert metrics["spam_support"] == 2
    assert 0 <= metrics["brier_score"] <= 1


def test_transfer_experiments_cover_all_domain_pairs() -> None:
    splits = {
        domain: {split: _frame(domain, split) for split in ("train", "validation", "test")}
        for domain in ("sms", "enron")
    }

    models, results = run_transfer_experiments(splits)

    assert set(models) == {"sms", "enron"}
    assert len(results) == 4
    assert set(zip(results["train_domain"], results["test_domain"], strict=True)) == {
        ("sms", "sms"),
        ("sms", "enron"),
        ("enron", "sms"),
        ("enron", "enron"),
    }
    assert results["f1"].eq(1.0).all()

    details = prediction_details(models["sms"], splits["enron"]["test"])
    assert {"prediction", "spam_probability", "correct"}.issubset(details.columns)
    assert details["correct"].all()
