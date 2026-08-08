import pandas as pd

from src.modeling import prediction_details, run_transfer_experiments


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
