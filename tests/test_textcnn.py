import pandas as pd

from src.textcnn import balanced_class_weights, run_textcnn_experiments


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
