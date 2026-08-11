from pathlib import Path

import pandas as pd


def test_textcnn_result_artifact_has_all_domain_pairs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results = pd.read_csv(project_root / "results" / "textcnn_results.csv")

    pairs = set(zip(results["train_domain"], results["test_domain"]))
    assert pairs == {
        ("sms", "sms"),
        ("sms", "enron"),
        ("enron", "sms"),
        ("enron", "enron"),
    }
    assert results["model"].eq("TextCNN").all()
    assert results[["accuracy", "precision", "recall", "f1", "roc_auc"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
