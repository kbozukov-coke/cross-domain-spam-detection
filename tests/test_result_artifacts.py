from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
DOMAIN_PAIRS = {
    ("sms", "sms"),
    ("sms", "enron"),
    ("enron", "sms"),
    ("enron", "enron"),
}
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/kaloyanbozukov/notebook3"


def test_textcnn_tuning_artifacts_are_complete() -> None:
    runs = pd.read_csv(RESULTS_DIRECTORY / "textcnn_tuning_results.csv")
    summary = pd.read_csv(RESULTS_DIRECTORY / "textcnn_tuning_summary.csv")

    assert len(runs) == 24
    assert set(runs["train_domain"]) == {"sms", "enron"}
    assert set(runs["training_seed"]) == {13, 42, 73}
    assert not runs.duplicated(["train_domain", "config_id", "training_seed"]).any()

    assert len(summary) == 8
    assert not summary.duplicated(["train_domain", "config_id"]).any()
    selected = summary.loc[summary["selected"]]
    assert selected.groupby("train_domain").size().eq(1).all()
    assert set(selected["config_id"]) == {"kernel_3"}
    assert runs["source_notebook"].eq(SOURCE_NOTEBOOK).all()
    assert summary["source_notebook"].eq(SOURCE_NOTEBOOK).all()


def test_textcnn_multiseed_artifacts_are_complete() -> None:
    runs = pd.read_csv(RESULTS_DIRECTORY / "textcnn_seed_results.csv")
    summary = pd.read_csv(RESULTS_DIRECTORY / "textcnn_seed_summary.csv")

    assert len(runs) == 20
    assert set(runs["training_seed"]) == {13, 42, 73, 101, 137}
    assert set(zip(runs["train_domain"], runs["test_domain"])) == DOMAIN_PAIRS
    assert not runs.duplicated(
        ["train_domain", "test_domain", "training_seed"]
    ).any()
    assert runs["model"].eq("TextCNN").all()
    assert runs["config_id"].eq("kernel_3").all()

    assert len(summary) == 4
    assert set(zip(summary["train_domain"], summary["test_domain"])) == DOMAIN_PAIRS
    assert summary["f1_count"].eq(5).all()
    assert runs["source_notebook"].eq(SOURCE_NOTEBOOK).all()
    assert summary["source_notebook"].eq(SOURCE_NOTEBOOK).all()
