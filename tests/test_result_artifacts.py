from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import binary_classification_metrics
from src.generative import generation_audit_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
DOMAIN_PAIRS = {
    ("sms", "sms"),
    ("sms", "enron"),
    ("enron", "sms"),
    ("enron", "enron"),
}
SOURCE_NOTEBOOK = "https://www.kaggle.com/code/kaloyanbozukov/notebook3"
GENERATIVE_SOURCE_NOTEBOOK = "notebooks/05_zero_shot_generative.ipynb"


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


def test_generative_artifacts_are_complete_and_reproducible() -> None:
    results = pd.read_csv(
        RESULTS_DIRECTORY / "generative_zero_shot_results.csv"
    )
    predictions = pd.read_csv(
        RESULTS_DIRECTORY / "generative_zero_shot_predictions.csv"
    )
    calibration = pd.read_csv(
        RESULTS_DIRECTORY / "generative_zero_shot_calibration.csv"
    )
    coverage = pd.read_csv(
        RESULTS_DIRECTORY / "generative_zero_shot_confidence_coverage.csv"
    )
    audit = pd.read_csv(
        RESULTS_DIRECTORY / "generative_generation_audit.csv"
    )

    assert len(results) == 2
    assert len(predictions) == 2_755
    assert len(calibration) == 20
    assert len(coverage) == 10
    assert len(audit) == 100
    assert set(results["test_domain"]) == {"sms", "enron"}
    assert predictions.groupby("test_domain").size().to_dict() == {
        "enron": 1_981,
        "sms": 774,
    }
    assert predictions["example_id"].is_unique
    assert predictions["example_id"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert {"text", "message", "body", "content"}.isdisjoint(
        predictions.columns
    )
    assert {"text", "message", "body", "content"}.isdisjoint(audit.columns)

    metadata_frames = (results, predictions, calibration, audit)
    assert all(
        frame["source_notebook"].eq(GENERATIVE_SOURCE_NOTEBOOK).all()
        for frame in metadata_frames
    )
    assert predictions["prompt_id"].eq("spam_definition_v1").all()
    assert predictions["label_score_normalization"].eq(
        "mean_token_log_probability_including_eos"
    ).all()

    indexed_results = results.set_index("test_domain")
    for domain, domain_predictions in predictions.groupby("test_domain"):
        recorded = indexed_results.loc[domain]
        recalculated = binary_classification_metrics(
            domain_predictions["label"],
            domain_predictions["spam_probability"],
        )
        for metric, value in recalculated.items():
            assert np.isclose(recorded[metric], value)

        domain_calibration = calibration.loc[
            calibration["test_domain"].eq(domain)
        ]
        assert len(domain_calibration) == 10
        assert np.isclose(domain_calibration["weight"].sum(), 1.0)
        assert np.isclose(
            domain_calibration["ece_contribution"].sum(),
            recorded["ece"],
        )

        domain_audit = audit.loc[audit["test_domain"].eq(domain)]
        assert domain_audit["label"].value_counts().to_dict() == {0: 25, 1: 25}
        recalculated_audit = generation_audit_summary(
            domain_audit["label"],
            domain_audit["likelihood_prediction"],
            domain_audit["generated_label_id"],
        )
        for metric, value in recalculated_audit.items():
            assert np.isclose(recorded[metric], value)
