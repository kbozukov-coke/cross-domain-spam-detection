"""Simple, reproducible baselines for cross-domain spam detection."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def build_tfidf_logistic_baseline(
    random_state: int = RANDOM_STATE,
    max_features: int = 50_000,
) -> Pipeline:
    """Build a transparent word TF-IDF and logistic-regression baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(
    model: ClassifierMixin,
    frame: pd.DataFrame,
    train_domain: str,
    test_domain: str,
) -> dict[str, object]:
    """Evaluate one fitted model and return classification metrics."""

    y_true = frame["label"]
    y_pred = model.predict(frame["text"])
    y_score = model.predict_proba(frame["text"])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "train_domain": train_domain,
        "test_domain": test_domain,
        "setting": "in-domain" if train_domain == test_domain else "cross-domain",
        "test_rows": len(frame),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_transfer_experiments(
    splits: Mapping[str, Mapping[str, pd.DataFrame]],
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Train one baseline per domain and evaluate it on both test domains."""

    required_domains = {"sms", "enron"}
    missing = required_domains - set(splits)
    if missing:
        raise ValueError(f"Missing domains: {sorted(missing)}")

    models: dict[str, Pipeline] = {}
    result_rows: list[dict[str, object]] = []

    for train_domain in ("sms", "enron"):
        train_frame = splits[train_domain]["train"]
        model = build_tfidf_logistic_baseline(random_state=random_state)
        model.fit(train_frame["text"], train_frame["label"])
        models[train_domain] = model

        for test_domain in ("sms", "enron"):
            result_rows.append(
                evaluate_model(
                    model,
                    splits[test_domain]["test"],
                    train_domain=train_domain,
                    test_domain=test_domain,
                )
            )

    results = pd.DataFrame(result_rows)
    return models, results


def prediction_details(model: ClassifierMixin, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach predictions and spam probabilities for error analysis."""

    details = frame.loc[:, ["text", "label", "source"]].copy()
    details["prediction"] = model.predict(details["text"])
    details["spam_probability"] = model.predict_proba(details["text"])[:, 1]
    details["correct"] = details["label"].eq(details["prediction"])
    return details
