"""Reproducible TextCNN training and cross-domain evaluation."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras


RANDOM_STATE = 42


def set_global_seed(random_state: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible training."""

    random.seed(random_state)
    np.random.seed(random_state)
    tf.keras.utils.set_random_seed(random_state)


def balanced_class_weights(labels: Sequence[int]) -> dict[int, float]:
    """Return inverse-frequency weights for binary labels."""

    label_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(label_array, minlength=2)
    if len(label_array) == 0 or np.any(counts == 0):
        raise ValueError("Both classes must be present to compute class weights.")
    return {
        class_id: float(len(label_array) / (2 * class_count))
        for class_id, class_count in enumerate(counts)
    }


def build_text_vectorizer(
    train_texts: Sequence[str],
    max_tokens: int = 30_000,
    sequence_length: int = 256,
) -> keras.layers.TextVectorization:
    """Fit a token vocabulary using training text only."""

    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        name="text_vectorization",
    )
    # Keep variable-length Python strings. Converting Enron to a fixed-width
    # Unicode array would allocate memory for every row at the maximum email
    # length (more than 20 GB for the current training split).
    text_array = np.asarray(train_texts, dtype=object)
    text_batches = tf.data.Dataset.from_tensor_slices(text_array).batch(256)
    vectorizer.adapt(text_batches)
    return vectorizer


def build_textcnn(
    vectorizer: keras.layers.TextVectorization,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout: float = 0.4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a compact TextCNN binary classifier."""

    inputs = keras.Input(shape=(), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=embedding_dim,
        name="embedding",
    )(x)
    x = keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        name="conv1d",
    )(x)
    x = keras.layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="textcnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


def fit_textcnn(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
    max_tokens: int = 30_000,
    sequence_length: int = 256,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout: float = 0.4,
    learning_rate: float = 1e-3,
    epochs: int = 8,
    batch_size: int = 64,
    patience: int = 2,
    verbose: int = 1,
) -> tuple[keras.Model, keras.callbacks.History, dict[str, float | int]]:
    """Fit one TextCNN without adapting any component on validation or test data."""

    set_global_seed(random_state)
    train_texts = train_frame["text"].astype(str).to_numpy(dtype=object)
    train_labels = train_frame["label"].to_numpy(dtype=np.float32)
    validation_texts = validation_frame["text"].astype(str).to_numpy(dtype=object)
    validation_labels = validation_frame["label"].to_numpy(dtype=np.float32)

    vectorizer = build_text_vectorizer(
        train_texts,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
    )
    model = build_textcnn(
        vectorizer,
        embedding_dim=embedding_dim,
        filters=filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]

    started_at = time.perf_counter()
    history = model.fit(
        train_texts,
        train_labels,
        validation_data=(validation_texts, validation_labels),
        class_weight=balanced_class_weights(train_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    training_seconds = time.perf_counter() - started_at
    training_info: dict[str, float | int] = {
        "epochs_trained": len(history.history["loss"]),
        "training_seconds": training_seconds,
        "vocabulary_size": len(vectorizer.get_vocabulary()),
        "parameters": model.count_params(),
    }
    return model, history, training_info


def evaluate_textcnn(
    model: keras.Model,
    frame: pd.DataFrame,
    train_domain: str,
    test_domain: str,
    training_info: Mapping[str, float | int] | None = None,
) -> dict[str, object]:
    """Evaluate a fitted TextCNN with the same metrics as the ML baseline."""

    y_true = frame["label"].to_numpy()
    test_texts = frame["text"].astype(str).to_numpy(dtype=object)
    y_score = model.predict(test_texts, verbose=0).reshape(-1)
    y_pred = (y_score >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result: dict[str, object] = {
        "model": "TextCNN",
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
    if training_info:
        result.update(training_info)
    return result


def run_textcnn_experiments(
    splits: Mapping[str, Mapping[str, pd.DataFrame]],
    **fit_kwargs: object,
) -> tuple[
    dict[str, keras.Model],
    dict[str, keras.callbacks.History],
    pd.DataFrame,
]:
    """Train one TextCNN per source and evaluate all four domain pairs."""

    required_domains = {"sms", "enron"}
    missing = required_domains - set(splits)
    if missing:
        raise ValueError(f"Missing domains: {sorted(missing)}")

    models: dict[str, keras.Model] = {}
    histories: dict[str, keras.callbacks.History] = {}
    result_rows: list[dict[str, object]] = []

    for train_domain in ("sms", "enron"):
        model, history, training_info = fit_textcnn(
            splits[train_domain]["train"],
            splits[train_domain]["validation"],
            **fit_kwargs,
        )
        models[train_domain] = model
        histories[train_domain] = history

        for test_domain in ("sms", "enron"):
            result_rows.append(
                evaluate_textcnn(
                    model,
                    splits[test_domain]["test"],
                    train_domain=train_domain,
                    test_domain=test_domain,
                    training_info=training_info,
                )
            )

    return models, histories, pd.DataFrame(result_rows)


def textcnn_prediction_details(model: keras.Model, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach TextCNN predictions and probabilities for error analysis."""

    details = frame.loc[:, ["text", "label", "source"]].copy()
    detail_texts = details["text"].astype(str).to_numpy(dtype=object)
    scores = model.predict(detail_texts, verbose=0).reshape(-1)
    details["prediction"] = (scores >= 0.5).astype(np.int8)
    details["spam_probability"] = scores
    details["correct"] = details["label"].eq(details["prediction"])
    return details
