"""DistilBERT fine-tuning and cross-domain evaluation utilities."""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .protocol import REFERENCE_TRAINING_SEED


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


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Convert two-class logits to the probability of the spam class."""

    logit_array = np.asarray(logits, dtype=np.float64)
    if logit_array.ndim != 2 or logit_array.shape[1] != 2:
        raise ValueError("Expected logits with shape (rows, 2).")
    shifted = logit_array - logit_array.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, 1] / exponentials.sum(axis=1)


def classification_metrics_from_logits(
    labels: Sequence[int],
    logits: np.ndarray,
) -> dict[str, float | int]:
    """Calculate the same binary metrics used by the previous models."""

    y_true = np.asarray(labels, dtype=np.int64)
    logit_array = np.asarray(logits)
    if len(y_true) != len(logit_array):
        raise ValueError("Labels and logits must contain the same number of rows.")

    y_score = probabilities_from_logits(logit_array)
    y_pred = logit_array.argmax(axis=1).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
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


def distilbert_prediction_details(
    frame: pd.DataFrame,
    logits: np.ndarray,
) -> pd.DataFrame:
    """Attach DistilBERT predictions and probabilities for error analysis."""

    if len(frame) != len(logits):
        raise ValueError("The frame and logits must contain the same number of rows.")
    details = frame.loc[:, ["text", "label", "source"]].copy()
    details["prediction"] = np.asarray(logits).argmax(axis=1).astype(np.int8)
    details["spam_probability"] = probabilities_from_logits(logits)
    details["correct"] = details["label"].eq(details["prediction"])
    return details


def _tokenize_frame(frame: pd.DataFrame, tokenizer: object, max_length: int) -> object:
    """Build a tokenized Hugging Face Dataset without padding every row."""

    from datasets import Dataset

    dataset = Dataset.from_pandas(
        frame.loc[:, ["text", "label"]],
        preserve_index=False,
    )

    def tokenize_batch(batch: Mapping[str, Sequence[str]]) -> dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized.rename_column("label", "labels")


def _validate_splits(splits: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    required_domains = {"sms", "enron"}
    missing_domains = required_domains - set(splits)
    if missing_domains:
        raise ValueError(f"Missing domains: {sorted(missing_domains)}")
    for domain in required_domains:
        missing_splits = {"train", "validation", "test"} - set(splits[domain])
        if missing_splits:
            raise ValueError(
                f"{domain} is missing splits: {sorted(missing_splits)}"
            )


def run_distilbert_experiments(
    splits: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    model_name: str = "distilbert-base-uncased",
    random_state: int = REFERENCE_TRAINING_SEED,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 16,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    require_gpu: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[tuple[str, str], pd.DataFrame],
]:
    """Fine-tune one model per domain and evaluate all four domain pairs.

    Checkpoints are temporary and each model is released before the next one is
    created. This keeps Kaggle memory and output storage use predictable.
    """

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _validate_splits(splits)
    if min(
        max_length,
        epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
    ) <= 0:
        raise ValueError("Lengths, epochs, and batch sizes must be positive.")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for this experiment.")
    gpu_count = torch.cuda.device_count()

    class WeightedTrainer(Trainer):
        def __init__(self, *args: object, class_weights: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: object,
            inputs: Mapping[str, object],
            return_outputs: bool = False,
            num_items_in_batch: object | None = None,
        ) -> object:
            del num_items_in_batch
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            outputs = model(**model_inputs)
            weights = self.class_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=weights,
            )
            return (loss, outputs) if return_outputs else loss

    set_seed(random_state)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    tokenized_tests = {
        domain: _tokenize_frame(splits[domain]["test"], tokenizer, max_length)
        for domain in ("sms", "enron")
    }
    histories: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    prediction_details: dict[tuple[str, str], pd.DataFrame] = {}

    def compute_metrics(evaluation: object) -> dict[str, float | int]:
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return classification_metrics_from_logits(evaluation.label_ids, logits)

    for train_domain in ("sms", "enron"):
        set_seed(random_state)
        train_frame = splits[train_domain]["train"]
        validation_frame = splits[train_domain]["validation"]
        tokenized_train = _tokenize_frame(train_frame, tokenizer, max_length)
        tokenized_validation = _tokenize_frame(
            validation_frame,
            tokenizer,
            max_length,
        )
        class_weight_values = balanced_class_weights(train_frame["label"])
        class_weight_tensor = torch.tensor(
            [class_weight_values[0], class_weight_values[1]],
            dtype=torch.float32,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        with tempfile.TemporaryDirectory(prefix=f"distilbert-{train_domain}-") as output_dir:
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                run_name=f"distilbert-{train_domain}",
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                num_train_epochs=epochs,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                report_to="none",
                seed=random_state,
                data_seed=random_state,
                dataloader_num_workers=2,
            )
            trainer = WeightedTrainer(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_validation,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                class_weights=class_weight_tensor,
            )

            train_output = trainer.train()
            history = pd.DataFrame(trainer.state.log_history)
            history["train_domain"] = train_domain
            histories[train_domain] = history
            epochs_trained = int(round(float(trainer.state.epoch or epochs)))

            for test_domain in ("sms", "enron"):
                prediction_output = trainer.predict(tokenized_tests[test_domain])
                logits = prediction_output.predictions
                if isinstance(logits, tuple):
                    logits = logits[0]
                test_frame = splits[test_domain]["test"]
                metrics = classification_metrics_from_logits(
                    test_frame["label"],
                    logits,
                )
                result_rows.append(
                    {
                        "model": "DistilBERT",
                        "train_domain": train_domain,
                        "test_domain": test_domain,
                        "setting": (
                            "in-domain"
                            if train_domain == test_domain
                            else "cross-domain"
                        ),
                        "test_rows": len(test_frame),
                        **metrics,
                        "epochs_trained": epochs_trained,
                        "training_seconds": float(
                            train_output.metrics.get("train_runtime", np.nan)
                        ),
                        "parameters": parameter_count,
                        "gpu_count": gpu_count,
                        "effective_train_batch_size": (
                            per_device_train_batch_size * max(1, gpu_count)
                        ),
                    }
                )
                prediction_details[(train_domain, test_domain)] = (
                    distilbert_prediction_details(test_frame, logits)
                )

        del trainer, model, tokenized_train, tokenized_validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return histories, pd.DataFrame(result_rows), prediction_details
