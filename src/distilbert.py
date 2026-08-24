"""DistilBERT fine-tuning and cross-domain evaluation utilities."""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd

from .evaluation import binary_classification_metrics, build_prediction_table
from .protocol import (
    DECISION_THRESHOLD,
    REFERENCE_TRAINING_SEED,
    SELECTION_METRIC,
)


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
    """Convert finite two-class logits to the probability of the spam class."""

    logit_array = np.asarray(logits, dtype=np.float64)
    if (
        logit_array.ndim != 2
        or logit_array.shape[1] != 2
        or len(logit_array) == 0
    ):
        raise ValueError("Expected logits with shape (rows, 2).")
    if not np.isfinite(logit_array).all():
        raise ValueError("Logits must contain only finite values.")
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
    return binary_classification_metrics(y_true, y_score)


def distilbert_prediction_details(
    frame: pd.DataFrame,
    logits: np.ndarray,
    *,
    training_seed: int | None = None,
    train_domain: str | None = None,
    test_domain: str | None = None,
) -> pd.DataFrame:
    """Attach DistilBERT predictions and probabilities for error analysis.

    Supplying both domains returns the shared stable prediction-table schema.
    Omitting both keeps the compact backwards-compatible diagnostic table.
    """

    if len(frame) != len(logits):
        raise ValueError("The frame and logits must contain the same number of rows.")
    if (train_domain is None) != (test_domain is None):
        raise ValueError("train_domain and test_domain must be supplied together.")

    scores = probabilities_from_logits(logits)
    if train_domain is not None and test_domain is not None:
        return build_prediction_table(
            frame,
            scores,
            model="DistilBERT",
            training_seed=training_seed,
            train_domain=train_domain,
            test_domain=test_domain,
        )

    details = frame.loc[:, ["text", "label", "source"]].copy()
    details["prediction"] = (scores >= DECISION_THRESHOLD).astype(np.int8)
    details["spam_probability"] = scores
    details["correct"] = details["label"].eq(details["prediction"])
    return details


def validate_max_lengths(
    max_lengths: Sequence[int],
    *,
    maximum_length: int,
) -> tuple[int, ...]:
    """Validate distinct positive input lengths against a model limit."""

    if isinstance(maximum_length, (bool, np.bool_)) or not isinstance(
        maximum_length, Integral
    ):
        raise TypeError("maximum_length must be a positive integer.")
    if maximum_length <= 0:
        raise ValueError("maximum_length must be a positive integer.")

    validated: list[int] = []
    for value in max_lengths:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("Every max length must be a positive integer.")
        numeric_value = int(value)
        if numeric_value <= 0:
            raise ValueError("Every max length must be a positive integer.")
        if numeric_value > maximum_length:
            raise ValueError(
                f"max_length={numeric_value} exceeds the model limit "
                f"of {maximum_length}."
            )
        if numeric_value in validated:
            raise ValueError("Evaluation max lengths must be distinct.")
        validated.append(numeric_value)

    if not validated:
        raise ValueError("At least one max length is required.")
    return tuple(validated)


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


def _validate_frame(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    missing_columns = {"text", "label"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty.")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError(f"{name} contains missing text or labels.")


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


def _validate_training_options(
    *,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
) -> None:
    integer_options = {
        "epochs": epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in integer_options.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative and finite.")
    if not np.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in the interval [0, 1).")


def run_distilbert_source_experiment(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    train_domain: str,
    evaluation_frames: Mapping[str, pd.DataFrame] | None = None,
    evaluation_max_lengths: Sequence[int] | None = None,
    detail_max_length: int | None = None,
    model_name: str = "distilbert-base-uncased",
    random_state: int = REFERENCE_TRAINING_SEED,
    train_max_length: int = 256,
    learning_rate: float = 2e-5,
    epochs: int = 2,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    require_gpu: bool = False,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    dict[tuple[str, int], pd.DataFrame],
]:
    """Train one source model and optionally evaluate supplied frames.

    The function owns the complete model lifecycle and returns only CPU-side
    tables. Passing no evaluation frames creates a validation-only path for
    hyperparameter selection and never accesses a test split.
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

    if not isinstance(train_domain, str) or not train_domain:
        raise ValueError("train_domain must be a non-empty string.")
    _validate_frame(train_frame, name="train_frame")
    _validate_frame(validation_frame, name="validation_frame")
    _validate_training_options(
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for this experiment.")

    evaluation_items = dict(evaluation_frames or {})
    for domain, frame in evaluation_items.items():
        if not isinstance(domain, str) or not domain:
            raise ValueError("Evaluation domain names must be non-empty strings.")
        _validate_frame(frame, name=f"evaluation_frames[{domain!r}]")
        if detail_max_length is not None and "source" not in frame.columns:
            raise ValueError(
                "Evaluation frames need a source column when details are retained."
            )

    requested_lengths = tuple(
        evaluation_max_lengths
        if evaluation_max_lengths is not None
        else (train_max_length,)
    )

    trainer = None
    model = None
    tokenizer = None
    tokenized_train = None
    tokenized_validation = None
    validation_prediction = None
    train_output = None
    history = pd.DataFrame()
    validation_result: dict[str, object] = {}
    evaluation_rows: list[dict[str, object]] = []
    prediction_details: dict[tuple[str, int], pd.DataFrame] = {}

    try:
        set_seed(random_state)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )
        model_limit = int(model.config.max_position_embeddings)
        validate_max_lengths((train_max_length,), maximum_length=model_limit)
        validated_evaluation_lengths = validate_max_lengths(
            requested_lengths,
            maximum_length=model_limit,
        )
        if (
            detail_max_length is not None
            and detail_max_length not in validated_evaluation_lengths
        ):
            raise ValueError(
                "detail_max_length must be one of evaluation_max_lengths."
            )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )
        tokenized_train = _tokenize_frame(
            train_frame,
            tokenizer,
            train_max_length,
        )
        tokenized_validation = _tokenize_frame(
            validation_frame,
            tokenizer,
            train_max_length,
        )
        class_weight_values = balanced_class_weights(train_frame["label"])
        class_weight_tensor = torch.tensor(
            [class_weight_values[0], class_weight_values[1]],
            dtype=torch.float32,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        class WeightedTrainer(Trainer):
            def __init__(
                self,
                *args: object,
                class_weights: object,
                **kwargs: object,
            ) -> None:
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

        def compute_metrics(evaluation: object) -> dict[str, float | int]:
            logits = evaluation.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            return classification_metrics_from_logits(
                evaluation.label_ids,
                logits,
            )

        with tempfile.TemporaryDirectory(
            prefix=f"distilbert-{train_domain}-{random_state}-"
        ) as output_dir:
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                run_name=f"distilbert-{train_domain}-{random_state}",
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=epochs,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                load_best_model_at_end=True,
                metric_for_best_model=SELECTION_METRIC,
                greater_is_better=True,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                report_to="none",
                seed=random_state,
                data_seed=random_state,
                dataloader_num_workers=2,
                disable_tqdm=not verbose,
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
            history["training_seed"] = random_state

            validation_prediction = trainer.predict(tokenized_validation)
            validation_logits = validation_prediction.predictions
            if isinstance(validation_logits, tuple):
                validation_logits = validation_logits[0]
            validation_metrics = classification_metrics_from_logits(
                validation_frame["label"],
                validation_logits,
            )

            evaluation_history = history.dropna(subset=["eval_macro_f1"])
            best_epoch = (
                float(
                    evaluation_history.loc[
                        evaluation_history["eval_macro_f1"].idxmax(),
                        "epoch",
                    ]
                )
                if not evaluation_history.empty
                else float("nan")
            )
            effective_train_batch_size = int(
                trainer.args.train_batch_size
                * trainer.args.gradient_accumulation_steps
                * trainer.args.world_size
            )
            training_info: dict[str, object] = {
                "model_name": model_name,
                "training_seed": random_state,
                "train_max_length": train_max_length,
                "learning_rate": learning_rate,
                "epochs_requested": epochs,
                "epochs_trained": float(trainer.state.epoch or epochs),
                "best_epoch": best_epoch,
                "best_validation_macro_f1": float(
                    validation_metrics["macro_f1"]
                ),
                "training_seconds": float(
                    train_output.metrics.get("train_runtime", np.nan)
                ),
                "parameters": parameter_count,
                "train_rows": len(train_frame),
                "train_ham": int(train_frame["label"].eq(0).sum()),
                "train_spam": int(train_frame["label"].eq(1).sum()),
                "gpu_count": torch.cuda.device_count(),
                "world_size": int(trainer.args.world_size),
                "fp16": bool(trainer.args.fp16),
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_train_batch_size": effective_train_batch_size,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
            }
            validation_result = {
                "model": "DistilBERT",
                "train_domain": train_domain,
                "validation_rows": len(validation_frame),
                **validation_metrics,
                **training_info,
            }

            for evaluation_max_length in validated_evaluation_lengths:
                for test_domain, evaluation_frame in evaluation_items.items():
                    tokenized_evaluation = _tokenize_frame(
                        evaluation_frame,
                        tokenizer,
                        evaluation_max_length,
                    )
                    prediction_output = trainer.predict(tokenized_evaluation)
                    logits = prediction_output.predictions
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    metrics = classification_metrics_from_logits(
                        evaluation_frame["label"],
                        logits,
                    )
                    evaluation_rows.append(
                        {
                            "model": "DistilBERT",
                            "train_domain": train_domain,
                            "test_domain": test_domain,
                            "setting": (
                                "in-domain"
                                if train_domain == test_domain
                                else "cross-domain"
                            ),
                            "test_rows": len(evaluation_frame),
                            "evaluation_max_length": evaluation_max_length,
                            **metrics,
                            **training_info,
                        }
                    )
                    if evaluation_max_length == detail_max_length:
                        prediction_details[
                            (test_domain, evaluation_max_length)
                        ] = distilbert_prediction_details(
                            evaluation_frame,
                            logits,
                            training_seed=random_state,
                            train_domain=train_domain,
                            test_domain=test_domain,
                        )
                    del tokenized_evaluation, prediction_output, logits

        return (
            history,
            validation_result,
            pd.DataFrame(evaluation_rows),
            prediction_details,
        )
    finally:
        if trainer is not None:
            trainer.optimizer = None
            trainer.lr_scheduler = None
        trainer = None
        model = None
        tokenizer = None
        tokenized_train = None
        tokenized_validation = None
        validation_prediction = None
        train_output = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    """Run the original one-seed, two-source benchmark.

    This compatibility wrapper delegates to the leakage-safe single-source
    runner. New tuning notebooks should call run_distilbert_source_experiment
    directly and pass no evaluation frames during model selection.
    """

    _validate_splits(splits)
    histories: dict[str, pd.DataFrame] = {}
    result_frames: list[pd.DataFrame] = []
    prediction_details: dict[tuple[str, str], pd.DataFrame] = {}

    for train_domain in ("sms", "enron"):
        history, _, results, details = run_distilbert_source_experiment(
            splits[train_domain]["train"],
            splits[train_domain]["validation"],
            train_domain=train_domain,
            evaluation_frames={
                domain: splits[domain]["test"] for domain in ("sms", "enron")
            },
            evaluation_max_lengths=(max_length,),
            detail_max_length=max_length,
            model_name=model_name,
            random_state=random_state,
            train_max_length=max_length,
            learning_rate=learning_rate,
            epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            require_gpu=require_gpu,
        )
        histories[train_domain] = history
        result_frames.append(results)
        for (test_domain, _), detail_frame in details.items():
            prediction_details[(train_domain, test_domain)] = detail_frame

    return (
        histories,
        pd.concat(result_frames, ignore_index=True),
        prediction_details,
    )
