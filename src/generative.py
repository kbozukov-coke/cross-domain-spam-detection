"""Utilities for zero-shot generative spam classification."""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import nullcontext
from numbers import Integral

import numpy as np
import pandas as pd


LABEL_NAMES = ("ham", "spam")
LABEL_TO_ID = {"ham": 0, "spam": 1}


def format_prompts(
    texts: Sequence[str],
    prompt_template: str,
) -> list[str]:
    """Insert each message into a template containing a text placeholder."""

    if not isinstance(prompt_template, str) or "{text}" not in prompt_template:
        raise ValueError("prompt_template must contain a {text} placeholder.")
    prompts: list[str] = []
    for text in texts:
        if text is None or pd.isna(text):
            raise ValueError("Prompt texts must not contain missing values.")
        prompts.append(prompt_template.format(text=str(text)))
    if not prompts:
        raise ValueError("At least one text is required.")
    return prompts


def spam_probabilities_from_label_scores(
    label_scores: Sequence[Sequence[float]],
    *,
    label_names: Sequence[str] = LABEL_NAMES,
) -> np.ndarray:
    """Softmax two finite label scores into spam pseudo-probabilities."""

    names = tuple(label_names)
    if len(names) != 2 or len(set(names)) != 2 or set(names) != set(LABEL_NAMES):
        raise ValueError("label_names must contain ham and spam exactly once.")
    scores = np.asarray(label_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 2 or len(scores) == 0:
        raise ValueError("label_scores must have shape (rows, 2).")
    if not np.isfinite(scores).all():
        raise ValueError("label_scores must contain only finite values.")

    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, names.index("spam")]


def normalize_generated_label(text: str) -> str | None:
    """Parse a strict one-label generated response."""

    if not isinstance(text, str):
        return None
    normalized = text.strip().lower()
    normalized = re.sub(r"""^[\s"']+|[\s"'.,!?;:]+$""", "", normalized)
    return normalized if normalized in LABEL_TO_ID else None


def parse_generated_labels(texts: Sequence[str]) -> np.ndarray:
    """Return integer labels and use -1 for invalid generations."""

    parsed = [
        LABEL_TO_ID.get(normalize_generated_label(text), -1)
        for text in texts
    ]
    return np.asarray(parsed, dtype=np.int8)


def confidence_coverage_table(
    labels: Sequence[int],
    spam_probabilities: Sequence[float],
    *,
    minimum_confidences: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """Summarize accuracy after abstaining below confidence thresholds."""

    y_true = np.asarray(labels)
    y_probability = np.asarray(spam_probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_probability.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(y_true) == 0 or len(y_true) != len(y_probability):
        raise ValueError("Labels and probabilities must be non-empty and aligned.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(y_probability).all() or np.any(
        (y_probability < 0) | (y_probability > 1)
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1.")

    thresholds: list[float] = []
    for value in minimum_confidences:
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Confidence thresholds must be between 0.5 and 1."
            ) from exc
        if not np.isfinite(threshold) or not 0.5 <= threshold <= 1:
            raise ValueError("Confidence thresholds must be between 0.5 and 1.")
        if threshold in thresholds:
            raise ValueError("Confidence thresholds must be distinct.")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one confidence threshold is required.")

    predictions = (y_probability >= 0.5).astype(np.int8)
    confidence = np.maximum(y_probability, 1 - y_probability)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        retained = confidence >= threshold
        retained_rows = int(retained.sum())
        accuracy = (
            float((predictions[retained] == y_true[retained]).mean())
            if retained_rows
            else float("nan")
        )
        rows.append(
            {
                "minimum_confidence": threshold,
                "retained_rows": retained_rows,
                "coverage": retained_rows / len(y_true),
                "accuracy": accuracy,
                "error_rate": 1 - accuracy if retained_rows else float("nan"),
                "mean_confidence": (
                    float(confidence[retained].mean())
                    if retained_rows
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def generation_audit_summary(
    labels: Sequence[int],
    likelihood_predictions: Sequence[int],
    generated_label_ids: Sequence[int],
) -> dict[str, float | int]:
    """Summarize strict generation compliance without dropping invalid rows."""

    y_true = np.asarray(labels)
    likelihood = np.asarray(likelihood_predictions)
    generated = np.asarray(generated_label_ids)
    if any(array.ndim != 1 for array in (y_true, likelihood, generated)):
        raise ValueError("Generation audit arrays must be one-dimensional.")
    if len(y_true) == 0 or len({len(y_true), len(likelihood), len(generated)}) != 1:
        raise ValueError("Generation audit arrays must be non-empty and aligned.")
    if not np.isin(y_true, (0, 1)).all() or not np.isin(
        likelihood, (0, 1)
    ).all():
        raise ValueError("True and likelihood labels must contain only 0 and 1.")
    if not np.isin(generated, (-1, 0, 1)).all():
        raise ValueError("Generated labels must contain only -1, 0 and 1.")

    valid = generated >= 0
    valid_rows = int(valid.sum())
    return {
        "audit_rows": len(y_true),
        "valid_generation_rows": valid_rows,
        "generation_compliance_rate": valid_rows / len(y_true),
        "generation_exact_accuracy_all_rows": float(
            (generated == y_true).mean()
        ),
        "generation_accuracy_valid_only": (
            float((generated[valid] == y_true[valid]).mean())
            if valid_rows
            else float("nan")
        ),
        "likelihood_generation_agreement_valid_only": (
            float((generated[valid] == likelihood[valid]).mean())
            if valid_rows
            else float("nan")
        ),
    }


def _validate_inference_options(
    *,
    max_length: int,
    batch_size: int,
    max_new_tokens: int | None = None,
) -> None:
    options = {"max_length": max_length, "batch_size": batch_size}
    if max_new_tokens is not None:
        options["max_new_tokens"] = max_new_tokens
    for name, value in options.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def _model_device(model: object, device: object | None) -> object:
    import torch

    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _autocast_context(device: object) -> object:
    import torch

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _candidate_token_lengths(
    tokenizer: object,
    candidate_labels: Sequence[str],
) -> tuple[int, ...]:
    lengths = []
    for candidate in candidate_labels:
        encoded = tokenizer(
            text_target=candidate,
            add_special_tokens=True,
        )
        token_ids = encoded["input_ids"]
        if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
            raise ValueError("Candidate labels must include the tokenizer EOS token.")
        lengths.append(len(token_ids))
    return tuple(lengths)


def score_candidate_labels(
    model: object,
    tokenizer: object,
    texts: Sequence[str],
    *,
    prompt_template: str,
    candidate_labels: Sequence[str] = LABEL_NAMES,
    max_length: int = 256,
    batch_size: int = 16,
    device: object | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score ham and spam by mean decoder-token log probability.

    The score includes EOS and is averaged across non-padding target tokens.
    Length normalization keeps verbalizers with different token counts
    comparable without replacing the human-readable ``ham``/``spam`` labels.
    """

    import torch

    _validate_inference_options(max_length=max_length, batch_size=batch_size)
    labels = tuple(candidate_labels)
    if len(labels) != 2 or set(labels) != set(LABEL_NAMES):
        raise ValueError("candidate_labels must contain ham and spam exactly once.")
    _candidate_token_lengths(tokenizer, labels)

    prompts = format_prompts(texts, prompt_template)
    input_device = _model_device(model, device)
    model.eval()
    collected_scores: list[np.ndarray] = []
    candidate_targets: dict[str, tuple[object, object]] = {}
    for candidate in labels:
        encoded_target = tokenizer(
            text_target=[candidate],
            padding=True,
            return_tensors="pt",
        )
        candidate_targets[candidate] = (
            encoded_target["input_ids"].to(input_device),
            encoded_target["attention_mask"].to(input_device),
        )

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            encoded_inputs = tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_inputs = {
                name: tensor.to(input_device)
                for name, tensor in encoded_inputs.items()
            }

            with _autocast_context(input_device):
                encoder_outputs = model.get_encoder()(**encoded_inputs)

            batch_label_scores: list[np.ndarray] = []
            for candidate in labels:
                base_target_ids, base_target_mask = candidate_targets[candidate]
                target_ids = base_target_ids.repeat(len(prompt_batch), 1)
                target_mask = base_target_mask.repeat(len(prompt_batch), 1)
                model_labels = target_ids.masked_fill(target_mask.eq(0), -100)

                with _autocast_context(input_device):
                    outputs = model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=encoded_inputs.get("attention_mask"),
                        labels=model_labels,
                        use_cache=False,
                    )
                token_log_probabilities = torch.log_softmax(
                    outputs.logits.float(),
                    dim=-1,
                )
                selected_log_probabilities = token_log_probabilities.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)
                sequence_log_probability = (
                    selected_log_probabilities * target_mask
                ).sum(dim=1)
                target_token_count = target_mask.sum(dim=1)
                sequence_scores = sequence_log_probability / target_token_count
                batch_label_scores.append(
                    sequence_scores.detach().cpu().numpy()
                )

            collected_scores.append(np.column_stack(batch_label_scores))
            if verbose:
                completed = min(start + batch_size, len(prompts))
                if (
                    completed == len(prompts)
                    or start == 0
                    or completed % (20 * batch_size) == 0
                ):
                    print(f"Scored {completed}/{len(prompts)} messages.")

    score_matrix = np.vstack(collected_scores)
    spam_probability = spam_probabilities_from_label_scores(
        score_matrix,
        label_names=labels,
    )
    output = pd.DataFrame(
        {
            f"{label}_score": score_matrix[:, index]
            for index, label in enumerate(labels)
        }
    )
    output["spam_probability"] = spam_probability
    output["prediction"] = (spam_probability >= 0.5).astype(np.int8)
    output["label_score_margin"] = (
        score_matrix[:, labels.index("spam")]
        - score_matrix[:, labels.index("ham")]
    )
    return output


def generate_label_responses(
    model: object,
    tokenizer: object,
    texts: Sequence[str],
    *,
    prompt_template: str,
    max_length: int = 256,
    batch_size: int = 16,
    max_new_tokens: int = 3,
    device: object | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate short deterministic responses and audit label compliance."""

    import torch

    _validate_inference_options(
        max_length=max_length,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    prompts = format_prompts(texts, prompt_template)
    input_device = _model_device(model, device)
    model.eval()
    generated_texts: list[str] = []

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            encoded_inputs = tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_inputs = {
                name: tensor.to(input_device)
                for name, tensor in encoded_inputs.items()
            }
            with _autocast_context(input_device):
                generated_ids = model.generate(
                    **encoded_inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                )
            generated_texts.extend(
                tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )
            if verbose:
                completed = min(start + batch_size, len(prompts))
                if (
                    completed == len(prompts)
                    or start == 0
                    or completed % (20 * batch_size) == 0
                ):
                    print(f"Generated {completed}/{len(prompts)} responses.")

    generated_label_ids = parse_generated_labels(generated_texts)
    return pd.DataFrame(
        {
            "generated_text": generated_texts,
            "generated_label": [
                LABEL_NAMES[label] if label >= 0 else None
                for label in generated_label_ids
            ],
            "generated_label_id": generated_label_ids,
            "valid_generation": generated_label_ids >= 0,
        }
    )
