from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # The pure helpers do not require the optional runtime.
    torch = None

from src.generative import (
    confidence_coverage_table,
    format_prompts,
    generate_label_responses,
    generation_audit_summary,
    normalize_generated_label,
    parse_generated_labels,
    score_candidate_labels,
    spam_probabilities_from_label_scores,
)


class FakeTokenizer:
    eos_token_id = 9

    def __call__(
        self,
        texts=None,
        *,
        text_target=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text_target is not None:
            targets = [text_target] if isinstance(text_target, str) else text_target
            token_ids = [
                [1, 4, self.eos_token_id]
                if target == "ham"
                else [2, self.eos_token_id]
                for target in targets
            ]
        else:
            prompts = [texts] if isinstance(texts, str) else texts
            token_ids = []
            for prompt in prompts:
                marker = 2 if "invalid example" in prompt else int("spam example" in prompt)
                token_ids.append([marker, self.eos_token_id])

        if return_tensors == "pt":
            ids = torch.tensor(token_ids, dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }
        return {"input_ids": token_ids[0] if isinstance(text_target, str) else token_ids}

    def batch_decode(self, generated_ids, *, skip_special_tokens=True):
        labels = {1: "ham", 2: "spam", 3: "This is spam"}
        return [labels[int(row[0])] for row in generated_ids]


class FakeSequenceModel:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.encoder_calls = 0
        self.decoder_calls = 0

    def parameters(self):
        yield self._parameter

    def eval(self):
        return self

    def get_encoder(self):
        def encode(*, input_ids, attention_mask):
            self.encoder_calls += 1
            return SimpleNamespace(marker=input_ids[:, 0])

        return encode

    def __call__(self, *, encoder_outputs, attention_mask, labels, use_cache):
        self.decoder_calls += 1
        batch_size, target_length = labels.shape
        logits = torch.zeros(batch_size, target_length, 10)
        for row, marker in enumerate(encoder_outputs.marker.tolist()):
            candidate_is_spam = int(labels[row, 0]) == 2
            candidate_is_preferred = candidate_is_spam == (marker == 1)
            target_logit = 5.0 if candidate_is_preferred else -5.0
            for position, token_id in enumerate(labels[row].tolist()):
                logits[row, position, token_id] = target_logit
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        do_sample,
        num_beams,
        max_new_tokens,
    ):
        generated = []
        for marker in input_ids[:, 0].tolist():
            generated.append([3 if marker == 2 else marker + 1])
        return torch.tensor(generated, dtype=torch.long)


def test_format_prompts_is_deterministic() -> None:
    template = "Classify as ham or spam.\nMessage:\n{text}"
    assert format_prompts(["hello", "claim prize"], template) == [
        "Classify as ham or spam.\nMessage:\nhello",
        "Classify as ham or spam.\nMessage:\nclaim prize",
    ]


def test_format_prompts_requires_placeholder_and_rows() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        format_prompts(["hello"], "Classify this message.")
    with pytest.raises(ValueError, match="At least one"):
        format_prompts([], "{text}")
    with pytest.raises(ValueError, match="missing"):
        format_prompts([None], "{text}")


def test_label_score_softmax_is_stable_and_maps_spam() -> None:
    probabilities = spam_probabilities_from_label_scores(
        [[1_000.0, 1_001.0], [-1_001.0, -1_000.0]]
    )
    assert np.isfinite(probabilities).all()
    assert probabilities[0] == pytest.approx(probabilities[1])
    assert probabilities[0] > 0.5

    reversed_probabilities = spam_probabilities_from_label_scores(
        [[1_001.0, 1_000.0]],
        label_names=("spam", "ham"),
    )
    assert reversed_probabilities[0] > 0.5


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [1.0, 2.0],
        [[1.0, 2.0, 3.0]],
        [[np.nan, 0.0]],
    ],
)
def test_label_score_softmax_rejects_invalid_scores(scores) -> None:
    with pytest.raises(ValueError):
        spam_probabilities_from_label_scores(scores)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("spam", "spam"),
        (" HAM. ", "ham"),
        ('"spam"', "spam"),
        ("not spam", None),
        ("spam or ham", None),
        ("This is spam", None),
        ("", None),
        (None, None),
    ],
)
def test_generated_label_parser_is_strict(text, expected) -> None:
    assert normalize_generated_label(text) == expected


def test_parse_generated_labels_retains_invalid_rows() -> None:
    parsed = parse_generated_labels(["ham", "spam", "not spam", ""])
    assert parsed.tolist() == [0, 1, -1, -1]


def test_confidence_coverage_keeps_all_rows_at_half() -> None:
    table = confidence_coverage_table(
        [0, 1, 0, 1],
        [0.1, 0.8, 0.6, 0.55],
        minimum_confidences=[0.5, 0.8, 0.95],
    )
    assert table["retained_rows"].tolist() == [4, 2, 0]
    assert table["coverage"].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert table.loc[0, "accuracy"] == pytest.approx(0.75)
    assert np.isnan(table.loc[2, "accuracy"])


def test_generation_audit_counts_invalid_outputs_as_wrong() -> None:
    summary = generation_audit_summary(
        labels=[0, 1, 1, 0],
        likelihood_predictions=[0, 1, 0, 0],
        generated_label_ids=[0, 1, -1, 1],
    )
    assert summary["audit_rows"] == 4
    assert summary["valid_generation_rows"] == 3
    assert summary["generation_compliance_rate"] == pytest.approx(0.75)
    assert summary["generation_exact_accuracy_all_rows"] == pytest.approx(0.5)
    assert summary["generation_accuracy_valid_only"] == pytest.approx(2 / 3)
    assert summary[
        "likelihood_generation_agreement_valid_only"
    ] == pytest.approx(2 / 3)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_candidate_scoring_preserves_rows_and_reuses_encoder() -> None:
    model = FakeSequenceModel()
    scores = score_candidate_labels(
        model,
        FakeTokenizer(),
        ["ordinary example", "spam example"],
        prompt_template="Message: {text}",
        max_length=16,
        batch_size=2,
        device="cpu",
    )

    assert scores["prediction"].tolist() == [0, 1]
    preferred_score = 5.0 - np.log(np.exp(5.0) + 9.0)
    rejected_score = -5.0 - np.log(np.exp(-5.0) + 9.0)
    expected_spam_probability = np.exp(rejected_score) / (
        np.exp(preferred_score) + np.exp(rejected_score)
    )
    assert scores["ham_score"].iloc[0] == pytest.approx(preferred_score)
    assert scores["spam_score"].iloc[0] == pytest.approx(rejected_score)
    assert scores["spam_probability"].iloc[0] == pytest.approx(
        expected_spam_probability
    )
    assert scores["spam_probability"].iloc[1] == pytest.approx(
        1 - expected_spam_probability
    )
    assert scores["label_score_margin"].iloc[0] < 0
    assert scores["label_score_margin"].iloc[1] > 0
    assert len(FakeTokenizer()(text_target="ham")["input_ids"]) == 3
    assert len(FakeTokenizer()(text_target="spam")["input_ids"]) == 2
    assert model.encoder_calls == 1
    assert model.decoder_calls == 2


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_generation_keeps_invalid_responses() -> None:
    generated = generate_label_responses(
        FakeSequenceModel(),
        FakeTokenizer(),
        ["ordinary example", "spam example", "invalid example"],
        prompt_template="Message: {text}",
        max_length=16,
        batch_size=3,
        max_new_tokens=4,
        device="cpu",
    )

    assert generated["generated_label_id"].tolist() == [0, 1, -1]
    assert generated["valid_generation"].tolist() == [True, True, False]
    assert generated.loc[2, "generated_text"] == "This is spam"
