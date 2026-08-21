from src.protocol import (
    CONFIRMATION_TRAINING_SEEDS,
    DATA_SPLIT_SEED,
    FINAL_TRAINING_SEEDS,
    HEADLINE_METRIC,
    POSITIVE_CLASS_LABEL,
    PRIMARY_TRANSFER,
    REFERENCE_TRAINING_SEED,
    SECONDARY_TRANSFER,
    SELECTION_METRIC,
    SELECTION_SPLIT,
)


def test_locked_seed_protocol_is_explicit_and_non_redundant():
    assert DATA_SPLIT_SEED == 42
    assert REFERENCE_TRAINING_SEED == 42
    assert FINAL_TRAINING_SEEDS == (13, 42, 73, 101, 137)
    assert len(FINAL_TRAINING_SEEDS) == len(set(FINAL_TRAINING_SEEDS))
    assert set(CONFIRMATION_TRAINING_SEEDS).issubset(FINAL_TRAINING_SEEDS)


def test_locked_transfer_and_metric_roles_are_distinct():
    assert PRIMARY_TRANSFER == ("sms", "enron")
    assert SECONDARY_TRANSFER == tuple(reversed(PRIMARY_TRANSFER))
    assert SELECTION_SPLIT == "validation"
    assert SELECTION_METRIC == "macro_f1"
    assert HEADLINE_METRIC == "f1"
    assert POSITIVE_CLASS_LABEL == 1
