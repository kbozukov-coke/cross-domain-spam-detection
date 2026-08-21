"""Locked choices for the robustness extension of the experiment."""

from typing import Final


# The data partitions stay identical across every model run. Training seeds
# control model initialization and data order; they never recreate the splits.
DATA_SPLIT_SEED: Final[int] = 42
REFERENCE_TRAINING_SEED: Final[int] = 42
CONFIRMATION_TRAINING_SEEDS: Final[tuple[int, ...]] = (13, 42, 73)
FINAL_TRAINING_SEEDS: Final[tuple[int, ...]] = (13, 42, 73, 101, 137)

# SMS -> Enron answers the project question. The reverse direction is retained
# as a secondary diagnostic rather than a second primary hypothesis.
PRIMARY_TRANSFER: Final[tuple[str, str]] = ("sms", "enron")
SECONDARY_TRANSFER: Final[tuple[str, str]] = ("enron", "sms")

# Hyperparameters are ranked on source-domain validation data. Binary F1 in
# this repository treats spam (label 1) as the positive class.
SELECTION_SPLIT: Final[str] = "validation"
SELECTION_METRIC: Final[str] = "macro_f1"
HEADLINE_METRIC: Final[str] = "f1"
POSITIVE_CLASS_LABEL: Final[int] = 1
POSITIVE_CLASS_NAME: Final[str] = "spam"
