"""Per-condition classification metrics.

Every metric here is defined for a single condition. Aggregating across conditions is a
presentation concern and is deliberately not offered at this layer: a suite-wide mean hides
exactly the per-condition failure the gate exists to catch.

Undefined metrics return NaN rather than 0.0. A specificity of 0.0 means "never correct on
negatives"; a specificity of NaN means "there were no negatives to be correct about". Collapsing
the second into the first is how an empty slice silently passes a gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ConfusionMatrix"]


def _ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or NaN when the denominator is empty."""
    if denominator == 0:
        return math.nan
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts for one condition, at one decision threshold."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def __post_init__(self) -> None:
        for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_labels(
        cls, y_true: Sequence[bool | int], y_pred: Sequence[bool | int]
    ) -> ConfusionMatrix:
        """Build a matrix from aligned ground-truth and predicted label sequences."""
        if len(y_true) != len(y_pred):
            raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions")

        tp = fp = tn = fn = 0
        for truth, pred in zip(y_true, y_pred, strict=True):
            if pred:
                if truth:
                    tp += 1
                else:
                    fp += 1
            elif truth:
                fn += 1
            else:
                tn += 1
        return cls(true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn)

    @property
    def total(self) -> int:
        """Number of cases the matrix was built from."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def positives(self) -> int:
        """Cases where the condition was actually present."""
        return self.true_positive + self.false_negative

    @property
    def negatives(self) -> int:
        """Cases where the condition was actually absent."""
        return self.true_negative + self.false_positive

    @property
    def sensitivity(self) -> float:
        """Of the cases that had the condition, the share the model flagged."""
        return _ratio(self.true_positive, self.positives)

    @property
    def specificity(self) -> float:
        """Of the cases that did not have it, the share the model left alone."""
        return _ratio(self.true_negative, self.negatives)

    @property
    def ppv(self) -> float:
        """Positive predictive value. Moves with prevalence; never compare it across cohorts."""
        return _ratio(self.true_positive, self.true_positive + self.false_positive)

    @property
    def npv(self) -> float:
        """Negative predictive value. Runs high on uncommon conditions and flatters weak models."""
        return _ratio(self.true_negative, self.true_negative + self.false_negative)

    @property
    def prevalence(self) -> float:
        """Share of evaluated cases in which the condition was present."""
        return _ratio(self.positives, self.total)

    @property
    def accuracy(self) -> float:
        """Reported for completeness. Never gate on it -- see README, section 1."""
        return _ratio(self.true_positive + self.true_negative, self.total)

    @property
    def balanced_accuracy(self) -> float:
        """Mean of sensitivity and specificity: what accuracy should have been."""
        return (self.sensitivity + self.specificity) / 2

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping for report generation and baseline serialisation."""
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "total": self.total,
            "prevalence": self.prevalence,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
        }
