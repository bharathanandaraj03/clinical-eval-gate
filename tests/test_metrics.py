"""Tests for per-condition metrics.

`test_accuracy_is_a_lie_at_low_prevalence` is the README's central argument expressed as an
executable assertion. If it ever stops holding, the README is wrong.
"""

import math

import pytest

from clinical_eval_gate import ConfusionMatrix


def test_known_matrix_values() -> None:
    cm = ConfusionMatrix(true_positive=80, false_positive=30, true_negative=870, false_negative=20)

    assert cm.total == 1000
    assert cm.positives == 100
    assert cm.negatives == 900
    assert cm.sensitivity == pytest.approx(0.80)
    assert cm.specificity == pytest.approx(870 / 900)
    assert cm.ppv == pytest.approx(80 / 110)
    assert cm.npv == pytest.approx(870 / 890)
    assert cm.prevalence == pytest.approx(0.10)
    assert cm.accuracy == pytest.approx(0.95)


def test_accuracy_is_a_lie_at_low_prevalence() -> None:
    """A model that never predicts the condition scores 98% accuracy and is worthless."""
    y_true = [True] * 20 + [False] * 980
    y_pred = [False] * 1000

    cm = ConfusionMatrix.from_labels(y_true, y_pred)

    assert cm.accuracy == pytest.approx(0.98)
    assert cm.sensitivity == pytest.approx(0.0)
    assert cm.balanced_accuracy == pytest.approx(0.50)


def test_from_labels_matches_manual_counts() -> None:
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]

    cm = ConfusionMatrix.from_labels(y_true, y_pred)

    assert (cm.true_positive, cm.false_negative) == (2, 1)
    assert (cm.false_positive, cm.true_negative) == (1, 2)


def test_from_labels_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        ConfusionMatrix.from_labels([True, False], [True])


def test_perfect_classifier() -> None:
    cm = ConfusionMatrix(true_positive=50, false_positive=0, true_negative=50, false_negative=0)

    assert cm.sensitivity == pytest.approx(1.0)
    assert cm.specificity == pytest.approx(1.0)
    assert cm.balanced_accuracy == pytest.approx(1.0)


def test_undefined_metrics_are_nan_not_zero() -> None:
    """No negatives in the slice means specificity is undefined, not failed."""
    cm = ConfusionMatrix(true_positive=10, false_positive=0, true_negative=0, false_negative=0)

    assert math.isnan(cm.specificity)
    assert math.isnan(cm.npv)
    assert cm.sensitivity == pytest.approx(1.0)


def test_empty_matrix_is_all_nan() -> None:
    cm = ConfusionMatrix(true_positive=0, false_positive=0, true_negative=0, false_negative=0)

    assert cm.total == 0
    assert all(math.isnan(cm.as_dict()[k]) for k in ("sensitivity", "specificity", "accuracy"))


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ConfusionMatrix(true_positive=-1, false_positive=0, true_negative=0, false_negative=0)


def test_as_dict_is_serialisable_shape() -> None:
    cm = ConfusionMatrix(true_positive=1, false_positive=2, true_negative=3, false_negative=4)
    d = cm.as_dict()

    assert d["total"] == 10
    assert set(d) >= {"sensitivity", "specificity", "ppv", "npv", "prevalence"}
