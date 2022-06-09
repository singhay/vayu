"""
Tests for vayu.score_calculations
"""
from math import sqrt
import pytest

from vayu.constants import Split
from vayu.models.classification import OptimalThresholdFinder
from vayu.score_calculations import ScoreCalculations


def test_score_calculations_exceptions():
    otf = OptimalThresholdFinder('fprs', 0.15, target_threshold=0.5)
    with pytest.raises(ValueError, match="Evaluation volume should be greater than 0"):
        _ = ScoreCalculations([], [], otf, Split.TRAIN)


def test_score_calculations():
    labels = [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.]
    predictions = [.76, 0.49, .52, .12, .89, .64, .86, .99, .41, .57]
    threshold = 0.5
    tp = 6.
    fp = 1.
    tn = 1.
    fn = 2.
    volume = tp + fp + tn + fn
    ar = (tp+fp)/volume
    fpr = fp / (fp + tn)
    pvr = ar/fpr
    tpfp_ratio = tp/fp
    mcc = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    otf = OptimalThresholdFinder('fprs', 0.15, target_threshold=threshold)
    metrics = ScoreCalculations(predictions, labels, otf, Split.TRAIN)

    assert metrics.split_type == Split.TRAIN
    assert volume == metrics.volume
    assert fpr == metrics.fpr
    assert recall == metrics.recall
    assert precision == metrics.precision
    assert (tp + fp) / volume == metrics.automation_rate
    assert (tn + fp) / volume == metrics.idr
    assert tpfp_ratio == metrics.tpfp_ratio
    assert pvr == metrics.pvr
    assert sqrt(tpfp_ratio * pvr) == metrics.yamm
    assert mcc == metrics.mcc
    assert (2 * precision * recall) / (precision + recall) == metrics.f1

