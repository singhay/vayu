"""Test for class vayu.classification.optimal_threshold_finder"""
from vayu.models.classification.optimal_threshold_finder import OptimalThresholdFinder


def test_compute_thresholds():
    otf = OptimalThresholdFinder(target_metric_type='fprs', target_metric_value=1/3)
    predictions = [0.3, .7, .9, .1, .8]
    targets = [0, 1, 1, 0, 0]
    otf.calculate_thresholds(predictions, targets)
    threshold = 0.7

    assert otf.target_threshold == threshold

