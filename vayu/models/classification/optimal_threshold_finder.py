r"""
Optimal Threshold Finder
========================
Compute optimal binary classification threshold for given target fpr and thresholds.
"""

import json
import logging
import os
from bisect import bisect_left, bisect_right
from typing import List, Union

import numpy as np
from pandas._typing import FrameOrSeries
from sklearn import metrics

from vayu.constants import (TARGET_THRESHOLD, STAY_UNDER_TARGET_METRIC, THRESHOLDS, THRESHOLDS_FILE,
                            OPTIMAL_THRESHOLD, PERCENTILE_THRESHOLDS, TARGET_METRIC, CalibrationMetricType,
                            TARGET_METRIC_TYPE, TARGET_METRIC_VALUES)

logger = logging.getLogger(__name__)


class OptimalThresholdFinder:
    r"""

    :param str target_metric_type: The type of target metric, either of `fprs` or `automation_rates`
    :param float target_metric_value: The metric to target to find the binary classification threshold for
    :param bool stay_under_target_metric: whether cutoff should be equal to target metric value or under
    :param List[float] target_metric_values: list of target metric values
    :param float target_threshold: property of class that is the computed target_threshold
    :param List[float] thresholds: list of cutoff for corresponding target metric values
    :param float optimal_threshold: model's best performing optimal threshold independent of target metric
    :param List[float] percentile_thresholds: percentage of predictions under a percentile cutoff

    Example::
        >>> import numpy as np
        >>> import pandas as pd
        >>> y_score = np.array([0.81, 0.27, 0.56, 0.98])
        >>> y_true = pd.Series([1, 0, 1, 1])
        >>> output_dir = "./outputs"
        >>> from vayu.models.classification.optimal_threshold_finder import OptimalThresholdFinder
        >>> otf = OptimalThresholdFinder(target_metric_type='fprs', target_metric_value=0.15)
        >>> otf.calculate_thresholds(y_score, y_true)
        >>> otf.target_threshold
        >>> otf.export_thresholds(output_dir=output_dir)
    """

    def __init__(self, target_metric_type: str,
                 target_metric_value: float, stay_under_target_metric=True,
                 target_metric_values=None,
                 target_threshold=None, thresholds=None,
                 optimal_threshold=None, percentile_thresholds=None):
        """Constructor method that also is used by classmethod from_precomputed_thresholds()"""
        self._thresholds = thresholds or []
        self._target_threshold = target_threshold
        self._percentile_thresholds = percentile_thresholds
        self.optimal_threshold = optimal_threshold
        self.target_metric_value = target_metric_value
        self.target_metric_type = self._validate_if_target_metric_type_supported(target_metric_type)
        self.target_metric_values = target_metric_values
        self.stay_under_target_metric = stay_under_target_metric
        self.thresholds_dict = {}

    def _validate_if_target_metric_type_supported(self, target_metric_type):
        """Tests whether the target metric type is supported"""
        if target_metric_type == CalibrationMetricType.FPR.value:
            return target_metric_type
        elif target_metric_type == CalibrationMetricType.AR.value:
            return target_metric_type
        else:
            raise NotImplementedError(f"{target_metric_type} not supported, choose either of "
                                      f"{list(v.value for v in CalibrationMetricType)}")

    @property
    def thresholds(self):
        """Returns thresholds for target fpr on data set"""
        return list(self._thresholds)

    @property
    def percentile_thresholds(self):
        """Returns percentile threshold for target fpr on data set"""
        return self._percentile_thresholds

    @property
    def target_threshold(self):
        """Returns optimal threshold for target fpr on data set"""
        return self._target_threshold

    @classmethod
    def from_precomputed_thresholds(cls, thresholds_path: str):
        """Initialize constructor by loading saved thresholds.json"""
        path = os.path.join(thresholds_path, THRESHOLDS_FILE)
        thresholds_dict = json.load(open(path))
        otf = cls(**thresholds_dict)
        otf.thresholds_dict = thresholds_dict
        logger.info("Loaded threshold of %s from %s", otf.target_threshold, path)
        return otf

    def calculate_thresholds(self, y_score: Union[List[float], np.ndarray],
                             y_true: Union[List[float], FrameOrSeries]):
        """Calculates thresholds using predicted probabilities and ground truth labels

        Args:
            y_score (Union[List[float], np.ndarray)]: list of predicted probabilities
            y_true (Union[List[float], FrameOrSeries]): list of ground truth labels
        """
        # thresholds sorted high to low
        self.fprs, self.tprs, self._thresholds = metrics.roc_curve(y_true, y_score)

        # Youden's J-Score
        optimal_idx = np.argmax(self.tprs - self.fprs)
        self.optimal_threshold = self._thresholds[optimal_idx]

        self._calculate_target_metric_values(y_score)

        self._percentile_thresholds = []
        sorted_predictions = sorted(y_score)
        for percentile in range(0, 101):
            threshold = np.percentile(sorted_predictions, percentile)
            self._percentile_thresholds.append(threshold)

        self.thresholds_dict = {THRESHOLDS: list(self._thresholds),
                                TARGET_METRIC_VALUES: self.target_metric_values,
                                TARGET_METRIC_TYPE: self.target_metric_type,
                                TARGET_METRIC: self.target_metric_value,
                                STAY_UNDER_TARGET_METRIC: self.stay_under_target_metric,
                                PERCENTILE_THRESHOLDS: self._percentile_thresholds}

        self._find_target_threshold(self.thresholds_dict[THRESHOLDS],
                                    self.thresholds_dict[TARGET_METRIC_VALUES])

        self.thresholds_dict[TARGET_THRESHOLD] = self._target_threshold
        self.thresholds_dict[OPTIMAL_THRESHOLD] = self.optimal_threshold
        logger.info("Optimal threshold at target %s: %s found to be %s" % (self.target_metric_type,
                                                                           self.target_metric_value,
                                                                           self._target_threshold))

    def _calculate_target_metric_values(self, y_score: Union[List[float], np.ndarray]):
        """Generates target metric values e.g. fprs for CDR, automation_rates for M0

        Args:
            y_score (Union[List[float], np.ndarray)]: list of predicted probabilities
        """
        if self.target_metric_type == CalibrationMetricType.FPR.value:
            self.target_metric_values = list(self.fprs)
        elif self.target_metric_type == CalibrationMetricType.AR.value:
            self.automation_rates = list()
            sorted_probas = sorted(y_score)
            num_probas = len(sorted_probas)
            for threshold in self.thresholds:
                num_approved = num_probas - bisect_left(sorted_probas, threshold)
                self.automation_rates.append(num_approved / num_probas)
            self.target_metric_values = self.automation_rates

    def _find_target_threshold(self, thresholds: List[float], metric_values: List[float]):
        """Find the target threshold for given metric

        Args:
            thresholds (List[float]): list of thresholds sorted descending order
            metric_values (List[float]): list of corresponding metric values sorted descending order
        """
        assert len(thresholds) != 0, 'No thresholds found'
        assert len(metric_values) != 0, 'No metric_values found'
        assert len(metric_values) == len(thresholds), 'Unequal number of elements'

        i = bisect_right(metric_values, self.target_metric_value) - 1

        if self.stay_under_target_metric:
            self._target_threshold = thresholds[i]
        else:
            under_diff = self.target_metric_value - metric_values[i]
            over_diff = metric_values[i + 1] - self.target_metric_value
            if under_diff < over_diff:
                self._target_threshold = thresholds[i]
            else:
                self._target_threshold = thresholds[i + 1]

    def export_thresholds(self, output_dir: str):
        """Export thresholds calculated to a json file

        Args:
            output_dir (str): path to the output directory where thresholds json will be persisted.
        """
        with open(os.path.join(output_dir, THRESHOLDS_FILE), 'w') as f:
            json.dump(self.thresholds_dict, f)
