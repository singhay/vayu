import logging
import os
from math import sqrt
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (confusion_matrix, auc, precision_recall_curve,
                             roc_auc_score, average_precision_score)

from vayu.constants import (POSITIVE_LABEL, NEGATIVE_LABEL, Split, METRICS_FILE, PROBABILITY_FILE,
                            PROBABILITY, TARGET_COLNAME, PREDICTION)
from vayu.models.classification import OptimalThresholdFinder
from vayu.utils import stringify

logger = logging.getLogger(__name__)


class ScoreCalculations:
    """Calculates metrics for evaluating CDR models

    Attributes:
       predictions: List[float] of predicted probabilities
       targets: List[float] of ground truth
       threshold: float between 0 and 1 of threshold to be used for classification
    """

    def __init__(self, predictions, targets, otf, split_type: Split):
        r"""
        Args:
            targets (List[float]): ground truth
            predictions (List[float]): predicted probabilities
            otf (OptimalThresholdFinder): OTF object comprising of threshold
            split_type (Split): The type of dataset split

        Raises:
            ValueError if the volume is 0
        """
        if len(targets) == 0 or len(predictions) == 0:
            raise ValueError("Evaluation volume should be greater than 0")

        self.split_type = split_type
        self.y_true, self.y_pred = targets, predictions
        self.threshold = otf.target_threshold
        self.otf = otf

        predictions = np.array(predictions)
        predictions[predictions >= self.threshold] = POSITIVE_LABEL
        predictions[predictions < self.threshold] = NEGATIVE_LABEL


        self._confusion_matrix = confusion_matrix(targets, predictions)
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix.ravel()
        self.volume = self.tp + self.fp + self.tn + self.fn
        logger.debug("FP:{}, TP:{}, FN:{}, TN:{}, TOTAL: {}".format(self.fp, self.tp, self.fn,
                                                                    self.tn, self.volume))
        if self.volume < 200:
            logger.warning("Total evaluation records are low: %s", self.volume)

    @property
    def confusion_matrix(self):
        """
        Returns:
            The confusion matrix
        """
        return self._confusion_matrix

    @property
    def fpr(self):
        """
        Returns:
            The false positive rate as a float between 0 and 1
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.

    @property
    def precision(self):
        """
        Returns:
            The precision as a float between 0 and 1
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.

    @property
    def recall(self):
        """
        Returns:
            The recall as a float between 0 and 1
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.

    @property
    def automation_rate(self):
        """
        Returns:
            The approval rate as a float between 0 and 1
        """
        return (self.tp + self.fp) / self.volume if self.volume > 0 else 0.

    @property
    def idr(self):
        """
        Returns:
            The IDR(denial rate)
        """
        return (self.tn + self.fp) / self.volume if self.volume > 0 else 0.

    @property
    def tpfp_ratio(self):
        """
        Returns:
            The ratio of true positive to false positives
        """
        return self.tp / self.fp if self.fp > 0 else 0.

    @property
    def mcc(self):
        r"""Matthews correlation coefficient is a binary classification metric for when there is a high class imbalance.
        https: // en.wikipedia.org/wiki/Matthews_correlation_coefficient

          - 1 = perfect prediction
          - 0 = random prediction
          - -1 = total disagreement

        Returns:
            The Matthews correlation coefficient
        """
        mcc_denominator = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        return ((self.tp * self.tn) - (self.fp * self.fn)) / sqrt(mcc_denominator) if mcc_denominator > 0 else 0.

    @property
    def pvr(self):
        """Performance vs. random calculation

        .. math: :
            PVR = approval_rate/false_positive_rate

        Returns:
            The performance vs. random
        """
        return self.automation_rate / self.fpr if self.fpr > 0. else 0.

    @property
    def yamm(self):
        r"""'Yet another modeling metric'

        .. math: :
            YAMM = \sqrt{TPFP_RATIO * PVR}

        Returns:
           The calculated YAMM score
        """
        return sqrt(self.tpfp_ratio * self.pvr)

    @property
    def f1(self):
        """F-measure

        Returns:
           Standard F1 measure
        """
        sum_pr = self.precision + self.recall
        return (2. * self.precision * self.recall) / sum_pr if sum_pr > 0. else 0.

    @property
    def auc_roc(self):
        """Calculates AUC RoC

        Returns:
            Area under curve for Receiver operator characteristic
        """
        return roc_auc_score(self.y_true, self.y_pred)

    @property
    def auc_pr(self):
        """Calculates AUC PR which is better than AUC RoC for imbalanced dataset
        Returns:
            Area under curve for Precision Recall
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        return auc(recall, precision)

    @property
    def average_precision(self):
        """Calculates AUC PR which is better than AUC RoC for imbalanced dataset
        Returns:
            Area under curve for Precision Recall
        """
        return average_precision_score(self.y_true, self.y_pred)

    def get_all_metrics(self) -> dict:
        """Returns all metrics in a dictionary"""
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn,
                'precision': self.precision, 'recall': self.recall, 'f1': self.f1,
                'auc_roc': self.auc_roc, 'auc_pr': self.auc_pr, 'mcc': self.mcc,
                'pvr': self.pvr, 'yamm': self.yamm,
                'tpfp_ratio': self.tpfp_ratio, 'volume': self.volume,
                'automation_rate': self.automation_rate, 'fpr': self.fpr,
                'average_precision': self.average_precision, 'idr': self.idr,
                'threshold': self.threshold}

    def export_metrics_to_yaml(self, output_dir: str, file_name: str = ''):
        """Writes scores to yaml at given output directory

        Args:
            output_dir (str): Path to the directory to save yaml
            file_name (str): Additional file name to add to output file
        """
        results = self.get_all_metrics()
        logger.info("Results from selected threshold: %s", self.threshold)
        for key, value in results.items():
            logger.info("\t%s_%s: %s", self.split_type.value, key, results[key])
            results[key] = stringify(results[key])

        path = os.path.join(output_dir, file_name + METRICS_FILE.format(self.split_type.value))
        with open(path, 'w') as f:
            # This is to preserve order of elements in the results dict
            yaml.add_representer(dict, lambda obj, data: yaml.representer.SafeRepresenter
                                 .represent_dict(obj, data.items()))
            yaml.dump(results, f)
            logger.info("%s results exported to %s", self.split_type.value, path)

    def export_predicted_probas_to_csv(self, output_dir: str, ids: List[str],
                                       features: List[str], file_name: str = '', is_export_features: bool = False):
        """Writes predicted probabilities to csv with header

        Args:
            output_dir (str): Path to the directory to save
            ids (List[str]): List of episodeIds
            features (List[str]): List of encodings
            file_name (str): Additional file name to add to output file
            is_export_features (bool): whether to export ID and features
        """
        dataset = pd.DataFrame()
        dataset[PROBABILITY] = np.array(self.y_pred)
        dataset[TARGET_COLNAME] = np.array(self.y_true)
        if is_export_features:
            if len(features) == len(self.y_true):
                dataset['features'] = np.array(features)
            if len(ids) == len(self.y_true):
                dataset['id'] = np.array(list(map(str, ids)))
        dataset.loc[dataset[PROBABILITY] >= self.threshold, PREDICTION] = POSITIVE_LABEL
        dataset.loc[dataset[PROBABILITY] < self.threshold, PREDICTION] = NEGATIVE_LABEL

        path = os.path.join(output_dir, file_name + PROBABILITY_FILE.format(self.split_type.value))

        dataset.to_csv(path, index=False, header=True, date_format='%Y/%m/%d %H:%m:%S')
        logger.info("%s scores exported to %s", self.split_type.value, path)

    def log_all_metrics_to_aml(self, run):
        """
        Upload metrics, confusion matrix and accuracy table to AML

        Args:
            run: azure ml run context used to upload metrics
        """
        self.log_metrics_to_aml(run)
        self.log_confusion_matrix_to_aml(run)
        self.log_accuracy_table_to_aml(run)

    def log_metrics_to_aml(self, run):
        """
        Log specified metrics to AML

        Args:
            run: azure ml run context used to upload metrics
        """
        for metric_name, metric_value in self.get_all_metrics().items():
            run.log(f'{self.split_type.value} {metric_name}', metric_value)

    def log_confusion_matrix_to_aml(self, run):
        """
        Log confusion matrix to AML

        Args:
            run: azure ml run context used to upload metrics
        """
        cm = {
            "schema_type": "confusion_matrix",
            "schema_version": "v1",
            "data": {
                "class_labels": ['0', '1'],
                "matrix": self.confusion_matrix.tolist()
            }
        }
        run.log_confusion_matrix(f'{self.split_type.value}_confusion_matrix', cm)

    def log_accuracy_table_to_aml(self, run):
        """
        Log accuracy table to AML

        Args:
            run: azure ml run context used to upload metrics
        """
        probability_tables = []
        for threshold in self.otf.thresholds:
            otf = OptimalThresholdFinder(self.otf.target_metric_type,
                                         self.otf.target_metric_value,
                                         target_threshold=threshold)
            score = ScoreCalculations(self.y_pred, self.y_true, otf, self.split_type)
            probability_tables.append([stringify(score.tp), stringify(score.fp),
                                       stringify(score.tn), stringify(score.fn)])

        percentile_tables = []
        for threshold in self.otf.percentile_thresholds:
            otf = OptimalThresholdFinder(self.otf.target_metric_type,
                                         self.otf.target_metric_value,
                                         target_threshold=threshold)
            score = ScoreCalculations(self.y_pred, self.y_true, otf, self.split_type)
            percentile_tables.append([stringify(score.tp), stringify(score.fp),
                                      stringify(score.tn), stringify(score.fn)])

        accuracy_table = {
            "schema_type": "accuracy_table",
            "schema_version": "v1",
            "data": {
                "probability_tables": [probability_tables],
                "percentile_tables": [percentile_tables],
                "probability_thresholds": self.otf.thresholds,
                "percentile_thresholds": self.otf.percentile_thresholds,
                "class_labels": ['1'],
            }
        }

        run.log_accuracy_table(f'{self.split_type.value}_accuracy_table', accuracy_table)
