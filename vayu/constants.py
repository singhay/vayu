from enum import Enum

from transformers import (
    BertConfig, BertModel, BertTokenizerFast,
    RobertaConfig, RobertaModel, RobertaTokenizerFast,
    TransfoXLConfig, TransfoXLModel, TransfoXLTokenizerFast)

MODEL_TYPE = 'ModelType'
CALIBRATION_METRIC = 'CalibrationMetric'
THRESHOLDS = 'thresholds'
PERCENTILE_THRESHOLDS = 'percentile_thresholds'
TARGET_THRESHOLD = 'target_threshold'
OPTIMAL_THRESHOLD = 'optimal_threshold'
TARGET_METRIC = 'target_metric_value'
TARGET_METRIC_VALUES = 'target_metric_values'
TARGET_METRIC_TYPE = 'target_metric_type'
STAY_UNDER_TARGET_METRIC = 'stay_under_target_metric'
TPRS = 'tprs'
FPRS = 'fprs'
TARGET_FPR = 'target_fpr'
STAY_UNDER_TARGET_FPR = 'stay_under_target_fpr'
AUTOMATION_RATES = 'automation_rates'
LABEL = 'label'
DOC_LEN = 'num_sentences'
PREDICTION = 'prediction'
PROBABILITY = 'probability'
THRESHOLD = 'threshold'
TOKENS_COLNAME = 'tokens'
DATE_TRAINED = 'datetime'
TARGET_COLNAME = 'target'

POSITIVE_LABEL = 1.
NEGATIVE_LABEL = 0.
NUM_CLASSES = 1

NAME = 'name'
VERSION = 'version'
DESCRIPTION = 'description'
ATTRIBUTES = 'attributes'
CONFIG = 'config'
EPISODE_ID = 'episodeId'
CPT_CODE = 'cptCode'
CONTACT_METHOD = 'contactMethod'
AUTH_STATUS = 'authStatus'
AUTH_DATE = 'authDate'
STATUS = 'status'

PATIENT_DOB = 'patientDOB'

# File name constants
MODEL_INFO_FILE = 'model_info_advanced.json'
THRESHOLDS_FILE = 'thresholds.json'
METRICS_FILE = "{}_results.yaml"
PROBABILITY_FILE = "{}_predictions.csv"

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizerFast),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizerFast),
    "transfoxl": (TransfoXLConfig, TransfoXLModel, TransfoXLTokenizerFast)
}


class Split(Enum):
    """Types of data split types"""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    EMBEDDING = "embedding"


class ColumnNotFoundException(Exception):
    """Exception for when column is not found in dataset"""
    def __init__(self, column_name: str, data_type: str):
        super(ColumnNotFoundException, self).__init__(f"{column_name} column not found in {data_type} dataset.")


class LabelLeakageException(Exception):
    """Exception for when there same records found in different dataset types"""
    pass


class CalibrationMetricType(Enum):
    """Type of calibration metric"""
    FPR = 'fprs'
    AR = 'automation_rates'
