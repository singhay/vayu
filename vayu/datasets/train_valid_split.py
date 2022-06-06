r"""
Train validation split
======================
Script used to split a single JsonL dataset into two datasets given split ratio

.. code-block:: bash

    $ python -m vayu.datasets.train_valid_split --file_path data.json --validation_split_ratio 0.2 --cut_eval --eval_months 2
    # Default split ratio is 0.1
"""

import argparse
import logging
import os
from typing import Tuple, Union

import pandas as pd
from sklearn import model_selection

from vayu.constants import AUTH_DATE

POSITIVE_LABEL = 1.
NEGATIVE_LABEL = 0.

LABEL = 'label'
DOC_LEN = 'doc_len'
DEFAULT_VALIDATION_SPLIT_RATIO = 0.1
DEFAULT_EVAL_MONTHS = 2

FI_TRUTH_CONFIG = {"NLA", "NLAY", "NLAW", "LA", "LAY", "LAW", "NA", "NAY", "NAW", "LNA", "LNAW", "LNAY", "NLFA",
                   "NLFAW",
                   "NLFAY", "LFA", "LFAY", "LFAW"}
APPROVAL_STATUS = 'A'

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def assign_legacy_label(row) -> float:
    """Assigns label based on status and authStatus using CDR truth configs

     :param row: Row object of pandas dataframe
     :return: float
     """
    if row.contactMethod == 'FAX':
        label = POSITIVE_LABEL if row.status in FI_TRUTH_CONFIG else NEGATIVE_LABEL
    else:
        label = POSITIVE_LABEL if row.authStatus == APPROVAL_STATUS else NEGATIVE_LABEL

    return label


def assign_label(row, label_smooth: float = 0.0) -> float:
    """Assigns label based on authStatus using CDR truth configs
    Label Smoothing And Logit Squeezing (Page 4 https://arxiv.org/pdf/1910.11585.pdf)

     :param row: Row object of pandas dataframe
     :param float label_smooth: amount of label smoothing to perform
     :return: float
     """
    label = POSITIVE_LABEL if row.authStatus == APPROVAL_STATUS else NEGATIVE_LABEL

    num_classes = 2  # TODO: is this really a magic constant when we always know it ?
    label = label - label_smooth * (label - (1/num_classes))
    return label


def split_dataset(file_path: str,
                  split_ratio: float = DEFAULT_VALIDATION_SPLIT_RATIO,
                  cut_eval: bool = True,
                  eval_start_datetime: int = 2) -> Tuple[pd.DataFrame,
                                                         pd.DataFrame, Union[pd.DataFrame, None]]:
    """Reads as input a jsonl file and stratified splits it into two files using given ratio

    :param str file_path: path to jsonl file
    :param float split_ratio: stratified split ratio [0, 1]
    :param bool cut_eval: whether to cut evaluation dataset as well
    :param int eval_start_datetime: inclusive start datetime of eval set e.g. '2019-12-21 23:59:59'
    :return: A tuple of (train, validation, eval) pandas dataframes
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, Union[pd.DataFrame,None]]
    """
    json_file_path, file_extension = os.path.splitext(file_path)
    if file_extension != ".json":
        raise ValueError("Only JSON files are supported")

    df = pd.read_json(file_path, lines=True)
    logger.info("Dataset of %s records read from %s", df.shape[0], file_path)

    df_eval = None
    if cut_eval:
        df, df_eval = df[df[AUTH_DATE] < eval_start_datetime], df[df[AUTH_DATE] >= eval_start_datetime]
        df.reset_index(drop=True, inplace=True)
        df_eval.reset_index(drop=True, inplace=True)

    if split_ratio == DEFAULT_VALIDATION_SPLIT_RATIO:
        logger.info("Using default validation split ratio: %s", split_ratio)
    else:
        logger.info("Using validation split ratio: %s", split_ratio)

    df[DOC_LEN] = df.tokens.apply(lambda x: len(x))
    df.drop(df.loc[df[DOC_LEN] == 0].index, inplace=True)

    df[LABEL] = df.apply(assign_label, axis=1)
    logger.info("Dataset cleaned by removing empty docs from %s records", df.shape[0])
    df_train, df_valid = model_selection.train_test_split(df, test_size=split_ratio,
                                                          random_state=42, stratify=df[LABEL])
    df_train.drop(columns=LABEL, inplace=True)
    df_valid.drop(columns=LABEL, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    train_size, valid_size = df_train.shape[0], df_valid.shape[0]

    if cut_eval:
        eval_size = df_eval.shape[0]
        logger.info("Train, valid, and eval dataset split into %s, %s and %s records correspondingly.",
                    train_size, valid_size, eval_size)
    else:
        logger.info("Train, and valid dataset split into %s, and %s records correspondingly.",
                    train_size, valid_size)

    return df_train, df_valid, df_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path",
        default=None, type=str,
        required=True,
        help="The input source json file."
    )
    parser.add_argument(
        "--validation_split_ratio",
        default=DEFAULT_VALIDATION_SPLIT_RATIO,
        type=float,
        help="Validation split ratio to be used to split dataset.",
    )
    parser.add_argument(
        "--cut_eval",
        action="store_true",
        help="Whether to cut evaluation dataset as well.",
    )
    parser.add_argument(
        "--eval_start_datetime",
        type=str,
        help="Inclusive start datetime of eval set <= e.g. '2019-12-21 23:59:59'",
    )
    args = parser.parse_args()

    df_train, df_valid, df_eval = split_dataset(args.file_path, args.validation_split_ratio,
                                                args.cut_eval, args.eval_start_datetime)
    train_size, valid_size = df_train.shape[0], df_valid.shape[0]
    json_file_path, file_extension = os.path.splitext(args.file_path)

    df_train.to_json(f"{json_file_path}_train_{train_size}records.json", orient='records', lines=True)
    df_valid.to_json(f"{json_file_path}_valid_{valid_size}records.json", orient='records', lines=True)

    if args.cut_eval:
        eval_size = df_eval.shape[0]
        df_valid.to_json(f"{json_file_path}_eval_{eval_size}records.json",
                         orient='records', lines=True)
    logger.info("Dataset split successful, please find the target files in %s", json_file_path)
