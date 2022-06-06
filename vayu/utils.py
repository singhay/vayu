"""
Utils
=====
A set of utility functions used throughout the framework.
"""
import logging
import os
import subprocess
from datetime import datetime
from glob import glob
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer, TransfoXLTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from vayu.constants import ColumnNotFoundException, Split
from vayu.tokenizer.cdr_tokenizer import CDRTokenizer

logger = logging.getLogger(__name__)


def stringify(obj):
    """Helps serialize numpy objects"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()
    else:
        return obj


def count_file_lines(file_path: str) -> int:
    """Counts the number of lines in a file using wc utility.

    :param str file_path: path to file
    :return: no of lines
    :rtype: int
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.split()
    return int(num[0])


def get_all_checkpoints(path: str) -> List[Tuple[float, str]]:
    """Extract path of checkpoints sorted in ascending order of validation loss

    :param str path: path to directory containing checkpoints with extension *.ckpt
    :return: list of tuples of [(validation_loss, checkpoint path)] sorted
    :rtype: List[Tuple[float, str]]
    """
    top_k = {}
    models = glob(os.path.join(path, "*.ckpt"))  # This does not change
    print(models)
    if not models:
        raise FileNotFoundError("No models found in %s" % path)
    for model in models:
        # This splitting logic operates on -> path/checkpoints-epoch=002-val_loss=0.693-loss=0.708.ckpt
        # This splitting logic operates on -> checkpoints-epoch=002-val_loss=0.693-loss=0.708.ckpt
        # This splitting logic operates on -> val_loss=0.693
        # This splitting logic operates on -> 0.693
        _, value = list(filter(lambda x: x.startswith('val'), model.split("/")[-1].split("-")))[0].split("=")
        top_k[float(value)] = model

    return sorted(top_k.items())  # Sort by validation loss in descending order and return top model


def get_top_checkpoint_path(path: str) -> Tuple[float, str]:
    """Extract path of checkpoint with lowest validation loss

    :param str path: path to directory containing checkpoints with extension *.ckpt
    :return: tuple of (validation loss, absolute path to checkpoint) of lowest validation loss model
    :rtype: Tuple[float, str]

    # TODO: What happens when two models have same validation loss ?
    * Choose the one with the highest validation + training loss avg
    * Choose the one with the highest epoch
    """
    checkpoints_sorted = get_all_checkpoints(path)
    logger.info("Picking top model from %s", checkpoints_sorted)
    return checkpoints_sorted[0]  # Return checkpoint having lowest loss


def resolve_tokenizer(tokenizer_class: Union[PreTrainedTokenizer, str], tokenizer_path: str,
                      lowercase: bool, **kwargs) -> Union[PreTrainedTokenizer, CDRTokenizer]:
    if tokenizer_class == RobertaTokenizer or tokenizer_class == 'bpe':
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path,
                                                     do_lower_case=lowercase,
                                                     add_prefix_space=True)
    elif tokenizer_class == BertTokenizer or tokenizer_class == 'wordpiece':
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path,
                                                  do_lower_case=lowercase)
    elif tokenizer_class == TransfoXLTokenizer:
        tokenizer = TransfoXLTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = CDRTokenizer(tokenizer_path, lowercase=lowercase, **kwargs)

    return tokenizer


def validate_is_id_column_present(column_name: str, dataset: pd.DataFrame, data_type: Split):
    if column_name not in dataset.columns:
        raise ColumnNotFoundException(column_name, data_type.value)

# def export_to_excel(excel_sheets):
#     metrics = {k: [] for k in excel_sheets}
#     metrics[excel_sheets.MAIN].append(model_info + stats)
#
#     wb = load_workbook(local_wb_path)
#     for sheet_name, data in metrics.items():
#         for row in data:
#             try:
#                 wb[sheet_name.value].append(row)
#             except ValueError:
#                 logger.debug("ValueError encountered, converting row %s to string before writing", row)
#                 wb[sheet_name.value].append([str(i) for i in row])
#
#     # Save it into a new file, not the template itself
#     output_file_name = GenerateEvaluationMetricsTask.EXCEL_EVAL_OUT_FILE_TEMPLATE \
#         .format(self.total_models, self.TIMESTAMP)
#     tmp_metric_file_path = os.path.join(self.temp_dir.path, output_file_name)
#     wb.save(filename=tmp_metric_file_path)
#
#     output_file_path = os.path.join(self.output_path, output_file_name)
#     logger.info("Writing excel file to hdfs {}".format(output_file_path))
#     if is_hdfs_path(self.output_path):
#         target = HdfsTarget(self.output_path)
#         target.fs.put(tmp_metric_file_path, self.output_path)
#         os.remove(tmp_metric_file_path)
#     else:
#         fs = LocalTarget(tmp_metric_file_path)
#         fs.move(output_file_path)
#
#     logger.info("Excel write successful, removing tmp data dump from {}".format(
#         tmp_data_path))
