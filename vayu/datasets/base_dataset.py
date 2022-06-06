r"""
Base dataset
============
Abstract base class representing attributes like train, valid, test datasets as well as processing.
"""

import logging
from abc import abstractmethod, ABC
from functools import partial
from typing import Union

import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from transformers import PreTrainedTokenizer

from vayu.constants import EPISODE_ID, Split, AUTH_DATE, LABEL, LabelLeakageException
from vayu.utils import resolve_tokenizer, validate_is_id_column_present

logger = logging.getLogger(__name__)


class BaseDatasetMixin(ABC):
    """

    :param Union[PreTrainedTokenizer, str] tokenizer_class: What type of tokenizer to use
    :param str tokenizer_path: path to tokenizer
    :param str train_path: path to training set
    :param str val_path: path to validation set
    :param str test_path: path to test set
    :param int batch_size: batch size (not including gradient accumulation)
    :param int num_workers: number of parallel processes to start
    :param kwargs: Arguments required for CDRTokenizer
    """

    def __init__(self,
                 tokenizer_class: Union[PreTrainedTokenizer, str],
                 tokenizer_path: str,
                 train_path: str, valid_path: str, test_path: str,
                 batch_size: int, num_workers: int, pct_of_data: float,
                 label_smooth: float, cat_encoding: dict = None,
                 id_column: str = EPISODE_ID, time_column: str = AUTH_DATE, **kwargs):

        self.id_column = id_column
        self.time_column = time_column
        self.tokenizer = resolve_tokenizer(tokenizer_class, tokenizer_path, **kwargs)

        self.train_path, self.valid_path, self.test_path = train_path, valid_path, test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader_cls = partial(DataLoader, collate_fn=self._collate_fn,
                                      batch_size=batch_size,
                                      num_workers=num_workers, pin_memory=True)

        # Let dataset_cls be initialized by child class
        self.dataset_cls = None
        self.train_dataset, self.valid_dataset = None, None
        self.label_smooth = label_smooth
        self.pct_of_data = pct_of_data
        self.cat_encoding = cat_encoding
        self._init_data_class()

    @abstractmethod
    def _init_data_class(self):
        """Initializes the data class that is going to be used"""

    def prepare_data(self):
        self.train_dataset = self.dataset_cls(data=self.train_path,
                                              data_split_type=Split.TRAIN,
                                              pct_of_data=self.pct_of_data,
                                              label_smooth=self.label_smooth,
                                              cat_encoding=self.cat_encoding)
        self.valid_dataset = self.dataset_cls(data=self.valid_path, data_split_type=Split.VALID,
                                              label_smooth=self.label_smooth, cat_encoding=self.cat_encoding)

        self.check_label_leakage()

    def train_dataloader(self):
        """Return pytorch train data loader initialized with _collate_fn(), batch_size and num_workers"""
        return self.dataloader_cls(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Return pytorch validation data loader initialized with _collate_fn(), batch_size and num_workers"""
        return self.dataloader_cls(dataset=self.valid_dataset,
                                   sampler=SequentialSampler(self.valid_dataset))

    @abstractmethod
    def _collate_fn(self, data):
        """Logic to club together batches (mostly used to sort and pad variable length sequences)"""

    def _set_num_labels(self):
        """Find unique labels in train and validation set
        TODO: this method is intended for future use"""

        self.labels = self.train_dataset.data[LABEL].unique().tolist()
        self.labels.extend(self.valid_dataset.data[LABEL].unique().tolist())

        self.labels = set(self.labels)
        self.num_labels = len(self.labels)

    def check_label_leakage(self):
        """Checks label overlap using id column, exits program if overlap found"""

        # ----------------
        # Check id overlap
        # ----------------
        validate_is_id_column_present(self.id_column, self.train_dataset.data, Split.TRAIN)
        validate_is_id_column_present(self.id_column, self.valid_dataset.data, Split.VALID)
        logger.info("Found %s column in all dataset splits, checking id overlap", self.id_column)
        train_ids = self.train_dataset.data[self.id_column].values
        valid_ids = self.valid_dataset.data[self.id_column].values

        common_ids = np.intersect1d(train_ids, valid_ids)
        if len(common_ids) != 0:
            # Stop training if ratio is >= 5%
            if len(common_ids) / len(valid_ids) >= 0.05:
                # raise LabelLeakageException("Overlap of %s ids found between train and validation set: %s",
                #                             len(common_ids), common_ids)
                # TODO: dataloader barfs because len of train and valid dataset is changed
                logger.info("Dropping overlapped %s ids from train set", len(common_ids))
                # self.valid_dataset.data.append(self.train_dataset.data[
                #                                    self.train_dataset.data[EPISODE_ID]
                #                                .isin(common_ids)])
                data = self.train_dataset.data.drop(self.train_dataset.data[
                                                        self.train_dataset.data[EPISODE_ID]
                                                    .isin(common_ids)].index).reset_index(drop=True)
                self.train_dataset = self.dataset_cls(data=data,
                                                      data_split_type=Split.TRAIN,
                                                      pct_of_data=self.pct_of_data,
                                                      label_smooth=self.label_smooth,
                                                      cat_encoding=self.cat_encoding)
                logger.info("Total records in train: %s, validation: %s",
                            len(self.train_dataset), len(self.valid_dataset))
            else:
                logger.warning("Overlap of %s ids found between train and validation set: %s",
                               len(common_ids), common_ids)

    def __repr__(self):
        return f"Total records in train: {len(self.train_dataset)}, " \
               f"validation: {len(self.valid_dataset)}"
