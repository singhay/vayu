r"""
Classification datasets
=======================

Set of classification dataset classes that extend :class:`~vayu.datasets.base_dataset.BaseDatasetMixin` that can be used for binary classification models

CDR dataset:
    1. JsonL dataset composed of tokens (List[List[str]) and some categorical variables
    2. Avg size of one cpt dataset ~100k records comprising of multiple contact methods

Processing data transformation to tensor:
    1. Each sentence is encoded using pre-trained tokenizer into sequence of ids e.g. 512 tokens
    2. Each document contains multiple sentences e.g. 2500 sentences x 512 tokens
    3. For any given dataset e.g. 100k documents x 2500 sentences x 512 tokens
    4. __getitem__() return a tensor of (2500 sentences x 512 tokens), (1 label) which is used
        by DataLoader to generate batches
"""

import json
import linecache
import logging
import os
import pickle
from functools import partial
from glob import glob
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from vayu.constants import (LABEL, DOC_LEN,
                            CONTACT_METHOD, TOKENS_COLNAME, NEGATIVE_LABEL, POSITIVE_LABEL,
                            CPT_CODE, STATUS, AUTH_STATUS, AUTH_DATE, PATIENT_DOB, Split, EPISODE_ID)
from vayu.datasets.base_dataset import BaseDatasetMixin
from vayu.datasets.train_valid_split import (assign_label, APPROVAL_STATUS, FI_TRUTH_CONFIG)
from vayu.tokenizer.cdr_tokenizer import CDRTokenizer
from vayu.tokenizer.tokenization_mixins import (FlatDatasetTokenizationMixin,
                                                ChunkedDatasetTokenizationMixin,
                                                ChunkedFirstDatasetTokenizationMixin)
from vayu.utils import count_file_lines

logger = logging.getLogger(__name__)


class LazyCDRJsonLDatasetMixin(Dataset):
    """
    :param str data: path to jsonl file
    :param Union[CDRTokenizer, PreTrainedTokenizer] tokenizer: tokenizer to be used

    TODO: Loading multiple JSON files
    """

    def __init__(self, data: str, tokenizer: Union[CDRTokenizer, PreTrainedTokenizer]):
        self.data = data
        self.tokenizer = tokenizer
        self.cpt_codes = set()
        self.contact_methods = set()

    def __len__(self) -> int:
        """Denotes the total number of samples"""
        return count_file_lines(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, pd.DataFrame, int, float, list]:
        """Generates one sample of data"""
        # linecache starts counting from one, not zero, +1 the given index
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx += 1

        js = json.loads(linecache.getline(self.data, idx))
        if CPT_CODE in js:
            self.cpt_codes.add(js[CPT_CODE])
        if CONTACT_METHOD in js:
            self.contact_methods.add(js[CONTACT_METHOD])

        # Load data and get label
        # tokenization to be implemented by :class:`vayu.tokenizer.tokenization_mixins`
        unique_id = self.data.loc[idx, EPISODE_ID]
        data, length = self.tokenize(js[TOKENS_COLNAME])
        label = self.get_label(js[STATUS], js[AUTH_STATUS], js[CONTACT_METHOD])
        # return data, length, label
        return unique_id, data, length, label, []

    def get_label(self, status, auth_status, contact_method):
        """Evaluates label

        :param str status: case status
        :param str auth_status: either A or D
        :param str contact_method: either FAX or other
        :return: binary ground truth
        :rtype: float
        """
        # if contact_method == 'FAX':
        #     label = POSITIVE_LABEL if status in FI_TRUTH_CONFIG else NEGATIVE_LABEL
        # else:
        #     label = POSITIVE_LABEL if auth_status == APPROVAL_STATUS else NEGATIVE_LABEL

        return POSITIVE_LABEL if auth_status == APPROVAL_STATUS else NEGATIVE_LABEL

    def get_cpt_codes(self) -> str:
        return ' '.join(self.cpt_codes)

    def get_contact_methods(self) -> str:
        return ' '.join(self.contact_methods)


class CDRJsonLDatasetMixin(Dataset):
    r"""This is memory intensive base class, use :class:`~vayu.datasets.classification.LazyCDRJsonLDatasetMixin`
     instead if dataset is big by setting `data.is_lazy: true` in configuration

    :param data: path to jsonl dataset with at least status, authStatus, tokens and contactMethod
    :param tokenizer: Transformers tokenizer object
    :param Split data_split_type: Type of split (train, tune, test)
    :param dict cat_encoding: path to pickled map of key: category -> value: encoding
    :param float pct_of_data: percentage of data to use
    :param float label_smooth: amount of label smooth to perform
    """

    def __init__(self, data: Union[str, pd.DataFrame, Tuple],
                 tokenizer: Union[PreTrainedTokenizer, CDRTokenizer],
                 data_split_type: Split, cat_encoding: dict,
                 pct_of_data: float = 1.0, label_smooth: float = 0.0):
        if isinstance(data, str):
            self.data, self.unlabelled_data = self.read_data(data, data_split_type,
                                                             pct_of_data, label_smooth)
        elif isinstance(data, Tuple):
            self.data, self.unlabelled_data = data
        else:
            self.data, self.unlabelled_data = self.split_data_chronologically(data, pct_of_data)

        self.tokenizer = tokenizer

        self.cat_encoder = {}
        if cat_encoding:
            for col_name, path in cat_encoding.items():
                self.cat_encoder[col_name] = pickle.load(open(glob(os.path.join(path, "*.pkl"))[0], 'rb'))

    def __len__(self):
        """Denotes the total number of samples"""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Generates one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load data and get label
        # tokenization to be implemented by :class:`vayu.tokenizer.tokenization_mixins`
        data, length = self.tokenize(self.data.loc[idx, TOKENS_COLNAME])
        unique_id = self.data.loc[idx, EPISODE_ID]
        return unique_id, data, length, self.data.loc[idx, LABEL], self.encode_categorical_features(idx)

    def encode_categorical_features(self, idx):
        """Lookup embedding for given index in data"""
        features = []
        for col_name in self.cat_encoder.keys():
            key = self.data.loc[idx, col_name]
            encoder_dict = self.cat_encoder[col_name]
            if key in encoder_dict:
                features.extend(encoder_dict[key])
            elif key.split('.')[0] in encoder_dict:  # T81.32 -> T81 ICD10CM
                features.extend(encoder_dict[key.split('.')[0]])
            else:
                logger.debug(f"{col_name}: {key} not found")
                features.extend(encoder_dict['<UNK>'])

        return features

    def get_label_weight(self) -> List[float]:
        """Returns label weight"""
        class_weight = compute_class_weight('balanced', np.unique(self.data[LABEL]), self.data[LABEL])
        return class_weight.tolist()

    def get_label_frequency(self) -> dict:
        """Returns label frequency"""
        return self.data['label'].value_counts().to_dict()

    def get_pos_label_weight(self) -> float:
        """Returns positive (approval) label weight"""
        return self.get_label_weight()[1]

    def get_cpt_codes(self) -> str:
        """Returns list of all cpt codes present in dataset"""
        return ' '.join(map(str, self.data[CPT_CODE].unique().tolist())) \
            if CPT_CODE in self.data.columns \
            else f'{CPT_CODE} column not present in dataframe'

    def get_contact_methods(self) -> str:
        """Returns list of all contact methods present in dataset"""
        return ' '.join(self.data[CONTACT_METHOD].unique().tolist()) \
            if CONTACT_METHOD in self.data.columns \
            else f'{CONTACT_METHOD} column not present in dataframe'

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the dataframe"""
        return self.data

    @staticmethod
    def read_data(data_path: str, dataset_split_type: Split = Split.TRAIN,
                  pct_of_data: float = 1.0, label_smooth: float = 0.0) -> Tuple:
        """Reads jsonl data from given path into pandas dataframe and removes empty documents

        :param str data_path: Path to jsonl dataset
        :param Split dataset_split_type: Type of dataset from train, tune or test
        :param float pct_of_data: percentage of data to use
        :param float label_smooth: amount of label smooth to perform
        :return: Pandas DataFrame sorted in ascending order of document length
        """
        if os.path.isfile(data_path):
            data_paths = [data_path]
        else:
            data_paths = glob(os.path.join(data_path, "*.json"))
            # if dataset_split_type == Split.TRAIN:
            #     data_paths = data_paths[:5]

        for i, symlink_data_path in enumerate(data_paths):
            if os.path.islink(symlink_data_path):  # Assuming symlink_data_path exists
                data_paths[i] = os.readlink(symlink_data_path)

        logger.info("Reading %s json file(s) %s", len(data_paths), data_paths)

        df = pd.concat((pd.read_json(file_path, lines=True,
                                     dtype=dict(cpt=int,
                                                contactMethod='string',
                                                icd9Code='string',
                                                authStatus='string',
                                                status='string',
                                                episodeId='string',
                                                patientSex='string',
                                                physicianSpecialty='string',
                                                pediatric='string',
                                                patientDOB='datetime64[ns]',
                                                authDate='datetime64[ns, UTC]',
                                                tokens=list(list('string'))),
                                     convert_dates=[PATIENT_DOB, AUTH_DATE])
                        for file_path in data_paths),
                       ignore_index=True)
        logger.info("Loaded dataset from %s with %s records", data_path, df.shape[0])

        # Sometimes pandas doesn't load timezone columns as datetime so convert manually.
        df[AUTH_DATE] = pd.to_datetime(df[AUTH_DATE], errors='raise', utc=True)

        df[DOC_LEN] = df.tokens.apply(lambda x: len(x))
        # Prune empty documents
        empty_doc_count = df[df[DOC_LEN] == 0].shape[0]
        if empty_doc_count > 0:
            df.drop(df.loc[df[DOC_LEN] == 0].index, inplace=True)
            df.reset_index(drop=True, inplace=True)
        logger.info("Dropped %s empty records, final dataset size %s", empty_doc_count, df.shape[0])

        """Following block is for experimental evaluation of ground truth labeling
        1. quick approval ground truth labelling for train and validation set of FAX contact_method 
        if dataset_split_type == Split.TEST:
            df[LABEL] = df.apply(assign_label, axis=1)
        else:
            df[LABEL] = df.apply(assign_legacy_label, axis=1)
            
        2. no quick approval
            df[LABEL] = df.apply(assign_label, axis=1)

        3. assign soft labels
        if dataset_split_type == Split.TEST or dataset_split_type == Split.TUNE:
            df[LABEL] = df.apply(assign_label, axis=1)
        else:
            df[LABEL] = df.apply(assign_label, axis=1)
        """
        if dataset_split_type == Split.TRAIN:
            df[LABEL] = df.apply(assign_label, axis=1, label_smooth=label_smooth)
        else:
            df[LABEL] = df.apply(assign_label, axis=1,
                                 label_smooth=0.0)  # Validation and test sets are not smoothed

        if dataset_split_type == Split.TRAIN:
            return CDRJsonLDatasetMixin.split_data_chronologically(df, pct_of_data)
        else:
            # Sort descending to allocate largest GPU memory chunk in beginning of run.
            return df.sort_values(by=DOC_LEN, ascending=False), None

    @staticmethod
    def split_data_chronologically(df, pct_of_data):
        """Split data chronologically"""
        if pct_of_data == 1:
            return df, pd.DataFrame()

        # df['month'] = (df[AUTH_DATE].dt.year - min(df[AUTH_DATE].dt.year)) * 12 + df[AUTH_DATE].dt.month
        # df['month'] -= min(df['month'])
        df['week'] = (df[AUTH_DATE].dt.year - min(df[AUTH_DATE].dt.year)) * 52 + df[AUTH_DATE].dt.week
        df['week'] -= min(df['week'])
        # logger.info("Training data is comprised of %s unique months", df['month'].unique().shape[0])
        logger.info("Training data is comprised of %s unique weeks", df['week'].unique().shape[0])

        if pct_of_data > 1:
            aprx_weeks = pct_of_data
        else:
            aprx_weeks = round(pct_of_data * df['week'].unique().shape[0])

        # aprx_months = round(pct_of_data * df['month'].unique().shape[0])
        # aprx_months = 1
        # a, b = df[df['month'] <= aprx_months], df[df['month'] > aprx_months]
        # aprx_weeks = 2
        # logger.info("%s%% percent will comprise of %s months of data", pct_of_data * 100, aprx_weeks)
        a, b = df[df['week'] <= aprx_weeks], df[df['week'] > aprx_weeks]
        logger.info("%s records picked from the first %s weeks of data and %s left out of %s",
                    a.shape[0], aprx_weeks, b.shape[0], df.shape[0])
        return a.reset_index(drop=True), b.reset_index(drop=True)


class CDRJsonLFlatDataset(CDRJsonLDatasetMixin, FlatDatasetTokenizationMixin):
    def __init__(self, max_length: int, **kwargs):
        FlatDatasetTokenizationMixin.__init__(self, max_length)
        CDRJsonLDatasetMixin.__init__(self, **kwargs)


class CDRJsonLChunkedDataset(CDRJsonLDatasetMixin, ChunkedDatasetTokenizationMixin):
    def __init__(self, max_chunks: int, max_chunk_size: int, **kwargs):
        ChunkedDatasetTokenizationMixin.__init__(self, max_chunks, max_chunk_size)
        CDRJsonLDatasetMixin.__init__(self, **kwargs)


class CDRJsonLChunkedFirstDataset(CDRJsonLDatasetMixin, ChunkedFirstDatasetTokenizationMixin):
    def __init__(self, max_chunks: int, max_chunk_size: int, **kwargs):
        ChunkedFirstDatasetTokenizationMixin.__init__(self, max_chunks, max_chunk_size)
        CDRJsonLDatasetMixin.__init__(self, **kwargs)


class LazyCDRJsonLFlatDataset(LazyCDRJsonLDatasetMixin, FlatDatasetTokenizationMixin):
    def __init__(self, max_length: int, **kwargs):
        FlatDatasetTokenizationMixin.__init__(self, max_length)
        LazyCDRJsonLDatasetMixin.__init__(self, **kwargs)


class LazyCDRJsonLChunkedDataset(LazyCDRJsonLDatasetMixin, ChunkedDatasetTokenizationMixin):
    def __init__(self, max_chunks: int, max_chunk_size: int, **kwargs):
        ChunkedFirstDatasetTokenizationMixin.__init__(self, max_chunks, max_chunk_size)
        LazyCDRJsonLDatasetMixin.__init__(self, **kwargs)


class LazyCDRJsonLChunkedFirstDatasetTokenization(LazyCDRJsonLDatasetMixin, ChunkedFirstDatasetTokenizationMixin):
    """This class tokenizes in a different way as follows:

    Because we have many short sentences
    This is to save documents from getting very long post subword tokenization
    Caveat here is this chunks bluntly i.e. it would break sentences ..
    ..which might lead to context fragmentation

    TODO: Generate overlapping chunks to handle context fragmentation
    """

    def __init__(self, max_chunks: int, max_chunk_size: int, **kwargs):
        ChunkedFirstDatasetTokenizationMixin.__init__(self, max_chunks, max_chunk_size)
        LazyCDRJsonLDatasetMixin.__init__(self, **kwargs)


class FlatClassificationDataset(BaseDatasetMixin):
    """Class that initializes flat documents (no sentences) for train, test and validation.

    :param int max_chunks: maximum number of chunks possible in a document, larger chunks are truncated
    :param int max_chunk_size: maximum size of a chunk
    :param kwargs: keyword arguments for :class:`~vayu.datasets.base_dataset.BaseDatasetMixin`
    """

    def __init__(self, max_chunks: int, max_chunk_size: int,
                 num_classes: int, is_lazy: bool, **kwargs):
        """Maximum length of a document is defined by: max_chunks * max_chunk_size"""
        self.max_length = max_chunks * max_chunk_size
        self.num_classes = num_classes
        if is_lazy:
            self.data_cls = LazyCDRJsonLFlatDataset
        else:
            self.data_cls = CDRJsonLFlatDataset
        BaseDatasetMixin.__init__(self, **kwargs)

    def _init_data_class(self):
        """Initializes the data class that is going to be used"""
        self.dataset_cls = partial(self.data_cls, tokenizer=self.tokenizer,
                                   max_length=self.max_length)

    def _collate_fn(self, data):
        r"""used by dataloader to club together variable length flat documents

        :param List[Tuple] data: is a list of tuple with (input, length, label)
                                where 'input' is a tensor of arbitrary shape
                                and label/length are scalars
        """
        _, lengths, labels = zip(*data)
        max_len = max(lengths)
        features = torch.zeros((len(data), max_len))
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)

        for i, (document, length, _) in enumerate(data):
            diff = max_len - length
            features[i] = torch.nn.functional.pad(document,
                                                  [0, diff],
                                                  mode='constant',
                                                  value=self.tokenizer.pad_token_id)

        return features.long(), labels.float(), lengths.long()


class ChunkedClassificationDataset(BaseDatasetMixin):
    """Class that initializes chunked documents.

    :param int max_chunks: maximum number of chunks possible in a document, larger chunks are truncated
    :param int max_chunk_size: maximum size of a chunk
    :param kwargs: keyword arguments for :class:`~vayu.datasets.base_dataset.BaseDatasetMixin`
    """

    def __init__(self, max_chunks: int, max_chunk_size: int,
                 num_classes: int, is_lazy: bool, **kwargs):
        self.max_chunks = max_chunks
        self.max_chunk_size = max_chunk_size
        self.num_classes = num_classes
        if is_lazy:
            self.data_cls = LazyCDRJsonLChunkedDataset
        else:
            self.data_cls = CDRJsonLChunkedDataset
        BaseDatasetMixin.__init__(self, **kwargs)

    def _init_data_class(self):
        """Initializes the data class that is going to be used"""
        self.dataset_cls = partial(self.data_cls, tokenizer=self.tokenizer,
                                   max_chunks=self.max_chunks, max_chunk_size=self.max_chunk_size)

    def _collate_fn(self, data):
        """
           data: is a list of tuple with (input, length, label)
                 where 'input' is a tensor of arbitrary shape
                 and label/length are scalars

        TODO: further improve this by not truncating based on the length of the longest sequence in the batch,
         but based on the 95% percentile of lengths within the sequence
        """
        ids, _, lengths, labels, cat_features = zip(*data)
        max_len = max(lengths)
        features = torch.zeros((len(data), max_len, self.max_chunk_size))
        labels = torch.tensor(labels)
        lengths = torch.tensor(lengths)

        for i, (_, document, length, _, _) in enumerate(data):
            diff = max_len - length
            features[i] = torch.nn.functional.pad(document,
                                                  (0, 0, 0, diff),
                                                  mode='constant',
                                                  value=self.tokenizer.pad_token_id)

        return list(ids), features.long(), labels.float(), lengths.long(), torch.tensor(cat_features)
