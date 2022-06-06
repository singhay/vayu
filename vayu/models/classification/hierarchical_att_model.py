r"""
Hierarchical Attention Network (HAN)
====================================

This model is hierarchical in nature in the way that it builds a representation for each chunk in document
and then uses another network on top to learn document representation using representation of all the chunks in document from previous layer.

Papers
------
* Original paper: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

Improvements
------------
* Add support to switch from GRU to LSTM and increase number of layers
* Use :class:`~vayu.models.classification.cnn` to extract features from document and concatenate
"""

from argparse import Namespace
from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from vayu.datasets.classification import ChunkedClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.classification.doc_att_model import DocAttNet
from vayu.models.classification.sent_att_model import SentAttNet


class HierAttNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""

    :param Union[Namespace,DictConfig] hparams: for pytorch lightning configuration
    :param int embedding_size:
    :param str pretrained_vector_path:
    :param bool is_pretrained_vector_freeze:
    :param float dropout:
    :param int sent_rnn_num_layers:
    :param bool sent_rnn_bidirectional:
    :param int sent_rnn_size:
    :param int doc_rnn_num_layers:
    :param bool doc_rnn_bidirectional:
    :param int doc_rnn_size:
    :param kwargs: keyword arguments used for initialization of dataset classes

    todo: Add support to switch from GRU to LSTM and increase number of layers
    """
    def __init__(self, hparams: Union[Namespace, DictConfig],
                 embedding_size: int,
                 pretrained_vector_path: str, is_pretrained_vector_freeze: bool,
                 dropout: float,
                 sent_rnn_num_layers: int,
                 sent_rnn_bidirectional: bool,
                 sent_rnn_size: int,
                 doc_rnn_num_layers: int,
                 doc_rnn_bidirectional: bool,
                 doc_rnn_size: int,
                 **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.sent_rnn_size = sent_rnn_size
        self.num_sent_rnn_directions = 2 if sent_rnn_bidirectional else 1
        self.num_doc_rnn_directions = 2 if doc_rnn_bidirectional else 1
        self.max_sent_length = self.max_chunks
        self.max_word_length = self.max_chunk_size
        self.sent_att_net = SentAttNet(pretrained_vector_path, is_pretrained_vector_freeze,
                                       self.tokenizer.vocab_size,
                                       embedding_size,
                                       sent_rnn_size,
                                       sent_rnn_num_layers,
                                       sent_rnn_bidirectional,
                                       self.tokenizer.pad_token_id
                                       )
        self.doc_att_net = DocAttNet(self.num_sent_rnn_directions, doc_rnn_size, sent_rnn_size,
                                     doc_rnn_num_layers, doc_rnn_bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.num_doc_rnn_directions * doc_rnn_size, self.num_classes)

    def forward(self, batch):
        """Forward pass, also takes care of variable length sequences"""
        data, labels, lengths = batch

        output = torch.ones((data.shape[0], data.shape[1],
                             self.num_sent_rnn_directions*self.sent_rnn_size),
                            dtype=torch.float, device=data.device)
        for i in range(data.shape[0]):
            output[i, :lengths[i]], final_hidden_state = self.sent_att_net(data[i, :lengths[i]])
        output = self.doc_att_net(output, lengths)
        logits = self.classifier(self.dropout(output))

        return logits, labels


class HierCNNAttNet(ChunkedClassificationDataset, CDRLightningMixin):
    """todo: Implement this model"""
    pass
