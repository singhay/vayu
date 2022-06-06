r"""
`Bag of Embedding <https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag>`_
==========================================================================================================

This model serves as a good baseline for neural network architectures

Papers
------
* Bag-of-embeddings for text classification: https://dl.acm.org/doi/10.5555/3060832.3061016

Improvements
------------
* Currently, model only has single linear layer on -> make network deeper
* Add activation layers when making network deeper
* Support modes other than mean, -> mode: ``sum`` takes in weights per sample into account
"""
from argparse import Namespace

import torch.nn as nn

from vayu.datasets.classification import FlatClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin


class BagOfEmbedding(FlatClassificationDataset, CDRLightningMixin):
    r"""Baseline bag of embedding model that averages embedding of all words and passes to linear layer

    :param Namespace hparams: for pytorch lightning configuration
    :param int embedding_size: dimensions of word embedding vectors
    :param float dropout: to regularize the network
    :param kwargs: keyword arguments used for initialization of dataset classes
    """
    def __init__(self, hparams: Namespace, embedding_size: int, dropout: float, **kwargs):
        FlatClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.embedding = nn.EmbeddingBag(self.tokenizer.vocab_size, embedding_size, sparse=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_size, self.num_classes)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, batch):
        data, labels, lengths = batch
        embedded = self.embedding(data)
        return self.classifier(embedded), labels
