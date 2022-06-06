r"""
`Recurrent Neural Networks (RNN) <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_
===========================================================================================

* Since RNN have vanishing gradient problems, we'll be using two better variants LSTM and GRU.
* CDR documents are long and RNN's ability to remember information decreases with increase in sequence length which in CDR's case is really long.
* To address length, we leverage :class:`~vayu.models.classification.cnn` module to extract features from each chunk of documents.
* Chunk here is a continuous block of text that has multiple sentences in it, see :class:`~vayu.datasets.classification.CDRJsonLDatasetMixin` for more details

Papers
------

* A Novel Neural Network-Based Method for Medical Text Classification: https://www.mdpi.com/1999-5903/11/12/255
* Long Length Document Classification by Local Convolutional Feature Aggregation: https://www.mdpi.com/1999-4893/11/8/109

Improvements
------------
* TODO: concat [conv, pool pre-trained embedding] before passing to RNN
"""

import logging
import os
from glob import glob
from typing import List

import gensim
import numpy as np
import torch
import torch.nn as nn

from vayu.datasets.classification import ChunkedClassificationDataset, FlatClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.layers import RNNFeatureExtractor, DeepFCClassifier

logger = logging.getLogger(__name__)


class CNNRNNNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""Extract features from chunk of document -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param str rnn_type: type of rnn either ``lstm`` or ``gru``, default: lstm
    :param int rnn_size: dimension of rnn layers
    :param bool rnn_bidirectional: concatenate forward and backward rnn states
    :param int rnn_num_layers: number of rnn layers to stack
    :param float rnn_dropout: recurrent dropout i.e. dropout between timesteps
    :param bool is_enable_rnn_bias: add bias to the rnn
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams, embedding_size: int,
                 pretrained_vector_path: str,
                 is_pretrained_vector_fine_tune: bool, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int],
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        self.total_kernels = sum(cnn_kernel_numbers) * 2  # max + avg cnn pool
        self.rnn_type = rnn_type

        if pretrained_vector_path:
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path)
            logger.warning("Note that index of words in pretrained_vector should be same in current model")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                          # If we want to fine-tune then not freeze layer
                                                          freeze=not is_pretrained_vector_fine_tune,
                                                          padding_idx=self.tokenizer.pad_token_id)
        else:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size,
                                          padding_idx=self.tokenizer.pad_token_id)

        self.convs = nn.ModuleList([nn.Conv2d(1, number, (size, embedding_size),
                                              padding=(size - 1, 0)) for (size, number) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers)])

        self.rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                       input_size=self.total_kernels,
                                       hidden_size=rnn_size, num_layers=rnn_num_layers,
                                       bidirectional=rnn_bidirectional,
                                       bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.classifier = DeepFCClassifier(input_size=self.rnn.total_output_size,
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def forward(self, batch):
        data, labels, lengths = batch
        batch_size = data.shape[0]

        # To collect CNN features
        pooled_outputs = torch.zeros((batch_size, data.shape[1], self.total_kernels),
                                     requires_grad=False, device=data.device)
        for i in range(batch_size):
            x = self.embedding(data[i, :lengths[i]])
            x = x.unsqueeze(1)  # Insert channel dimension
            x = [torch.nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x] + \
                [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
            pooled_outputs[i, :lengths[i]] = torch.cat(x, dim=1)

        rnn_features = self.rnn(pooled_outputs, lengths)
        logits = self.classifier(rnn_features)

        return logits, labels

    def resolve_params(self):
        if self.is_pretrained_vector_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'embedding' not in x[0],
                                                      self.pretrained_model.named_parameters()))},
                {'params': self.embedding.parameters(),
                 'lr': self.pretrained_vector_fine_tune_learning_rate}
            ]
        else:
            params = self.parameters()

        return params


class CNNRNNFlatNet(FlatClassificationDataset, CDRLightningMixin):
    r"""Extract features from chunk of document -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param str rnn_type: type of rnn either ``lstm`` or ``gru``, default: lstm
    :param int rnn_size: dimension of rnn layers
    :param bool rnn_bidirectional: concatenate forward and backward rnn states
    :param int rnn_num_layers: number of rnn layers to stack
    :param float rnn_dropout: recurrent dropout i.e. dropout between timesteps
    :param bool is_enable_rnn_bias: add bias to the rnn
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams, embedding_size: int,
                 pretrained_vector_path: str,
                 is_pretrained_vector_fine_tune: bool, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int],
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        FlatClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        self.total_kernels = sum(cnn_kernel_numbers) * 2  # max + avg cnn pool
        self.rnn_type = rnn_type

        if pretrained_vector_path:
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path)
            logger.warning("Note that index of words in pretrained_vector should be same in current model")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                          # If we want to fine-tune then not freeze layer
                                                          freeze=not is_pretrained_vector_fine_tune,
                                                          padding_idx=self.tokenizer.pad_token_id)
        else:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size,
                                          padding_idx=self.tokenizer.pad_token_id)

        self.convs = nn.ModuleList([nn.Conv1d(1, number, (size, embedding_size),
                                              padding=(size - 1, 0)) for (size, number) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers)])

        self.rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                       input_size=self.total_kernels,
                                       hidden_size=rnn_size, num_layers=rnn_num_layers,
                                       bidirectional=rnn_bidirectional,
                                       bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.classifier = DeepFCClassifier(input_size=self.rnn.total_output_size + self.total_kernels,
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def forward(self, batch):
        data, labels, lengths = batch

        embedding = self.embedding(data)
        cnn_features = embedding.unsqueeze(1)  # Insert channel dimension
        cnn_features = [torch.nn.functional.relu(conv(cnn_features)).squeeze(3) for conv in self.convs]
        cnn_features = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_features] + \
                       [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in cnn_features]

        rnn_features = self.rnn(embedding, lengths)

        logits = self.classifier(torch.cat([cnn_features, rnn_features], dim=1))

        return logits, labels

    def resolve_params(self):
        if self.is_pretrained_vector_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'embedding' not in x[0],
                                                      self.pretrained_model.named_parameters()))},
                {'params': self.embedding.parameters(),
                 'lr': self.pretrained_vector_fine_tune_learning_rate}
            ]
        else:
            params = self.parameters()

        return params


class PretrainedCNNRNNNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""Extract features from chunk of document -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param str pretrained_model_path: pretrained model path
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams, pretrained_model_path,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool,
                 is_pretrained_model_fine_tune: bool,
                 pretrained_model_fine_tune_learning_rate: float, **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_model_fine_tune = is_pretrained_model_fine_tune
        self.pretrained_model_fine_tune_learning_rate = pretrained_model_fine_tune_learning_rate

        self.pretrained_model = CNNRNNNet.from_pretrained(pretrained_model_path)

        if not self.is_pretrained_model_fine_tune:
            self.pretrained_model.eval()

        # Overwrite last linear layer of pretrained model
        self.pretrained_model.classifier = DeepFCClassifier(input_size=self.pretrained_model.rnn.total_output_size,
                                                            # Convert from listConfig
                                                            middle_layers=list(fc_middle_layers),
                                                            output_size=self.num_classes,
                                                            is_enable_bias=is_enable_fc_layer_bias,
                                                            dropout=fc_dropout,
                                                            activation=fc_activation)

    def forward(self, batch):
        return self.pretrained_model(batch)

    def resolve_params(self):
        if self.is_pretrained_model_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'classifier' not in x[0],
                                                      self.pretrained_model.named_parameters())),
                 'lr': self.pretrained_model_fine_tune_learning_rate},
                {'params': self.pretrained_model.classifier.parameters()}
            ]
        else:
            params = self.parameters()

        return params


class CNN1dDilationRNNNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""Extract features from chunk of document -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param List[int] cnn_kernel_dilations: number of cnn kernel dilations, default: 1, 1, 1
    :param str rnn_type: type of rnn either ``lstm`` or ``gru``, default: lstm
    :param int rnn_size: dimension of rnn layers
    :param bool rnn_bidirectional: concatenate forward and backward rnn states
    :param int rnn_num_layers: number of rnn layers to stack
    :param float rnn_dropout: recurrent dropout i.e. dropout between timesteps
    :param bool is_enable_rnn_bias: add bias to the rnn
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams, embedding_size: int,
                 pretrained_vector_path: str,
                 is_pretrained_vector_fine_tune: bool, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int], cnn_kernel_dilations: List[int],
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        self.total_kernels = sum(cnn_kernel_numbers) * 2  # max + avg cnn pool
        self.rnn_type = rnn_type

        embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size,
                                 padding_idx=self.tokenizer.pad_token_id)
        self.embeddings = nn.ModuleList([embedding])

        if pretrained_vector_path:
            path = glob(os.path.join(pretrained_vector_path, "*.npy"))[0]
            logger.warning("Note that index of words in pretrained_vector should be same as tokenizer")
            static_embed = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(path)),
                                                        # If we want to fine-tune then not freeze layer
                                                        freeze=not is_pretrained_vector_fine_tune,
                                                        padding_idx=self.tokenizer.pad_token_id)

            logger.info("Loaded pretrained embeddings of dimension %s from %s",
                        static_embed.embedding_dim, path)

            # Override dimensions from config
            embedding_size += static_embed.embedding_dim

            self.embeddings = nn.ModuleList([embedding, static_embed])

        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, number, kernel_size=size,
                                              padding=size, dilation=dilation) for (size, number, dilation) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers, cnn_kernel_dilations)]
                                   )

        self.rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                       input_size=self.total_kernels,
                                       hidden_size=rnn_size, num_layers=rnn_num_layers,
                                       bidirectional=rnn_bidirectional,
                                       bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.classifier = DeepFCClassifier(input_size=self.rnn.total_output_size + kwargs.pop('cat_encoding_size'),
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def resolve_params(self):
        """set layer specific optimization parameters"""
        if self.is_pretrained_vector_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'embedding' not in x[0],
                                                      self.named_parameters()))},
                {'params': self.embedding.parameters(),
                 'lr': self.pretrained_vector_fine_tune_learning_rate}
            ]
        else:
            params = self.parameters()

        return params

    def forward(self, batch, is_return_features=False):
        """forward pass data"""
        ids, data, labels, lengths, cat_features = batch
        batch_size = data.shape[0]

        # To collect CNN features
        pooled_outputs = torch.zeros((batch_size, data.shape[1], self.total_kernels),
                                     requires_grad=False, device=data.device)
        for i in range(batch_size):
            x = torch.cat([embedding(data[i, :lengths[i]]) for embedding in self.embeddings], dim=2)
            x = [torch.nn.functional.relu(conv(x.permute(0, 2, 1))) for conv in self.convs]
            x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x] + \
                [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
            pooled_outputs[i, :lengths[i]] = torch.cat(x, dim=1)

        rnn_features = self.rnn(pooled_outputs, lengths)
        features = torch.cat([rnn_features, cat_features], dim=1)
        logits = self.classifier(features)

        if is_return_features:
            return ids, logits, labels, features
        else:
            return logits, labels


class CNN1dRNNNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""Extract features from chunk of document -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param str rnn_type: type of rnn either ``lstm`` or ``gru``, default: lstm
    :param int rnn_size: dimension of rnn layers
    :param bool rnn_bidirectional: concatenate forward and backward rnn states
    :param int rnn_num_layers: number of rnn layers to stack
    :param float rnn_dropout: recurrent dropout i.e. dropout between timesteps
    :param bool is_enable_rnn_bias: add bias to the rnn
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams, embedding_size: int,
                 pretrained_vector_path: str,
                 is_pretrained_vector_fine_tune: bool, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int],
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        self.total_kernels = sum(cnn_kernel_numbers) * 2  # max + avg cnn pool
        self.rnn_type = rnn_type

        if pretrained_vector_path:
            path = glob(os.path.join(pretrained_vector_path, "*.npy"))[0]
            logger.warning("Note that index of words in pretrained_vector should be same as tokenizer")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(path)),
                                                          # If we want to fine-tune then not freeze layer
                                                          freeze=not is_pretrained_vector_fine_tune,
                                                          padding_idx=self.tokenizer.pad_token_id)

            # Override dimensions from config
            logger.info("Loaded pretrained embeddings of dimension %s from %s", embedding_size, path)
            embedding_size = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size,
                                          padding_idx=self.tokenizer.pad_token_id)

        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, number, kernel_size=size,
                                              padding=size) for (size, number) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers)])

        self.rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                       input_size=self.total_kernels,
                                       hidden_size=rnn_size, num_layers=rnn_num_layers,
                                       bidirectional=rnn_bidirectional,
                                       bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.classifier = DeepFCClassifier(input_size=self.rnn.total_output_size,
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def resolve_params(self):
        """set layer specific optimization parameters"""
        if self.is_pretrained_vector_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'embedding' not in x[0],
                                                      self.named_parameters()))},
                {'params': self.embedding.parameters(),
                 'lr': self.pretrained_vector_fine_tune_learning_rate}
            ]
        else:
            params = self.parameters()

        return params

    def forward(self, batch):
        """forward pass data"""
        data, labels, lengths = batch
        batch_size = data.shape[0]

        # To collect CNN features
        pooled_outputs = torch.zeros((batch_size, data.shape[1], self.total_kernels),
                                     requires_grad=False, device=data.device)
        for i in range(batch_size):
            x = self.embedding(data[i, :lengths[i]])
            x = [torch.nn.functional.relu(conv(x.permute(0, 2, 1))) for conv in self.convs]
            x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x] + \
                [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
            pooled_outputs[i, :lengths[i]] = torch.cat(x, dim=1)

        rnn_features = self.rnn(pooled_outputs, lengths)
        logits = self.classifier(rnn_features)

        return logits, labels
