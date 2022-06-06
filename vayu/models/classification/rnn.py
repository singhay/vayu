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
from typing import List

import gensim
import torch
import torch.nn as nn

from vayu.datasets.classification import FlatClassificationDataset, ChunkedClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.layers import RNNFeatureExtractor, DeepFCClassifier

logger = logging.getLogger(__name__)


class RNNNet(FlatClassificationDataset, CDRLightningMixin):
    r"""Embed tokens of flat documents -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
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
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        FlatClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        self.rnn_type = rnn_type

        if pretrained_vector_path:
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path)
            logger.warning("Note that index of words in pretrained_vector should be same in current model")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                          # If we want to fine-tune then don't freeze layer
                                                          freeze=not is_pretrained_vector_fine_tune,
                                                          padding_idx=self.tokenizer.pad_token_id)
        else:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size,
                                          padding_idx=self.tokenizer.pad_token_id)

        self.rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                       input_size=embedding_size,
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
        rnn_features = self.rnn(self.embedding(data), lengths)
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


class HRNNNet(ChunkedClassificationDataset, CDRLightningMixin):
    r"""Hierarchical RNNs
     Embed tokens of chunked documents -> RNN for modeling order -> Classification

    :param Union[Namespace,Config] hparams: required by lightning module
    :param int embedding_size: dimension of the embedding
    :param float dropout: dropout before passing to fully connected classification layer
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
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
                 rnn_type: str, rnn_size: int, rnn_bidirectional: bool, rnn_num_layers: int,
                 rnn_dropout: float, is_enable_rnn_bias: bool,
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)
        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

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

        self.chunk_rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                             input_size=embedding_size,
                                             hidden_size=rnn_size, num_layers=rnn_num_layers,
                                             bidirectional=rnn_bidirectional,
                                             bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.doc_rnn = RNNFeatureExtractor(rnn_type=rnn_type,
                                           input_size=self.chunk_rnn.total_output_size,
                                           hidden_size=rnn_size, num_layers=rnn_num_layers,
                                           bidirectional=rnn_bidirectional,
                                           bias=is_enable_rnn_bias, dropout=rnn_dropout)

        self.classifier = DeepFCClassifier(input_size=self.doc_rnn.total_output_size,
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def forward(self, batch):
        data, labels, lengths = batch
        chunk_rnn_features = torch.ones((data.shape[0], data.shape[1],
                                         self.num_sent_rnn_directions * self.sent_rnn_size),
                                        dtype=torch.float, device=data.device)

        final_hidden_state = None
        for i in range(data.shape[0]):
            chunk_rnn_features[i, :lengths[i]], final_hidden_state = self.chunk_rnn(self.embedding(data[i]),
                                                                                    final_hidden_state)

        chunk_rnn_features = self.chunk_rnn()
        doc_rnn_features = self.doc_rnn(chunk_rnn_features, lengths)
        logits = self.classifier(doc_rnn_features)

        return logits, labels
