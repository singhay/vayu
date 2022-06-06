r"""
`Convolutional Neural Networks (CNN) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_
===================================================================================================

* The filter region size can have a large effect on performance, and should be tuned.
* Vectors are better than one-hot encoding; non-static better than static
* The number of feature maps can also play an important role in the performance, and increasing the number of feature maps will increase the training time of the model.
* 1-max pooling uniformly outperforms other pooling strategies.
* Regularization has relatively little effect on the performance of the model.
* Line-search over the single filter region size to find the ‘best’ single region size. A reasonable range might be 1∼10. However, for datasets with very long sentences like CR, it may be worth exploring larger filter region sizes. Once this ‘best’ region size is identified, it may be worth exploring combining multiple filters using regions sizes near this single best size, given that empirically multiple ‘good’ region sizes always outperformed using only the single best region size.
* Alter the number of feature maps for each filter region size from 100 to 600, and when this is being explored, use a small dropout rate (0.0-0.5) and a large max norm constraint. Note that increasing the number of feature maps will increase the running time, so there is a trade-off to consider. Also pay attention whether the best value found is near the border of the range. If the best value is near 600, it may be worth trying larger values.
* When increasing the number of feature maps begins to reduce performance, try imposing stronger regularization, e.g., a dropout out rate larger than 0.5


Papers
------

* Convolutional Neural Networks for Sentence Classification: https://arxiv.org/pdf/1408.5882.pdf
* A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification: https://arxiv.org/pdf/1510.03820
* Long Length Document Classification by Local Convolutional Feature Aggregation: https://www.mdpi.com/1999-4893/11/8/109
* Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding: https://arxiv.org/abs/1504.01255
* Summary: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
* Word2Vec + UMLS + CNN: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0781-4

Improvements
------------
* TODO: Convert 2D convolution to 1D convolution (update stride=embedding_size)
* TODO: deeper CNN

"""

import logging
from argparse import Namespace
from typing import List, Union

import gensim
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vayu.datasets.classification import FlatClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.layers import DeepFCClassifier

logger = logging.getLogger(__name__)


class TextCNN(FlatClassificationDataset, CDRLightningMixin):
    r"""Implementation with some improvements over Convolutional Neural Networks for Sentence Classification:
    https://arxiv.org/pdf/1408.5882.pdf

    :param Union[Namespace,DictConfig] hparams: required by lightning module
    :param Union[int,None] embedding_size: dimension of the embedding, ``None`` if loading pretrained
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param bool is_pretrained_vector_fine_tune: whether to fine-tune vectors along with rest of network
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams: Union[Namespace, DictConfig], embedding_size: Union[int, None],
                 pretrained_vector_path: str,
                 is_pretrained_vector_fine_tune: bool, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int],
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        FlatClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)

        self.is_pretrained_vector_fine_tune = is_pretrained_vector_fine_tune
        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        if pretrained_vector_path:
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path, binary=True)
            logger.info("Pretrained word2vec model loaded with dimensions: %s", model.vector_size)
            logger.warning("Note that index of words in pretrained_vector should be same in current model")
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                          # If we want to fine-tune then not freeze layer
                                                          freeze=not is_pretrained_vector_fine_tune,
                                                          padding_idx=self.tokenizer.pad_token_id)
            embedding_size = model.vector_size
        else:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_size)

        self.convs = nn.ModuleList([nn.Conv2d(1, number, (size, embedding_size),
                                              padding=(size - 1, 0)) for (size, number) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers)])

        self.classifier = DeepFCClassifier(input_size=sum(cnn_kernel_numbers) * 2,  # 2: max + avg pool
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def forward(self, batch):
        data, labels, lengths = batch
        x = self.embedding(data)
        x = x.unsqueeze(1)  # Insert channel dimension
        x = [torch.nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x] + \
            [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        logits = self.classifier(x)

        return logits, labels

    def resolve_params(self):
        if self.is_pretrained_vector_fine_tune:
            # Here we set a specific learning rate for fine tuning layer and keep default for rest
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'embedding' not in x[0],
                                                      self.pretrained_model.named_parameters()))},
                {'params': self.embedding.parameters(),
                 'lr': self.pretrained_vector_fine_tune_learning_rate}
            ]
        else:
            params = self.parameters()

        return params


class TextCNNMulti(FlatClassificationDataset, CDRLightningMixin):
    r"""A TextCNN model that has multiple channel features
    1) trainable embeddings
    2) pretrained embeddings
    3) pretrained embeddings that are fine-tuned (trained at lower learning rate)

    TODO: Add pos embedding channel [CD, JJ, MD, NN, NNP, PRP, RB, VB]
    TODO: Add lemma embedding to multi-channel network
    TODO: Maybe add stopword / punctuation binary channel
    TODO: Maybe add probability of present in a bucket of document section (PPD) channel
    TODO: Add (to preserve dimensions) NER embedding one of the channels

    This is an in-house brew that adds multiple channels as improvement to TextCNN in that
     it has static embedding, pretrained embeddings, another channel that fine-tunes pretrained embeddings

    :param Union[Namespace,DictConfig] hparams: required by lightning module
    :param str pretrained_vector_path: path to pretrained Word2Vec / fastText of type  :py:class:`gensim.models.KeyedVectors`
    :param float pretrained_vector_fine_tune_learning_rate: fine tune learning rate, default 1e-5
    :param List[int] cnn_kernel_sizes: size of cnn kernel spans, default: 3,4,5
    :param List[int] cnn_kernel_numbers: number of cnn kernel spans, default: 100, 100, 100
    :param List[int] fc_middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param float fc_dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str fc_activation: what kind of activation function to apply between fully connected layers {relu, gelu, elu, leakyRelu}
    :param bool is_enable_fc_layer_bias: compute bias along with the rest of network
    :param dict kwargs: keyword arguments for initializing datasets
    """

    def __init__(self, hparams: Union[Namespace, DictConfig],
                 pretrained_vector_path: str, pretrained_vector_fine_tune_learning_rate: float,
                 cnn_kernel_sizes: List[int], cnn_kernel_numbers: List[int],
                 fc_dropout: float, fc_activation: str, fc_middle_layers: List[int],
                 is_enable_fc_layer_bias: bool, **kwargs):
        FlatClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)

        self.pretrained_vector_fine_tune_learning_rate = pretrained_vector_fine_tune_learning_rate

        model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vector_path, binary=True)
        logger.info("Pretrained word2vec model loaded with dimensions: %s", model.vector_size)
        logger.warning("Note that index of words in pretrained_vector should be same in current model")

        self.embedding = nn.Embedding(self.tokenizer.vocab_size, model.vector_size,
                                      padding_idx=self.tokenizer.pad_token_id)
        self.static_embed = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors), freeze=True,
                                                         padding_idx=self.tokenizer.pad_token_id)
        self.non_static_embed = nn.Embedding.from_pretrained(torch.FloatTensor(model.vectors),
                                                             freeze=False,
                                                             padding_idx=self.tokenizer.pad_token_id)
        self.embeddings = nn.ModuleList([self.embedding, self.static_embed, self.non_static_embed])

        self.convs = nn.ModuleList([nn.Conv2d(len(self.embeddings), number, (size, model.vector_size),
                                              padding=(size - 1, 0)) for (size, number) in
                                    zip(cnn_kernel_sizes, cnn_kernel_numbers)])

        self.classifier = DeepFCClassifier(input_size=sum(cnn_kernel_numbers) * 2,  # 2: max + avg pool
                                           middle_layers=list(fc_middle_layers),  # Convert from listConfig
                                           output_size=self.num_classes,
                                           is_enable_bias=is_enable_fc_layer_bias,
                                           dropout=fc_dropout, activation=fc_activation)

    def forward(self, batch):
        data, labels, lengths = batch
        x = torch.stack([embedding(data) for embedding in self.embeddings], dim=1)
        x = [torch.nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x] + \
            [torch.nn.functional.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        logits = self.classifier(x)

        return logits, labels

    def resolve_params(self):
        """ Prepare optimizer and schedule (reduce LR when necessary) """
        params = [
            # Here we set a specific learning rate for fine tuning layer and keep default for rest
            {'params': map(lambda x: x[1], filter(lambda x: 'non_static_embed' not in x[0],
                                                  self.named_parameters()))},
            {'params': self.non_static_embed.parameters(),
             'lr': self.pretrained_vector_fine_tune_learning_rate}
        ]

        return params
