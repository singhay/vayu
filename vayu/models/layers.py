"""
Layers
======

:py:class:`~torch.nn.Module` that supports two variants of Recurrent Neural Networks
the features from whose are concatenation of max, avg pool and last hidden state of RNN
"""
from typing import List

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNFeatureExtractor(torch.nn.Module):
    """Module for aggregating features from RNN layer for use in downstream tasks

    :param str rnn_type: ``lstm`` or ``gru``
    :param input_size: the dimension of the input, usually equal to embedding size
    :param hidden_size: the size of RNN hidden layers
    :param num_layers: number of RNN layers to stack, features from previous layer are passed to next layer
    :param bidirectional: whether to concatenate forward and backward directions
    :param bias: compute bias along with the rest of network
    :param dropout: recurrent dropout to apply between sequences
    """
    def __init__(self, rnn_type, input_size, hidden_size, num_layers,
                 bidirectional, bias, dropout):
        super().__init__()

        if rnn_type.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(input_size, batch_first=True,
                                     hidden_size=hidden_size, num_layers=num_layers,
                                     bidirectional=bidirectional,
                                     bias=bias, dropout=dropout)
        elif rnn_type.lower() == 'gru':
            self.rnn = torch.nn.GRU(input_size, batch_first=True,
                                    hidden_size=hidden_size, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    bias=bias, dropout=dropout)
        else:
            raise ValueError("%s rnn type is not supported" % rnn_type)

        self.num_directions = 2 if bidirectional else 1
        self.total_output_size = hidden_size * self.num_directions * 3  # 3: avg_pool, max_pool, rnn_out[:, -1]

    def forward(self, data, lengths, hidden=None):
        batch_size, total_chunks = data.shape[0], data.shape[1]

        # Pack for variable length sequences
        pooled_outputs = pack_padded_sequence(data, batch_first=True,
                                              lengths=lengths.cpu(), enforce_sorted=False)
        # Reason why lengths has .cpu() - https://github.com/pytorch/pytorch/issues/43227
        self.rnn.flatten_parameters()  # To bring all parameters to the same device
        rnn_out, _ = self.rnn(pooled_outputs)
        rnn_out, lengths = pad_packed_sequence(rnn_out, batch_first=True)
        avg_pool = torch.nn.functional.adaptive_avg_pool1d(rnn_out.permute(0, 2, 1), 1).view(batch_size, -1)
        max_pool = torch.nn.functional.adaptive_max_pool1d(rnn_out.permute(0, 2, 1), 1).view(batch_size, -1)
        concat = torch.cat([avg_pool, max_pool, rnn_out[:, -1]], dim=1)
        return concat


class DeepFCClassifier(torch.nn.Module):
    """Class that enables creating deep neural network with blocks of (Dropout->Linear->Activation)
    An example network would be ``[input_size, dropout, activation, middle_layers[0], dropout, activation, middle_layers[1], dropout, activation, output_size]``

    :param int input_size: size of incoming layer
    :param List[int] middle_layers: number of linear layers to stack, e.g. [input_size, 32, 64, output_size]
    :param int output_size: output size of the last layer e.g. binary classification: 1
    :param bool is_enable_bias: compute bias along with the rest of network
    :param float dropout: probability [0,1] of dropout to apply between fully connected layers
    :param str activation: what kind of activation function to use {relu, gelu, elu, leakyRelu}
    """
    def __init__(self, input_size: int, middle_layers: List[int], output_size: int,
                 is_enable_bias: bool, dropout: float, activation: str):
        super().__init__()
        activation_layer = self.resolve_activation(activation)
        dropout_layer = torch.nn.Dropout(dropout)

        classifiers = []
        prev_layer_size = input_size
        for layer in middle_layers:
            classifiers.append(dropout_layer)
            classifiers.append(torch.nn.Linear(prev_layer_size, layer, bias=is_enable_bias))
            classifiers.append(activation_layer)
            prev_layer_size = layer

        # If length of classifiers is still 0 that means middle_layers is empty
        # hence only insert the dropout layer as activation is not needed here
        if len(classifiers) == 0:
            classifiers.append(dropout_layer)

        # Finally insert the last layer of network, activation is taken care of by for loop above
        classifiers.append(torch.nn.Linear(prev_layer_size, output_size, bias=is_enable_bias))

        self.classifier = torch.nn.Sequential(*classifiers)

    def forward(self, batch):
        return self.classifier(batch)

    @staticmethod
    def resolve_activation(activation):
        activation = activation.lower()
        if activation == 'relu':
            return torch.nn.ReLU()
        if activation == 'gelu':
            return torch.nn.GELU()
        if activation == 'elu':
            return torch.nn.ELU()
        if activation == 'lrelu':
            return torch.nn.LeakyReLU()
        else:
            raise NotImplementedError(f"{activation} is not supported")

