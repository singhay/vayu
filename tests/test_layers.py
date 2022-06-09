import pytest
import torch

from vayu.models.layers import RNNFeatureExtractor, DeepFCClassifier
from tests.test_helpers import _var_change_helper


def test_rnn_extractor_lstm():
    input_size = 2
    rnn_extractor = RNNFeatureExtractor(rnn_type='lstm',
                                        input_size=input_size,
                                        hidden_size=6, num_layers=2,
                                        bidirectional=True,
                                        bias=True, dropout=0.1)
    batch = torch.randint(2, 50, (2, 10, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens
    lengths = torch.randint(5, 10, (2,))  # document length before padding

    batch_size = batch.shape[0]
    with torch.no_grad():
        # output <- forward pass(inputs)
        rnn_features = rnn_extractor(batch, lengths)
    assert rnn_features.shape == torch.Size([batch_size, 36])  # hidden_size * bidirectional * 3 (avg, max, last)

    _var_change_helper(True, rnn_extractor, batch, rnn_extractor.named_parameters(),
                       lengths=lengths)


def test_rnn_extractor_gru():
    input_size = 2
    rnn_extractor = RNNFeatureExtractor(rnn_type='gru',
                                        input_size=input_size,
                                        hidden_size=5, num_layers=2,
                                        bidirectional=True,
                                        bias=True, dropout=0.1)
    batch = torch.randint(2, 50, (2, 10, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens
    lengths = torch.randint(2, 10, (2,))  # document length before padding

    _var_change_helper(True, rnn_extractor, batch, rnn_extractor.named_parameters(),
                       lengths=lengths)


def test_rnn_extractor_gru_single_layer():
    input_size = 2
    rnn_extractor = RNNFeatureExtractor(rnn_type='gru',
                                        input_size=input_size,
                                        hidden_size=5, num_layers=1,
                                        bidirectional=True,
                                        bias=True, dropout=0.0)
    batch = torch.randint(2, 50, (2, 10, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens
    lengths = torch.randint(2, 10, (2,))  # document length before padding

    _var_change_helper(True, rnn_extractor, batch, rnn_extractor.named_parameters(),
                       lengths=lengths)


def test_rnn_extractor_gru_unidirectional():
    input_size = 2
    rnn_extractor = RNNFeatureExtractor(rnn_type='gru',
                                        input_size=input_size,
                                        hidden_size=5, num_layers=2,
                                        bidirectional=False,
                                        bias=True, dropout=0.1)
    batch = torch.randint(2, 50, (2, 10, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens
    lengths = torch.randint(2, 10, (2,))  # document length before padding

    _var_change_helper(True, rnn_extractor, batch, rnn_extractor.named_parameters(),
                       lengths=lengths)


def test_rnn_extractor_not_supported():
    rnn_type = 'sha-rnn'
    with pytest.raises(ValueError, match="%s rnn type is not supported" % rnn_type):
        _ = RNNFeatureExtractor(rnn_type=rnn_type,
                                input_size=2,
                                hidden_size=5, num_layers=2,
                                bidirectional=True,
                                bias=True, dropout=0.1)


def test_deep_classifier():
    input_size, output_size = 32, 4
    model = DeepFCClassifier(input_size, [16, 8], output_size, is_enable_bias=True,
                             dropout=0.2, activation='gelu')
    batch = torch.randint(2, 50, (2, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens

    batch_size = batch.shape[0]
    with torch.no_grad():
        features = model(batch)
    assert features.shape == torch.Size([batch_size, output_size])

    _var_change_helper(True, model, batch, model.named_parameters())


def test_deep_classifier_not_deep():
    input_size, output_size = 32, 4
    model = DeepFCClassifier(input_size, [], output_size, is_enable_bias=True,
                             dropout=0.2, activation='gelu')
    batch = torch.randint(2, 50, (2, input_size),
                          dtype=torch.float)  # num documents x num sentences x num tokens

    batch_size = batch.shape[0]
    with torch.no_grad():
        features = model(batch)
    assert features.shape == torch.Size([batch_size, output_size])

    _var_change_helper(True, model, batch, model.named_parameters())


def test_deep_classifier_resolve_activation():
    assert isinstance(DeepFCClassifier.resolve_activation('relu'), torch.nn.ReLU)
    assert isinstance(DeepFCClassifier.resolve_activation('gelu'), torch.nn.GELU)
    assert isinstance(DeepFCClassifier.resolve_activation('elu'), torch.nn.ELU)
    assert isinstance(DeepFCClassifier.resolve_activation('lrelu'), torch.nn.LeakyReLU)


def test_deep_classifier_resolve_activation_not_implemented_error():
    with pytest.raises(NotImplementedError, match="prelu is not supported"):
        _ = DeepFCClassifier.resolve_activation('prelu')
