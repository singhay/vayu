import pytest
import torch
from omegaconf import DictConfig

from vayu.models.classification.bag_of_embedding import BagOfEmbedding
from vayu.models.classification.cnn import TextCNN
from vayu.models.classification.cnn_rnn import CNNRNNNet, CNN1dRNNNet
from vayu.models.classification.hierarchical_att_model import HierAttNet
from vayu.models.classification.rnn import RNNNet
from vayu.models.classification.transformer import TransformerLinear, TransformerLSTM, TransformerAttention, \
    TransformerXLLSTM
from tests.test_helpers import _var_change_helper, _forward_step


@pytest.fixture()
def make_kwargs():
    def _make_kwargs():
        hparams = DictConfig({'batch_size': 4, 'num_workers': 0})
        kwargs = {
                  'max_chunks': 2, 'max_chunk_size': 10,
                  'tokenizer_class': 'bpe',
                  'tokenizer_path': './vayu/tokenizer/resources/65k_roberta_bpe',
                  'stem': False, 'lowercase': False, 'normalize': False, 'is_lazy': True,
                  'train_path': './vayu/tests/resources/cdr_data.json',
                  'valid_path': './vayu/tests/resources/cdr_data.json',
                  'test_path': './vayu/tests/resources/cdr_data.json',
                  'num_classes': 1, 'label_smooth': 0.0, 'pct_of_data': 0.9
                  }
        kwargs.update(hparams)
        return hparams, kwargs

    return _make_kwargs


def _test_flat_data_model(model):
    batch = (torch.randint(2, 5, (2, 100)),  # num documents x num tokens
             torch.randint(0, 2, (2,)),  # num document labels
             torch.randint(20, 90, (2,)))  # document length before padding

    _forward_step(model, batch)

    static_params = [np for np in model.named_parameters() if not np[1].requires_grad]
    _var_change_helper(False, model, batch, static_params)

    # pretrained_vector should not be trained hence weights should remain the same
    params = [np for np in model.named_parameters() if np[1].requires_grad]
    _var_change_helper(True, model, batch, params)


def _test_chunked_data_model(model):
    batch = (torch.randint(2, 5, (2, 10, 100)),  # num documents x num sentences x num tokens
             torch.randint(0, 2, (2,)),  # num document labels
             torch.randint(2, 10, (2,)))  # document length before padding

    _forward_step(model, batch)

    static_params = [np for np in model.named_parameters() if not np[1].requires_grad]
    _var_change_helper(False, model, batch, static_params)

    params = [np for np in model.named_parameters() if np[1].requires_grad]
    _var_change_helper(True, model, batch, params)


def test_boe(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = BagOfEmbedding(hparams, embedding_size=100, dropout=0.2, **kwargs)
    _test_flat_data_model(model)


def test_cnn(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TextCNN(hparams, embedding_size=10,
                    pretrained_vector_path='',
                    is_pretrained_vector_fine_tune=False, pretrained_vector_fine_tune_learning_rate=1e-5,
                    cnn_kernel_sizes=[2, 4], cnn_kernel_numbers=[5, 5],
                    fc_dropout=0.2, fc_activation='relu',
                    fc_middle_layers=[], is_enable_fc_layer_bias=True,
                    **kwargs)
    _test_flat_data_model(model)


def test_rnn(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = RNNNet(hparams, embedding_size=10,
                   pretrained_vector_path='',
                   is_pretrained_vector_fine_tune=False, pretrained_vector_fine_tune_learning_rate=1e-5,
                   rnn_type='lstm', rnn_size=5, rnn_bidirectional=True, rnn_num_layers=2,
                   rnn_dropout=0.1, is_enable_rnn_bias=True,
                   fc_dropout=0.2, fc_activation='relu',
                   fc_middle_layers=[], is_enable_fc_layer_bias=True, **kwargs)
    _test_flat_data_model(model)


def test_cnn_pretrained_vector(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TextCNN(hparams, embedding_size=None,
                    pretrained_vector_path='./tests/resources/w2v.bin',
                    is_pretrained_vector_fine_tune=True, pretrained_vector_fine_tune_learning_rate=1e-5,
                    cnn_kernel_sizes=[2, 4], cnn_kernel_numbers=[5, 5],
                    fc_dropout=0.2, fc_activation='relu',
                    fc_middle_layers=[], is_enable_fc_layer_bias=True,
                    **kwargs)

    _test_flat_data_model(model)


def test_cnn_multi(make_kwargs):
    pass
    # Figure out why even after pretrained_vector_fine_tune=False; static.embedding.weight is changing
    # hparams, kwargs = make_kwargs()
    # # TODO: Investigate why frozen layers are changing leading to test fail
    # model = TextCNNMulti(hparams, dropout=0.2,
    #                      pretrained_vector_path='./tests/resources/w2v.bin', pretrained_vector_fine_tune=False,
    #                      pretrained_vector_fine_tune_learning_rate=1e-5,
    #                      cnn_kernel_sizes=[2, 5], cnn_kernel_numbers=[5, 5],
    #                 fc_dropout=0.2, fc_activation='relu',
    #                 fc_middle_layers=[], is_enable_fc_layer_bias=True,
    #                      **kwargs)
    # _test_flat_data_model(model)


def test_cnn_lstm(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = CNNRNNNet(hparams, embedding_size=10,
                      pretrained_vector_path='', pretrained_vector_fine_tune='',
                      is_pretrained_vector_fine_tune=False, pretrained_vector_fine_tune_learning_rate=1e-5,
                      cnn_kernel_sizes=[2, 5], cnn_kernel_numbers=[5, 5],
                      rnn_type='lstm', rnn_size=5, rnn_bidirectional=True, rnn_num_layers=2,
                      rnn_dropout=0.1, is_enable_rnn_bias=True,
                      fc_dropout=0.2, fc_activation='relu',
                      fc_middle_layers=[], is_enable_fc_layer_bias=True, **kwargs)

    _test_chunked_data_model(model)


def test_cnn1d_lstm(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = CNN1dRNNNet(hparams, embedding_size=10,
                        pretrained_vector_path='', pretrained_vector_fine_tune='',
                        is_pretrained_vector_fine_tune=False,
                        pretrained_vector_fine_tune_learning_rate=1e-5,
                        cnn_kernel_sizes=[2, 5], cnn_kernel_numbers=[5, 5],
                        rnn_type='lstm', rnn_size=5, rnn_bidirectional=True, rnn_num_layers=2,
                        rnn_dropout=0.1, is_enable_rnn_bias=True,
                        fc_dropout=0.2, fc_activation='relu',
                        fc_middle_layers=[], is_enable_fc_layer_bias=True, **kwargs)

    _test_chunked_data_model(model)


def test_cnn_lstm_multilayer_rnn(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = CNNRNNNet(hparams, embedding_size=10,
                      pretrained_vector_path='',
                      is_pretrained_vector_fine_tune=False, pretrained_vector_fine_tune_learning_rate=1e-5,
                      cnn_kernel_sizes=[2, 5], cnn_kernel_numbers=[5, 5],
                      rnn_type='lstm', rnn_size=5, rnn_bidirectional=True, rnn_num_layers=3,
                      rnn_dropout=0.1, is_enable_rnn_bias=True,
                      fc_dropout=0.2, fc_activation='relu',
                      fc_middle_layers=[], is_enable_fc_layer_bias=True, **kwargs)

    _test_chunked_data_model(model)


def test_cnn_lstm_defrosted_vector(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = CNNRNNNet(hparams, embedding_size=10,
                      pretrained_vector_path='',
                      is_pretrained_vector_fine_tune=True, pretrained_vector_fine_tune_learning_rate=1e-5,
                      cnn_kernel_sizes=[2, 5], cnn_kernel_numbers=[5, 5],
                      rnn_type='lstm', rnn_size=5, rnn_bidirectional=True, rnn_num_layers=2,
                      rnn_dropout=0.1, is_enable_rnn_bias=True,
                      fc_dropout=0.2, fc_activation='relu',
                      fc_middle_layers=[], is_enable_fc_layer_bias=True, **kwargs)
    _test_chunked_data_model(model)


def test_han_bidirectional(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = HierAttNet(hparams, embedding_size=10, dropout=0.2,
                       pretrained_vector_path='',
                       is_pretrained_vector_freeze=False,
                       sent_rnn_num_layers=1,
                       sent_rnn_bidirectional=True,
                       sent_rnn_size=5,
                       doc_rnn_num_layers=1,
                       doc_rnn_bidirectional=True,
                       doc_rnn_size=5, **kwargs)
    _test_chunked_data_model(model)


def test_han_unidirectional(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = HierAttNet(hparams, embedding_size=10, dropout=0.2,
                       pretrained_vector_path='',
                       is_pretrained_vector_freeze=False,
                       sent_rnn_num_layers=1,
                       sent_rnn_bidirectional=False,
                       sent_rnn_size=5,
                       doc_rnn_num_layers=1,
                       doc_rnn_bidirectional=False,
                       doc_rnn_size=5, **kwargs)
    _test_chunked_data_model(model)


def test_han_pretrained_vector(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = HierAttNet(hparams, embedding_size=100, dropout=0.2,
                       pretrained_vector_path='./tests/resources/w2v.bin',
                       is_pretrained_vector_freeze=True,
                       sent_rnn_num_layers=1,
                       sent_rnn_bidirectional=False,
                       sent_rnn_size=5,
                       doc_rnn_num_layers=1,
                       doc_rnn_bidirectional=False,
                       doc_rnn_size=5, **kwargs)
    _test_chunked_data_model(model)


def test_transformer_linear(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TransformerLinear(hparams=hparams, config_name='./tests/resources/roberta-lm-dummy/config.json',
                              model_type='roberta', model_name_or_path='./tests/resources/roberta-lm-dummy/',
                              is_pretrained_model_fine_tune=False,
                              pretrained_model_fine_tune_learning_rate=1e-5, mode='linear',
                              **kwargs)

    _test_chunked_data_model(model)


def test_transformer_linear_defrosted_transformer_mean(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TransformerLinear(hparams=hparams,
                              config_name='./tests/resources/roberta-lm-dummy/config.json',
                              model_type='roberta',
                              model_name_or_path='./tests/resources/roberta-lm-dummy/',
                              is_pretrained_model_fine_tune=False,
                              pretrained_model_fine_tune_learning_rate=1e-5, mode='mean',
                              **kwargs)
    _test_chunked_data_model(model)


def test_transformer_linear_frozen_transformer_max(make_kwargs):
    hparams, kwargs = make_kwargs()
    with pytest.raises(NotImplementedError) as _:
        _ = TransformerLinear(hparams=hparams,
                              config_name='./tests/resources/roberta-lm-dummy/config.json',
                              model_type='roberta',
                              model_name_or_path='./tests/resources/roberta-lm-dummy/',
                              is_pretrained_model_fine_tune=False,
                              pretrained_model_fine_tune_learning_rate=1e-5,
                              mode='max',
                              **kwargs)


def test_transformer_lstm(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TransformerLSTM(hparams=hparams,
                            config_name='./tests/resources/roberta-lm-dummy/config.json',
                            model_type='roberta',
                            model_name_or_path='./tests/resources/roberta-lm-dummy/',
                            is_pretrained_model_fine_tune=False,
                            pretrained_model_fine_tune_learning_rate=1e-5,
                            rnn_size=5, rnn_bidirectional=True, rnn_num_layers=3,
                            rnn_is_enable_bias=True, rnn_dropout=0.2,
                            **kwargs)

    _test_chunked_data_model(model)


def test_transformer_attention(make_kwargs):
    hparams, kwargs = make_kwargs()
    model = TransformerAttention(hparams=hparams,
                                 config_name='./tests/resources/roberta-lm-dummy/config.json',
                                 model_type='roberta',
                                 model_name_or_path='./tests/resources/roberta-lm-dummy/',
                                 is_pretrained_model_fine_tune=False,
                                 pretrained_model_fine_tune_learning_rate=1e-5,
                                 num_attention_heads=2, num_attention_layers=2,
                                 **kwargs)
    _test_chunked_data_model(model)


def test_transformer_xl(make_kwargs):
    hparams, kwargs = make_kwargs()
    kwargs.pop('tokenizer_path')
    model = TransformerXLLSTM(hparams=hparams,
                              config_name='./tests/resources/transfoxl-lm-dummy/config.json',
                              model_type='transfoxl',
                              model_name_or_path='./tests/resources/transfoxl-lm-dummy/',
                              is_pretrained_model_fine_tune=False,
                              pretrained_model_fine_tune_learning_rate=1e-5,
                              tokenizer_path='./vayu/tokenizer/resources/265k_transfoxl',
                              rnn_size=5, rnn_bidirectional=True, rnn_num_layers=3,
                              rnn_is_enable_bias=True, rnn_dropout=0.2,
                              **kwargs)
    _test_chunked_data_model(model)
