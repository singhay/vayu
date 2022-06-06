r"""
`Transformer <https://arxiv.org/abs/1706.03762>`_
=================================================
Set of transformer models adapted for classification problems with the ability to load pretrained language models and fine-tune as and when required.

Since transformer models can only take 512 tokens at a time and are O(n^2) where n is length of incoming document. We bypass this limitation for long documents by splitting our  documents into chunks that are easily consumed by transformer models. The embedding extracted per chunk is then fed to a linear logistic or lstm layer that learns weights and biases for doing efficient prediction.

Papers
------
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
* Robustly Optimized BERT Pretraining Approach: https://arxiv.org/abs/1907.11692
* Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context: https://arxiv.org/abs/1901.02860
* How to Fine-Tune BERT for Text Classification?: https://arxiv.org/pdf/1905.05583.pdf
* Hierarchical Transformers For Long Document Classification: https://arxiv.org/pdf/1910.10781.pdf
* Transformers: https://huggingface.co/transformers/

Improvements
------------
* TODO: Add ability to fine-tune rest of the transformer network at low learning rate
* TODO: Add schedule to unfreeze transformer later in the training layer by layer
* TODO: Instead of taking sentence embedding ([CLS]) take mean of embedding of all words in sentence
* TODO: Weighted combination of different layers of transformer instead of last layer which as been found to more effective.
"""

import logging
from abc import ABC

import torch
from transformers.modeling_bert import BertPooler

from vayu.constants import MODEL_CLASSES
from vayu.datasets.classification import ChunkedClassificationDataset
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.layers import RNNFeatureExtractor

logger = logging.getLogger(__name__)


class CDRTransformerBase(ChunkedClassificationDataset, CDRLightningMixin, ABC):
    """

    :param Namespace hparams: required by lightning module for persistence
    :param str config_name: custom configuration of the model_type if needed
    :param str model_type: bert, roberta or transfoxl
    :param str model_name_or_path: either a model name or pretrained transformer path
    :param bool is_pretrained_model_fine_tune: whether to train the transformer along with classification models
    :param bool pretrained_model_fine_tune_learning_rate: learning rate for transformer model
    :param kwargs: keyword arguments used for initialization of dataset classes
    """
    def __init__(self, hparams, config_name, model_type, model_name_or_path,
                 is_pretrained_model_fine_tune: bool,
                 pretrained_model_fine_tune_learning_rate: float, **kwargs):
        self.is_pretrained_model_fine_tune = is_pretrained_model_fine_tune
        self.pretrained_model_fine_tune_learning_rate = pretrained_model_fine_tune_learning_rate

        # Load appropriate tokenizer and initialize config
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type.lower()]
        kwargs['tokenizer_class'] = tokenizer_class
        ChunkedClassificationDataset.__init__(self, **kwargs)
        CDRLightningMixin.__init__(self, hparams=hparams)

        self.pretrained_transformer_config = config_class.from_pretrained(
            config_name if config_name else model_name_or_path,
            pad_idx=self.tokenizer.pad_token_id
        )
        self.pretrained_transformer = model_class.from_pretrained(
            model_name_or_path,
            config=self.pretrained_transformer_config
        )

        if is_pretrained_model_fine_tune:
            self.unfreeze_transformer_encoder()
        else:
            self.freeze_transformer_encoder()

    def freeze_transformer_encoder(self):
        """Freezes all parameters of the transformer"""
        for param in self.pretrained_transformer.parameters():
            param.requires_grad = False
        logger.info("Transformer parameters frozen")

    def unfreeze_transformer_encoder(self):
        """Defrosts all parameters of the transformer so they can be trained."""
        for param in self.pretrained_transformer.parameters():
            param.requires_grad = True
        logger.info("Transformer parameters defrosted")

    def forward(self, batch):
        """Passes a document (List[List[str]]) via transformer and returns it in a 3D tensor"""
        data, labels, lengths = batch
        batch_size = data.shape[0]
        pooled_outputs = torch.zeros((batch_size, data.shape[1],
                                      self.pretrained_transformer_config.hidden_size),
                                     device=data.device)
        for i in range(batch_size):
            last_hidden_states, pooler_output = self.pretrained_transformer(input_ids=data[i, :lengths[i]])

            """Experimental Average pooling.
            last_hidden_state = mask_fill(0.0, tokens, last_hidden_states, self.tokenizer.pad_token_id)
            pooler_output = torch.sum(last_hidden_states, 1)
            sum_mask = mask.unsqueeze(-1).expand(last_hidden_states.size()).float().sum(1)
            pooler_output = pooler_output / sum_mask
            """
            pooled_outputs[i, :lengths[i]] = pooler_output

        return pooled_outputs

    def resolve_params(self):
        if self.is_pretrained_model_fine_tune:
            params = [
                {'params': map(lambda x: x[1], filter(lambda x: 'classifier' not in x[0],
                                                      self.pretrained_transformer.named_parameters())),
                 'lr': self.pretrained_model_fine_tune_learning_rate},
                {'params': self.classifier.parameters()}
            ]
        else:
            params = self.parameters()

        return params


class TransformerLinear(CDRTransformerBase):
    def __init__(self, hparams, mode='linear', **kwargs):
        self.mode = mode
        super().__init__(hparams=hparams, **kwargs)

        if self.mode == 'linear':
            # TODO: Pending logic for dealing properly with variable length!
            # right now, we avg pool sequences which looses a lot of signal
            self.classifier = torch.nn.Linear(self.pretrained_transformer_config.hidden_size,
                                              self.num_classes,
                                              bias=True)
        elif self.mode == 'mean':
            self.classifier = torch.nn.Linear(self.pretrained_transformer_config.hidden_size, self.max_chunks,
                                              bias=True)
            self.ensemble = torch.nn.Linear(self.max_chunks, self.num_classes, bias=True)
        elif self.mode == 'max':
            raise NotImplementedError("Linear Max not implemented, logic for threshold is pending!")

    def forward(self, batch):
        """Extracts embeddings of each chunk in a document via transformer
        and either sums embeddings or ensembles them using fully connected layer
        """
        data, labels, lengths = batch
        pooled_outputs = super().forward(batch)
        if self.mode == 'linear':
            # batch_size x # sequences x self.pretrained_transformer_config.hidden_size ->
            # batch_size x self.pretrained_transformer_config.hidden_size
            doc_pooled = torch.nn.functional.adaptive_avg_pool1d(pooled_outputs.permute(0, 2, 1), 1).squeeze(2)
            logits = self.classifier(doc_pooled)  # -> (self.pretrained_transformer_config.hidden_size, 1)
            return logits, labels
        elif self.mode == 'mean':
            logits = self.classifier(pooled_outputs)  # -> (batch_size, sequence_length, max_chunks)
            ensemble = self.ensemble(logits)  # -> (batch_size, max_chunks, 1)
            return torch.mean(ensemble, dim=1), labels  # -> (batch_size, 1)
        elif self.mode == 'max':
            # TODO: Majority voting baseline, what threshold to use ?
            # logits = self.classifier(pooled_outputs)
            # sigmoids = torch.sigmoid(logits)
            # sigmoids[sigmoids >= threshold] = POSITIVE_LABEL
            # sigmoids[sigmoids < threshold] = NEGATIVE_LABEL
            # logits, indices = torch.mode(sigmoids, dim=1)
            # return logits, labels
            raise NotImplementedError("Linear Max not implemented, logic for threshold is pending!")


class TransformerLSTM(CDRTransformerBase):
    r"""RoBERT in Sec 3.2 https://arxiv.org/pdf/1910.10781.pdf

    .. code-block:: python

        python -m main do_train=true do_eval=true \
        model=classification/transformer_lstm \
        model.params.model_name_or_path=path/to/lm_74177_roberta_65kvocab_12L/
    """

    def __init__(self, hparams, rnn_size, rnn_bidirectional, rnn_num_layers,
                 rnn_is_enable_bias, rnn_dropout, **kwargs):
        super().__init__(hparams=hparams, **kwargs)

        self.lstm = RNNFeatureExtractor(rnn_type='lstm',
                                        input_size=self.pretrained_transformer_config.hidden_size,
                                        hidden_size=rnn_size, num_layers=rnn_num_layers,
                                        bidirectional=rnn_bidirectional,
                                        bias=rnn_is_enable_bias, dropout=rnn_dropout)
        self.classifier = torch.nn.Linear(self.lstm.total_output_size, self.num_classes)

    def forward(self, batch):
        """Extracts embeddings of each chunk in a document via transformer
        and models sequential nature of chunks in document using LSTM network"""
        data, labels, lengths = batch
        pooled_outputs = super().forward(batch)
        rnn_features = self.lstm(pooled_outputs, lengths)
        logits = self.classifier(rnn_features)
        return logits, labels


class TransformerAttention(CDRTransformerBase):
    """ ToBERT in Sec 3.3 https://arxiv.org/pdf/1910.10781.pdf

    :param num_attention_heads: number of multi head attention heads
    :param num_attention_layers: number of attention layers
    :param kwargs: keyword arguments used for initialization of :class:`~vayu.models.transformer.CDRTransformerBase`
    """

    def __init__(self, hparams, num_attention_heads, num_attention_layers, **kwargs):
        super().__init__(hparams=hparams, **kwargs)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.pretrained_transformer_config.hidden_size,
            nhead=num_attention_heads)
        self.tobert_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)
        self.tobert_pooler = BertPooler(self.pretrained_transformer_config)
        self.classifier = torch.nn.Linear(self.pretrained_transformer_config.hidden_size, self.num_classes)

    def forward(self, batch):
        """Extracts embeddings of each chunk in a document via transformer
        and models sequential nature of chunks in document using Transformer network"""
        data, labels, lengths = batch
        pooled_outputs = super().forward(batch)
        hidden_states = self.tobert_encoder(pooled_outputs)
        logits = self.classifier(self.tobert_pooler(hidden_states))

        return logits, labels


class TransformerXLLSTM(CDRTransformerBase):
    """XL + LSTM head on top: main benefit is recurrence and relative positional embeddings"""

    def __init__(self, hparams, rnn_size, rnn_bidirectional, rnn_num_layers,
                 rnn_is_enable_bias, rnn_dropout, **kwargs):
        super().__init__(hparams=hparams, **kwargs)

        self.lstm = RNNFeatureExtractor(rnn_type='lstm',
                                        input_size=self.pretrained_transformer_config.hidden_size,
                                        hidden_size=rnn_size, num_layers=rnn_num_layers,
                                        bidirectional=rnn_bidirectional,
                                        bias=rnn_is_enable_bias, dropout=rnn_dropout)
        self.dropout = torch.nn.Dropout2d(self.pretrained_transformer_config.dropout, inplace=True)
        self.classifier = torch.nn.Linear(self.lstm.total_output_size, self.num_classes)

    def forward(self, batch):
        r"""Extracts embeddings and memory of each chunk in a document via :py:class:`transformers.TransfoXLModel`
            and models sequential nature of chunks in document by passing memory back
            along with next chunk.
        """
        data, labels, lengths = batch
        batch_size, total_chunks = data.shape[0], data.shape[1]
        pooled_outputs = torch.zeros((batch_size, total_chunks,
                                      self.pretrained_transformer_config.hidden_size),
                                     device=data.device)
        for i in range(batch_size):
            mems = None
            for j in range(lengths[i]):
                output = self.pretrained_transformer(input_ids=data[i, j].unsqueeze(0), mems=mems)
                last_hidden_states, mems = output[:2]
                pooled_sentence = torch.nn.functional.adaptive_avg_pool1d(last_hidden_states
                                                                          .permute(0, 2, 1), 1).squeeze()
                pooled_outputs[i, j] = pooled_sentence

        self.dropout(pooled_outputs)
        rnn_features = self.lstm(pooled_outputs, lengths)
        logits = self.classifier(rnn_features)
        return logits, labels
