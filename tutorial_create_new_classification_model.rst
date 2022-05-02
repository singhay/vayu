How to add a new model
~~~~~~~~~~~~~~~~~~~~~~

There are four main components of a model:

1. :ref:`Datasets <_base_dataset:Datasets>`
2. Preprocessing on the dataset from step 1
3. Algorithm
4. Optimization
5. Configuration that allows easy experimentation

Let us create a new multi-class classification model:
* Datasets: A JsonL file of records where each record has a document containing multiple sentences containing multiple words (``List[List[str]]``)
* Preprocessing on dataset:
    1. tokenize the document using custom tokenizer
    2. you want document flattened (``List[str]``) so you can directly model full document sequentially
* Algorithm: RNN specifically GRU (ideally this model is not optimal)
* Optimization: Adam optimizer is state of art, so that is the only one supported in library for now.
* Configuration: we will use yaml to configure each aspect of the model

WIP
===

.. code-block:: python

    class CDRJsonLFlatDataset(CDRJsonLDatasetMixin, FlatDatasetTokenizationMixin):
        def __init__(self, max_length: int, **kwargs):
            FlatDatasetTokenizationMixin.__init__(self, max_length)
            CDRJsonLDatasetMixin.__init__(self, **kwargs)


    class FlatClassificationDataset(BaseDatasetMixin):
        """Class that initializes flat documents (no sentences) for train, test and validation.

        :param int max_chunks: maximum number of chunks possible in a document, larger chunks are truncated
        :param int max_chunk_size: maximum size of a chunk
        :param dict truth_configs: configs for resolving labels
        :param kwargs: keyword arguments for :class:`~vayu.datasets.base_dataset.BaseDatasetMixin`
        """

        def __init__(self, max_chunks, max_chunk_size, truth_configs, is_lazy: bool, **kwargs):
            """Maximum length of a document is defined by: max_chunks * max_chunk_size"""
            self.max_length = max_chunks * max_chunk_size
            self.truth_configs = truth_configs
            self.num_classes = 1
            if is_lazy:
                self.data_cls = LazyCDRJsonLFlatDataset
            else:
                self.data_cls = CDRJsonLFlatDataset
            BaseDatasetMixin.__init__(self, **kwargs)

        def _init_data_class(self):
            """Initializes the data class that is going to be used"""
            self.dataset_cls = partial(self.data_cls, tokenizer=self.tokenizer,
                                       truth_configs=self.truth_configs,
                                       max_length=self.max_length)

        def _collate_fn(self, data):
            """
               data: is a list of tuple with (input, length, label)
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


.. code-block:: python

    class BagOfEmbedding(FlatClassificationDataset, CDRLightningMixin):
    """

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

    def forward(self, batch):
        data, labels, lengths = batch
        embedded = self.embedding(data)
        return self.classifier(embedded), labels