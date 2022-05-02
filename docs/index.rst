.. Vayu documentation master file, created by
   sphinx-quickstart on Wed Jun  3 19:44:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vayu's documentation!
================================
Vayu_ is a framework built on top of torch_ for training binary classification, named entity recognition (NER) and language models. source_code_

It can be used to (see introduction_guide):
1. Train: train models on multiple GPUs including saving checkpoints, learning rate schedule and early stopping
2. Tune: load trained / checkpointed models to find optimal threshold for target fpr
3. Eval: finally, the optimal threshold can be used to evaluate on held out test dataset

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Start Here

   introduction_guide
   tutorial_create_new_classification_model

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets
   classification_datasets
   language_modeling_datasets
   tokenization_mixins
   train_valid_split

.. toctree::
   :maxdepth: 2
   :caption: Models

   cdr_lightning_mixin
   bag_of_embedding
   cnn
   cnn_rnn
   han
   transformer
   optimal_threshold_finder
   layers

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Scripts

   language_model
   ner
   utils

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Contributing

   checklist

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _torch: https://pytorch.org/docs/stable/index.html
.. _vayu: https://en.wikipedia.org/wiki/Vayu
.. _source_code: https://gitlab.qpidhealth.net/qaas/deep-learning
