r"""
CDRLightningMixin
=================

Mixin class for classification models primarily binary (support for multiple classes in future releases)

"""

import logging
import os
import random
import numpy as np
from abc import abstractmethod, ABC
from argparse import Namespace
from collections import OrderedDict
from functools import partial
from typing import Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from scipy.stats import entropy
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
from transformers import AdamW

from vayu.constants import Split
from vayu.utils import get_top_checkpoint_path

logger = logging.getLogger(__name__)


class CDRLightningMixin(pl.LightningModule, ABC):
    r"""Trainer class for preparing data, training and persisting models

    :param Union[Namespace,DictConfig] hparams: hyperparameters for configuring lightning module
    hparams also end up being saved as yaml
    """

    def __init__(self, hparams: Union[Namespace, DictConfig]):
        super(CDRLightningMixin, self).__init__()
        hparams.model_type = self.__class__.__name__

        # hparams will go away soon https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        if isinstance(hparams, DictConfig):
            self.hparams = Namespace(**OmegaConf.to_container(hparams, resolve=True))
        else:
            # When loading a saved model, hparams is passed in as Namespace object
            self.hparams = hparams

        self.model_type = hparams.model_type
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.logit_squeeze = hparams.logit_squeeze

        self._build_loss()

    @abstractmethod
    def forward(self, batch):
        """To be implemented by subclasses"""

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (reduce LR when necessary) """
        optimizer = self.resolve_optimizer()
        optimizer = optimizer(params=self.resolve_params())

        scheduler = self.resolve_scheduler()
        scheduler = scheduler(optimizer=optimizer)

        return [optimizer], [scheduler]

    def _build_loss(self):
        """Binary cross entropy loss"""
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    def loss(self, logits, labels):
        """Binary cross entropy loss with logits, here logits are passed w/o sigmoid
        BCE with logits is more numerically stable using log-sum-exp trick than naive BCE

        Label Smoothing and Logit Squeezing: A Replacement for Adversarial Training?
        https://arxiv.org/abs/1910.11585
        Basically method suggests taking frobenius norm of mini-batch logits
        """
        return self.bce_with_logits_loss(self.squeeze(logits) +
                                         self.logit_squeeze * torch.norm(logits, p="fro"),
                                         labels)

    def training_step(self, batch, batch_idx):
        logits, y = self.forward(batch)
        loss = self.loss(logits, y)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # torch.cuda.empty_cache()
        tensorboard_logs = {'train_loss': loss}
        return OrderedDict({"loss": loss, "progress_bar": tensorboard_logs, 'log': tensorboard_logs})

    def training_epoch_end(self, outputs: list) -> dict:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}

        # OrderedDict so that logs are printed to stdout in consistent manner
        return OrderedDict({'loss': avg_loss, "progress_bar": tensorboard_logs, 'log': tensorboard_logs})

    def on_epoch_end(self) -> None:
        # Build Active learning set
        if self.current_epoch % self.hparams.active_learning['add_data_every_n_epoch'] == 0 and \
                self.train_dataset.unlabelled_data.shape[0] > 0 and \
                self.current_epoch > self.hparams.active_learning['start_after_epoch'] and \
                self.hparams.active_learning['is_enable']:
            data = self.train_dataset.unlabelled_data
            logger.info("Training set of %s records", self.train_dataset.data.shape[0])
            logger.info("Building Active learning set from a pool of %s records", data.shape[0])
            data_cls = self.data_cls(data=data,
                                     data_split_type=Split.TRAIN,
                                     pct_of_data=8,  # 4 weeks == 1 month
                                     label_smooth=self.label_smooth,
                                     cat_encoding=self.cat_encoding,
                                     tokenizer=self.tokenizer,
                                     max_chunks=self.max_chunks,
                                     max_chunk_size=self.max_chunk_size
                                     )
            ids, probas, features = [], [], []
            with torch.no_grad():
                # self.enable_dropout()  # MCDropout
                for batch in self.dataloader_cls(data_cls):
                    # for _ in range(10):  # MCDropout
                    batch = [batch[0]] + [data.to(self.device) for data in batch[1:]]
                    idxs, logits, labels, feature = self.forward(**dict(batch=batch, is_return_features=True))
                    ids.extend(idxs)
                    probs = torch.sigmoid(self.squeeze(logits)).detach().cpu().tolist()
                    probas.extend(probs)
                    features.extend(feature.detach().cpu().tolist())

            # -------------------------START-------------------------
            # num_samples = round(data.shape[0] * self.current_epoch * 0.01)
            num_samples = round(len(data_cls) * 0.5)
            logger.info("Starting %s acquisition", self.hparams.active_learning['type'])
            if self.hparams.active_learning['type'] == 'random':
                # TODO: stratified random sample
                candidates = random.choices(ids, k=num_samples)
            elif self.hparams.active_learning['type'] == 'uncertainty':
                probas = map(lambda x: x if x > .5 else 1 - x, probas)
                candidates = list(map(lambda x: x[0],
                                      sorted(zip(ids, probas), key=lambda x: x[1])
                                      )
                                  )[:num_samples]
            elif self.hparams.active_learning['type'] == 'entropy':
                candidates = list(map(lambda x: x[0],
                                      sorted(zip(ids, map(lambda prob: entropy([prob, 1 - prob], base=2), probas)),
                                             key=lambda x: x[1], reverse=True)
                                      )
                                  )[:num_samples]
            elif self.hparams.active_learning['type'] == 'k-means':
                n_clusters = self.hparams.active_learning['n_clusters']
                features = np.array(features)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(features)
                euclidean_dists = []
                predictions = kmeans.predict(features)
                for feature in features:
                    for center in kmeans.cluster_centers_:
                        dist = np.linalg.norm(feature - center)
                        euclidean_dists.append(dist)
                # Sample equal number of candidates from each cluster
                samples = zip(ids, predictions, euclidean_dists)
                num_samples_per_cluster = num_samples // n_clusters
                candidates = set()
                for i in range(n_clusters):
                    cluster_samples = list(filter(lambda x: x[1] == i, samples))
                    if cluster_samples:
                        # Choose some samples closest to cluster representing the cluster
                        candidates.update(
                            map(lambda x: x[0],  # Get id key
                                # Sort by euclidean distance from center of cluster
                                sorted(cluster_samples, key=lambda x: x[2])[:round(num_samples_per_cluster * .25)]))

                        # Choose the rest at random from cluster
                        candidates.update(map(lambda x: x[0],  # Get id key
                                              random.choices(list(filter(lambda x: x[0] not in candidates,
                                                                         cluster_samples)),
                                                             k=round(num_samples_per_cluster - len(candidates)))))

            # -------------------------END-------------------------

            logger.info("%s records used out of %s: %s%%", len(candidates), len(ids),
                        round(len(candidates) * 100 / len(ids), 4))

            # Remove duplicate candidates already existing in training set
            # candidates = set(candidates)
            # candidates = candidates.difference(set(self.train_dataset.data[self.id_column].tolist()))

            # Forget past
            # data = data.loc[data[self.id_column].isin(candidates), :].reset_index(drop=True)

            # Remember past
            # # Add candidates to existing training set
            active_dataset = self.train_dataset.data.append(data.loc[data[self.id_column].isin(candidates), :],
                                                            ignore_index=True)

            # # Remove candidates from unlabelled_pool of datasets
            # # # Remove only the ones added to training set
            # unlabelled_pool = data.drop(data[data[self.id_column].isin(candidates)].index).reset_index(drop=True)

            # # # Remove all the samples selected for this round from unlabelled pool
            unlabelled_pool = data.drop(data[data[self.id_column].isin(ids)].index).reset_index(drop=True)
            logger.info("New dataset size: %s and pool size: %s", active_dataset.shape[0], unlabelled_pool.shape[0])

            # Overwrite training and unlabelled_pool datasets
            self.train_dataset = self.data_cls(data=(active_dataset, unlabelled_pool),
                                               data_split_type=Split.TRAIN,
                                               pct_of_data=None,
                                               label_smooth=self.label_smooth,
                                               cat_encoding=self.cat_encoding,
                                               tokenizer=self.tokenizer,
                                               max_chunks=self.max_chunks,
                                               max_chunk_size=self.max_chunk_size)

    def validation_step(self, batch, batch_idx):
        logits, y = self.forward(batch)
        loss = self.loss(logits, y)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tensorboard_logs = {"val_loss": loss}
        return OrderedDict({"val_loss": loss, "progress_bar": tensorboard_logs, 'log': tensorboard_logs})

    def validation_epoch_end(self, outputs: list) -> dict:
        avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        self.logger.experiment.log('val_loss', avg_loss.item())
        if avg_loss < self.hparams.sota_val_loss:
            logger.log("SoTA Loss achieved with a decrease of %s%%",
                       (self.hparams.sota_val_loss - avg_loss) * 100 / self.hparams.sota_val_loss)
        # OrderedDict so that logs are printed to stdout in consistent manner
        return OrderedDict({'val_loss': avg_loss, "progress_bar": tensorboard_logs, 'log': tensorboard_logs})

    def test_step(self, batch, batch_idx):
        logits, y = self.forward(batch)
        loss = self.loss(logits, y)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        return OrderedDict({'loss': loss})

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': test_loss_mean}

        result = {
            "test_loss": test_loss_mean,
            "progress_bar": tensorboard_logs,
            "log": tensorboard_logs
        }
        return OrderedDict(result)

    @staticmethod
    def squeeze(logits):
        """ Handle edge case where squeeze() removes all elements of a tensor shape [1,1] -> scalar
        which makes operations like .item() not work
        """
        if len(logits) == 1:
            return logits.squeeze(0)
        else:
            return logits.squeeze()

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config = OmegaConf.load(os.path.join(pretrained_model_path, ".hydra", "config.yaml"))
        config.training.hparams.merge_with(config.model.params, config.data)
        model = cls(config.training.hparams, **config.training.hparams)
        loss, model_path = get_top_checkpoint_path(pretrained_model_path)
        model.load_from_checkpoint(model_path, **config.training.hparams)

        return model

    def resolve_optimizer(self):
        if self.hparams.optimizer['type'] == 'Adam':
            optimizer = partial(AdamW, **self.hparams.optimizer['params'])
        else:
            raise NotImplementedError("Only Adam is supported at this time")

        return optimizer

    def resolve_scheduler(self):
        if self.hparams.lr_scheduler['type'] == 'ReduceLROnPlateau':
            scheduler = partial(ReduceLROnPlateau, **self.hparams.lr_scheduler['params'])
        elif self.hparams.lr_scheduler['type'] == 'StepLR':
            scheduler = partial(StepLR, **self.hparams.lr_scheduler['params'])
        elif self.hparams.lr_scheduler['type'] == 'ExponentialLR':
            scheduler = partial(ExponentialLR, **self.hparams.lr_scheduler['params'])
        else:
            raise NotImplementedError("Only ReduceLROnPlateau/ExponentialLR is supported at this time")

        return scheduler

    def resolve_params(self):
        return filter(lambda param: param.requires_grad, self.parameters())

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
