import argparse
import json
import logging
import os
from datetime import datetime
from typing import Union, Dict, Optional, Any

import torch
from azureml.core import Run
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

import vayu
from vayu.constants import (VERSION, NAME, DESCRIPTION, ATTRIBUTES, CONFIG,
                            CPT_CODE, CONTACT_METHOD, DATE_TRAINED, MODEL_INFO_FILE)

logger = logging.getLogger(__name__)


class PersistModelProperties(Callback):
    """Exports all of model metadata."""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_train_end(self, trainer, pl_module):
        """ Exports model info required by Luigi excel task"""
        self.export_model_info(pl_module.hparams.model_type, vayu.__version__,
                               "pyTorch model",
                               pl_module.train_dataset.get_cpt_codes(),
                               pl_module.train_dataset.get_contact_methods())

    def export_model_info(self, model_name, version, description, cpt, contact_method):
        """Export model information to a json"""
        model_info_dict = {NAME: model_name,
                           VERSION: version,
                           DESCRIPTION: description,
                           ATTRIBUTES: {CPT_CODE: cpt,
                                        CONTACT_METHOD: contact_method},
                           DATE_TRAINED: datetime.now().strftime("%Y%m%d%H%M%S")
                           }
        path = os.path.join(self.output_dir, MODEL_INFO_FILE)
        with open(path, 'w') as f:
            json.dump(model_info_dict, f)
            logger.info("Model info exported to %s" % path)


class AMLogger(LightningLoggerBase):
    """Logger for azure machine learning"""
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(self, save_dir: str, run: Run = None):
        super().__init__()
        self.save_dir = save_dir
        self._experiment = run
        self._prev_epoch = -1
        self.hparams = {}

    @property
    def log_dir(self) -> str:
        """
        The directory for this run's tensorboard checkpoint. By default, it is named
        ``'version_${self.version}'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def experiment(self) -> Any:
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._prev_epoch != metrics['epoch']:
            self._prev_epoch = metrics.pop('epoch')
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.experiment.log(k, v)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], argparse.Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)

        # store params to output
        self.hparams.update(params)

    @rank_zero_only
    def save(self) -> None:
        super().save()

        # prepare the file path
        hparams_file = os.path.join(self.save_dir, self.NAME_HPARAMS_FILE)

        # save the metatags file
        save_hparams_to_yaml(hparams_file, self.hparams)

    @property
    def name(self) -> str:
        return ''

    @property
    def version(self) -> str:
        return ''
