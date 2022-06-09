import numpy as np
import pytest
from omegaconf import DictConfig

from vayu.datasets import assign_label, NEGATIVE_LABEL, POSITIVE_LABEL
from vayu.utils import get_top_checkpoint_path
from vayu.utils import stringify
from datetime import datetime


def test_stringify_numpy():
    np_arr = np.ones((1, 2))
    assert isinstance(stringify(np_arr), list)
    assert isinstance(stringify(np.ones((1, 1), dtype='float')), list)


def test_stringify_datetime():
    dt = datetime.now()
    assert stringify(dt) == str(dt)


def test_get_top_checkpoint_path():
    top_model_path = './tests/resources/model/checkpoints-epoch=000-val_loss=0.726-loss=0.737.ckpt'
    val_loss, model_path = get_top_checkpoint_path("./tests/resources/model")
    assert model_path == top_model_path


def test_get_top_checkpoint_path_no_model_found():
    with pytest.raises(FileNotFoundError, match="No models found in ./tests/resources/"):
        _, _ = get_top_checkpoint_path("./tests/resources/")


def test_assign_labels():
    row = DictConfig({'authStatus': 'A'})
    assert assign_label(row) == POSITIVE_LABEL
    row = DictConfig({'authStatus': 'other'})
    assert assign_label(row) == NEGATIVE_LABEL


def test_assign_labels_label_smoothing():
    row = DictConfig({'authStatus': 'A'})
    assert assign_label(row, label_smooth=0.1) == 0.95
    assert assign_label(row, label_smooth=0.2) == 0.90

    row = DictConfig({'authStatus': 'other'})
    assert assign_label(row, label_smooth=0.1) == 0.05
    assert assign_label(row, label_smooth=0.2) == 0.1
