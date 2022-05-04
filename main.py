import logging
import os
import shutil

import hydra
import pytorch_lightning as pl
import torch
from azureml.core import Run
from hydra import utils
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.backends import cudnn
from torch.utils.data import SequentialSampler, DataLoader

from vayu.callbacks import PersistModelProperties, AMLogger
from vayu.constants import Split, THRESHOLDS_FILE
from vayu.models.cdr_lightning_mixin import CDRLightningMixin
from vayu.models.classification.optimal_threshold_finder import OptimalThresholdFinder
from vayu.score_calculations import ScoreCalculations
from vayu.utils import get_top_checkpoint_path

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

logger = logging.getLogger(__name__)


def train(model: CDRLightningMixin, cfg: dict, run: Run) -> None:
    """
    Main training routine specific for this project
    :param model: child of CDRLightningMixin
    :param cfg: dictionary of hyperparameters
    :param run: azureml run object
    """
    # ------------------------
    # INIT CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(**cfg['callbacks']['early_stop'])

    # ------------------------
    # INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(**cfg['pl_trainer'],
                         early_stop_callback=early_stop_callback if cfg['is_early_stop'] else None,
                         resume_from_checkpoint=cfg['pl_trainer']['checkpoint_path'] or None,
                         callbacks=[PersistModelProperties(cfg['output_dir'])],
                         logger=AMLogger(cfg['output_dir'], run),
                         reload_dataloaders_every_epoch=cfg['data']['active_learning']['is_enable']
                         )

    # --------------------------------
    # INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    ckpt_path = os.path.join(
        trainer.default_save_path,
        "checkpoints-{epoch:03d}-{val_loss:.3f}-{loss:.3f}",
    )
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        **cfg['callbacks']['checkpoint']
    )
    trainer.checkpoint_callback = checkpoint_callback

    # ------------------------
    # START TRAINING
    # ------------------------
    trainer.fit(model)

    # ------------------------
    # START TUNING
    # ------------------------
    dataloader = model.train_dataloader() if cfg['is_calibrate_on_train'] else model.val_dataloader()
    # dataloader = read_data_from_local_file(model, cfg['data']['train_path'], Split.TRAIN)
    # Load best model from checkpoint
    loss, model_path = get_top_checkpoint_path(cfg['output_dir'])
    logger.info("Loading model from %s", model_path)
    model = model.load_from_checkpoint(model_path,
                                       hparams_file=os.path.join(trainer.weights_save_path, "hparams.yaml"),
                                       map_location=torch.device('cpu') if cfg['n_gpus'] == 0 else device,
                                       **cfg['training']['hparams'])

    calibration_set_type = Split.TRAIN if cfg['is_calibrate_on_train'] else Split.VALID
    ids, predictions, targets, features = predict(model, dataloader, cfg['n_gpus'],
                                                  calibration_set_type.value + "_calibration_set")

    otf = OptimalThresholdFinder(**cfg['callbacks']['threshold'])
    otf.calculate_thresholds(predictions, targets)
    otf.export_thresholds(output_dir=cfg['output_dir'])

    scores = ScoreCalculations(predictions, targets, otf, calibration_set_type)
    scores.export_metrics_to_yaml(cfg['output_dir'])
    scores.export_predicted_probas_to_csv(cfg['output_dir'], ids=ids, features=features,
                                          is_export_features=cfg['is_export_features'])

    # ------------------------
    # START EVALUATING
    # ------------------------
    evaluate(model, cfg['data']['test_path'], cfg['output_dir'], otf, cfg['n_gpus'], run, cfg['is_export_features'])


def evaluate(model, test_data_path: str, test_output_dir: str,
             otf: OptimalThresholdFinder, n_gpus: int, run: Run, is_export_features: bool) -> None:
    logger.info("Starting evaluation")

    ids, predictions, targets, features = [], [], [], []

    # If single file is passed, the individual eval is already complete
    if os.path.isdir(test_data_path):
        # The following code outputs a yaml and predictions csv in a directory
        if len(os.listdir(test_data_path)) > 0:
            individual_evals_out_dir_path = os.path.join(test_output_dir, "individual_evals")
            os.mkdir(individual_evals_out_dir_path)
            for eval_jsonl_file_name in os.listdir(test_data_path):
                eval_jsonl_path = os.path.join(test_data_path, eval_jsonl_file_name)
                logger.info("Starting evaluation for %s", eval_jsonl_path)
                file_name, extension = os.path.splitext(eval_jsonl_file_name)

                dataloader = read_data_from_local_file(model, eval_jsonl_path, Split.TEST)
                ids_indiv, predictions_indiv, targets_indiv, features_indiv = predict(model,
                                                                                      dataloader,
                                                                                      n_gpus, file_name)
                scores = ScoreCalculations(predictions_indiv, targets_indiv, otf, Split.TEST)

                scores.export_metrics_to_yaml(individual_evals_out_dir_path, file_name + "_")
                scores.export_predicted_probas_to_csv(individual_evals_out_dir_path, ids_indiv,
                                                      features_indiv, file_name + "_", is_export_features)

                ids.extend(ids_indiv)
                predictions.extend(predictions_indiv)
                targets.extend(targets_indiv)
                features.extend(features_indiv)
        else:
            raise ValueError(f"{test_data_path} is empty, please pass a directory"
                             "that contains jsonl files for evaluating individually.")
    else:
        test_dataloader = read_data_from_local_file(model, test_data_path, Split.TEST)
        ids, predictions, targets, features = predict(model, test_dataloader, n_gpus,
                                                      os.path.basename(test_data_path))

    scores = ScoreCalculations(predictions, targets, otf, Split.TEST)
    scores.export_metrics_to_yaml(test_output_dir)
    scores.export_predicted_probas_to_csv(test_output_dir, ids=ids, features=features,
                                          is_export_features=is_export_features)
    scores.log_metrics_to_aml(run)


def read_data_from_local_file(model, data: str, data_split_type: Split):
    dataset = model.dataset_cls(data=data, data_split_type=data_split_type,
                                cat_encoding=model.cat_encoding)
    dataloader = model.dataloader_cls(dataset, sampler=SequentialSampler(dataset))

    return dataloader


def predict(model, dataloader: DataLoader, n_gpus: int, tqdm_desc: str):
    if n_gpus != 0:
        model.cuda()
    model.freeze()
    # multi-gpu evaluate
    if n_gpus > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.eval()

    ids, predictions, targets, features = [], [], [], []
    # for batch in tqdm(dataloader, desc=tqdm_desc, miniters=round(0.1 * len(dataloader))):
    logger.info("Predicting on %s", tqdm_desc)
    for batch in dataloader:
        batch = [batch[0]] + [data.to(device) for data in batch[1:]]
        idxs, logits, labels, encodings = model(**dict(batch=batch, is_return_features=True))
        ids.extend(idxs)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(torch.sigmoid(CDRLightningMixin.squeeze(logits)).detach().cpu().tolist())
        features.extend([','.join(map(str, encoding)) for encoding in encodings.detach().cpu().tolist()])

    return ids, predictions, targets, features


def get_dataset(input_datasets: dict, dataset_type: str) -> str:
    """Fetches the local path of dataset from all the datasets in AML run

    Args:
        input_datasets (dict): A dictionary of input datasets from Azure run
        dataset_type (str): The type of dataset (either of train, tune or test)

    Returns:
        str: The path of dataset in local filesystem
    """
    for dataset_name, dataset_path in input_datasets.items():
        if dataset_type in dataset_name:
            return dataset_path
    raise ValueError(f"{dataset_type} not found in datasets")


@hydra.main(config_path="conf/config.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    # So that model, lightning and hydra logs are written to the same place
    cfg.output_dir = os.getcwd()
    cfg.n_gpus = torch.cuda.device_count()
    seed_everything(cfg.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        cfg.n_gpus,
        cfg.use_fp16
    )

    # ------------------------
    # DATA LOADING
    # if not os.path.exists(cfg.data.val_path):  # TODO: data validation checks
    #    raise FileNotFoundError(f"No validation set found at {cfg.data.val_path}")
    # ------------------------
    # Get the experiment run context
    run = Run.get_context(allow_offline=True)
    if cfg.is_azure_train:
        cfg.data.train_path = get_dataset(run.input_datasets, Split.TRAIN.value)
        cfg.data.valid_path = get_dataset(run.input_datasets, Split.VALID.value)
        cfg.data.test_path = get_dataset(run.input_datasets, Split.TEST.value)
        cfg.data.tokenizer_path = utils.to_absolute_path(cfg.data.tokenizer_path)
        if 'pretrained_vector_path' in cfg.model.params and cfg.model.params.pretrained_vector_path != '':
                cfg.model.params.pretrained_vector_path = get_dataset(run.input_datasets, Split.EMBEDDING.value)

        if cfg.data.cat_encoding:
            for key in cfg.data.cat_encoding:
                path = cfg.data.cat_encoding[key]
                cfg.data.cat_encoding[key] = get_dataset(run.input_datasets, path)

    if 'pretrained_vector_path' in cfg.model.params and cfg.model.params.pretrained_vector_path != '':
        shutil.copytree(cfg.model.params.pretrained_vector_path,
                        os.path.join(cfg.output_dir, Split.EMBEDDING.value))
        cfg.model.params.pretrained_vector_path = os.path.join(cfg.output_dir, Split.EMBEDDING.value)

    if cfg.pretty_print:
        print(cfg.pretty())
    # ------------------------
    # INIT MODEL
    # ------------------------
    cfg.training.hparams.merge_with(cfg.model.params, cfg.data)

    if cfg.do_train:
        model = hydra.utils.instantiate(cfg.model,
                                        **cfg.training,  # redundant but required by lightning module
                                        **cfg.training.hparams)
        train(model, OmegaConf.to_container(cfg, resolve=True), run)
    elif cfg.do_eval_pretrained_model:
        # ------------------------
        # LOAD MODEL
        # Load model config
        # Load best model from checkpoints with lowest validation loss
        # ------------------------
        cfg.pretrained_model_path = utils.to_absolute_path(cfg.pretrained_model_path)
        # TODO: add ability to evaluate multiple checkpoints
        loss, model_path = get_top_checkpoint_path(cfg.pretrained_model_path)
        logger.info("Loading model from %s", model_path)

        pretrained_config_path = os.path.join(cfg.pretrained_model_path, '.hydra/config.yaml')
        if os.path.exists(pretrained_config_path):
            # TODO: this part needs an integration test so bad
            pretrained_config = OmegaConf.load(pretrained_config_path)
            pretrained_config.training.hparams.merge_with(pretrained_config.model.params,
                                                          pretrained_config.data)
            model = hydra.utils.instantiate(pretrained_config.model,
                                            **pretrained_config.training,  # redundant but required by lightning module
                                            **pretrained_config.training.hparams)
            model = model.load_from_checkpoint(model_path,
                                               map_location=torch.device('cpu') if
                                               cfg.n_gpus == 0 else None,
                                               **pretrained_config.training.hparams)
            # This is so we can support command line argument overrides
            cfg = OmegaConf.merge(pretrained_config, cfg)
        else:
            raise FileNotFoundError("No config file found at `.hydra/config.yaml`")

        # ----------------------------------------
        # START TUNING
        # Load threshold file if present else tune
        # ----------------------------------------
        logger.info("Starting tuning")
        if os.path.exists(os.path.join(cfg.pretrained_model_path, THRESHOLDS_FILE)):
            otf = OptimalThresholdFinder.from_precomputed_thresholds(cfg.pretrained_model_path)
        else:
            if not os.path.exists(cfg.data.val_path):
                raise FileNotFoundError(f"No validation set found at {cfg.data.val_path}")

            predictions, targets = predict(model, cfg.data.val_path, cfg.n_gpus, Split.VALID.value)
            otf = OptimalThresholdFinder(**cfg.callbacks.threshold)
            otf.calculate_thresholds(predictions, targets)
            otf.export_thresholds(output_dir=cfg.output_dir)

        # ------------------------------------------------
        # START EVALUATING
        # Write predictions to a csv and metrics to a yaml
        # ------------------------------------------------
        evaluate(model, cfg.data.test_path, cfg.output_dir, otf, cfg.n_gpus, run, is_export_features=True)


def set_environment_variables_for_nccl_backend(single_node=False, master_port=6105):
    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]

        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"

    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NODE_RANK"] = os.environ[
        "OMPI_COMM_WORLD_RANK"
    ]  # node rank is the world_rank from mpi run


if __name__ == "__main__":
    # set_environment_variables_for_nccl_backend(single_node=False)
    main()
