"""Integration tests are WIP because compose API is still experimental"""


def test_training(tmpdir):
    # properties = tmpdir.join("applicationEvaluationFake.yml")
    # properties.write("Some fake properties in here")
    # main_wrapper = hydra.main("./vayu/tests/resources/model/.hydra", strict=False)
    # main_wrapper(main)()
    pass
    # initialize(config_dir="./vayu/tests/resources/model/.hydra", strict=False)
    # cfg = compose("config.yaml", overrides=[])
    # main(cfg)
    # directory named by day of test run/time of run
    #   main.log
    #   tune.yaml
    #   test.yaml
    #   thresholds.json
    #   model_info_advanced.json
    # directory named lightning_logs/version_0 with
    #   three checkpoints
    #   hparams.yaml
    #   events file for tensorboard
    # .hydra directory
    #   config.yaml
    #   hydra.yaml
    #   overrides.yaml


def test_tuning():
    """Test if system can tune (generate thresholds) for a trained model"""
    pass


def test_evaluation():
    """Test whether we can load a pretrained model and evaluate on a dataset"""
    pass


def test_resume_from_checkpoint():
    """Test whether we can resume training a model from a checkpoint"""
    pass


def test_pretraining():
    """Test whether we can load an already trained model and fine tune it on given task"""
    pass

