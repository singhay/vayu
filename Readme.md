<div align="center">

![Logo](docs/_images/logos/logo_100x100.png)

# [Vayu](https://en.wikipedia.org/wiki/Vayu)

** [Vayu](https://en.wikipedia.org/wiki/Vayu) is a framework built on top of pytorch for training binary classification models. **

[![pipeline status](https://gitlab.qpidhealth.net/qaas/deep-learning/badges/master/pipeline.svg)](https://gitlab.qpidhealth.net/qaas/deep-learning/-/commits/master)
[![coverage report](https://gitlab.qpidhealth.net/qaas/deep-learning/badges/master/coverage.svg)](https://gitlab.qpidhealth.net/qaas/deep-learning/-/commits/master)

</div>




 
 Support for named entity
recognition (NER), language and knowledge graph models is planned for future releases which are now standalone scripts.


### Docs
- [latest](http://10.205.63.29/docs/vayu/latest) 
- [stable](http://10.205.63.29/docs/vayu/stable)
- [0.1.1](http://10.205.63.29/docs/vayu/0.1.1)
- [0.1.0](http://10.205.63.29/docs/vayu/0.1.0)


### Requirements

This project requires Python 3.6+

Create a conda environment (preferred)
```bash
conda env create --file conda_requirements.yml
conda activate vayu
```


### Data
* JsonL : One line per record in json format exported using following scala script using spark
* Having metadata columns like `cptCode` and `contactMethod` is helpful if present to export `model_info_advanced.json`
* The tokens column is `List[List[str]]` / Doc[sentences] where doc is List[sentence], sentence is `List[str]`
* Datasets should already be split into train, validation and test in (BIO CoNLL format for NER) for the scripts to
 ingest. `vayu/datasets/train_valid_split.py` can be used to split datasets.
* Having a static validation and test set instead of random is important for measure performance of new models
 compared to earlier ones. 

### Train

`python -m main do_train=true`
1. Training flow by default trains -> tunes -> evaluates
2. By default, training gets configured using yaml situated in `conf/config.yaml`. 
3. To override any aspect of training, either pass via cmdline e.g. `python -m main training.optimizer_learning_rate=0.1 model=classification/cnn_lstm model.params.embedding_size=10`
4. Configs are composed from multiple hierarchical small config files along with their defaults. This allows highly
 flexible configuration e.g. having same training, data params but different params for different models.
5. After training, system outputs the path of trained model which consists of: 


    a. complete config.yaml
    b. lightning_logs/version_0 will have by default top 3 models found during training based on validation loss.
    c. model_info_advanced.json, thresholds.json, test_results.yaml, tune_results.yaml as well as train.log which
     comprises of stdout of training.

### Hyperparameter sweeps
```
python -m main do_train=true --multirun \ 
model.params.embedding_size=10,50,100 \ 
hydra.sweep.dir=multiruns/embedding_effect \ 
hydra.sweep.subdir=\${model.embedding_size}
```
Above will create separate directories for each embedding_size, this saves developer effort in running multiple configs.
     
### Resume training from checkpoint
```
python -m main do_train=true \
model=classification/boe \
pl_trainer.checkpoint_path=path/to/checkpoint
```

### Evaluate a pretrained model
You can evaluate on a directory of json files or single file
* This assumes your config is present in `outputs/2020-05-27/16-28-42/.hydra/config.yaml`
* If you want to evaluate model on a separate threshold, simply update `threshold` param in `thresholds.json`
* If you want to tune again on different dataset:
    1. backup and delete `thresholds.json` in pretrained model dir.
    2. If `thresholds.json` is not deleted, then it will be used for prediction.
    3. New threshold will be exported to a new directory to avoid overwriting.
```bash
python -m main do_eval=true do_train=false \ 
pretrained_model_path=outputs/2020-05-27/16-28-42 \
data.test_path=path \  # path can be dir of files or single json file
data.val_path=path  # if you want to tune on a new dataset
```



### Tensorboard
Here you can compare experiments 
```bash
tensorboard --logdir="outputs"
```

### Todo
* [x] Does not support batch size > 1 when training throws OOM for a bigger batch size, if you lower max_chunks and
 chunk_size which will support larger batch size
    * [x] Custom collate functions for Dataloader
* [ ] Categorical data featurization (pytorch-tabnet)
* [x] Support for streaming json datasets loading w/o blowing up memory
* [x] Support for loading multiple json datasets
* [x] Training takes evaluation set as param from beginning, which is a limitation for now since we would want to evaluate on any dataset post training. Workaround for that is load model manually in an interpreter and run custom evaluate dataset through it
* [ ] Imbalanced batch sampler
 
### Milestones
- [x] CNN + classification head
- [x] CNN + LSTM + classification head
- [x] HAN + classification head
- [ ] CNN + HAN + classification head
- [x] Support gensim pretrained vectors (w2v, d2v, fastText and glove)
- [x] Transformers + Classification Head
- [ ] CDR custom tokenizer
- [ ] Hybrid CDR tokenizer that leverages a whitelist along with BPE/wordpiece tokenization

### Core libraries
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Hydra](https://hydra.cc)
- [Transformers](https://huggingface.co/transformers/index.html)
