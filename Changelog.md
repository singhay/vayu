# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[v0.2.0] - 11/11/2020
---
### Added
* Scripts to register results of any model as a run in any AML experiment [source MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/21)
* Ability to load embeddings using azure ml dataset [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Calibration on training set by default [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Log metrics to Azure ML [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Output calibration set predictions on optimal threshold [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* AML configuration and scripts to train models on azure
* Experiment tags [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* RunID in slack message upon experiment completion [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)

### Changed
* Copy embedding to model dir on serialization [!1](https://inqbator-gitlab.innovate.lan/singhay/vayu/-/merge_requests/1)
* Calibration tqdm update progress bar every 1/10th time [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Avoid uploading accuracy tables because they don't load in UI [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Moved aml.docker to respective cpu, gpu configs [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Renamed `val_path` to `valid_path` [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)

### Removed
* Gensim dependency, instead use numpy matrix to load pretrained embeddings [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)
* Unwanted nested checkpoint saving dirs `lightning_logs/version_0` [AML177 MR](https://gitlab.qpidhealth.net/qaas/vayu/-/merge_requests/20)

[v0.1.1] - 07/15/2020
---
### Added
- Label smoothing
- Label squeezing
- Add new model that uses 1d convolution instead of 2d increasing performance by 46% and reduced training time from 3h to 2h for 360k records.
- Auto build sphinx docs and push to web server using CI
- Add gitlab badges for test coverage

### Changed
- Make tuning on training set default instead of tuning set
- Moved tests directory outside of source directory
- Moved one-off scripts into their own "scripts" module from "vayu"

### Removed
- Configurable truth configs because they are detrimental to performance
- Task yaml because it was not needed
- requirements.txt

