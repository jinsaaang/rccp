# Configs

Canonical benchmark configs live here.

The active entrypoints in `scripts/benchmark_core` read `config/default_config.yaml`
and `config/default_fclstm_train.yaml` directly. Dataset, forecaster,
uncertainty-method, task, and evaluation groups are kept as flat subdirectories
under this folder.

## Layout

- `dataset/`: public dataset descriptors. Data files are not tracked.
- `model_fc/`: shared forecaster configs, currently global GRU/RNN, global transformer decoder, and ResCP-native backbones.
- `model_uc/`: conformal methods and adaptive baselines.
- `task/`: chronological split/evaluation task configs.
- `evaluation/`: output metric/plot settings.

## Main Methods

The paper-grid runner supports:

- `conf_default`
- `enbpi`
- `spic`
- `nextcp`
- `hopcpt`
- `rescp`
- `ct_ssf`
- `agaci`
- `dtaci`
- `rccp`

Method-specific hyperparameter grids live in `scripts/benchmarks/run_paper_grid.py`.
SPCI/SPIC follows the paper/default setting with `past_window_len=100` and the
default quantile forest depth.
