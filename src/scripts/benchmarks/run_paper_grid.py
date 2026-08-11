"""Paper-style HP search and final evaluation.

This is the ``jh`` validation-search flow adapted for the current codebase:

1. Train the shared forecaster once per dataset / architecture / size / seed.
2. Run HP search on the first seed only, using a held-out validation slice.
3. Select HP by validation Winkler score.
4. Re-run the final test for all requested seeds with the first-seed HP.

Final-table defaults are:
    dataset: air
    forecasters: global_lstm, global_transformer_decoder
    sizes: lstm128, transformer128_256
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "benchmark_core" / "run_internal_benchmark.py"
TRAIN_FC = REPO_ROOT / "scripts" / "benchmark_core" / "train_lstm_forecaster.py"
RCCP_FAST_RUNNER = REPO_ROOT / "scripts" / "benchmark_core" / "run_rccp_fast_from_artifacts.py"
DATA_ROOT = REPO_ROOT / "data" / "source"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
GENERATED_FC_ROOT = REPO_ROOT / "models_save" / "lstm_fc"
DEFAULT_RCCP_FAST_ONLINE_BLOCK_SIZE = 1

DEFAULT_DATASETS = ["air"]
DEFAULT_METHODS = [
    "conf_default",
    "enbpi",
    "spic",
    "nextcp",
    "hopcpt",
    "rescp",
    "ct_ssf",
    "rccp",
    "agaci",
    "dtaci",
]
SUPPORTED_FORECASTERS = ["global_lstm", "global_rnn", "global_transformer_decoder"]
DEFAULT_FORECASTERS = ["global_lstm", "global_transformer_decoder"]
DEFAULT_FORECASTER_SIZES = {
    "global_lstm": [(128, None)],
    "global_rnn": [(32, None)],
    "global_transformer_decoder": [(128, 256)],
}
DEFAULT_SIZES = [32, 64, 128, 256]
DEFAULT_TRANSFORMER_FF_SIZE = 256

DATASET_CONFIGS = {
    "air": ("bejing_air_pm10", "default_3year_gn"),
    "air10": ("bejing_air_pm10", "default_3year_gn"),
    "beijing": ("bejing_air_pm10", "default_3year_gn"),
    "bejing": ("bejing_air_pm10", "default_3year_gn"),
    "air_rescp": ("rescp_bejing_air_pm10", "default_3year_gn"),
    "air_rescp_repro": ("rescp_bejing_air_pm10", "rescp_beijing_repro"),
    "air_rescp_paper": ("rescp_bejing_air_pm10", "rescp_beijing_paper_eval"),
    "beijing_rescp": ("rescp_bejing_air_pm10", "default_3year_gn"),
    "bejing_rescp": ("rescp_bejing_air_pm10", "default_3year_gn"),
    "solar": ("nsdb2018-20_60m", "default_3year_gn"),
    "solar3y": ("nsdb2018-20_60m", "default_3year_gn"),
    "wind": ("wind", "default_3year_gn"),
    "weather": ("weather", "default_3year_gn"),
    "amazon": ("amazon", "default_3year_gn"),
    "stock": ("amazon", "default_3year_gn"),
    "amzn": ("amazon", "default_3year_gn"),
    "apple": ("apple", "default_3year_gn"),
    "aapl": ("apple", "default_3year_gn"),
    "google": ("google", "default_3year_gn"),
    "goog": ("google", "default_3year_gn"),
    "elec": ("elec_normalized", "default_3year_gn"),
    "elec_norm": ("elec_normalized", "default_3year_gn"),
    "elec_normalized": ("elec_normalized", "default_3year_gn"),
    "electricity": ("elec_normalized", "default_3year_gn"),
    "electricity_normalized": ("elec_normalized", "default_3year_gn"),
    "exchange": ("exchange_rate", "default_3year_gn"),
    "exchange_rate": ("exchange_rate", "default_3year_gn"),
    "exchange_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
    "exchange_ot1_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
}

SEARCH_METHODS = {"enbpi", "nextcp", "hopcpt", "rescp", "rccp", "agaci", "dtaci"}

HOPCPT_NIPS23_GLOBAL_LSTM_OVERRIDES = {
    # methods/upstream/hopcpt/experiments_neurips23.md, global_lstm_* blocks.
    "air": [
        "config.model_uc.batch_size=2",
        "config.model_uc.ctx_encode_dropout=null",
        "config.trainer.trainer_config.optim.lr=0.001",
    ],
    "air10": [
        "config.model_uc.batch_size=2",
        "config.model_uc.ctx_encode_dropout=null",
        "config.trainer.trainer_config.optim.lr=0.001",
    ],
    "beijing": [
        "config.model_uc.batch_size=2",
        "config.model_uc.ctx_encode_dropout=null",
        "config.trainer.trainer_config.optim.lr=0.001",
    ],
    "bejing": [
        "config.model_uc.batch_size=2",
        "config.model_uc.ctx_encode_dropout=null",
        "config.trainer.trainer_config.optim.lr=0.001",
    ],
    "solar": [
        "config.trainer.trainer_config.optim.lr=0.01",
        "config.model_uc.pos_encode.mode=rel-simple",
    ],
    "solar3y": [
        "config.trainer.trainer_config.optim.lr=0.01",
        "config.model_uc.pos_encode.mode=rel-simple",
    ],
}

RESCP_RECOMMENDED_OVERRIDES = {
    # Paper/native ResCP RNN settings from upstream configs:
    # configs/conformal_predictor/rexcp_sampling_{beijing,solar}_rnn.yaml
    # configs/sampler/{beijing,solar}_rnn.yaml
    "air": [
        "config.model_uc.spectral_radius=1.3",
        "config.model_uc.leaking_rate=0.95",
        "config.model_uc.input_scaling=0.25",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3200",
    ],
    "air10": [
        "config.model_uc.spectral_radius=1.3",
        "config.model_uc.leaking_rate=0.95",
        "config.model_uc.input_scaling=0.25",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3200",
    ],
    "beijing": [
        "config.model_uc.spectral_radius=1.3",
        "config.model_uc.leaking_rate=0.95",
        "config.model_uc.input_scaling=0.25",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3200",
    ],
    "bejing": [
        "config.model_uc.spectral_radius=1.3",
        "config.model_uc.leaking_rate=0.95",
        "config.model_uc.input_scaling=0.25",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3200",
    ],
    "solar": [
        "config.model_uc.spectral_radius=0.9",
        "config.model_uc.leaking_rate=0.75",
        "config.model_uc.input_scaling=0.7",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3900",
    ],
    "solar3y": [
        "config.model_uc.spectral_radius=0.9",
        "config.model_uc.leaking_rate=0.75",
        "config.model_uc.input_scaling=0.7",
        "config.model_uc.temperature=0.1",
        "config.model_uc.sliding_window=3900",
    ],
}


@dataclass
class GridSpec:
    axes: List[tuple[str, Sequence]] = field(default_factory=list)

    def points(self):
        if not self.axes:
            yield [], "default"
            return
        keys = [k for k, _ in self.axes]
        values = [list(v) for _, v in self.axes]
        for combo in itertools.product(*values):
            overrides = []
            tag_parts = []
            for key, value in zip(keys, combo):
                override_value = "null" if value is None else value
                overrides.append(f"config.{key}={override_value}")
                tag_parts.append(_safe_token(key, value))
            yield overrides, "__".join(tag_parts)


def _safe_token(key: str, value) -> str:
    short_key = key.split(".")[-1]
    short_key = {
        "gamma_grid_name": "ggrid",
        "past_window_len": "win",
        "ctx_encode_dropout": "dropout",
        "sliding_window": "window",
        "spectral_radius": "rho",
        "leaking_rate": "leak",
        "input_scaling": "scale",
        "max_depth": "depth",
        "temperature": "temp",
        "ewma_beta": "ewma",
        "correction_floor": "cfloor",
        "base_key_source": "key",
        "similarity_metric": "sim",
        "use_target_summary": "ysum",
        "use_residual_context": "rctx",
    }.get(short_key, short_key)
    if value is None:
        value_token = "null"
    elif isinstance(value, bool):
        value_token = "t" if value else "f"
    elif isinstance(value, float):
        value_token = f"{value:g}".replace(".", "p").replace("-", "neg")
    else:
        value_token = {
            "paper": "p",
            "wide": "w",
            "agaci_figure_fast": "fast",
            "agaci_figure_extended": "ext",
        }.get(str(value), str(value)).replace(".", "p")
    return f"{short_key}-{value_token}"


def _dedupe_overrides(overrides: Sequence[str]) -> List[str]:
    """Keep one Hydra override per key, with later overrides winning."""
    by_key: Dict[str, str] = {}
    order: List[str] = []
    for override in overrides:
        key = override.split("=", 1)[0]
        if key not in by_key:
            order.append(key)
        by_key[key] = override
    return [by_key[key] for key in order]


def _rccp_fast_cli_from_overrides(overrides: Sequence[str]) -> List[str]:
    option_names = {
        "k": "--k",
        "temperature": "--temperature",
        "ewma_beta": "--ewma-beta",
        "correction_floor": "--correction-floor",
        "correction_mode": "--correction-mode",
        "base_key_source": "--base-key-source",
        "similarity_metric": "--similarity-metric",
        "min_memory": "--min-memory",
        "proposal_min_radius": "--proposal-min-radius",
        "ewma_min_scale": "--ewma-min-scale",
        "state_save_overhead_sec": "--state-save-overhead-sec",
    }
    cli: List[str] = []
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key.startswith("config."):
            key = key[len("config.") :]
        if key == "task.data_splits":
            cli.extend(["--data-splits", value])
            continue
        if not key.startswith("model_uc."):
            continue
        model_key = key[len("model_uc.") :]
        if model_key == "include_state_save_overhead":
            if str(value).lower() in {"1", "true", "yes", "on"}:
                cli.append("--include-state-save-overhead")
            elif str(value).lower() in {"0", "false", "no", "off"}:
                cli.append("--no-include-state-save-overhead")
            continue
        if model_key == "location_batch":
            if str(value).lower() in {"1", "true", "yes", "on"}:
                cli.append("--location-batch")
            elif str(value).lower() in {"0", "false", "no", "off"}:
                cli.append("--no-location-batch")
            continue
        option = option_names.get(model_key)
        if option is not None:
            cli.extend([option, value])
    return cli


GRIDS: Dict[str, GridSpec] = {
    "conf_default": GridSpec(),
    "enbpi": GridSpec(axes=[
        ("model_uc.past_window_len", [200, 150, 100, 50, 10]),
    ]),
    "spic": GridSpec(),
    "nextcp": GridSpec(axes=[
        ("model_uc.rho", [0.999, 0.99, 0.95, 0.9]),
    ]),
    "hopcpt": GridSpec(axes=[
        ("trainer.trainer_config.optim.lr", [0.01, 0.001]),
        ("model_uc.ctx_encode_dropout", [0.3, 0.0]),
        ("model_uc.pos_encode.mode", ["rel-simple", None]),
    ]),
    "rescp": GridSpec(axes=[
        ("model_uc.spectral_radius", [0.5, 0.75, 0.9, 0.99, 1.1, 1.5]),
        ("model_uc.leaking_rate", [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]),
        ("model_uc.input_scaling", [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]),
        ("model_uc.temperature", [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]),
        ("model_uc.sliding_window", [250, 500, 1000, 2000, 4000, 8000, "all"]),
    ]),
    # Keep final-table CT-SSF fixed-parameter and search-free. inv_lr=0.001
    # and inv_step=80 are config defaults, not search axes.
    "ct_ssf": GridSpec(),
    "rccp": GridSpec(axes=[
        ("model_uc.k", [32, 64, 128]),
        ("model_uc.temperature", [0.75, 1.0, 1.25]),
        ("model_uc.ewma_beta", [0.9, 0.95, 0.99]),
    ]),
    "agaci": GridSpec(axes=[
        ("model_uc.gamma_grid_name", ["paper", "agaci_figure_fast", "agaci_figure_extended", "wide"]),
    ]),
    "dtaci": GridSpec(axes=[
        ("model_uc.gamma_grid_name", ["paper", "agaci_figure_extended", "wide"]),
        ("model_uc.eta", [1.0, 2.72, 3.19]),
    ]),
}

RESCP_QUICK_GRID = GridSpec(axes=[
    ("model_uc.spectral_radius", [0.9, 0.99, 1.1]),
    ("model_uc.leaking_rate", [0.9, 1.0]),
    ("model_uc.input_scaling", [0.25, 0.5, 0.75]),
    ("model_uc.temperature", [0.05, 0.25, 1.0]),
    ("model_uc.sliding_window", [250, 1000, "all"]),
])

RESCP_RECOMMENDED_GRID = GridSpec()


@dataclass(frozen=True)
class ForecasterVariant:
    family: str
    size: int
    ff_size: Optional[int] = None

    @property
    def hydra_config(self) -> str:
        return self.family

    @property
    def tag(self) -> str:
        if self.family == "global_rnn":
            return f"rnn{self.size}"
        if self.family == "global_lstm":
            return f"lstm{self.size}"
        if self.family == "global_transformer_decoder":
            ff = self.ff_size if self.ff_size is not None else self.size * 2
            return f"transformer{self.size}_{ff}"
        raise ValueError(f"Unsupported forecaster family: {self.family}")

    def train_overrides(self) -> List[str]:
        common = [
            "config.model_fc.model_params.seq_len=50",
            "config.model_fc.model_params.batch_size=256",
            "config.model_fc.model_params.loss=mse",
            "config.model_fc.model_params.dropout=0.1",
            "config.model_fc.model_params.train_split=0.75",
            "config.model_fc.model_params.train_with_calib=false",
        ]
        if self.family == "global_rnn":
            return common + [
                f"config.model_fc.model_params.rnn_conf.hidden_size={self.size}",
                "config.model_fc.model_params.rnn_conf.num_layers=1",
                "config.model_fc.model_params.rnn_conf.cell_type=gru",
            ]
        if self.family == "global_lstm":
            return common + [
                f"config.model_fc.model_params.lstm_conf.hidden_size={self.size}",
                "config.model_fc.model_params.lstm_conf.num_layers=1",
                "config.model_fc.model_params.lstm_conf.dropout=0.0",
            ]
        if self.family == "global_transformer_decoder":
            ff_size = self.ff_size if self.ff_size is not None else self.size * 2
            return common + [
                f"config.model_fc.model_params.transformer_conf.hidden_size={self.size}",
                f"config.model_fc.model_params.transformer_conf.ff_size={ff_size}",
                "config.model_fc.model_params.transformer_conf.dropout=0.1",
            ]
        raise ValueError(f"Unsupported forecaster family: {self.family}")


@dataclass
class JobSpec:
    method: str
    dataset: str
    forecaster: ForecasterVariant
    alpha: float
    seed: int
    tag: str
    extra_overrides: List[str]
    rccp_fast_online_block_size: int = DEFAULT_RCCP_FAST_ONLINE_BLOCK_SIZE
    artifact_tag: Optional[str] = None

    def build_cmd(self, gpu_id: int, extra_cli: List[str]) -> List[str]:
        if self.method == "rccp":
            artifact_tag = self.artifact_tag or self.tag.split("__s", 1)[0]
            rccp_overrides = _dedupe_overrides(self.extra_overrides + extra_cli)
            return [
                sys.executable,
                str(RCCP_FAST_RUNNER),
                "--dataset",
                self.dataset,
                "--forecaster",
                self.forecaster.hydra_config,
                "--forecaster-tag",
                self.forecaster.tag,
                "--seed",
                str(self.seed),
                "--tag",
                self.tag,
                "--artifact-tag",
                artifact_tag,
                "--alpha",
                str(self.alpha),
                "--out",
                str(self.result_path()),
                "--data-base-dir",
                DATA_ROOT.as_posix(),
                "--online-block-size",
                str(self.rccp_fast_online_block_size),
            ] + _rccp_fast_cli_from_overrides(rccp_overrides)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--method",
            self.method,
            "--dataset",
            self.dataset,
            "--forecaster",
            self.forecaster.hydra_config,
            "--alpha",
            str(self.alpha),
            "--seed",
            str(self.seed),
            "--tag",
            self.tag,
        ]
        for ov in self.extra_overrides:
            cmd.extend(["--override", ov])
        cmd.extend(["--override", f"config.experiment_data.gpu_id={gpu_id}"])
        for ov in extra_cli:
            cmd.extend(["--override", ov])
        return cmd

    def result_path(self) -> Path:
        return _result_path(self.method, self.dataset, self.forecaster.hydra_config, self.tag)


def _result_path(method: str, dataset: str, forecaster: str, tag: str) -> Path:
    return REPO_ROOT / "results" / "raw" / f"{method}_{dataset}_{forecaster}_{tag}.json"


def _split_override(search: bool) -> List[str]:
    if search:
        return ["config.task.data_splits=[0.4,0.2,0.2,0.2]"]
    return ["config.task.data_splits=[0.4,0.4,0.2]"]


def _fc_experiment_name(dataset: str, forecaster: ForecasterVariant, seed: int, tag: str) -> str:
    raw = f"paper_fc_{dataset}_{forecaster.tag}_seed{seed}_{tag}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _fc_artifacts(exp_name: str) -> Dict[str, Path]:
    checkpoint_model = CHECKPOINT_ROOT / f"model_{exp_name}"
    generated_model = GENERATED_FC_ROOT / f"model_{exp_name}"
    model_path = checkpoint_model if checkpoint_model.exists() else generated_model
    return {
        "model": model_path,
        "prediction": GENERATED_FC_ROOT / f"prediction_{exp_name}.pt",
        "state": GENERATED_FC_ROOT / f"state_{exp_name}.pt",
    }


def _fc_artifact_overrides(exp_name: str, forecaster: ForecasterVariant) -> List[str]:
    artifacts = _fc_artifacts(exp_name)
    return [
        f"config.model_fc.model_params.pre_trained_model_path={artifacts['model'].as_posix()}",
        f"config.model_fc.model_params.pre_trained_predictions_paths={artifacts['prediction'].as_posix()}",
        f"config.model_fc.model_params.pre_trained_states_path={artifacts['state'].as_posix()}",
        *forecaster.train_overrides(),
    ]


def _train_forecaster(
    dataset: str,
    forecaster: ForecasterVariant,
    seed: int,
    tag: str,
    gpu_id: int,
    force: bool,
    extra_overrides: List[str],
) -> None:
    dataset_conf, task_conf = DATASET_CONFIGS[dataset]
    exp_name = _fc_experiment_name(dataset, forecaster, seed, tag)
    artifacts = _fc_artifacts(exp_name)
    if not force and all(path.exists() for path in artifacts.values()):
        print(f"[fc-skip] {dataset}/{forecaster.tag} seed={seed}")
        return
    cmd = [
        sys.executable,
        str(TRAIN_FC),
        f"model_fc@config.model_fc={forecaster.hydra_config}",
        f"dataset@config.dataset={dataset_conf}",
        f"task@config.task={task_conf}",
        f"config.experiment_data.seed={seed}",
        f"config.experiment_data.experiment_name={exp_name}",
        f"config.experiment_data.base_proj_dir={REPO_ROOT.as_posix()}/",
        f"config.experiment_data.data_dir={DATA_ROOT.as_posix()}",
        f"config.experiment_data.gpu_id={gpu_id}",
        "config.trainer.trainer_config.n_epochs=150",
        "config.trainer.trainer_config.val_every=5",
        "config.trainer.trainer_config.early_stopping_patience=10000",
        "config.trainer.trainer_config.optim._target_=torch.optim.Adam",
        "config.trainer.trainer_config.optim.lr=0.001",
        "config.trainer.trainer_config.optim.weight_decay=0.0",
        "config.trainer.trainer_config.lr_scheduler=null",
        *_split_override(search=False),
        *forecaster.train_overrides(),
    ]
    if artifacts["model"].exists():
        cmd.append(f"config.model_fc.model_params.pre_trained_model_path={artifacts['model'].as_posix()}")
    cmd.extend(extra_overrides)
    print(f"[fc-train] gpu={gpu_id} {dataset}/{forecaster.tag} seed={seed}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _method_fixed_overrides(
    method: str,
    forecaster: ForecasterVariant,
    dataset: str,
    rescp_grid: str,
) -> List[str]:
    if method == "hopcpt":
        overrides = [
            "config.trainer.trainer_config.n_epochs=3000",
            "config.trainer.trainer_config.val_every=5",
            "config.trainer.trainer_config.early_stopping_patience=100",
            "config.trainer.trainer_config.optim._target_=torch.optim.AdamW",
            "config.trainer.trainer_config.optim.lr=0.001",
            "config.trainer.trainer_config.optim.weight_decay=0.01",
            "config.trainer.trainer_config.lr_scheduler=null",
            "config.trainer.trainer_config.model_selection=threshold-pi",
            "config.model_uc.eps_mem_size=8000",
            "config.model_uc.batch_size=4",
            "config.model_uc.batch_mode=one_ts",
        ]
        if forecaster.family == "global_lstm":
            overrides += HOPCPT_NIPS23_GLOBAL_LSTM_OVERRIDES.get(dataset, [])
        if dataset == "weather":
            overrides += [
                "config.model_uc.limit_train_seq_to_mem_size=true",
                "config.model_uc.limit_train_seq_stride=282",
            ]
        return _dedupe_overrides(overrides)
    if method == "spic":
        return ["config.model_uc.past_window_len=100"]
    if method == "rescp":
        overrides = [
            "config.model_uc.hidden_size=512",
            "config.model_uc.n_quantiles=100",
            "config.model_uc.density=0.2",
            "config.model_uc.decay=linear",
            "config.model_uc.similarity=cosine",
        ]
        if rescp_grid == "recommended":
            overrides += RESCP_RECOMMENDED_OVERRIDES.get(dataset, [])
        return overrides
    return []


def _dispatch(jobs: List[JobSpec], gpus: List[int], extra_cli: List[str], parallel: int) -> None:
    pending = [j for j in jobs if not j.result_path().exists()]
    for j in jobs:
        if j.result_path().exists():
            print(f"[skip] {j.result_path()}")
    if not pending:
        return
    if parallel <= 1:
        for j in pending:
            gpu = gpus[0] if gpus else 0
            print(f"[run] gpu={gpu} {j.method}/{j.dataset}/{j.forecaster.tag} tag={j.tag}")
            subprocess.run(j.build_cmd(gpu, extra_cli), cwd=REPO_ROOT, check=True)
        return

    worker_gpu: Dict[int, int] = {}
    lock = threading.Lock()
    next_worker = [0]

    def assign_gpu() -> int:
        ident = threading.get_ident()
        with lock:
            if ident not in worker_gpu:
                wid = next_worker[0]
                next_worker[0] += 1
                worker_gpu[ident] = gpus[wid % len(gpus)]
            return worker_gpu[ident]

    def run_one(job: JobSpec):
        gpu = assign_gpu()
        print(f"[run] gpu={gpu} {job.method}/{job.dataset}/{job.forecaster.tag} tag={job.tag}")
        proc = subprocess.run(job.build_cmd(gpu, extra_cli), cwd=REPO_ROOT)
        return job, proc.returncode

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        for job, rc in pool.map(run_one, pending):
            if rc != 0:
                print(f"[fail] {job.method}/{job.dataset}/{job.forecaster.tag} rc={rc}")
                failures += 1
    if failures:
        raise SystemExit(f"{failures} jobs failed")


def _load_results(method: str, dataset: str, forecaster: ForecasterVariant, tag_prefix: str) -> List[dict]:
    prefix = f"{method}_{dataset}_{forecaster.hydra_config}_{tag_prefix}"
    rows = []
    for path in (REPO_ROOT / "results" / "raw").glob(f"{prefix}*.json"):
        try:
            payload = json.loads(path.read_text())
            payload["_path"] = str(path)
            payload["_hp_tag"] = path.stem[len(prefix):].lstrip("_")
            rows.append(payload)
        except Exception as exc:
            print(f"[warn] cannot parse {path}: {exc}", file=sys.stderr)
    return rows


def _select_by_winkler(rows: List[dict]) -> Optional[dict]:
    scored = [r for r in rows if r.get("winkler") is not None and math.isfinite(float(r["winkler"]))]
    if scored:
        return min(scored, key=lambda r: float(r["winkler"]))
    return None


def _build_search_jobs(args, datasets, methods, forecasters, selection_seed: int) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    artifact_tag = args.artifact_tag or args.tag
    for dataset in datasets:
        for forecaster in forecasters:
            exp_name = _fc_experiment_name(dataset, forecaster, selection_seed, artifact_tag)
            fc_overrides = _fc_artifact_overrides(exp_name, forecaster)
            for method in methods:
                if method not in SEARCH_METHODS:
                    continue
                grid = GRIDS.get(method) or GridSpec()
                for overrides, suffix in grid.points():
                    tag = f"{args.tag}__s{selection_seed}__{forecaster.tag}__search__{suffix}"
                    jobs.append(JobSpec(
                        method=method,
                        dataset=dataset,
                        forecaster=forecaster,
                        alpha=args.alpha,
                        seed=selection_seed,
                        tag=tag,
                        extra_overrides=(
                            fc_overrides
                            + _split_override(search=True)
                            + _method_fixed_overrides(method, forecaster, dataset, args.rescp_grid)
                            + overrides
                        ),
                        rccp_fast_online_block_size=int(args.rccp_fast_online_block_size),
                        artifact_tag=artifact_tag,
                    ))
    return jobs


def _phase_search(args, datasets, methods, forecasters, base_overrides, gpus, selection_seed: int):
    jobs = _build_search_jobs(args, datasets, methods, forecasters, selection_seed)
    _dispatch(jobs, gpus, base_overrides, args.parallel)


def _phase_select(args, datasets, methods, forecasters, selection_seed: int) -> Dict[str, dict]:
    selected: Dict[str, dict] = {}
    for dataset in datasets:
        for forecaster in forecasters:
            tag_prefix = f"{args.tag}__s{selection_seed}__{forecaster.tag}__search__"
            for method in methods:
                if method not in SEARCH_METHODS:
                    continue
                rows = _load_results(method, dataset, forecaster, tag_prefix)
                best = _select_by_winkler(rows)
                if best is None:
                    print(f"[select] empty for {method}/{dataset}/{forecaster.tag}")
                    continue
                key = _selection_key(dataset, method, forecaster)
                selected[key] = {"hp": best.get("_hp_tag"), "winkler": best.get("winkler")}
                print(
                    f"[select] {method}/{dataset}/{forecaster.tag} -> {best.get('_hp_tag') or 'default'} "
                    f"(winkler={best.get('winkler')})"
                )
    _save_selection(args.tag, selection_seed, selected)
    return selected


def _selection_key(dataset: str, method: str, forecaster: ForecasterVariant) -> str:
    return f"{dataset}/{method}/{forecaster.tag}"


def _save_selection(tag: str, selection_seed: int, selected: Dict[str, dict]) -> None:
    out_dir = REPO_ROOT / "results" / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _selection_path(tag, selection_seed)
    merged = {}
    if path.exists():
        merged.update(json.loads(path.read_text()))
    merged.update(selected)
    path.write_text(json.dumps(merged, indent=2))


def _selection_path(tag: str, selection_seed: int) -> Path:
    return REPO_ROOT / "results" / "selected" / f"selection_{tag}_s{selection_seed}.json"


def _load_selection(tag: str, selection_seed: int) -> Dict[str, dict]:
    path = _selection_path(tag, selection_seed)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _hp_overrides_from_tag(method: str, hp_tag: Optional[str]) -> List[str]:
    if not hp_tag or hp_tag == "default":
        return []
    grid = GRIDS.get(method) or GridSpec()
    for overrides, suffix in grid.points():
        if suffix == hp_tag:
            return overrides
    raise ValueError(f"HP tag {hp_tag!r} not found in grid for {method}")


def _phase_final(args, datasets, methods, forecasters, base_overrides, selected, gpus, seeds):
    jobs: List[JobSpec] = []
    artifact_tag = args.artifact_tag or args.tag
    for seed in seeds:
        for dataset in datasets:
            for forecaster in forecasters:
                exp_name = _fc_experiment_name(dataset, forecaster, seed, artifact_tag)
                fc_overrides = _fc_artifact_overrides(exp_name, forecaster)
                for method in methods:
                    hp_overrides = []
                    if method in SEARCH_METHODS:
                        body = selected.get(_selection_key(dataset, method, forecaster), {})
                        hp_overrides = _hp_overrides_from_tag(method, body.get("hp"))
                    tag = f"{args.tag}__s{seed}__{forecaster.tag}__final"
                    jobs.append(JobSpec(
                        method=method,
                        dataset=dataset,
                        forecaster=forecaster,
                        alpha=args.alpha,
                        seed=seed,
                        tag=tag,
                        extra_overrides=(
                            fc_overrides
                            + _split_override(search=False)
                            + _method_fixed_overrides(method, forecaster, dataset, args.rescp_grid)
                            + hp_overrides
                        ),
                        rccp_fast_online_block_size=int(args.rccp_fast_online_block_size),
                        artifact_tag=artifact_tag,
                    ))
    _dispatch(jobs, gpus, base_overrides, args.parallel)


def _calibration_time_from_payload(method: str, payload: dict) -> Optional[float]:
    if payload.get("calibration_time_sec") is not None:
        return float(payload["calibration_time_sec"])
    method_time = payload.get("method_wall_clock_time_sec")
    if method_time is None:
        return None
    value = float(method_time)
    if method == "rccp" and payload.get("state_save_overhead_sec") is not None:
        value += float(payload["state_save_overhead_sec"])
    return value


def _write_summary(args, datasets, methods, forecasters, seeds) -> None:
    rows = []
    for seed in seeds:
        for dataset in datasets:
            for forecaster in forecasters:
                for method in methods:
                    path = _result_path(method, dataset, forecaster.hydra_config, f"{args.tag}__s{seed}__{forecaster.tag}__final")
                    if not path.exists():
                        continue
                    payload = json.loads(path.read_text())
                    rows.append({
                        "method": method,
                        "dataset": dataset,
                        "forecaster": forecaster.family,
                        "size": forecaster.size,
                        "ff_size": forecaster.ff_size,
                        "seed": seed,
                        "coverage": payload.get("coverage"),
                        "delta_cov": payload.get("delta_cov"),
                        "width": payload.get("width"),
                        "winkler": payload.get("winkler"),
                        "winkler_norm": payload.get("winkler_norm"),
                        "time_sec": payload.get("time_sec"),
                        "wall_clock_time_sec": payload.get("wall_clock_time_sec"),
                        "method_wall_clock_time_sec": payload.get("method_wall_clock_time_sec"),
                        "calibration_time_sec": _calibration_time_from_payload(method, payload),
                        "method_wall_clock_excluding_g_time_sec": payload.get("method_wall_clock_excluding_g_time_sec"),
                        "method_wall_clock_including_g_time_sec": payload.get("method_wall_clock_including_g_time_sec"),
                        "ct_ssf_g_fit_time_sec": payload.get("ct_ssf_g_fit_time_sec"),
                        "state_save_overhead_sec": payload.get("state_save_overhead_sec"),
                        "time_sec_without_state_save_overhead": payload.get("time_sec_without_state_save_overhead"),
                        "wall_clock_time_sec_without_state_save_overhead": payload.get("wall_clock_time_sec_without_state_save_overhead"),
                    })
    out_dir = REPO_ROOT / "results" / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"summary_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[summary] {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--method", action="append", default=[], choices=list(GRIDS.keys()))
    parser.add_argument("--forecaster", action="append", default=[], choices=SUPPORTED_FORECASTERS)
    parser.add_argument("--size", action="append", type=int, default=[], choices=DEFAULT_SIZES)
    parser.add_argument("--transformer-ff-size", type=int, default=DEFAULT_TRANSFORMER_FF_SIZE)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed to evaluate. Repeat for multiple seeds; the first seed is used for HP selection.",
    )
    parser.add_argument("--selection-seed", type=int, default=None, help="Override the first seed used for HP search/select.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--tag", default="papergrid")
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Read pretrained forecaster artifacts from this tag while writing benchmark results under --tag.",
    )
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override for method runs.")
    parser.add_argument("--train-override", action="append", default=[], help="Extra Hydra override for forecaster training.")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--phase", choices=["all", "train", "search", "select", "final", "summary"], default="all")
    parser.add_argument(
        "--rescp-grid",
        choices=["recommended", "quick", "full"],
        default="recommended",
        help=(
            "ResCP HP policy. recommended uses one paper/native setting per dataset; "
            "quick/full run explicit validation searches."
        ),
    )
    parser.add_argument(
        "--rccp-fast-online-block-size",
        type=int,
        default=DEFAULT_RCCP_FAST_ONLINE_BLOCK_SIZE,
        help=(
            "RCCP artifact fast-runner block size for final rows. "
            "The default 1 is exact prequential online memory. Values >1 are "
            "explicit batched-online speed approximations and should be logged "
            "as such in experiment notes."
        ),
    )
    args = parser.parse_args()

    if args.rescp_grid == "recommended":
        GRIDS["rescp"] = RESCP_RECOMMENDED_GRID
        SEARCH_METHODS.discard("rescp")
    elif args.rescp_grid == "quick":
        GRIDS["rescp"] = RESCP_QUICK_GRID

    datasets = args.dataset or DEFAULT_DATASETS
    methods = args.method or DEFAULT_METHODS
    forecaster_families = args.forecaster or DEFAULT_FORECASTERS
    if args.size:
        forecasters = [
            ForecasterVariant(
                family,
                size,
                args.transformer_ff_size if family == "global_transformer_decoder" else None,
            )
            for family in forecaster_families
            for size in args.size
        ]
    else:
        forecasters = [
            ForecasterVariant(family, size, ff_size)
            for family in forecaster_families
            for size, ff_size in DEFAULT_FORECASTER_SIZES[family]
        ]
    seeds = args.seed or [10]
    selection_seed = args.selection_seed if args.selection_seed is not None else seeds[0]
    train_seeds = list(dict.fromkeys([selection_seed] + seeds))
    gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    train_gpu = gpus[0] if gpus else 0

    if args.phase in ("all", "train") and not args.skip_train:
        for seed in train_seeds:
            for dataset in datasets:
                for forecaster in forecasters:
                    _train_forecaster(
                        dataset=dataset,
                        forecaster=forecaster,
                        seed=seed,
                        tag=args.tag,
                        gpu_id=train_gpu,
                        force=args.force_train,
                        extra_overrides=args.train_override,
                    )
    if args.phase in ("all", "search"):
        _phase_search(args, datasets, methods, forecasters, args.override, gpus, selection_seed)
    selected: Dict[str, dict] = {}
    if args.phase in ("all", "select"):
        selected = _phase_select(args, datasets, methods, forecasters, selection_seed)
    if args.phase in ("all", "final"):
        if not selected:
            selected = _load_selection(args.tag, selection_seed)
        _phase_final(args, datasets, methods, forecasters, args.override, selected, gpus, seeds)
    if args.phase in ("all", "summary"):
        _write_summary(args, datasets, methods, forecasters, seeds)


if __name__ == "__main__":
    main()
