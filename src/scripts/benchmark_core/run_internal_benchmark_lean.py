from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / "data" / "source"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
GENERATED_FC_ROOT = REPO_ROOT / "models_save" / "lstm_fc"

from evaluation.benchmark_results import BenchmarkResult, dump_result


DATASET_CONFIGS = {
    "air": ("bejing_air_pm10", "default_3year_gn"),
    "solar": ("nsdb2018-20_60m", "default_3year_gn"),
    "wind": ("wind", "default_3year_gn"),
    "electricity": ("elec_normalized", "default_3year_gn"),
}


def _fc_experiment_name(dataset: str, forecaster_tag: str, seed: int, tag: str) -> str:
    raw = f"paper_fc_{dataset}_{forecaster_tag}_seed{seed}_{tag}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _fc_artifact_overrides(exp_name: str, forecaster_tag: str) -> List[str]:
    checkpoint_model = CHECKPOINT_ROOT / f"model_{exp_name}"
    generated_model = GENERATED_FC_ROOT / f"model_{exp_name}"
    model_path = checkpoint_model if checkpoint_model.exists() else generated_model
    overrides = [
        f"config.model_fc.model_params.pre_trained_model_path={model_path.as_posix()}",
        f"config.model_fc.model_params.pre_trained_predictions_paths={(GENERATED_FC_ROOT / f'prediction_{exp_name}.pt').as_posix()}",
        f"config.model_fc.model_params.pre_trained_states_path={(GENERATED_FC_ROOT / f'state_{exp_name}.pt').as_posix()}",
        "config.model_fc.model_params.seq_len=50",
        "config.model_fc.model_params.batch_size=256",
        "config.model_fc.model_params.loss=mse",
        "config.model_fc.model_params.dropout=0.1",
        "config.model_fc.model_params.train_split=0.75",
        "config.model_fc.model_params.train_with_calib=false",
    ]
    if forecaster_tag.startswith("lstm"):
        hidden = forecaster_tag.replace("lstm", "")
        overrides += [
            f"config.model_fc.model_params.lstm_conf.hidden_size={hidden}",
            "config.model_fc.model_params.lstm_conf.num_layers=1",
            "config.model_fc.model_params.lstm_conf.dropout=0.0",
        ]
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Run lean post-forecast conformal timing.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--forecaster", default="global_lstm")
    parser.add_argument("--forecaster-tag", default="lstm128")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    dataset_conf, task_conf = DATASET_CONFIGS[args.dataset]
    artifact_exp = _fc_experiment_name(args.dataset, args.forecaster_tag, args.seed, args.artifact_tag)
    experiment_name = f"{args.method}_{args.dataset}_{args.forecaster}_{args.tag}"
    hydra_job_name = f"lean_{hashlib.sha1(experiment_name.encode()).hexdigest()[:10]}_{os.getpid()}"
    task_alpha = f"[{args.alpha}]" if args.method == "hopcpt" else str(args.alpha)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_core" / "internal_main_lean_runtime.py"),
        f"hydra.job.name={hydra_job_name}",
        f"model_fc@config.model_fc={args.forecaster}",
        f"model_uc@config.model_uc={args.method}",
        f"dataset@config.dataset={dataset_conf}",
        f"task@config.task={task_conf}",
        f"config.task.alpha={task_alpha}",
        f"config.experiment_data.seed={args.seed}",
        f"config.experiment_data.experiment_name={experiment_name}",
        f"config.experiment_data.base_proj_dir={REPO_ROOT.as_posix()}/",
        f"config.experiment_data.data_dir={DATA_ROOT.as_posix()}",
        "config.experiment_data.evaluate=true",
        "config.task.data_splits=[0.4,0.4,0.2]",
    ]
    cmd.extend(_fc_artifact_overrides(artifact_exp, args.forecaster_tag))
    for override in args.override:
        cmd.append(override)

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)

    run_dir = REPO_ROOT / "outputs" / hydra_job_name
    metrics_path = run_dir / "lean_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing lean_metrics.json in {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = BenchmarkResult(
        method=args.method,
        dataset=args.dataset,
        forecaster=args.forecaster,
        alpha=float(metrics["alpha"]),
        coverage=float(metrics["mean_coverage"]),
        delta_cov=float(metrics["mean_coverage_eps"]),
        width=float(metrics["mean_pi_width"]),
        winkler=float(metrics["winkler_score"]),
        winkler_norm=float(metrics["winkler_score_norm"]),
        time_sec=elapsed,
        wall_clock_time_sec=elapsed,
        method_wall_clock_time_sec=float(metrics["method_wall_clock_time_sec"]),
        calibration_time_sec=float(metrics["method_wall_clock_time_sec"]),
        run_dir=str(run_dir),
        notes="lean post-forecast conformal inference timing; excludes forecast/data/artifact loading and logging/table construction",
    )
    out = REPO_ROOT / "results" / "raw" / f"{experiment_name}.json"
    dump_result(result, out)
    print(out)


if __name__ == "__main__":
    main()
