from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / "data" / "source"

from evaluation.benchmark_results import BenchmarkResult, dump_result, parse_conformal_run_dir


_DATASET_MAP = {
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
    "exchange": ("exchange_rate", "default_3year_gn"),
    "exchange_rate": ("exchange_rate", "default_3year_gn"),
    "exchange_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
    "exchange_ot1_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
    "amazon": ("amazon", "default_3year_gn"),
    "stock": ("amazon", "default_3year_gn"),
    "amzn": ("amazon", "default_3year_gn"),
    "apple": ("apple", "default_3year_gn"),
    "aapl": ("apple", "default_3year_gn"),
    "google": ("google", "default_3year_gn"),
    "goog": ("google", "default_3year_gn"),
    "elec": ("elec_normalized", "default_3year_gn"),
    "elec_norm": ("elec_normalized", "default_3year_gn"),
    "electricity": ("elec_normalized", "default_3year_gn"),
    "elec_normalized": ("elec_normalized", "default_3year_gn"),
    "electricity_normalized": ("elec_normalized", "default_3year_gn"),
}

_DEFAULT_FC_ARTIFACTS = {
    "air10": {
        "model": "models_save/lstm_fc/model_trainsweep_air10_full_seed10-seed10-c0",
        "prediction": "models_save/lstm_fc/prediction_trainsweep_air10_full_seed10-seed10-c0.pt",
        "state": "models_save/lstm_fc/state_trainsweep_air10_full_seed10-seed10-c0.pt",
    },
    "solar3y": {
        "model": "models_save/lstm_fc/model_trainsweep_solar3y_full_seed10-seed10-c0",
        "prediction": "models_save/lstm_fc/prediction_trainsweep_solar3y_full_seed10-seed10-c0.pt",
        "state": "models_save/lstm_fc/state_trainsweep_solar3y_full_seed10-seed10-c0.pt",
    },
}


def _make_command(
    method: str,
    dataset: str,
    forecaster: str,
    alpha: float,
    seed: int,
    experiment_name: str,
    base_proj_dir: Path,
    hydra_job_name: str,
    extra_overrides: List[str],
) -> List[str]:
    dataset_conf, task_conf = _DATASET_MAP[dataset]
    task_alpha = f"[{alpha}]" if method == "hopcpt" else str(alpha)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_core" / "internal_main.py"),
        f"hydra.job.name={hydra_job_name}",
        f"model_fc@config.model_fc={forecaster}",
        f"model_uc@config.model_uc={method}",
        f"dataset@config.dataset={dataset_conf}",
        f"task@config.task={task_conf}",
        f"config.task.alpha={task_alpha}",
        f"config.experiment_data.seed={seed}",
        f"config.experiment_data.experiment_name={experiment_name}",
        f"config.experiment_data.base_proj_dir={base_proj_dir.as_posix()}/",
        f"config.experiment_data.data_dir={DATA_ROOT.as_posix()}",
        "config.experiment_data.evaluate=true",
    ]
    if forecaster.startswith("global_lstm"):
        artifacts = _DEFAULT_FC_ARTIFACTS.get(dataset)
        if artifacts is not None:
            cmd.extend(
                [
                    f"config.model_fc.model_params.pre_trained_model_path={base_proj_dir.as_posix()}/{artifacts['model']}",
                    f"config.model_fc.model_params.pre_trained_predictions_paths={base_proj_dir.as_posix()}/{artifacts['prediction']}",
                    f"config.model_fc.model_params.pre_trained_states_path={base_proj_dir.as_posix()}/{artifacts['state']}",
                ]
            )
    cmd.extend(extra_overrides)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run a conformal internal benchmark job.")
    parser.add_argument("--method", required=True, help="Hydra model_uc config name, e.g. rccp, rescp, hopcpt, enbpi.")
    parser.add_argument("--dataset", required=True, choices=sorted(_DATASET_MAP.keys()))
    parser.add_argument("--forecaster", required=True, help="Hydra model_fc config name.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--tag", default="bench")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override. Can be repeated.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    outputs_dir = repo_root / "outputs"
    before = {p.name for p in outputs_dir.iterdir() if p.is_dir()} if outputs_dir.exists() else set()

    experiment_name = f"{args.method}_{args.dataset}_{args.forecaster}_{args.tag}"
    hydra_job_name = f"bm_{hashlib.sha1(experiment_name.encode()).hexdigest()[:10]}_{os.getpid()}"
    cmd = _make_command(
        method=args.method,
        dataset=args.dataset,
        forecaster=args.forecaster,
        alpha=args.alpha,
        seed=args.seed,
        experiment_name=experiment_name,
        base_proj_dir=repo_root,
        hydra_job_name=hydra_job_name,
        extra_overrides=args.override,
    )

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    after = {p.name for p in outputs_dir.iterdir() if p.is_dir()} if outputs_dir.exists() else set()
    created = sorted(after - before)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    if not created:
        raise RuntimeError("Run succeeded but no new output directory was created.")

    expected_run_dir = outputs_dir / hydra_job_name
    if expected_run_dir.exists():
        run_dir = expected_run_dir
    elif created:
        run_dir = max((outputs_dir / name for name in created), key=lambda p: p.stat().st_mtime)
    else:
        exact_matches = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith(f"{experiment_name}_")]
        if not exact_matches:
            raise RuntimeError(f"Run succeeded but no output directory matched {experiment_name}.")
        run_dir = max(exact_matches, key=lambda p: p.stat().st_mtime)
    metrics = parse_conformal_run_dir(run_dir)
    result = BenchmarkResult(
        method=args.method,
        dataset=args.dataset,
        forecaster=args.forecaster,
        alpha=float(metrics["alpha"]),
        coverage=float(metrics.get("mean_coverage")) if metrics.get("mean_coverage") is not None else None,
        delta_cov=float(metrics.get("mean_coverage_eps")) if metrics.get("mean_coverage_eps") is not None else None,
        width=float(metrics.get("mean_pi_width")) if metrics.get("mean_pi_width") is not None else None,
        winkler=float(metrics.get("winkler_score")) if metrics.get("winkler_score") is not None else None,
        winkler_norm=float(metrics.get("winkler_score_norm")) if metrics.get("winkler_score_norm") is not None else None,
        time_sec=elapsed,
        wall_clock_time_sec=elapsed,
        method_wall_clock_time_sec=(
            float(metrics.get("method_wall_clock_time_sec"))
            if metrics.get("method_wall_clock_time_sec") is not None
            else None
        ),
        calibration_time_sec=(
            float(metrics.get("method_wall_clock_time_sec"))
            if metrics.get("method_wall_clock_time_sec") is not None
            else None
        ),
        method_wall_clock_excluding_g_time_sec=(
            float(metrics.get("method_wall_clock_excluding_g_time_sec"))
            if metrics.get("method_wall_clock_excluding_g_time_sec") is not None
            else None
        ),
        method_wall_clock_including_g_time_sec=(
            float(metrics.get("method_wall_clock_including_g_time_sec"))
            if metrics.get("method_wall_clock_including_g_time_sec") is not None
            else None
        ),
        ct_ssf_g_fit_time_sec=(
            float(metrics.get("ct_ssf_g_fit_time_sec"))
            if metrics.get("ct_ssf_g_fit_time_sec") is not None
            else None
        ),
        run_dir=str(run_dir),
    )

    result_path = repo_root / "results" / "raw" / f"{experiment_name}.json"
    dump_result(result, result_path)
    print(result_path)


if __name__ == "__main__":
    main()
