import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from internal_main import _collect_method_runtime_info
from internal_main_utils import _init_fc, _init_uc, _interval_membership_stats, _setup
from loader.generator import DataGenerator
from models.uncertainty.pi_base import PIPredictionStepData
from models.uncertainty.score_service import get_score_param

LOGGER = logging.getLogger(__name__)


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _rescale(value, mean, std):
    return (_as_float(value) * float(std)) + float(mean)


def _lean_evaluate(uc_service, datasets, alphas, evaluation_subset=None):
    total_covered = 0
    total_n = 0
    total_width = 0.0
    total_width_norm = 0.0
    total_miss_dist = 0.0
    total_miss_dist_norm = 0.0
    alpha_seen = None

    for dataset_no, dataset in enumerate(datasets):
        if evaluation_subset is not None:
            if len(evaluation_subset) == 2 and (dataset_no < evaluation_subset[0] or dataset_no >= evaluation_subset[1]):
                continue
            if len(evaluation_subset) != 2 and dataset.ts_id not in evaluation_subset:
                continue
        other_datasets = [d for d in datasets if d.ts_id != dataset.ts_id]
        for alpha in alphas:
            alpha_seen = float(alpha)
            start_step, pre_predict_len, max_window_len, eps = uc_service.pre_predict(dataset, alpha, other_datasets)
            X = dataset.X_full
            Y = dataset.Y_full
            pred_steps = X.shape[0] - start_step - pre_predict_len
            y_mean, y_std = dataset.Y_normalize_props
            mask_full = getattr(dataset, "mask_full", None)
            mask_values = None
            if mask_full is not None:
                mask_values = mask_full.reshape(-1).detach().cpu().numpy().astype(bool)

            for pred_step in range(pred_steps):
                overall_step = pred_step + start_step + pre_predict_len
                if mask_values is not None and overall_step < mask_values.shape[0] and not mask_values[overall_step]:
                    continue
                start_past = max(0, overall_step - max_window_len)
                pred_data = PIPredictionStepData(
                    ts_id=dataset.ts_id,
                    X_step=X[overall_step].unsqueeze(0),
                    X_past=X[start_past:overall_step],
                    Y_past=Y[start_past:overall_step],
                    eps_past=(torch.Tensor(eps[-max_window_len:]) if len(eps) > max_window_len else torch.Tensor(eps)) if eps is not None else None,
                    step_offset_prediction=pred_step,
                    step_offset_overall=overall_step,
                    alpha=alpha,
                    mix_ts=uc_service.pack_mix_data(
                        dataset.ts_id,
                        alpha,
                        mix_datasets=other_datasets,
                        max_past=max_window_len,
                        step_after_start=pred_step + pre_predict_len,
                    ),
                    score_param=get_score_param(dataset),
                )
                Y_step = Y[overall_step].unsqueeze(0)
                prediction = uc_service.predict_step(Y_step, pred_data)

                y_norm = _as_float(Y_step)
                y_real = _rescale(Y_step, y_mean, y_std)
                if prediction.pred_set is not None:
                    pred_set_norm = [(float(low), float(high)) for low, high in prediction.pred_set]
                    pred_set = [
                        ((float(low) * float(y_std)) + float(y_mean), (float(high) * float(y_std)) + float(y_mean))
                        for low, high in pred_set_norm
                    ]
                    covered, miss_dist, width = _interval_membership_stats(pred_set, y_real)
                    _, miss_dist_norm, width_norm = _interval_membership_stats(pred_set_norm, y_norm)
                else:
                    low_norm = _as_float(prediction.pred_interval[0])
                    high_norm = _as_float(prediction.pred_interval[1])
                    low = (low_norm * float(y_std)) + float(y_mean)
                    high = (high_norm * float(y_std)) + float(y_mean)
                    covered = bool(low <= y_real <= high)
                    width = abs(high - low)
                    width_norm = abs(high_norm - low_norm)
                    miss_dist = 0.0 if covered else min(abs(high - y_real), abs(y_real - low))
                    miss_dist_norm = 0.0 if covered else min(abs(high_norm - y_norm), abs(y_norm - low_norm))

                total_n += 1
                total_covered += int(covered)
                total_width += float(width)
                total_width_norm += float(width_norm)
                total_miss_dist += float(miss_dist)
                total_miss_dist_norm += float(miss_dist_norm)

                if eps is not None:
                    eps = eps + ((Y_step - prediction.fc_Y_hat).tolist())

    if total_n == 0 or alpha_seen is None:
        raise RuntimeError("Lean evaluation produced no test points.")
    coverage = total_covered / total_n
    return {
        "alpha": alpha_seen,
        "mean_coverage": coverage,
        "mean_coverage_eps": alpha_seen - (1.0 - coverage),
        "mean_pi_width": total_width / total_n,
        "winkler_score": (total_width + (2.0 * total_miss_dist / alpha_seen)) / total_n,
        "winkler_score_norm": (total_width_norm + (2.0 * total_miss_dist_norm / alpha_seen)) / total_n,
        "n_test_points": total_n,
    }


@hydra.main(version_base=None, config_path="../../config", config_name="default_config.yaml")
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"
    uc_persist_dir = f"{cfg.experiment_data.model_dir}/uc"

    datasets = DataGenerator.get_data(cfg.dataset, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha
    if cfg.dataset.add_config is not None and cfg.dataset.add_config.get("subset_before_prepare", False):
        datasets = list(filter(lambda d: d.ts_id in cfg.dataset.add_config["eval_subset"], datasets))

    fc_service = _init_fc(
        fc_conf=cfg.model_fc,
        data_conf=cfg.dataset,
        task_conf=cfg.task,
        trainer_conf=cfg.trainer,
        experiment_conf=cfg.experiment_data,
        datasets=datasets,
        fc_persist_dir=fc_persist_dir,
    )
    datasets = fc_service.prepare(datasets, alphas)

    method_started = time.perf_counter()
    uc_service = _init_uc(
        uc_conf=cfg.model_uc,
        data_conf=cfg.dataset,
        task_conf=cfg.task,
        fc_service=fc_service,
        datasets=datasets,
        uc_persist_dir=uc_persist_dir,
        fc_state_dim=fc_service.fc_state_dim,
        record_attention=False,
    )
    uc_service.prepare(datasets, alphas, experiment_config=cfg.experiment_data, calib_trainer_config=cfg.trainer)

    eval_subset = cfg.dataset.add_config["eval_subset"] if cfg.dataset.add_config is not None and "eval_subset" in cfg.dataset.add_config else None
    metrics = _lean_evaluate(uc_service, datasets, alphas, evaluation_subset=eval_subset)
    method_elapsed = time.perf_counter() - method_started
    runtime_payload = {
        "method_wall_clock_time_sec": method_elapsed,
        "timer_start": "after_fc_service_prepare",
        "timer_end": "after_uc_prepare_and_lean_evaluate",
        "includes": [
            "uncertainty-method calibration/training",
            "method-specific state/residual construction",
            "test interval generation",
            "online method updates",
            "coverage/width/winkler metric computation",
        ],
        "excludes": [
            "process startup",
            "dataset loading",
            "shared forecaster loading/training",
            "prediction/state artifact loading",
            "tqdm/logging/pandas table construction",
            "raw benchmark JSON serialization",
        ],
    }
    runtime_payload.update(_collect_method_runtime_info(uc_service))
    metrics.update(runtime_payload)
    Path("lean_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("method_runtime.json").write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    LOGGER.info(f"Lean evaluation metrics: {metrics}")


if __name__ == "__main__":
    my_app()
