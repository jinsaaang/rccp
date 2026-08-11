import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import logging
import time

import hydra
from omegaconf import DictConfig

from loader.generator import DataGenerator
from internal_main_utils import _init_fc, _init_uc, _setup, Evaluator

LOGGER = logging.getLogger(__name__)


def _collect_method_runtime_info(uc_service) -> dict:
    info = {}
    seen = set()
    uc_models = getattr(uc_service, "_uc_models", {})
    for per_alpha in uc_models.values():
        for model in per_alpha.values():
            marker = id(model)
            if marker in seen:
                continue
            seen.add(marker)
            getter = getattr(model, "method_runtime_info", None)
            if not callable(getter):
                continue
            model_info = getter() or {}
            for key, value in model_info.items():
                if isinstance(value, (int, float)):
                    info[key] = float(info.get(key, 0.0)) + float(value)
                else:
                    info[key] = value
    return info


@hydra.main(version_base=None, config_path='../../config', config_name='default_config.yaml')
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"
    uc_persist_dir = f"{cfg.experiment_data.model_dir}/uc"

    datasets = DataGenerator.get_data(cfg.dataset, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha

    for d in datasets:
        print(f"Calib size: {d.no_calib_steps}")
    if cfg.dataset.add_config is not None and cfg.dataset.add_config.get('subset_before_prepare', False):
        datasets = list(filter(lambda d: d.ts_id in cfg.dataset.add_config['eval_subset'], datasets))
    # Prepare (Create, Train,..) underlying forcast models
    fc_service = _init_fc(fc_conf=cfg.model_fc, data_conf=cfg.dataset, task_conf=cfg.task, trainer_conf=cfg.trainer,
                          experiment_conf=cfg.experiment_data, datasets=datasets, fc_persist_dir=fc_persist_dir)
    datasets = fc_service.prepare(datasets, alphas)

    # Method runtime starts after the shared forecaster artifacts/data are ready.
    # This is the comparable conformal-method cost used for ResCP/CT-SSF/etc.
    method_started = time.perf_counter()

    # Calibrate/Train UC
    uc_service = _init_uc(uc_conf=cfg.model_uc, data_conf=cfg.dataset, task_conf=cfg.task, fc_service=fc_service,
                          datasets=datasets, uc_persist_dir=uc_persist_dir, fc_state_dim=fc_service.fc_state_dim,
                          record_attention=cfg.evaluation['att_plot_vega'] or cfg.evaluation['att_hist_matplot'])
    uc_service.prepare(datasets, alphas, experiment_config=cfg.experiment_data, calib_trainer_config=cfg.trainer)

    # Evaluate
    if cfg.experiment_data.evaluate:
        if cfg.dataset.add_config is not None and 'eval_subset' in cfg.dataset.add_config:
            eval_subset = cfg.dataset.add_config['eval_subset']
            LOGGER.info(f"Evaluate only Subset of Dataset: {eval_subset}")
        else:
            LOGGER.info("Evaluate full dataset!")
            eval_subset = None
        Evaluator.evaluate(uc_service, datasets, alphas, cfg.evaluation, mix_mem_data=None,
                           evaluation_subset=eval_subset)
    else:
        model = cfg.model_uc._target_.split(".")[-1]
        if model in ['EnbPIModel']:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=True)
        elif model in ['AdaptiveCI', 'NexCP', 'SPICModel', 'EpsSelectionPIStat', 'Bluecat', 'DefaultConformal', 'DefaultConformalPlusRecent', 'CPRPModel', 'RACPModel', 'CPTCModel']:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=False)
        elif model == 'EpsPredictionHopfield' and cfg.model_uc.use_adaptiveci:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=False, prefix="extraval")
        else:
            LOGGER.info("Skip Evaluation")

    method_elapsed_including_optional_setup = time.perf_counter() - method_started
    method_runtime_info = _collect_method_runtime_info(uc_service)
    g_fit_time_sec = float(method_runtime_info.get("ct_ssf_g_fit_time_sec", 0.0) or 0.0)
    method_elapsed = max(method_elapsed_including_optional_setup - g_fit_time_sec, 0.0)
    runtime_payload = {
        "method_wall_clock_time_sec": method_elapsed,
        "method_wall_clock_excluding_g_time_sec": method_elapsed if g_fit_time_sec > 0 else None,
        "method_wall_clock_including_g_time_sec": method_elapsed_including_optional_setup,
        "ct_ssf_g_fit_time_sec": g_fit_time_sec if g_fit_time_sec > 0 else None,
        "timer_start": "after_fc_service_prepare",
        "timer_end": "after_uc_prepare_and_evaluate",
        "timer_default_policy": "exclude_ct_ssf_g_fit_time_when_present",
        "includes": [
            "uncertainty-method calibration/training after optional CT-SSF g fit",
            "method-specific state/residual construction",
            "interval generation",
            "central PIModel evaluation loop",
            "metric logging",
        ],
        "excludes": [
            "process startup",
            "dataset loading",
            "shared forecaster loading/training",
            "pretrained prediction/state artifact loading",
            "CT-SSF g head fitting when ct_ssf_g_fit_time_sec is present",
            "raw benchmark JSON serialization",
        ],
    }
    runtime_payload.update(method_runtime_info)
    Path("method_runtime.json").write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    my_app()
