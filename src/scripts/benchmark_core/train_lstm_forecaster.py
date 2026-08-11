import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import re
import shutil
import hydra
from omegaconf import DictConfig

from loader.generator import DataGenerator
from models.forcast.forcast_service import ForcastService
from utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../../config', config_name='default_fclstm_train.yaml')
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"

    datasets = DataGenerator.get_data(cfg.dataset, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha

    # Prepare (Create, Train,..) underlying forcast models
    fc_service = _init_fc(cfg, datasets, fc_persist_dir, )
    _ = fc_service.prepare(datasets, alphas)
    _publish_lstm_artifacts(cfg)
    return


def _setup(config):
    config.experiment_data.experiment_dir = Path().cwd()
    set_seed(config.experiment_data.seed)
    LOGGER.info("Experiment logging uses local files only.")


def _init_fc(config, datasets, fc_persist_dir) -> ForcastService:
    LOGGER.info('Initialize forcast service.')
    return ForcastService(lambda: hydra.utils.instantiate(config.model_fc, no_x_features=datasets[0].no_x_features,
                                                          alpha=config.task.alpha, train_only=False, save_predictions=True),
                          data_config=config.dataset, task_config=config.task, model_config=config.model_fc,
                          persist_dir=fc_persist_dir, save_new_reg_bak=True, trainer_config=config.trainer,
                          experiment_config=config.experiment_data)


def _publish_lstm_artifacts(config) -> None:
    """
    Standardize LSTM artifact locations expected by global_lstm_* configs:
    - models_save/lstm_fc/model
    - models_save/lstm_fc/prediction.pt
    - models_save/lstm_fc/state.pt
    """
    exp_dir = Path(config.experiment_data.experiment_dir)
    target_dir = Path(config.experiment_data.model_dir) / "lstm_fc"
    target_dir.mkdir(parents=True, exist_ok=True)

    best_epoch_fp = exp_dir / "best_epoch.txt"
    src_model = None
    if best_epoch_fp.exists():
        best_epoch = int(best_epoch_fp.read_text(encoding="utf-8").strip())
        candidate = exp_dir / f"model_epoch_{best_epoch:03d}"
        if candidate.exists():
            src_model = candidate
    if src_model is None:
        model_candidates = sorted(exp_dir.glob("model_epoch_*"))
        if model_candidates:
            src_model = model_candidates[-1]
    if src_model is not None and src_model.exists():
        model_main = target_dir / "model"
        shutil.copy2(src_model, model_main)
        exp_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.experiment_data.experiment_name))
        model_tagged = target_dir / f"model_{exp_tag}"
        shutil.copy2(src_model, model_tagged)
        LOGGER.info(f"Saved best model to {model_main} and {model_tagged}")
    else:
        LOGGER.warning("No model checkpoint found to publish.")

    pred_src = exp_dir / "predictions.pt"
    if pred_src.exists():
        pred_main = target_dir / "prediction.pt"
        shutil.copy2(pred_src, pred_main)
        exp_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.experiment_data.experiment_name))
        pred_tagged = target_dir / f"prediction_{exp_tag}.pt"
        shutil.copy2(pred_src, pred_tagged)
        LOGGER.info(f"Saved predictions to {pred_main} and {pred_tagged}")
    else:
        LOGGER.warning("No predictions.pt found to publish.")

    state_src = exp_dir / "states.pt"
    if state_src.exists():
        state_main = target_dir / "state.pt"
        shutil.copy2(state_src, state_main)
        exp_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.experiment_data.experiment_name))
        state_tagged = target_dir / f"state_{exp_tag}.pt"
        shutil.copy2(state_src, state_tagged)
        LOGGER.info(f"Saved states to {state_main} and {state_tagged}")
    else:
        LOGGER.warning("No states.pt found to publish.")


if __name__ == "__main__":
    my_app()
