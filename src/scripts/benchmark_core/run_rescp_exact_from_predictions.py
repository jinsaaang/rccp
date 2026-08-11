from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / "data" / "source"

from loader.generator import DataGenerator
from scripts.benchmark_core.run_rescp_exact_replay import _CopiedReservoir, _metrics, _run_sampler


def _load_predictions(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run the ResCP exact residual sampler from saved central forecaster predictions."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions.pt saved by train_lstm_forecaster.py.")
    parser.add_argument("--dataset-config", default=str(REPO_ROOT / "config/dataset/rescp_bejing_air_pm10.yaml"))
    parser.add_argument("--task-config", default=str(REPO_ROOT / "config/task/default_3year_gn.yaml"))
    parser.add_argument("--data-base-dir", default=str(DATA_ROOT))
    parser.add_argument("--dataset-name", default="air_rescp")
    parser.add_argument("--forecaster-name", default="rescp_native_rnn")
    parser.add_argument("--out", default=str(REPO_ROOT / "results/raw/rescp_exact_from_predictions_air_442.json"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--split", default="0.4,0.4,0.2")
    parser.add_argument(
        "--prediction-offset",
        type=int,
        default=24,
        help="First absolute time index covered by the saved predictions; usually backbone seq_len/window.",
    )
    parser.add_argument("--reservoir-seed", type=int, default=0, help="Random seed for the ResCP reservoir sampler.")
    args = parser.parse_args()

    split = [float(x) for x in args.split.split(",")]
    if len(split) != 3:
        raise ValueError("--split must have train,calib,test fractions.")

    dataset_cfg = OmegaConf.load(args.dataset_config)
    task_cfg = OmegaConf.load(args.task_config)
    task_cfg.data_splits = split
    datasets = DataGenerator.get_data(dataset_cfg, task_cfg, replace_base_dir=str(Path(args.data_base_dir)))
    ts_ids = [d.ts_id for d in datasets]
    y = torch.stack([d.Y_full.float().reshape(-1) for d in datasets], dim=1)
    mask = torch.stack(
        [
            getattr(d, "mask_full", torch.ones_like(d.Y_full, dtype=torch.bool)).float().reshape(-1)
            for d in datasets
        ],
        dim=1,
    ).bool()
    y_mean, y_std = datasets[0].Y_normalize_props
    y_mean = torch.as_tensor(y_mean).float().reshape(1, 1)
    y_std = torch.as_tensor(y_std).float().reshape(1, 1)
    n_steps, n_nodes = y.shape

    train_end = int(n_steps * split[0] / sum(split))
    test_start = int(n_steps * (split[0] + split[1]) / sum(split))
    test_end = n_steps
    calib_indices = torch.arange(train_end, test_start)
    test_indices = torch.arange(test_start, test_end)

    predictions = _load_predictions(Path(args.predictions))
    y_hat = torch.full((n_steps, n_nodes), torch.nan)
    for node, ts_id in enumerate(ts_ids):
        if ts_id not in predictions:
            raise KeyError(f"Missing prediction for {ts_id}. Available keys include: {list(predictions)[:5]}")
        pred = predictions[ts_id].detach().cpu().float().reshape(-1)
        start = int(args.prediction_offset)
        end = min(n_steps, start + pred.numel())
        y_hat[start:end, node] = pred[: end - start]

    needed = torch.cat([calib_indices, test_indices])
    if torch.isnan(y_hat[needed]).any():
        bad = torch.isnan(y_hat[needed]).nonzero()[0].tolist()
        raise RuntimeError(f"Saved predictions do not cover 4/4/2 calib/test positions; first missing relative index={bad}")

    residual = y - y_hat
    residual_used = residual[needed].clone()
    lower = torch.quantile(residual_used, 0.005, dim=0)
    upper = torch.quantile(residual_used, 0.995, dim=0)
    residual_clipped = residual.clone()
    residual_clipped[:, :] = torch.clamp(residual_clipped, lower, upper)

    cal_residuals = residual_clipped[calib_indices].unsqueeze(-1)
    test_residuals = residual_clipped[test_indices]
    y_test = y[test_indices]
    y_hat_test = (y_test - test_residuals).unsqueeze(-1)
    y_true = y_test.unsqueeze(-1)
    test_mask = mask[test_indices]

    state_input = residual_clipped[needed].unsqueeze(-1).numpy().astype(np.float32)
    cal_len = int(calib_indices.numel())
    scaler_mean = np.nanmean(state_input[:cal_len])
    scaler_std = np.nanstd(state_input[:cal_len])
    scaled = (state_input - scaler_mean) / scaler_std
    scaled = np.nan_to_num(scaled, nan=0.0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = _CopiedReservoir(seed=int(args.reservoir_seed)).to(device)
    states_after = encoder.get_states(torch.from_numpy(scaled).float().permute(1, 0, 2).to(device))
    states_after = states_after.permute(1, 0, 2)
    zero = torch.zeros(1, n_nodes, states_after.shape[-1], device=device)
    states_before = torch.cat([zero, states_after[:-1]], dim=0)
    norm = torch.linalg.norm(states_before, dim=-1, keepdim=True)
    states_before = torch.where(norm > 0, states_before / norm, states_before)

    started = time.perf_counter()
    lower_bound, upper_bound = _run_sampler(
        cal_states=states_before[:cal_len],
        cal_residuals=cal_residuals.to(device),
        test_states=states_before[cal_len:],
        y_hat=y_hat_test.to(device),
        y_true=y_true.to(device),
        alpha=float(args.alpha),
    )
    result_norm = _metrics(lower_bound, upper_bound, y_test.to(device), test_mask.to(device), float(args.alpha))
    lower_raw = lower_bound.cpu() * y_std + y_mean
    upper_raw = upper_bound.cpu() * y_std + y_mean
    y_test_raw = y_test * y_std + y_mean
    result = _metrics(
        lower_raw.to(device),
        upper_raw.to(device),
        y_test_raw.to(device),
        test_mask.to(device),
        float(args.alpha),
    )
    sampler_elapsed = time.perf_counter() - started
    wall_elapsed = time.perf_counter() - wall_started
    result.update(
        {
            "method": "rescp_exact_from_predictions",
            "dataset": args.dataset_name,
            "forecaster": args.forecaster_name,
            "split": args.split,
            "scale": "raw_pm10",
            "coverage_norm": result_norm["coverage"],
            "width_norm": result_norm["width"],
            "winkler_norm_scale": result_norm["winkler"],
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "prediction_offset": int(args.prediction_offset),
            "reservoir_seed": int(args.reservoir_seed),
            "prediction_path": str(Path(args.predictions).resolve()),
            "sampler_time_sec": sampler_elapsed,
            "calibration_time_sec": sampler_elapsed,
            "time_sec": wall_elapsed,
            "wall_clock_time_sec": wall_elapsed,
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
