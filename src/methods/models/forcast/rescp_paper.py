"""Shared ResCP-paper backbone adapter for central baselines.

This is not an upstream import. It reads the base-model residual artifact and
reconstructs the paper RNN point forecasts so any central uncertainty method can
run on the same Beijing/ResCP backbone and split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from loader.dataset import ChronoSplittedTsDataset
from models.forcast.forcast_base import (
    FCModelPrediction,
    FCPredictionData,
    FcSingleModelPrediction,
    ForcastMode,
    ForcastModel,
    PredictionOutputType,
)


class ResCPPaperForecaster(ForcastModel):
    def __init__(self, no_x_features, model_params, **kwargs):
        super().__init__(
            forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
            supported_outputs=(PredictionOutputType.POINT,),
        )
        self._artifact_dir = Path(model_params["artifact_dir"]).expanduser()
        self._clip_residuals = bool(model_params.get("clip_residuals", True))
        self._predictions: dict[str, torch.Tensor] = {}

    @property
    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return False

    @property
    def fc_state_dim(self):
        return None

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs):
        raise NotImplementedError("ResCP paper forecaster is prepared with train_global().")

    def train_global(
        self,
        datasets: Iterable[ChronoSplittedTsDataset],
        alphas,
        trainer_config,
        experiment_config,
    ):
        datasets = list(datasets)
        residuals = _normalize_columns(pd.read_hdf(self._artifact_dir / "residuals.h5", key="target"))
        if self._clip_residuals:
            residuals_input = _normalize_columns(pd.read_hdf(self._artifact_dir / "residuals.h5", key="input"))
            lower_clip = residuals_input.quantile(0.005, axis=0)
            upper_clip = residuals_input.quantile(0.995, axis=0)
            residuals = residuals.clip(lower_clip, upper_clip, axis=1)

        indices = np.load(self._artifact_dir / "indices.npz")
        valid_target_indices = indices["valid_target_indices"].astype(int)

        for dataset in datasets:
            station = _station_from_ts_id(dataset.ts_id, residuals)
            col = _residual_column(residuals, station)
            residual_values = residuals[col].to_numpy(dtype=np.float32)
            mean, std = dataset.Y_normalize_props
            mean = _as_float(mean)
            std = _as_float(std)
            y_full = dataset.Y_full.detach().cpu().reshape(-1, 1).float()
            pred_full = torch.full_like(y_full, float("nan"))
            usable = valid_target_indices < y_full.shape[0]
            raw_idx = valid_target_indices[usable]
            residual_norm = torch.from_numpy(residual_values[usable]).float().reshape(-1, 1) / std
            pred_full[raw_idx] = y_full[raw_idx] - residual_norm

            missing_cal = torch.isnan(pred_full[dataset.calib_step : dataset.test_step])
            if missing_cal.any():
                cal_slice = pred_full[dataset.calib_step : dataset.test_step]
                y_slice = y_full[dataset.calib_step : dataset.test_step]
                cal_slice[missing_cal] = y_slice[missing_cal]
                pred_full[dataset.calib_step : dataset.test_step] = cal_slice
            if torch.isnan(pred_full[dataset.test_step :]).any():
                raise ValueError(f"Missing paper forecaster predictions in test window for {dataset.ts_id}.")
            self._predictions[dataset.ts_id] = pred_full

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        start = int(pred_data.step_offset)
        end = start + int(pred_data.no_fc_steps)
        prediction = self._predictions[pred_data.ts_id][start:end]
        return FcSingleModelPrediction(point=prediction.to(device=pred_data.X_step.device))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=["node", "channel"])
    return df


def _station_from_ts_id(ts_id: str, residuals: pd.DataFrame) -> str:
    stations = list(residuals.columns.get_level_values(0).unique())
    matches = [station for station in stations if ts_id.endswith(str(station))]
    if len(matches) != 1:
        raise ValueError(f"Cannot map {ts_id!r} to one ResCP station; matches={matches}")
    return str(matches[0])


def _residual_column(residuals: pd.DataFrame, station: str):
    columns = [col for col in residuals.columns if str(col[0]) == station]
    if len(columns) != 1:
        raise ValueError(f"Expected one residual column for {station}, got {columns}")
    return columns[0]


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])
