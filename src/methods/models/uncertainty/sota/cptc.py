from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PICalibArtifacts,
    PICalibData,
    PIModelPrediction,
    PIPredictionStepData,
)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_features(x) -> np.ndarray:
    x = _as_numpy(x).astype(np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x.reshape(x.shape[0], -1)


def _interval_union_stats(intervals, y_value):
    if not intervals:
        return False, 0.0
    merged = []
    for low, high in sorted(intervals):
        low = float(low)
        high = float(high)
        if not merged or low > merged[-1][1]:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    for low, high in merged:
        if low <= y_value <= high:
            return True, min(y_value - low, high - y_value)
    dist = min(min(abs(y_value - low), abs(y_value - high)) for low, high in merged)
    return False, float(dist)


class CPTCModel(PIModel):
    """
    CPTC adapter for the conformal framework.

    The online update rule follows the original CPTC code:
    - state-specific residual pools
    - state-specific alpha updates
    - union of state intervals until probability mass reaches 1 - alpha

    Adaptation to the conformal loaders:
    - state probabilities come from a Gaussian mixture fitted on forecast states
      (or raw x-features when no forecast state exists)
    - state means are forecast means adjusted by per-state residual bias
    """

    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._n_states = int(kwargs.get("n_states", 3))
        self._prob_threshold = float(kwargs.get("prob_threshold", 0.3))
        self._gamma = float(kwargs.get("gamma", 0.03))
        self._min_residuals = int(kwargs.get("min_residuals", 25))
        self._covariance_type = str(kwargs.get("covariance_type", "diag"))
        self._reg_covar = float(kwargs.get("reg_covar", 1e-6))
        self._state_source = str(kwargs.get("state_source", "auto"))
        self._random_state = int(kwargs.get("random_state", 10))
        self._max_radius_scale = float(kwargs.get("max_radius_scale", 1.0))

        self._gmm: Optional[GaussianMixture] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._state_bias: Optional[np.ndarray] = None
        self._state_alpha: Optional[np.ndarray] = None
        self._state_residuals: Optional[Dict[int, List[float]]] = None
        self._all_residuals: Optional[List[float]] = None
        self._max_radius: Optional[float] = None
        self._last_probs: Optional[np.ndarray] = None
        self._last_region: Optional[List[tuple]] = None
        self._last_point_pred: Optional[float] = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        return None

    def _resolve_feature_source(self, fc_state) -> str:
        if self._state_source == "fc_state":
            if fc_state is None:
                raise ValueError("CPTC state_source='fc_state' but forecast model returned no state")
            return "fc_state"
        if self._state_source == "x_step":
            return "x_step"
        if self._state_source != "auto":
            raise ValueError(f"Unsupported CPTC state_source: {self._state_source}")
        return "fc_state" if fc_state is not None else "x_step"

    def _extract_features(self, fc_state, x_batch) -> np.ndarray:
        source = self._resolve_feature_source(fc_state)
        return _flatten_features(fc_state if source == "fc_state" else x_batch)

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        return (feats - self._feature_mean) / self._feature_std

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del calib_artifact, mix_calib_data, mix_calib_artifact
        fc_result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=calib_data.ts_id,
                X_past=calib_data.X_pre_calib,
                Y_past=calib_data.Y_pre_calib,
                X_step=calib_data.X_calib,
                step_offset=calib_data.step_offset,
            )
        )
        y_hat_cal = _as_numpy(fc_result.point).reshape(-1)
        y_cal = _as_numpy(calib_data.Y_calib).reshape(-1)
        fc_state = getattr(fc_result, "state", None)
        features = self._extract_features(fc_state, calib_data.X_calib)
        self._feature_mean = features.mean(axis=0, keepdims=True)
        self._feature_std = features.std(axis=0, keepdims=True) + 1e-6
        norm_features = self._normalize_features(features)
        self._gmm = GaussianMixture(
            n_components=self._n_states,
            covariance_type=self._covariance_type,
            reg_covar=self._reg_covar,
            random_state=self._random_state,
        )
        self._gmm.fit(norm_features)
        probs = self._gmm.predict_proba(norm_features)
        residual = y_cal - y_hat_cal
        abs_residual = np.abs(residual)
        bias_denom = np.clip(probs.sum(axis=0), 1e-6, None)
        self._state_bias = (probs.T @ residual) / bias_denom
        self._state_alpha = np.full(self._n_states, float(alpha), dtype=np.float32)
        max_radius = max(float(np.max(abs_residual)) * self._max_radius_scale, 1e-6)
        self._max_radius = max_radius
        self._state_residuals = {z: [max_radius] for z in range(self._n_states)}
        for t in range(abs_residual.shape[0]):
            for z, p_z_t in enumerate(probs[t]):
                if p_z_t > self._prob_threshold:
                    self._state_residuals[z].append(float(abs_residual[t]))
        self._all_residuals = abs_residual.astype(np.float32).tolist() + [max_radius]
        self._last_probs = None
        self._last_region = None
        self._last_point_pred = None
        return PICalibArtifacts(fc_Y_hat=fc_result.point, eps=(y_cal - y_hat_cal), fc_state_step=fc_state)

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._state_alpha = np.full(self._n_states, float(kwargs["alpha"]), dtype=np.float32)
        self._last_probs = None
        self._last_region = None
        self._last_point_pred = None

    def _build_region(self, y_hat: float, probs: np.ndarray):
        intervals = []
        for z in range(self._n_states):
            p_z = float(probs[z])
            if p_z <= 0:
                continue
            residuals = self._state_residuals[z]
            if len(residuals) < self._min_residuals:
                residuals = self._all_residuals
            q = float(np.clip(1.0 - self._state_alpha[z], 0.0, 1.0))
            radius = float(np.quantile(np.asarray(residuals, dtype=np.float32), q))
            center = float(y_hat + self._state_bias[z])
            intervals.append((center, radius, p_z))
        intervals = sorted(intervals, key=lambda x: x[2])
        selected = []
        running_sum = 0.0
        target_mass = 1.0 - float(np.mean(self._state_alpha))
        for interval in intervals:
            if running_sum < target_mass:
                selected.append(interval)
                running_sum += interval[2]
            else:
                break
        if not selected and intervals:
            selected.append(intervals[-1])
        pred_set = [(center - radius, center + radius) for center, radius, _ in selected]
        low = min(low for low, _ in pred_set)
        high = max(high for _, high in pred_set)
        return selected, pred_set, low, high

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        fc_result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=pred_data.X_past,
                Y_past=pred_data.Y_past,
                X_step=pred_data.X_step,
                step_offset=pred_data.step_offset_overall,
            )
        )
        y_hat = _as_numpy(fc_result.point).reshape(-1)[0]
        fc_state = getattr(fc_result, "state", None)
        features = self._extract_features(fc_state, pred_data.X_step)
        probs = self._gmm.predict_proba(self._normalize_features(features))[0]
        region, pred_set, low, high = self._build_region(y_hat, probs)
        self._last_probs = probs
        self._last_region = region
        self._last_point_pred = y_hat
        return PIModelPrediction(
            pred_interval=(torch.tensor([[low]], dtype=torch.float32), torch.tensor([[high]], dtype=torch.float32)),
            fc_Y_hat=fc_result.point,
            pred_set=[(float(low_i), float(high_i)) for low_i, high_i in pred_set],
        )

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        y_true = float(_as_numpy(Y_step).reshape(-1)[0])
        covered, _ = _interval_union_stats(pred_result.pred_set or [], y_true)
        err_t = int(not covered)
        sampled_state = int(np.random.choice(np.arange(self._n_states), p=self._last_probs))
        self._state_alpha[sampled_state] = self._state_alpha[sampled_state] + (self._gamma * (pred_data.alpha - err_t))
        self._state_alpha[sampled_state] = float(np.clip(self._state_alpha[sampled_state], 1e-4, 1 - 1e-4))
        abs_resid = abs(y_true - float(self._last_point_pred))
        self._state_residuals[sampled_state].append(float(abs_resid))
        self._all_residuals.append(float(abs_resid))

    def model_ready(self):
        return self._gmm is not None and self._state_bias is not None and self._state_alpha is not None

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None

    @property
    def can_handle_different_alpha(self):
        return False
