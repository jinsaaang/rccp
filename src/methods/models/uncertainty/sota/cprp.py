from typing import Optional, List, Tuple

import numpy as np
import torch

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
    PIModelPrediction,
)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return scores
    temp = max(float(temperature), 1e-6)
    scores = scores / temp
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    denom = np.sum(exp_scores)
    if denom <= 0:
        return np.ones_like(exp_scores) / exp_scores.shape[0]
    return exp_scores / denom


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if values.size != weights.size:
        raise ValueError("weights must match values length")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(max(int(idx), 0), sorted_values.shape[0] - 1)
    return float(sorted_values[idx])


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6
    return (mat / norms).astype(np.float32)


def _pearson_similarity_rows(memory_keys: np.ndarray, query_key: np.ndarray) -> np.ndarray:
    memory_centered = memory_keys - memory_keys.mean(axis=1, keepdims=True)
    query_centered = query_key - query_key.mean()
    memory_norm = np.linalg.norm(memory_centered, axis=1) + 1e-6
    query_norm = np.linalg.norm(query_centered) + 1e-6
    return ((memory_centered @ query_centered) / (memory_norm * query_norm)).astype(np.float32)


class CPRPModel(PIModel):
    """
    CPRP-style retrieval conformal predictor for conformal framework.
    - key: forecast feature/state (if available) or feature fallback
    - value: standardized absolute residual score
    - scale: EWMA absolute-residual scale (leak-free online update)
    """

    def __init__(self, **kwargs):
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._k = int(kwargs.get("k", 50))
        self._temperature = float(kwargs.get("temperature", 1.0))
        self._ewma_beta = float(kwargs.get("ewma_beta", 0.95))
        self._ewma_min_scale = float(kwargs.get("ewma_min_scale", 1e-3))
        self._eps = float(kwargs.get("eps", 1e-6))
        self._online_memory = bool(kwargs.get("online_memory", True))
        self._normalize_error = bool(kwargs.get("normalize_error", True))
        self._key_source = str(kwargs.get("key_source", "auto"))
        self._similarity_metric = str(kwargs.get("similarity_metric", "cosine")).lower()
        self._memory_use_calib_frac = float(kwargs.get("memory_use_calib_frac", 1.0))
        self._use_recent_calib = bool(kwargs.get("use_recent_calib", True))
        self._max_memory = kwargs.get("max_memory", None)
        if self._max_memory is not None:
            self._max_memory = int(self._max_memory)
            if self._max_memory <= 0:
                self._max_memory = None

        self._memory_keys: Optional[np.ndarray] = None
        self._memory_keys_cos: Optional[np.ndarray] = None
        self._memory_keys_pearson_centered: Optional[np.ndarray] = None
        self._memory_keys_pearson_norm: Optional[np.ndarray] = None
        self._memory_score_std: Optional[np.ndarray] = None
        self._ewma_scale: Optional[float] = None
        self._last_query_key: Optional[np.ndarray] = None
        self._last_scale_before_update: Optional[float] = None
        self._effective_key_source: Optional[str] = None
        if self._similarity_metric not in {"cosine", "l2", "pearson"}:
            raise ValueError(f"Unsupported similarity_metric: {self._similarity_metric}")

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        # No global calibration is required; done per dataset in calibrate_individual.
        return None

    def _resolve_key_source(self, fc_state) -> str:
        if self._key_source == "fc_state":
            if fc_state is None:
                raise ValueError("key_source='fc_state' but forecast model returned no state")
            return "fc_state"
        if self._key_source == "x_step":
            return "x_step"
        if self._key_source != "auto":
            raise ValueError(f"Unsupported key_source: {self._key_source}")
        return "fc_state" if fc_state is not None else "x_step"

    def _extract_key_batch(self, fc_state, x_step_batch) -> np.ndarray:
        source = self._resolve_key_source(fc_state)
        self._effective_key_source = source
        if source == "fc_state":
            keys = _as_numpy(fc_state)
        else:
            keys = _as_numpy(x_step_batch)
        keys = np.asarray(keys, dtype=np.float32)
        if keys.ndim == 1:
            keys = keys[None, :]
        else:
            keys = keys.reshape(keys.shape[0], -1)
        return keys.astype(np.float32)

    def _compute_similarity_scores(self, query_key: np.ndarray) -> np.ndarray:
        query_key = np.asarray(query_key, dtype=np.float32).reshape(-1)
        if self._similarity_metric == "cosine":
            query_norm = _l2_normalize_rows(query_key)[0]
            return (self._memory_keys_cos @ query_norm).astype(np.float32)
        if self._similarity_metric == "l2":
            memory_keys = np.asarray(self._memory_keys, dtype=np.float32)
            return (-np.linalg.norm(memory_keys - query_key[None, :], axis=1)).astype(np.float32)
        query_centered = query_key - query_key.mean()
        query_norm = np.linalg.norm(query_centered) + 1e-6
        return ((self._memory_keys_pearson_centered @ query_centered) / (self._memory_keys_pearson_norm * query_norm)).astype(np.float32)

    def _init_scale(self, abs_resid_cal: np.ndarray, memory_start_idx: int) -> float:
        if memory_start_idx > 0:
            warmup = abs_resid_cal[:memory_start_idx]
            if warmup.size > 0:
                return float(max(np.mean(warmup), self._ewma_min_scale))
        if abs_resid_cal.size > 0:
            return float(max(np.mean(abs_resid_cal), self._ewma_min_scale))
        return float(self._ewma_min_scale)

    def _ewma_update(self, scale: float, abs_resid: float) -> float:
        updated = self._ewma_beta * float(scale) + (1.0 - self._ewma_beta) * float(abs_resid)
        return float(max(updated, self._ewma_min_scale))

    def _append_memory(self, key: np.ndarray, score_std: float):
        key = np.asarray(key, dtype=np.float32).reshape(1, -1)
        score_std = float(score_std)
        key_cos = _l2_normalize_rows(key)
        key_centered = key - key.mean(axis=1, keepdims=True)
        key_centered_norm = (np.linalg.norm(key_centered, axis=1) + 1e-6).astype(np.float32)
        if self._memory_keys is None or self._memory_score_std is None:
            self._memory_keys = key
            self._memory_keys_cos = key_cos
            self._memory_keys_pearson_centered = key_centered.astype(np.float32)
            self._memory_keys_pearson_norm = key_centered_norm
            self._memory_score_std = np.array([score_std], dtype=np.float32)
        else:
            self._memory_keys = np.vstack([self._memory_keys, key])
            self._memory_keys_cos = np.vstack([self._memory_keys_cos, key_cos])
            self._memory_keys_pearson_centered = np.vstack([self._memory_keys_pearson_centered, key_centered.astype(np.float32)])
            self._memory_keys_pearson_norm = np.append(self._memory_keys_pearson_norm, key_centered_norm)
            self._memory_score_std = np.append(self._memory_score_std, np.float32(score_std))
        if self._max_memory is not None and self._memory_score_std.shape[0] > self._max_memory:
            self._memory_keys = self._memory_keys[-self._max_memory :]
            self._memory_keys_cos = self._memory_keys_cos[-self._max_memory :]
            self._memory_keys_pearson_centered = self._memory_keys_pearson_centered[-self._max_memory :]
            self._memory_keys_pearson_norm = self._memory_keys_pearson_norm[-self._max_memory :]
            self._memory_score_std = self._memory_score_std[-self._max_memory :]

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del alpha, calib_artifact, mix_calib_data, mix_calib_artifact

        self._memory_keys = None
        self._memory_keys_cos = None
        self._memory_keys_pearson_centered = None
        self._memory_keys_pearson_norm = None
        self._memory_score_std = None
        self._ewma_scale = None
        self._last_query_key = None
        self._last_scale_before_update = None

        fc_result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=calib_data.ts_id,
                X_past=calib_data.X_pre_calib,
                Y_past=calib_data.Y_pre_calib,
                X_step=calib_data.X_calib,
                step_offset=calib_data.step_offset,
            )
        )
        y_hat_cal = fc_result.point
        y_cal = _as_numpy(calib_data.Y_calib).reshape(-1)
        y_hat_cal_np = _as_numpy(y_hat_cal).reshape(-1)
        abs_resid_cal = np.abs(y_cal - y_hat_cal_np).astype(np.float32)
        keys_cal = self._extract_key_batch(fc_result.state, calib_data.X_calib)
        if keys_cal.shape[0] != abs_resid_cal.shape[0]:
            raise ValueError(
                f"Mismatch between key rows ({keys_cal.shape[0]}) and residuals ({abs_resid_cal.shape[0]})"
            )

        calib_size = abs_resid_cal.shape[0]
        keep_frac = float(np.clip(self._memory_use_calib_frac, 0.0, 1.0))
        if calib_size == 0 or keep_frac == 0.0:
            raise ValueError("Calibration set for CPRP memory is empty after filtering")
        keep_count = max(1, int(round(calib_size * keep_frac)))
        memory_start_idx = calib_size - keep_count if self._use_recent_calib else 0
        memory_end_idx = calib_size if self._use_recent_calib else keep_count

        ewma_scale = self._init_scale(abs_resid_cal, memory_start_idx)
        for idx in range(memory_start_idx, memory_end_idx):
            b_t = max(float(ewma_scale), self._ewma_min_scale)
            if self._normalize_error:
                s_t = float(abs_resid_cal[idx] / (b_t + self._eps))
            else:
                s_t = float(abs_resid_cal[idx])
            self._append_memory(keys_cal[idx], s_t)
            ewma_scale = self._ewma_update(ewma_scale, float(abs_resid_cal[idx]))
        self._ewma_scale = float(ewma_scale)

        artifact = PICalibArtifacts(fc_Y_hat=y_hat_cal, eps=abs_resid_cal.reshape(-1, 1))
        artifact.add_info = {
            "memory_start_idx": int(memory_start_idx),
            "memory_end_idx": int(memory_end_idx),
            "key_source_used": self._effective_key_source,
            "similarity_metric": self._similarity_metric,
            "memory_size": int(self._memory_score_std.shape[0]) if self._memory_score_std is not None else 0,
        }
        return artifact

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if not self.model_ready():
            raise ValueError("CPRP model is not calibrated")

        fc_result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=pred_data.X_past,
                Y_past=pred_data.Y_past,
                X_step=pred_data.X_step,
                step_offset=pred_data.step_offset_overall,
            )
        )
        y_hat = fc_result.point
        query_key = self._extract_key_batch(fc_result.state, pred_data.X_step)[0]

        b_t = max(float(self._ewma_scale), self._ewma_min_scale)
        sims = self._compute_similarity_scores(query_key)
        if self._k > 0 and self._k < sims.shape[0]:
            top_idx = np.argpartition(-sims, self._k)[: self._k]
        else:
            top_idx = np.arange(sims.shape[0])
        weights = _softmax(sims[top_idx], temperature=self._temperature)
        c_t = _weighted_quantile(
            self._memory_score_std[top_idx], q=1.0 - float(pred_data.alpha), weights=weights
        )
        if self._normalize_error:
            radius = float(b_t * c_t)
        else:
            radius = float(c_t)
        pred_int = y_hat - radius, y_hat + radius

        self._last_query_key = query_key.astype(np.float32)
        self._last_scale_before_update = float(b_t)
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=y_hat)

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        del pred_data, kwargs
        if self._ewma_scale is None:
            return
        y_step = _as_numpy(Y_step).reshape(-1)
        y_hat = _as_numpy(pred_result.fc_Y_hat).reshape(-1)
        if y_step.size == 0 or y_hat.size == 0:
            return
        abs_resid = float(np.abs(y_step[0] - y_hat[0]))
        b_t = (
            self._last_scale_before_update
            if self._last_scale_before_update is not None
            else max(float(self._ewma_scale), self._ewma_min_scale)
        )
        if self._online_memory and self._last_query_key is not None:
            if self._normalize_error:
                s_new = float(abs_resid / (b_t + self._eps))
            else:
                s_new = float(abs_resid)
            self._append_memory(self._last_query_key, s_new)
        self._ewma_scale = self._ewma_update(self._ewma_scale, abs_resid)
        self._last_query_key = None
        self._last_scale_before_update = None

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.X_step is not None

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return (
            self._memory_keys is not None
            and self._memory_keys_cos is not None
            and self._memory_keys_pearson_centered is not None
            and self._memory_keys_pearson_norm is not None
            and self._memory_score_std is not None
            and self._memory_score_std.shape[0] > 0
            and self._ewma_scale is not None
        )
