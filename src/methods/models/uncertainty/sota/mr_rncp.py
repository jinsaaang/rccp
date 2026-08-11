from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.components.mass_regularized_scale import (
    MassRegularizedScaleEstimator,
    estimate_bandwidth,
    estimate_lambda_from_loo_mass,
    weighted_quantile,
)
from models.uncertainty.components.retrieval_index import create_retrieval_index
from models.uncertainty.components.retrieval_keys import as_numpy, build_raw_window_forecast_key, safe_1d_float
from models.uncertainty.pi_base import (
    PICalibArtifacts,
    PICalibData,
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
)


def _conformal_multiplier(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("calibration scores must be non-empty")
    rank = int(np.ceil((scores.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), scores.size)
    return float(np.partition(scores, rank - 1)[rank - 1])


def _next_capacity(current: int, needed: int) -> int:
    capacity = max(int(current), 64)
    while capacity < int(needed):
        capacity *= 2
    return capacity


def _safe_multiplier(scores: np.ndarray, alpha: float, fallback: float) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return float(fallback)
    return _conformal_multiplier(scores, alpha)


def _trend_feature(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return 0.0
    idx = np.arange(values.size, dtype=np.float32)
    idx = idx - idx.mean()
    denom = float(np.sum(idx ** 2) + 1e-6)
    return float(np.sum(idx * (values - values.mean())) / denom)


def _residual_signature(eps_history: Optional[np.ndarray], steps: int) -> np.ndarray:
    steps = max(int(steps), 1)
    if eps_history is None:
        return np.zeros((steps + 3,), dtype=np.float32)
    eps_history = np.asarray(eps_history, dtype=np.float32).reshape(-1)
    if eps_history.size == 0:
        return np.zeros((steps + 3,), dtype=np.float32)
    tail = eps_history[-steps:]
    if tail.size < steps:
        tail = np.pad(tail, (steps - tail.size, 0), mode="constant", constant_values=0.0)
    scale = float(np.mean(np.abs(tail)) + 1e-6)
    norm_tail = (tail / scale).astype(np.float32)
    extra = np.array(
        [
            float(np.mean(np.abs(tail))),
            float(np.std(tail)),
            _trend_feature(tail),
        ],
        dtype=np.float32,
    )
    return np.concatenate([norm_tail, extra]).astype(np.float32)


def _target_summary(y_history: np.ndarray, window: int) -> np.ndarray:
    y_history = np.asarray(y_history, dtype=np.float32).reshape(-1)
    if y_history.size == 0:
        return np.zeros((6,), dtype=np.float32)
    tail = y_history[-max(int(window), 1) :]
    last = float(tail[-1])
    prev = float(tail[-2]) if tail.size > 1 else last
    return np.array(
        [
            float(np.mean(tail)),
            float(np.std(tail)),
            last,
            float(last - prev),
            float(np.max(tail) - np.min(tail)),
            _trend_feature(tail),
        ],
        dtype=np.float32,
    )


def _state_row(fc_state, idx: int):
    if fc_state is None:
        return None
    state = as_numpy(fc_state)
    if state.ndim == 1:
        return state
    return state.reshape(state.shape[0], -1)[idx]


class MRRNCPModel(PIModel):
    """
    Mass-Regularized Retrieval-Normalized Conformal Prediction.

    The model uses retrieval only to estimate a local residual scale. The
    conformal multiplier is a global split conformal quantile over normalized
    calibration scores.
    """

    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._context_length = int(kwargs.get("context_length", 96))
        self._key_mode = str(kwargs.get("key", "raw_window_plus_forecast"))
        self._distance = str(kwargs.get("distance", kwargs.get("retrieval_metric", "cosine"))).lower()
        self._kernel = str(kwargs.get("kernel", "gaussian")).lower()
        self._shrinkage_mode = str(kwargs.get("shrinkage_mode", "distribution")).lower()
        self._bandwidth_rule = str(kwargs.get("bandwidth_rule", "memory_distance_quantile"))
        self._bandwidth_quantile = float(kwargs.get("bandwidth_quantile", 0.10))
        self._bandwidth = kwargs.get("bandwidth", None)
        self._lambda_rule = str(kwargs.get("lambda_rule", "median_loo_mass"))
        self._lambda_value = kwargs.get("lambda_value", None)
        self._lambda_scale = float(kwargs.get("lambda_scale", 1.0))
        self._scale_quantile = float(kwargs.get("scale_quantile", kwargs.get("scale_functional_quantile", 0.5)))
        self._side_specific_scale = bool(kwargs.get("side_specific_scale", False))
        self._retrieval_backend = str(kwargs.get("retrieval_backend", "faiss_topm")).lower()
        self._candidate_count = kwargs.get("candidate_count", None)
        self._candidate_count_rule = str(kwargs.get("candidate_count_rule", "max_1024_4sqrt"))
        self._memory_fraction = float(kwargs.get("memory_fraction", 2.0 / 3.0))
        self._min_memory = int(kwargs.get("min_memory", 5))
        self._min_calibration = int(kwargs.get("min_calibration", 5))
        self._eps_rule = str(kwargs.get("epsilon_rule", "train_std_times_1e_minus_6"))
        self._eps = kwargs.get("eps", None)
        self._online_memory = bool(kwargs.get("online_memory", False))
        self._use_forecast_value = bool(kwargs.get("use_forecast_value", False))
        self._use_target_summary = bool(kwargs.get("use_target_summary", False))
        self._target_summary_window = int(kwargs.get("target_summary_window", self._context_length))
        self._use_residual_context = bool(kwargs.get("use_residual_context", False))
        self._residual_context_steps = int(kwargs.get("residual_context_steps", 8))
        self._interval_mode = str(kwargs.get("interval_mode", kwargs.get("mode", "symmetric"))).lower()
        self._q_mode = str(kwargs.get("q_mode", "global")).lower()
        self._quantile_alpha_offset = float(kwargs.get("quantile_alpha_offset", 0.0))
        self._online_base_scale_update_interval = int(kwargs.get("online_base_scale_update_interval", 0))
        self._rolling_window = int(kwargs.get("rolling_window", 256))
        self._rolling_min = int(kwargs.get("rolling_min", max(20, min(self._rolling_window, 64))))

        if self._distance not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported distance: {self._distance}")
        if self._kernel != "gaussian":
            raise ValueError(f"Unsupported kernel: {self._kernel}")
        if self._interval_mode not in {"symmetric", "two_sided", "twosided"}:
            raise ValueError(f"Unsupported interval_mode: {self._interval_mode}")
        if self._q_mode not in {"global", "rolling"}:
            raise ValueError(f"Unsupported q_mode: {self._q_mode}")
        self._index = None
        self._scale_estimator: Optional[MassRegularizedScaleEstimator] = None
        self._memory_keys: Optional[np.ndarray] = None
        self._memory_residuals: Optional[np.ndarray] = None
        self._memory_keys_buffer: Optional[np.ndarray] = None
        self._memory_residuals_buffer: Optional[np.ndarray] = None
        self._memory_size = 0
        self._online_appends_since_base_update = 0
        self._calib_scores: Optional[np.ndarray] = None
        self._calib_scores_pos: Optional[np.ndarray] = None
        self._calib_scores_neg: Optional[np.ndarray] = None
        self._rolling_scores = []
        self._rolling_scores_pos = []
        self._rolling_scores_neg = []
        self._last_scale_for_update: Optional[float] = None
        self._last_key_for_update: Optional[np.ndarray] = None
        self._active_alpha: Optional[float] = None
        self._active_q: Optional[float] = None
        self._active_q_pos: Optional[float] = None
        self._active_q_neg: Optional[float] = None
        self._candidate_k: Optional[int] = None
        self._bandwidth_value: Optional[float] = None
        self._lambda_value_resolved: Optional[float] = None
        self._eps_value: Optional[float] = None
        self._last_scale_info = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        return None

    def _split_memory_calibration(self, size: int) -> int:
        if size < 2:
            raise ValueError("MR-RNCP needs at least two calibration points to split memory and calibration")
        memory_end = int(round(size * float(np.clip(self._memory_fraction, 0.05, 0.95))))
        if size >= self._min_memory + self._min_calibration:
            memory_end = min(max(memory_end, self._min_memory), size - self._min_calibration)
        else:
            memory_end = min(max(memory_end, 1), size - 1)
        return int(memory_end)

    def _candidate_count_for_memory(self, memory_size: int) -> int:
        if self._retrieval_backend == "exact_full":
            return int(memory_size)
        if self._candidate_count is not None:
            return min(max(int(self._candidate_count), 1), int(memory_size))
        if self._candidate_count_rule == "max_1024_4sqrt":
            count = max(1024, int(4 * np.ceil(np.sqrt(memory_size))))
        elif self._candidate_count_rule == "sqrt":
            count = int(np.ceil(np.sqrt(memory_size)))
        else:
            count = int(memory_size)
        return min(max(count, 1), int(memory_size))

    def _resolve_eps(self, y_pre: np.ndarray, y_cal: np.ndarray) -> float:
        if self._eps is not None:
            return float(self._eps)
        ref = y_pre if y_pre.size > 1 else y_cal
        scale = float(np.std(ref) + 1e-6)
        if self._eps_rule == "train_std_times_1e_minus_6":
            return float(1e-6 * scale)
        return 1e-6

    def _resolve_bandwidth(self, memory_keys: np.ndarray) -> float:
        if self._bandwidth is not None:
            return float(self._bandwidth)
        if self._bandwidth_rule != "memory_distance_quantile":
            raise ValueError(f"Unsupported bandwidth_rule: {self._bandwidth_rule}")
        return estimate_bandwidth(
            memory_keys,
            metric=self._distance,
            quantile=self._bandwidth_quantile,
        )

    def _resolve_lambda(self, memory_keys: np.ndarray, bandwidth: float) -> float:
        if self._lambda_value is not None:
            return float(self._lambda_value)
        if self._lambda_rule == "zero":
            return 0.0
        if self._lambda_rule != "median_loo_mass":
            raise ValueError(f"Unsupported lambda_rule: {self._lambda_rule}")
        return float(
            self._lambda_scale
            * estimate_lambda_from_loo_mass(
                memory_keys,
                metric=self._distance,
                bandwidth=bandwidth,
                kernel=self._kernel,
                quantile=0.50,
            )
        )

    def _build_key(self, y_history, y_hat, fc_state=None, eps_history=None) -> np.ndarray:
        key = build_raw_window_forecast_key(
            y_history=y_history,
            y_hat=y_hat,
            context_length=self._context_length,
            mode=self._key_mode,
            fc_state=fc_state,
        )
        parts = [key]
        if self._use_forecast_value:
            parts.append(safe_1d_float(y_hat)[:1].astype(np.float32))
        if self._use_target_summary:
            parts.append(_target_summary(y_history, self._target_summary_window))
        if self._use_residual_context:
            parts.append(_residual_signature(eps_history, self._residual_context_steps))
        return np.concatenate(parts).astype(np.float32)

    def _build_calibration_keys(self, y_pre: np.ndarray, y_cal: np.ndarray, y_hat_cal: np.ndarray, fc_state, eps_cal: np.ndarray) -> np.ndarray:
        keys = []
        for idx in range(y_cal.shape[0]):
            y_history = np.concatenate([y_pre, y_cal[:idx]]).astype(np.float32)
            keys.append(
                self._build_key(
                    y_history=y_history,
                    y_hat=np.array([y_hat_cal[idx]], dtype=np.float32),
                    fc_state=_state_row(fc_state, idx),
                    eps_history=eps_cal[:idx],
                )
            )
        return np.stack(keys, axis=0).astype(np.float32)

    def _fit_memory(self, memory_keys: np.ndarray, memory_residuals: np.ndarray):
        memory_keys = memory_keys.astype(np.float32)
        memory_residuals = memory_residuals.astype(np.float32)
        self._memory_size = int(memory_keys.shape[0])
        capacity = _next_capacity(0, self._memory_size)
        self._memory_keys_buffer = np.empty((capacity, memory_keys.shape[1]), dtype=np.float32)
        self._memory_residuals_buffer = np.empty((capacity, memory_residuals.shape[1]), dtype=np.float32)
        self._memory_keys_buffer[: self._memory_size] = memory_keys
        self._memory_residuals_buffer[: self._memory_size] = memory_residuals
        self._refresh_memory_views()
        self._bandwidth_value = self._resolve_bandwidth(self._memory_keys)
        self._lambda_value_resolved = self._resolve_lambda(self._memory_keys, self._bandwidth_value)
        self._index = create_retrieval_index(self._retrieval_backend, metric=self._distance)
        self._index.fit(self._memory_keys)
        self._candidate_k = self._candidate_count_for_memory(self._memory_keys.shape[0])
        self._scale_estimator = MassRegularizedScaleEstimator(
            residuals=self._memory_residuals,
            bandwidth=self._bandwidth_value,
            lambda_value=self._lambda_value_resolved,
            metric=self._distance,
            kernel=self._kernel,
            scale_quantile=self._scale_quantile,
            shrinkage_mode=self._shrinkage_mode,
            ignore_zero_values=self._side_specific_scale,
        )

    def _refresh_memory_views(self):
        if self._memory_keys_buffer is None or self._memory_residuals_buffer is None:
            self._memory_keys = None
            self._memory_residuals = None
            return
        self._memory_keys = self._memory_keys_buffer[: self._memory_size]
        self._memory_residuals = self._memory_residuals_buffer[: self._memory_size]
        if self._scale_estimator is not None:
            self._scale_estimator.residuals = self._memory_residuals

    def _ensure_memory_capacity(self, needed: int, key_dim: int, residual_dim: int):
        current = 0 if self._memory_keys_buffer is None else int(self._memory_keys_buffer.shape[0])
        if current >= int(needed):
            return
        new_capacity = _next_capacity(current, needed)
        new_keys = np.empty((new_capacity, key_dim), dtype=np.float32)
        new_residuals = np.empty((new_capacity, residual_dim), dtype=np.float32)
        if self._memory_size > 0:
            new_keys[: self._memory_size] = self._memory_keys_buffer[: self._memory_size]
            new_residuals[: self._memory_size] = self._memory_residuals_buffer[: self._memory_size]
        self._memory_keys_buffer = new_keys
        self._memory_residuals_buffer = new_residuals
        self._refresh_memory_views()

    def _scale_for_key(self, key: np.ndarray):
        if self._index is None or self._scale_estimator is None or self._candidate_k is None:
            raise ValueError("MR-RNCP memory is not fitted")
        result = self._index.search(np.asarray(key, dtype=np.float32).reshape(1, -1), self._candidate_k)
        scale_result = self._scale_estimator.compute(result.indices[0], result.scores[0])
        self._last_scale_info = scale_result
        return scale_result

    def _memory_residual_row(self, signed_residual: float) -> np.ndarray:
        signed_residual = float(signed_residual)
        if self._side_specific_scale:
            return np.asarray([max(signed_residual, 0.0), max(-signed_residual, 0.0)], dtype=np.float32)
        return np.asarray([abs(signed_residual)], dtype=np.float32)

    def _append_memory(self, key: np.ndarray, signed_residual: float):
        if self._index is None or self._scale_estimator is None:
            return
        key = np.asarray(key, dtype=np.float32).reshape(1, -1)
        residual_arr = self._memory_residual_row(signed_residual).reshape(1, -1)
        self._ensure_memory_capacity(self._memory_size + 1, key.shape[1], residual_arr.shape[1])
        insert_idx = self._memory_size
        self._memory_keys_buffer[insert_idx] = key[0]
        self._memory_residuals_buffer[insert_idx] = residual_arr[0]
        self._memory_size += 1
        self._refresh_memory_views()
        self._index.add(key)
        self._online_appends_since_base_update += 1
        if (
            self._online_base_scale_update_interval > 0
            and self._online_appends_since_base_update >= self._online_base_scale_update_interval
        ):
            self._scale_estimator.base_scale = self._scale_estimator._compute_base_scale()
            self._online_appends_since_base_update = 0

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del calib_artifact, mix_calib_data, mix_calib_artifact

        self._index = None
        self._scale_estimator = None
        self._memory_keys = None
        self._memory_residuals = None
        self._memory_keys_buffer = None
        self._memory_residuals_buffer = None
        self._memory_size = 0
        self._online_appends_since_base_update = 0
        self._calib_scores = None
        self._calib_scores_pos = None
        self._calib_scores_neg = None
        self._rolling_scores = []
        self._rolling_scores_pos = []
        self._rolling_scores_neg = []
        self._last_scale_for_update = None
        self._last_key_for_update = None
        self._active_alpha = None
        self._active_q = None
        self._active_q_pos = None
        self._active_q_neg = None
        self._last_scale_info = None

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
        y_cal = safe_1d_float(calib_data.Y_calib)
        y_hat_cal_np = safe_1d_float(y_hat_cal)
        if y_cal.shape[0] != y_hat_cal_np.shape[0]:
            raise ValueError(f"Mismatch between calibration labels ({y_cal.shape[0]}) and forecasts ({y_hat_cal_np.shape[0]})")
        y_pre = safe_1d_float(calib_data.Y_pre_calib) if calib_data.Y_pre_calib is not None else np.zeros((0,), dtype=np.float32)
        abs_resid_cal = np.abs(y_cal - y_hat_cal_np).astype(np.float32)
        signed_resid_cal = (y_cal - y_hat_cal_np).astype(np.float32)
        all_keys = self._build_calibration_keys(y_pre, y_cal, y_hat_cal_np, fc_result.state, abs_resid_cal)

        memory_end = self._split_memory_calibration(abs_resid_cal.shape[0])
        memory_keys = all_keys[:memory_end]
        if self._side_specific_scale:
            memory_residuals = np.stack(
                [
                    np.maximum(signed_resid_cal[:memory_end], 0.0),
                    np.maximum(-signed_resid_cal[:memory_end], 0.0),
                ],
                axis=1,
            ).astype(np.float32)
        else:
            memory_residuals = abs_resid_cal[:memory_end].reshape(-1, 1)
        self._eps_value = self._resolve_eps(y_pre, y_cal)
        self._fit_memory(memory_keys, memory_residuals)

        scores = []
        scores_pos = []
        scores_neg = []
        masses = []
        rhos = []
        scales = []
        for idx in range(memory_end, abs_resid_cal.shape[0]):
            scale_result = self._scale_for_key(all_keys[idx])
            signed_resid = float(y_cal[idx] - y_hat_cal_np[idx])
            if self._side_specific_scale:
                scale_pos = float(scale_result.scale[0])
                scale_neg = float(scale_result.scale[1])
                score_pos = float(max(signed_resid, 0.0) / (scale_pos + self._eps_value))
                score_neg = float(max(-signed_resid, 0.0) / (scale_neg + self._eps_value))
                score = score_pos if signed_resid >= 0 else score_neg
                scale = 0.5 * (scale_pos + scale_neg)
            else:
                scale = float(scale_result.scale[0])
                denom = scale + self._eps_value
                score = float(abs(signed_resid) / denom)
                score_pos = float(max(signed_resid, 0.0) / denom)
                score_neg = float(max(-signed_resid, 0.0) / denom)
            scores.append(score)
            scores_pos.append(score_pos)
            scores_neg.append(score_neg)
            masses.append(float(scale_result.mass))
            rhos.append(float(scale_result.rho))
            scales.append(scale)
            if self._online_memory:
                self._append_memory(all_keys[idx], signed_resid)
        self._calib_scores = np.asarray(scores, dtype=np.float32)
        self._calib_scores_pos = np.asarray(scores_pos, dtype=np.float32)
        self._calib_scores_neg = np.asarray(scores_neg, dtype=np.float32)
        self._rolling_scores = list(self._calib_scores[-self._rolling_window :])
        self._rolling_scores_pos = list(self._calib_scores_pos[-self._rolling_window :])
        self._rolling_scores_neg = list(self._calib_scores_neg[-self._rolling_window :])
        self.pre_predict(alpha=alpha)

        artifact = PICalibArtifacts(fc_Y_hat=y_hat_cal, eps=abs_resid_cal.reshape(-1, 1))
        artifact.add_info = {
            "method": "mr_rncp",
            "memory_start_idx": 0,
            "memory_end_idx": int(memory_end),
            "calibration_start_idx": int(memory_end),
            "calibration_end_idx": int(abs_resid_cal.shape[0]),
            "memory_size": int(self._memory_size),
            "calibration_score_size": int(self._calib_scores.shape[0]),
            "retrieval_backend": self._retrieval_backend,
            "distance": self._distance,
            "kernel": self._kernel,
            "shrinkage_mode": self._shrinkage_mode,
            "side_specific_scale": self._side_specific_scale,
            "candidate_k": int(self._candidate_k),
            "bandwidth": float(self._bandwidth_value),
            "lambda": float(self._lambda_value_resolved),
            "eps": float(self._eps_value),
            "active_alpha": float(alpha),
            "effective_alpha": float(self._active_alpha),
            "quantile_alpha_offset": float(self._quantile_alpha_offset),
            "active_q": float(self._active_q),
            "active_q_pos": float(self._active_q_pos) if self._active_q_pos is not None else None,
            "active_q_neg": float(self._active_q_neg) if self._active_q_neg is not None else None,
            "interval_mode": self._interval_mode,
            "q_mode": self._q_mode,
            "use_forecast_value": self._use_forecast_value,
            "use_target_summary": self._use_target_summary,
            "target_summary_window": int(self._target_summary_window),
            "use_residual_context": self._use_residual_context,
            "residual_context_steps": int(self._residual_context_steps),
            "rolling_window": int(self._rolling_window),
            "rolling_min": int(self._rolling_min),
            "mean_mass_calib": float(np.mean(masses)) if masses else None,
            "mean_rho_calib": float(np.mean(rhos)) if rhos else None,
            "mean_scale_calib": float(np.mean(scales)) if scales else None,
        }
        return artifact

    def pre_predict(self, **kwargs):
        alpha = kwargs.get("alpha", None)
        if alpha is None:
            return
        if self._calib_scores is None:
            return
        self._active_alpha = max(float(alpha) - float(self._quantile_alpha_offset), 1e-6)
        self._active_q = _conformal_multiplier(self._calib_scores, self._active_alpha)
        side_alpha = self._active_alpha / 2.0 if self._interval_mode in {"two_sided", "twosided"} else self._active_alpha
        self._active_q_pos = _safe_multiplier(self._calib_scores_pos, side_alpha, self._active_q)
        self._active_q_neg = _safe_multiplier(self._calib_scores_neg, side_alpha, self._active_q)

    def _side_alpha(self) -> float:
        if self._active_alpha is None:
            raise ValueError("MR-RNCP alpha is not initialized")
        if self._interval_mode in {"two_sided", "twosided"}:
            return float(self._active_alpha) / 2.0
        return float(self._active_alpha)

    def _active_multipliers(self):
        if self._active_alpha is None or self._active_q is None:
            raise ValueError("MR-RNCP quantile is not initialized")
        side_alpha = self._side_alpha()
        if self._q_mode == "rolling" and len(self._rolling_scores) >= self._rolling_min:
            q = _safe_multiplier(np.asarray(self._rolling_scores, dtype=np.float32), self._active_alpha, self._active_q)
            q_pos = _safe_multiplier(np.asarray(self._rolling_scores_pos, dtype=np.float32), side_alpha, self._active_q_pos or q)
            q_neg = _safe_multiplier(np.asarray(self._rolling_scores_neg, dtype=np.float32), side_alpha, self._active_q_neg or q)
            return q, q_pos, q_neg
        return self._active_q, self._active_q_pos or self._active_q, self._active_q_neg or self._active_q

    def _append_rolling_scores(self, signed_resid: float, scale: float):
        if self._q_mode != "rolling" or self._eps_value is None:
            return
        denom = float(scale) + float(self._eps_value)
        score = float(abs(signed_resid) / denom)
        score_pos = float(max(signed_resid, 0.0) / denom)
        score_neg = float(max(-signed_resid, 0.0) / denom)
        self._rolling_scores.append(score)
        self._rolling_scores_pos.append(score_pos)
        self._rolling_scores_neg.append(score_neg)
        if len(self._rolling_scores) > self._rolling_window:
            self._rolling_scores = self._rolling_scores[-self._rolling_window :]
            self._rolling_scores_pos = self._rolling_scores_pos[-self._rolling_window :]
            self._rolling_scores_neg = self._rolling_scores_neg[-self._rolling_window :]

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if not self.model_ready():
            raise ValueError("MR-RNCP model is not calibrated")
        if self._active_q is None or self._eps_value is None:
            self.pre_predict(alpha=pred_data.alpha)
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
        y_hat_np = safe_1d_float(y_hat)
        key = self._build_key(
            y_history=safe_1d_float(pred_data.Y_past),
            y_hat=y_hat_np[:1],
            fc_state=_state_row(fc_result.state, 0),
            eps_history=safe_1d_float(pred_data.eps_past) if pred_data.eps_past is not None else None,
        )
        scale_result = self._scale_for_key(key)
        if self._side_specific_scale:
            scale_pos = float(scale_result.scale[0]) + self._eps_value
            scale_neg = float(scale_result.scale[1]) + self._eps_value
            scale_for_update = 0.5 * (float(scale_result.scale[0]) + float(scale_result.scale[1]))
        else:
            scale = float(scale_result.scale[0]) + self._eps_value
            scale_pos = scale_neg = scale
            scale_for_update = float(scale_result.scale[0])
        q, q_pos, q_neg = self._active_multipliers()
        if self._interval_mode in {"two_sided", "twosided"}:
            lower_radius = float(q_neg * scale_neg)
            upper_radius = float(q_pos * scale_pos)
        else:
            shared_scale = max(scale_pos, scale_neg) if self._side_specific_scale else scale_pos
            lower_radius = upper_radius = float(q * shared_scale)
        pred_int = y_hat - lower_radius, y_hat + upper_radius
        self._last_scale_for_update = float(scale_for_update)
        self._last_key_for_update = key.astype(np.float32)
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=y_hat)

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        del pred_data, kwargs
        if self._last_scale_for_update is None:
            return
        y_step = safe_1d_float(Y_step)
        y_hat = safe_1d_float(pred_result.fc_Y_hat)
        if y_step.size == 0 or y_hat.size == 0:
            return
        signed_resid = float(y_step[0] - y_hat[0])
        self._append_rolling_scores(signed_resid, self._last_scale_for_update)
        if self._online_memory and self._last_key_for_update is not None:
            self._append_memory(self._last_key_for_update, signed_resid)
        self._last_scale_for_update = None
        self._last_key_for_update = None

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.X_step is not None

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return (
            self._index is not None
            and self._scale_estimator is not None
            and self._memory_residuals is not None
            and self._calib_scores is not None
            and self._calib_scores.shape[0] > 0
        )
