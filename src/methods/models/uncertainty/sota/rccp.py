from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from models.forcast.forcast_base import FCPredictionData
from models.uncertainty.pi_base import (
    PICalibArtifacts,
    PICalibData,
    PIModelPrediction,
    PIPredictionStepData,
)
from models.uncertainty.sota.racp import (
    RACPModel,
    _as_numpy,
    _conformal_quantile_level,
    _l2_normalize_rows,
    _recent_target_summary,
    _residual_signature,
    _safe_float_array,
    _softmax,
    _weighted_quantile,
    faiss,
)


def _positive_part(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float32), 0.0).astype(np.float32)


def _order_stat_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute conformal quantile from an empty calibration score set")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), values.size)
    return float(np.sort(values)[rank - 1])


class RCCPModel(RACPModel):
    """
    Retrieval-Corrected Conformal Prediction.

    Retrieval produces a local proposal radius. Calibration conformalizes the
    multiplicative proposal error with prequential scores. The default max-score
    correction maps coverage to a single scalar event:
    max((y-yhat)_+ / R^+, (yhat-y)_+ / R^-) <= c.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._interval_mode = str(kwargs.get("interval_mode", "one_sided")).lower()
        self._correction_mode = str(kwargs.get("correction_mode", "max_score")).lower()
        self._min_memory = int(kwargs.get("min_memory", 5))
        self._proposal_finite_sample_correction = bool(kwargs.get("proposal_finite_sample_correction", False))
        self._proposal_local_sample_size_mode = str(kwargs.get("proposal_local_sample_size_mode", "effective")).lower()
        self._correction_floor = float(kwargs.get("correction_floor", 0.0))
        self._proposal_min_radius = float(kwargs.get("proposal_min_radius", self._eps))
        self._initial_scale_mode = str(kwargs.get("initial_scale_mode", "pre_calib_diff")).lower()
        self._use_forecast_value = bool(kwargs.get("use_forecast_value", True))
        self._use_target_summary = bool(kwargs.get("use_target_summary", True))

        if self._interval_mode not in {"one_sided", "symmetric"}:
            raise ValueError(f"Unsupported interval_mode: {self._interval_mode}")
        if self._correction_mode in {"none", "no_correction", "proposal_only"}:
            self._correction_mode = "none"
        if self._correction_mode in {"separate", "separate_one_sided", "one_sided"}:
            self._correction_mode = "separate"
        if self._correction_mode not in {"max_score", "separate", "none"}:
            raise ValueError(f"Unsupported correction_mode: {self._correction_mode}")
        if self._interval_mode == "symmetric" and self._correction_mode == "separate":
            raise ValueError("correction_mode='separate' is only valid with interval_mode='one_sided'")
        if self._proposal_local_sample_size_mode not in {"effective", "raw"}:
            raise ValueError(f"Unsupported proposal_local_sample_size_mode: {self._proposal_local_sample_size_mode}")
        if self._min_memory < 1:
            raise ValueError("min_memory must be at least 1")

        self._memory_score_pos: Optional[np.ndarray] = None
        self._memory_score_neg: Optional[np.ndarray] = None
        self._memory_score_pos_buffer: Optional[np.ndarray] = None
        self._memory_score_neg_buffer: Optional[np.ndarray] = None
        self._memory_score_pos_queue = None
        self._memory_score_neg_queue = None
        self._memory_score_pos_by_id = {}
        self._memory_score_neg_by_id = {}
        self._correction_abs: Optional[float] = None
        self._correction_pos: Optional[float] = None
        self._correction_neg: Optional[float] = None
        self._last_alpha: Optional[float] = None

    def _build_situation_key(
        self,
        base_key: np.ndarray,
        y_hat: float,
        y_history: np.ndarray,
        eps_history: Optional[np.ndarray],
    ) -> np.ndarray:
        key_parts = [np.asarray(base_key, dtype=np.float32).reshape(-1)]
        if self._use_forecast_value:
            key_parts.append(np.asarray([float(y_hat)], dtype=np.float32))
        if self._use_target_summary:
            key_parts.append(_recent_target_summary(y_history, window=self._context_window))
        if self._use_residual_context:
            key_parts.append(_residual_signature(eps_history, steps=self._residual_context_steps))
        return np.concatenate(key_parts).astype(np.float32)

    def required_past_len(self) -> Tuple[int, int]:
        max_len = 1
        if self._use_target_summary:
            max_len = max(max_len, self._context_window)
        if self._use_residual_context:
            max_len = max(max_len, self._residual_context_steps)
        return 0, max_len

    def _reset_memory_store(self):
        super()._reset_memory_store()
        self._memory_score_pos = None
        self._memory_score_neg = None
        self._memory_score_pos_buffer = None
        self._memory_score_neg_buffer = None
        self._memory_score_pos_queue = None
        self._memory_score_neg_queue = None
        self._memory_score_pos_by_id = {}
        self._memory_score_neg_by_id = {}

    def _refresh_public_memory_views_from_queue(self):
        if self._memory_ids is None or len(self._memory_ids) == 0:
            self._memory_keys = None
            self._memory_keys_cos = None
            self._memory_score = None
            self._memory_score_pos = None
            self._memory_score_neg = None
            self._public_views_dirty = False
            return
        self._memory_keys = np.stack(list(self._memory_key_queue), axis=0).astype(np.float32)
        self._memory_keys_cos = np.stack(list(self._memory_key_cos_queue), axis=0).astype(np.float32)
        self._memory_score = np.asarray(list(self._memory_score_queue), dtype=np.float32)
        self._memory_score_pos = np.asarray(list(self._memory_score_pos_queue), dtype=np.float32)
        self._memory_score_neg = np.asarray(list(self._memory_score_neg_queue), dtype=np.float32)
        self._public_views_dirty = False

    def _refresh_public_memory_views(self):
        if self._uses_faiss_id_map():
            self._refresh_public_memory_views_from_queue()
            return
        super()._refresh_public_memory_views()
        if self._memory_size <= 0 or self._memory_score_pos_buffer is None or self._memory_score_neg_buffer is None:
            self._memory_score_pos = None
            self._memory_score_neg = None
            return
        self._memory_score_pos = self._memory_score_pos_buffer[: self._memory_size]
        self._memory_score_neg = self._memory_score_neg_buffer[: self._memory_size]

    def _ensure_memory_capacity_triplet(self, dim: int, target_size: int):
        self._ensure_memory_capacity(dim=dim, target_size=target_size)
        if self._uses_faiss_id_map():
            return
        target_capacity = int(self._memory_keys_buffer.shape[0])
        for attr in ("_memory_score_pos_buffer", "_memory_score_neg_buffer"):
            current = getattr(self, attr)
            if current is not None and int(current.shape[0]) >= target_capacity:
                continue
            new_buffer = np.empty((target_capacity,), dtype=np.float32)
            if self._memory_size > 0 and current is not None:
                new_buffer[: self._memory_size] = current[: self._memory_size]
            setattr(self, attr, new_buffer)
        self._refresh_public_memory_views()

    def _trim_oldest_memory_triplet(self):
        if self._uses_faiss_id_map():
            if self._memory_ids is None or len(self._memory_ids) == 0:
                return
            oldest_id = int(self._memory_ids.popleft())
            self._memory_key_queue.popleft()
            self._memory_key_cos_queue.popleft()
            self._memory_score_queue.popleft()
            self._memory_score_pos_queue.popleft()
            self._memory_score_neg_queue.popleft()
            self._memory_score_by_id.pop(oldest_id, None)
            self._memory_score_pos_by_id.pop(oldest_id, None)
            self._memory_score_neg_by_id.pop(oldest_id, None)
            if self._faiss_index is not None:
                removed = self._faiss_index.remove_ids(
                    faiss.IDSelectorBatch(np.asarray([oldest_id], dtype=np.int64))
                )
                if removed != 1:
                    self._faiss_index = None
            self._mark_public_views_dirty()
            return

        if self._memory_size <= 0:
            return
        old_size = self._memory_size
        if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            removed = self._faiss_index.remove_ids(faiss.IDSelectorRange(0, 1))
            if removed != 1:
                self._faiss_index = None
        if old_size > 1:
            self._memory_keys_buffer[: old_size - 1] = self._memory_keys_buffer[1:old_size]
            self._memory_keys_cos_buffer[: old_size - 1] = self._memory_keys_cos_buffer[1:old_size]
            self._memory_score_buffer[: old_size - 1] = self._memory_score_buffer[1:old_size]
            self._memory_score_pos_buffer[: old_size - 1] = self._memory_score_pos_buffer[1:old_size]
            self._memory_score_neg_buffer[: old_size - 1] = self._memory_score_neg_buffer[1:old_size]
        self._memory_size -= 1
        self._refresh_public_memory_views()

    def _append_memory_triplet(self, key: np.ndarray, score_abs: float, score_pos: float, score_neg: float):
        self._ensure_search_backend()
        key = np.asarray(key, dtype=np.float32).reshape(1, -1)
        score_abs = float(score_abs)
        score_pos = float(score_pos)
        score_neg = float(score_neg)
        key_cos = _l2_normalize_rows(key)

        if self._uses_faiss_id_map():
            if self._memory_ids is None:
                self._memory_ids = deque()
                self._memory_key_queue = deque()
                self._memory_key_cos_queue = deque()
                self._memory_score_queue = deque()
                self._memory_score_pos_queue = deque()
                self._memory_score_neg_queue = deque()
                self._memory_score_by_id = {}
                self._memory_score_pos_by_id = {}
                self._memory_score_neg_by_id = {}
            if self._memory_ids and self._memory_key_queue[0].shape[0] != key.shape[1]:
                raise ValueError(f"Memory key dimension changed from {self._memory_key_queue[0].shape[0]} to {key.shape[1]}")
            if self._faiss_index is None:
                self._faiss_index = self._new_faiss_index(key.shape[1])
            if self._max_memory is not None and len(self._memory_ids) >= self._max_memory:
                self._trim_oldest_memory_triplet()
                if self._faiss_index is None:
                    self._rebuild_search_backend_from_public_memory()
            memory_id = int(self._next_memory_id)
            self._next_memory_id += 1
            self._memory_ids.append(memory_id)
            self._memory_key_queue.append(key[0].copy())
            self._memory_key_cos_queue.append(key_cos[0].copy())
            self._memory_score_queue.append(np.float32(score_abs))
            self._memory_score_pos_queue.append(np.float32(score_pos))
            self._memory_score_neg_queue.append(np.float32(score_neg))
            self._memory_score_by_id[memory_id] = np.float32(score_abs)
            self._memory_score_pos_by_id[memory_id] = np.float32(score_pos)
            self._memory_score_neg_by_id[memory_id] = np.float32(score_neg)
            indexed_key = key_cos if self._similarity_metric == "cosine" else key
            self._faiss_index.add_with_ids(
                np.ascontiguousarray(indexed_key),
                np.asarray([memory_id], dtype=np.int64),
            )
            self._mark_public_views_dirty()
            return

        if self._memory_size > 0 and self._memory_keys is not None and self._memory_keys.shape[1] != key.shape[1]:
            raise ValueError(f"Memory key dimension changed from {self._memory_keys.shape[1]} to {key.shape[1]}")
        if self._max_memory is not None and self._memory_size >= self._max_memory:
            self._trim_oldest_memory_triplet()
            if self._use_faiss and self._memory_size > 0 and self._faiss_index is None:
                self._rebuild_search_backend_from_public_memory()
        self._ensure_memory_capacity_triplet(dim=key.shape[1], target_size=self._memory_size + 1)
        insert_idx = self._memory_size
        self._memory_keys_buffer[insert_idx] = key[0]
        self._memory_keys_cos_buffer[insert_idx] = key_cos[0]
        self._memory_score_buffer[insert_idx] = np.float32(score_abs)
        self._memory_score_pos_buffer[insert_idx] = np.float32(score_pos)
        self._memory_score_neg_buffer[insert_idx] = np.float32(score_neg)
        self._memory_size += 1
        self._refresh_public_memory_views()
        if self._use_faiss:
            if self._faiss_index is None:
                self._faiss_index = self._new_faiss_index(key.shape[1])
            indexed_key = key_cos if self._similarity_metric == "cosine" else key
            self._faiss_index.add(np.ascontiguousarray(indexed_key))

    def _selected_memory_scores(self, selection_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        selection_idx = np.asarray(selection_idx, dtype=np.int64).reshape(-1)
        if selection_idx.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, empty
        if self._uses_faiss_id_map():
            score_abs = np.asarray([self._memory_score_by_id[int(memory_id)] for memory_id in selection_idx], dtype=np.float32)
            score_pos = np.asarray([self._memory_score_pos_by_id[int(memory_id)] for memory_id in selection_idx], dtype=np.float32)
            score_neg = np.asarray([self._memory_score_neg_by_id[int(memory_id)] for memory_id in selection_idx], dtype=np.float32)
            return score_abs, score_pos, score_neg
        self._ensure_public_memory_views()
        return (
            self._memory_score[selection_idx].astype(np.float32),
            self._memory_score_pos[selection_idx].astype(np.float32),
            self._memory_score_neg[selection_idx].astype(np.float32),
        )

    def _memory_count(self) -> int:
        if self._uses_faiss_id_map():
            return 0 if self._memory_ids is None else len(self._memory_ids)
        return int(self._memory_size)

    def _proposal_alpha(self, alpha: float) -> float:
        return float(alpha) / 2.0 if self._interval_mode == "one_sided" else float(alpha)

    def _correction_alpha(self, alpha: float) -> float:
        if self._interval_mode == "one_sided" and self._correction_mode == "separate":
            return float(alpha) / 2.0
        return float(alpha)

    def _proposal_quantile_level(self, alpha: float, weights: np.ndarray) -> float:
        proposal_alpha = self._proposal_alpha(alpha)
        if not self._proposal_finite_sample_correction:
            return 1.0 - proposal_alpha
        if self._proposal_local_sample_size_mode == "effective":
            denom = float(np.sum(weights ** 2) + 1e-6)
            local_n = max(int(round(float(np.sum(weights) ** 2) / denom)), 1)
        else:
            local_n = max(int(weights.shape[0]), 1)
        return _conformal_quantile_level(proposal_alpha, local_n)

    def _proposal_radius(self, query_key: np.ndarray, alpha: float, scale: float) -> Tuple[float, float, float, int]:
        top_scores, top_idx = self._search_memory(query_key, self._k)
        if top_idx.size == 0:
            raise ValueError("RCCP memory is empty; cannot build retrieval proposal")
        weights = _softmax(top_scores, temperature=self._temperature)
        q_level = self._proposal_quantile_level(alpha, weights)
        score_abs, score_pos, score_neg = self._selected_memory_scores(top_idx)
        proposal_abs = _weighted_quantile(score_abs, q=q_level, weights=weights)
        proposal_pos = _weighted_quantile(score_pos, q=q_level, weights=weights)
        proposal_neg = _weighted_quantile(score_neg, q=q_level, weights=weights)
        if self._normalize_score:
            proposal_abs = float(scale * proposal_abs)
            proposal_pos = float(scale * proposal_pos)
            proposal_neg = float(scale * proposal_neg)
        proposal_abs = max(float(proposal_abs), self._proposal_min_radius)
        proposal_pos = max(float(proposal_pos), self._proposal_min_radius)
        proposal_neg = max(float(proposal_neg), self._proposal_min_radius)
        return proposal_abs, proposal_pos, proposal_neg, int(top_idx.size)

    def _init_prequential_scale(self, y_pre: np.ndarray) -> float:
        y_pre = _safe_float_array(y_pre)
        if self._initial_scale_mode == "unit":
            return float(max(1.0, self._ewma_min_scale))
        if self._initial_scale_mode == "pre_calib_std" and y_pre.size > 1:
            return float(max(np.std(y_pre), self._ewma_min_scale))
        if self._initial_scale_mode == "pre_calib_diff" and y_pre.size > 1:
            return float(max(np.mean(np.abs(np.diff(y_pre))), self._ewma_min_scale))
        if y_pre.size > 1:
            return float(max(np.std(y_pre), self._ewma_min_scale))
        return float(self._ewma_min_scale)

    def _calibrate(self, calib_data: List[PICalibData], alphas, **kwargs) -> List[PICalibArtifacts]:
        return None

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del calib_artifact, mix_calib_data, mix_calib_artifact

        self._reset_memory_store()
        self._ewma_scale = None
        self._last_query_key = None
        self._last_scale_before_update = None
        self._last_base_key_source = None
        self._correction_abs = None
        self._correction_pos = None
        self._correction_neg = None
        self._last_alpha = float(alpha)

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
        signed_resid_cal = (y_cal - y_hat_cal_np).astype(np.float32)
        abs_resid_cal = np.abs(signed_resid_cal).astype(np.float32)
        pos_resid_cal = _positive_part(signed_resid_cal)
        neg_resid_cal = _positive_part(-signed_resid_cal)
        base_keys = self._extract_base_key_batch(fc_result.state, calib_data.X_calib)
        if base_keys.shape[0] != abs_resid_cal.shape[0]:
            raise ValueError(f"Mismatch between key rows ({base_keys.shape[0]}) and residuals ({abs_resid_cal.shape[0]})")

        y_pre = _safe_float_array(_as_numpy(calib_data.Y_pre_calib).reshape(-1))
        correction_abs_scores = []
        correction_pos_scores = []
        correction_neg_scores = []
        proposal_neighbor_counts = []
        ewma_scale = self._init_prequential_scale(y_pre)

        if not self._use_target_summary and not self._use_residual_context:
            key_parts = [base_keys]
            if self._use_forecast_value:
                key_parts.append(y_hat_cal_np.reshape(-1, 1).astype(np.float32))
            situation_keys = np.concatenate(key_parts, axis=1).astype(np.float32)
        else:
            situation_keys = []
            for idx in range(abs_resid_cal.shape[0]):
                y_history = np.concatenate([y_pre, y_cal[:idx]]).astype(np.float32) if self._use_target_summary else None
                eps_history = abs_resid_cal[:idx] if self._use_residual_context else None
                situation_keys.append(
                    self._build_situation_key(
                        base_key=base_keys[idx],
                        y_hat=float(y_hat_cal_np[idx]),
                        y_history=y_history,
                        eps_history=eps_history,
                    )
                )
            situation_keys = np.stack(situation_keys, axis=0).astype(np.float32)

        for idx in range(abs_resid_cal.shape[0]):
            situation_key = situation_keys[idx]
            b_t = max(float(ewma_scale), self._ewma_min_scale)
            if self._memory_count() >= self._min_memory:
                proposal_abs, proposal_pos, proposal_neg, neighbor_count = self._proposal_radius(
                    query_key=situation_key,
                    alpha=float(alpha),
                    scale=b_t,
                )
                if self._correction_mode == "none":
                    pass
                elif self._interval_mode == "symmetric":
                    correction_abs_scores.append(float(abs_resid_cal[idx] / (proposal_abs + self._eps)))
                elif self._correction_mode == "max_score":
                    correction_abs_scores.append(
                        max(
                            float(pos_resid_cal[idx] / (proposal_pos + self._eps)),
                            float(neg_resid_cal[idx] / (proposal_neg + self._eps)),
                        )
                    )
                else:
                    correction_pos_scores.append(float(pos_resid_cal[idx] / (proposal_pos + self._eps)))
                    correction_neg_scores.append(float(neg_resid_cal[idx] / (proposal_neg + self._eps)))
                proposal_neighbor_counts.append(neighbor_count)

            if self._normalize_score:
                score_abs = float(abs_resid_cal[idx] / (b_t + self._eps))
                score_pos = float(pos_resid_cal[idx] / (b_t + self._eps))
                score_neg = float(neg_resid_cal[idx] / (b_t + self._eps))
            else:
                score_abs = float(abs_resid_cal[idx])
                score_pos = float(pos_resid_cal[idx])
                score_neg = float(neg_resid_cal[idx])
            self._append_memory_triplet(situation_key, score_abs=score_abs, score_pos=score_pos, score_neg=score_neg)
            ewma_scale = self._ewma_update(ewma_scale, float(abs_resid_cal[idx]))

        if self._correction_mode == "none":
            self._correction_abs = 1.0
            self._correction_pos = 1.0
            self._correction_neg = 1.0
        elif self._interval_mode == "symmetric" or self._correction_mode == "max_score":
            if len(correction_abs_scores) == 0:
                raise ValueError(f"Not enough calibration points for RCCP correction: min_memory={self._min_memory}")
            self._correction_abs = max(
                _order_stat_quantile(np.asarray(correction_abs_scores), self._correction_alpha(alpha)),
                self._correction_floor,
            )
        else:
            if len(correction_pos_scores) == 0 or len(correction_neg_scores) == 0:
                raise ValueError(f"Not enough calibration points for RCCP correction: min_memory={self._min_memory}")
            side_alpha = self._correction_alpha(alpha)
            self._correction_pos = max(_order_stat_quantile(np.asarray(correction_pos_scores), side_alpha), self._correction_floor)
            self._correction_neg = max(_order_stat_quantile(np.asarray(correction_neg_scores), side_alpha), self._correction_floor)
        self._ewma_scale = float(ewma_scale)

        artifact = PICalibArtifacts(fc_Y_hat=y_hat_cal, eps=abs_resid_cal.reshape(-1, 1))
        self._ensure_public_memory_views()
        artifact.add_info = {
            "method": "rccp",
            "interval_mode": self._interval_mode,
            "correction_mode": self._correction_mode,
            "correction_abs": None if self._correction_abs is None else float(self._correction_abs),
            "correction_pos": None if self._correction_pos is None else float(self._correction_pos),
            "correction_neg": None if self._correction_neg is None else float(self._correction_neg),
            "correction_score_count": int(len(correction_abs_scores) or len(correction_pos_scores)),
            "proposal_neighbor_count_mean": float(np.mean(proposal_neighbor_counts)) if proposal_neighbor_counts else 0.0,
            "base_key_source_used": self._last_base_key_source,
            "similarity_metric": self._similarity_metric,
            "use_residual_context": self._use_residual_context,
            "memory_size": int(self._memory_count()),
            "min_memory": int(self._min_memory),
            "retrieval_backend": "faiss" if self._use_faiss else "numpy",
            "normalize_score": bool(self._normalize_score),
            "initial_scale_mode": self._initial_scale_mode,
        }
        return artifact

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if not self.model_ready():
            raise ValueError("RCCP model is not calibrated")

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
        y_hat_scalar = float(_as_numpy(y_hat).reshape(-1)[0])
        base_key = self._extract_base_key_batch(fc_result.state, pred_data.X_step)[0]
        y_history = _safe_float_array(_as_numpy(pred_data.Y_past).reshape(-1)) if self._use_target_summary else None
        eps_history = (
            _safe_float_array(_as_numpy(pred_data.eps_past).reshape(-1))
            if (self._use_residual_context and pred_data.eps_past is not None)
            else None
        )
        query_key = self._build_situation_key(
            base_key=base_key,
            y_hat=y_hat_scalar,
            y_history=y_history,
            eps_history=eps_history,
        )

        b_t = max(float(self._ewma_scale), self._ewma_min_scale)
        proposal_abs, proposal_pos, proposal_neg, _ = self._proposal_radius(
            query_key=query_key,
            alpha=float(pred_data.alpha),
            scale=b_t,
        )
        if self._correction_mode == "none":
            if self._interval_mode == "symmetric":
                pred_int = y_hat - proposal_abs, y_hat + proposal_abs
            else:
                pred_int = y_hat - proposal_neg, y_hat + proposal_pos
        elif self._interval_mode == "symmetric":
            radius = float(self._correction_abs * proposal_abs)
            pred_int = y_hat - radius, y_hat + radius
        elif self._correction_mode == "max_score":
            radius_pos = float(self._correction_abs * proposal_pos)
            radius_neg = float(self._correction_abs * proposal_neg)
            pred_int = y_hat - radius_neg, y_hat + radius_pos
        else:
            radius_pos = float(self._correction_pos * proposal_pos)
            radius_neg = float(self._correction_neg * proposal_neg)
            pred_int = y_hat - radius_neg, y_hat + radius_pos

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
        signed_resid = float(y_step[0] - y_hat[0])
        abs_resid = abs(signed_resid)
        pos_resid = max(signed_resid, 0.0)
        neg_resid = max(-signed_resid, 0.0)
        b_t = (
            self._last_scale_before_update
            if self._last_scale_before_update is not None
            else max(float(self._ewma_scale), self._ewma_min_scale)
        )
        if self._online_memory and self._last_query_key is not None:
            if self._normalize_score:
                score_abs = float(abs_resid / (b_t + self._eps))
                score_pos = float(pos_resid / (b_t + self._eps))
                score_neg = float(neg_resid / (b_t + self._eps))
            else:
                score_abs = float(abs_resid)
                score_pos = float(pos_resid)
                score_neg = float(neg_resid)
            self._append_memory_triplet(self._last_query_key, score_abs=score_abs, score_pos=score_pos, score_neg=score_neg)
        self._ewma_scale = self._ewma_update(self._ewma_scale, abs_resid)
        self._last_query_key = None
        self._last_scale_before_update = None

    @property
    def can_handle_different_alpha(self):
        return False

    def model_ready(self):
        memory_ready = (
            self._memory_keys is not None
            and self._memory_keys_cos is not None
            and self._memory_score is not None
            and self._memory_score_pos is not None
            and self._memory_score_neg is not None
            and self._memory_score.shape[0] > 0
            and self._memory_score_pos.shape[0] == self._memory_score.shape[0]
            and self._memory_score_neg.shape[0] == self._memory_score.shape[0]
            and self._ewma_scale is not None
        )
        if self._correction_mode == "none":
            return memory_ready
        if self._interval_mode == "symmetric" or self._correction_mode == "max_score":
            return memory_ready and self._correction_abs is not None
        return memory_ready and self._correction_pos is not None and self._correction_neg is not None
