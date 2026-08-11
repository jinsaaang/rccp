from collections import deque
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

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


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


def _conformal_quantile_level(alpha: float, sample_size: int) -> float:
    sample_size = max(int(sample_size), 1)
    rank = int(np.ceil((sample_size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), sample_size)
    return float(rank / sample_size)


def _effective_sample_size(weights: np.ndarray) -> int:
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    denom = float(np.sum(weights ** 2) + 1e-6)
    ess = float((np.sum(weights) ** 2) / denom)
    return max(int(round(ess)), 1)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6
    return (mat / norms).astype(np.float32)


def _safe_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return x


def _buffer_capacity(target_size: int) -> int:
    return max(int(target_size), 64)


def _trend_feature(values: np.ndarray) -> float:
    values = _safe_float_array(values)
    if values.size <= 1:
        return 0.0
    idx = np.arange(values.size, dtype=np.float32)
    idx = idx - idx.mean()
    denom = float(np.sum(idx ** 2) + 1e-6)
    return float(np.sum(idx * (values - values.mean())) / denom)


def _recent_target_summary(y_history: np.ndarray, window: int) -> np.ndarray:
    y_history = _safe_float_array(y_history)
    if y_history.size == 0:
        return np.zeros((6,), dtype=np.float32)
    tail = y_history[-max(int(window), 1) :]
    last = float(tail[-1])
    prev = float(tail[-2]) if tail.size > 1 else last
    summary = np.array(
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
    return summary


def _residual_signature(eps_history: Optional[np.ndarray], steps: int) -> np.ndarray:
    steps = max(int(steps), 1)
    if eps_history is None:
        return np.zeros((steps + 3,), dtype=np.float32)
    eps_history = _safe_float_array(eps_history)
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


class RACPModel(PIModel):
    """
    Retrieval-Augmented Conformal Prediction.
    Core idea:
    - build a causal "situation" key for each step
    - retrieve similar historical situations
    - use their residual scores to form a local conformal interval
    Optional ablation:
    - append recent residual signature to the situation key
    """

    def __init__(self, **kwargs):
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._k = int(kwargs.get("k", 64))
        self._temperature = float(kwargs.get("temperature", 1.0))
        self._ewma_beta = float(kwargs.get("ewma_beta", 0.95))
        self._ewma_min_scale = float(kwargs.get("ewma_min_scale", 1e-3))
        self._eps = float(kwargs.get("eps", 1e-6))
        self._online_memory = bool(kwargs.get("online_memory", True))
        self._normalize_score = bool(kwargs.get("normalize_score", True))
        self._base_key_source = str(kwargs.get("base_key_source", "auto"))
        self._similarity_metric = str(kwargs.get("similarity_metric", "cosine")).lower()
        self._memory_use_calib_frac = float(kwargs.get("memory_use_calib_frac", 1.0))
        self._use_recent_calib = bool(kwargs.get("use_recent_calib", True))
        self._context_window = int(kwargs.get("context_window", 32))
        self._use_residual_context = bool(kwargs.get("use_residual_context", False))
        self._residual_context_steps = int(kwargs.get("residual_context_steps", 8))
        self._use_finite_sample_correction = bool(kwargs.get("use_finite_sample_correction", False))
        self._local_sample_size_mode = str(kwargs.get("local_sample_size_mode", "effective")).lower()
        self._use_global_score_floor = bool(kwargs.get("use_global_score_floor", False))
        self._max_memory = kwargs.get("max_memory", None)
        if self._max_memory is not None:
            self._max_memory = int(self._max_memory)
            if self._max_memory <= 0:
                self._max_memory = None
        retrieval_backend = str(kwargs.get("retrieval_backend", "") or "").lower()
        requested_use_faiss = kwargs.get("use_faiss", None)
        if requested_use_faiss is None and retrieval_backend.startswith("faiss"):
            requested_use_faiss = faiss is not None
        if requested_use_faiss is None:
            self._use_faiss = faiss is not None
        else:
            self._use_faiss = bool(requested_use_faiss)
        if self._use_faiss and faiss is None:
            raise ImportError("RACPModel use_faiss=True requires faiss. Install faiss-cpu or faiss-gpu.")
        if self._similarity_metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported similarity_metric: {self._similarity_metric}")
        if self._local_sample_size_mode not in {"effective", "raw"}:
            raise ValueError(f"Unsupported local_sample_size_mode: {self._local_sample_size_mode}")

        self._memory_keys: Optional[np.ndarray] = None
        self._memory_keys_cos: Optional[np.ndarray] = None
        self._memory_score: Optional[np.ndarray] = None
        self._ewma_scale: Optional[float] = None
        self._last_query_key: Optional[np.ndarray] = None
        self._last_scale_before_update: Optional[float] = None
        self._last_base_key_source: Optional[str] = None
        self._memory_keys_buffer: Optional[np.ndarray] = None
        self._memory_keys_cos_buffer: Optional[np.ndarray] = None
        self._memory_score_buffer: Optional[np.ndarray] = None
        self._memory_size = 0
        self._faiss_index = None
        self._memory_ids = None
        self._memory_key_queue = None
        self._memory_key_cos_queue = None
        self._memory_score_queue = None
        self._memory_score_by_id = {}
        self._next_memory_id = 0
        self._public_views_dirty = False

    def _resolve_key_source(self, fc_state) -> str:
        if self._base_key_source == "fc_state":
            if fc_state is None:
                raise ValueError("base_key_source='fc_state' but forecast model returned no state")
            return "fc_state"
        if self._base_key_source == "x_step":
            return "x_step"
        if self._base_key_source != "auto":
            raise ValueError(f"Unsupported base_key_source: {self._base_key_source}")
        return "fc_state" if fc_state is not None else "x_step"

    def _extract_base_key_batch(self, fc_state, x_step_batch) -> np.ndarray:
        source = self._resolve_key_source(fc_state)
        self._last_base_key_source = source
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

    def _build_situation_key(
        self,
        base_key: np.ndarray,
        y_hat: float,
        y_history: np.ndarray,
        eps_history: Optional[np.ndarray],
    ) -> np.ndarray:
        base_key = np.asarray(base_key, dtype=np.float32).reshape(-1)
        target_summary = _recent_target_summary(y_history, window=self._context_window)
        key_parts = [
            base_key,
            np.array([float(y_hat)], dtype=np.float32),
            target_summary,
        ]
        if self._use_residual_context:
            key_parts.append(_residual_signature(eps_history, steps=self._residual_context_steps))
        return np.concatenate(key_parts).astype(np.float32)

    def required_past_len(self) -> Tuple[int, int]:
        max_len = 1
        if self._use_residual_context:
            max_len = max(max_len, self._residual_context_steps)
        return 0, max_len

    def _compute_similarity_scores(self, query_key: np.ndarray) -> np.ndarray:
        self._ensure_public_memory_views()
        query_key = np.asarray(query_key, dtype=np.float32).reshape(-1)
        if self._similarity_metric == "cosine":
            query_norm = _l2_normalize_rows(query_key)[0]
            return (self._memory_keys_cos @ query_norm).astype(np.float32)
        memory_keys = np.asarray(self._memory_keys, dtype=np.float32)
        return (-np.linalg.norm(memory_keys - query_key[None, :], axis=1)).astype(np.float32)

    def _uses_faiss_id_map(self) -> bool:
        return bool(self._use_faiss and self._max_memory is not None)

    def _new_faiss_index(self, dim: int):
        if not self._use_faiss:
            return None
        if self._similarity_metric == "cosine":
            base = faiss.IndexFlatIP(int(dim))
        else:
            base = faiss.IndexFlatL2(int(dim))
        if self._uses_faiss_id_map():
            return faiss.IndexIDMap2(base)
        return base

    def _mark_public_views_dirty(self):
        self._public_views_dirty = True

    def _refresh_public_memory_views_from_queue(self):
        if self._memory_ids is None or len(self._memory_ids) == 0:
            self._memory_keys = None
            self._memory_keys_cos = None
            self._memory_score = None
            self._public_views_dirty = False
            return
        self._memory_keys = np.stack(list(self._memory_key_queue), axis=0).astype(np.float32)
        self._memory_keys_cos = np.stack(list(self._memory_key_cos_queue), axis=0).astype(np.float32)
        self._memory_score = np.asarray(list(self._memory_score_queue), dtype=np.float32)
        self._public_views_dirty = False

    def _ensure_public_memory_views(self):
        if self._uses_faiss_id_map():
            if self._public_views_dirty:
                self._refresh_public_memory_views_from_queue()
            return
        if self._memory_size <= 0:
            self._memory_keys = None
            self._memory_keys_cos = None
            self._memory_score = None

    def _refresh_public_memory_views(self):
        if self._uses_faiss_id_map():
            self._refresh_public_memory_views_from_queue()
            return
        if self._memory_size <= 0:
            self._memory_keys = None
            self._memory_keys_cos = None
            self._memory_score = None
            return
        self._memory_keys = self._memory_keys_buffer[: self._memory_size]
        self._memory_keys_cos = self._memory_keys_cos_buffer[: self._memory_size]
        self._memory_score = self._memory_score_buffer[: self._memory_size]

    def _reset_memory_store(self):
        self._memory_keys = None
        self._memory_keys_cos = None
        self._memory_score = None
        self._memory_keys_buffer = None
        self._memory_keys_cos_buffer = None
        self._memory_score_buffer = None
        self._memory_size = 0
        self._faiss_index = None
        self._memory_ids = None
        self._memory_key_queue = None
        self._memory_key_cos_queue = None
        self._memory_score_queue = None
        self._memory_score_by_id = {}
        self._next_memory_id = 0
        self._public_views_dirty = False

    def _ensure_memory_capacity(self, dim: int, target_size: int):
        if self._uses_faiss_id_map():
            return
        dim = int(dim)
        target_size = int(target_size)
        current_capacity = 0 if self._memory_keys_buffer is None else int(self._memory_keys_buffer.shape[0])
        if current_capacity >= target_size:
            return
        new_capacity = _buffer_capacity(target_size if current_capacity == 0 else max(target_size, current_capacity * 2))
        keys = np.empty((new_capacity, dim), dtype=np.float32)
        keys_cos = np.empty((new_capacity, dim), dtype=np.float32)
        scores = np.empty((new_capacity,), dtype=np.float32)
        if self._memory_size > 0:
            keys[: self._memory_size] = self._memory_keys_buffer[: self._memory_size]
            keys_cos[: self._memory_size] = self._memory_keys_cos_buffer[: self._memory_size]
            scores[: self._memory_size] = self._memory_score_buffer[: self._memory_size]
        self._memory_keys_buffer = keys
        self._memory_keys_cos_buffer = keys_cos
        self._memory_score_buffer = scores
        self._refresh_public_memory_views()

    def _backend_matches_public_memory(self) -> bool:
        if self._uses_faiss_id_map():
            expected = 0 if self._memory_ids is None else len(self._memory_ids)
            if expected == 0:
                return self._faiss_index is None or self._faiss_index.ntotal == 0
            return self._faiss_index is not None and self._faiss_index.ntotal == expected
        if self._memory_score is None:
            return self._memory_size == 0
        public_size = int(np.asarray(self._memory_score).shape[0])
        if public_size == 0:
            return self._memory_size == 0
        if (
            self._memory_keys is None
            or self._memory_keys_cos is None
            or self._memory_keys_buffer is None
            or self._memory_keys_cos_buffer is None
            or self._memory_score_buffer is None
            or self._memory_size != public_size
        ):
            return False
        if not np.shares_memory(self._memory_keys, self._memory_keys_buffer):
            return False
        if not np.shares_memory(self._memory_keys_cos, self._memory_keys_cos_buffer):
            return False
        if not np.shares_memory(self._memory_score, self._memory_score_buffer):
            return False
        if self._use_faiss and (self._faiss_index is None or self._faiss_index.ntotal != public_size):
            return False
        return True

    def _rebuild_search_backend_from_public_memory(self):
        if self._uses_faiss_id_map():
            self._ensure_public_memory_views()
            if self._memory_score is None:
                self._reset_memory_store()
                return
            scores = np.asarray(self._memory_score, dtype=np.float32).reshape(-1)
            if scores.size == 0:
                self._reset_memory_store()
                return
            keys = np.asarray(self._memory_keys, dtype=np.float32)
            keys_cos = np.asarray(self._memory_keys_cos, dtype=np.float32)
            if keys.ndim == 1:
                keys = keys[None, :]
            if keys_cos.ndim == 1:
                keys_cos = keys_cos[None, :]
            self._memory_ids = deque()
            self._memory_key_queue = deque()
            self._memory_key_cos_queue = deque()
            self._memory_score_queue = deque()
            self._memory_score_by_id = {}
            self._faiss_index = self._new_faiss_index(keys.shape[1])
            for key, key_cos, score in zip(keys, keys_cos, scores):
                memory_id = int(self._next_memory_id)
                self._next_memory_id += 1
                self._memory_ids.append(memory_id)
                self._memory_key_queue.append(np.asarray(key, dtype=np.float32).copy())
                self._memory_key_cos_queue.append(np.asarray(key_cos, dtype=np.float32).copy())
                self._memory_score_queue.append(np.float32(score))
                self._memory_score_by_id[memory_id] = np.float32(score)
                indexed_key = key_cos if self._similarity_metric == "cosine" else key
                self._faiss_index.add_with_ids(
                    np.ascontiguousarray(np.asarray(indexed_key, dtype=np.float32).reshape(1, -1)),
                    np.asarray([memory_id], dtype=np.int64),
                )
            self._public_views_dirty = True
            return
        if self._memory_score is None:
            self._reset_memory_store()
            return
        scores = np.asarray(self._memory_score, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            self._reset_memory_store()
            return
        keys = np.asarray(self._memory_keys, dtype=np.float32)
        keys_cos = np.asarray(self._memory_keys_cos, dtype=np.float32)
        if keys.ndim == 1:
            keys = keys[None, :]
        if keys_cos.ndim == 1:
            keys_cos = keys_cos[None, :]
        if keys.shape[0] != scores.shape[0] or keys_cos.shape[0] != scores.shape[0]:
            raise ValueError("Memory arrays are inconsistent and cannot be rebuilt")
        self._memory_size = int(scores.shape[0])
        self._memory_keys_buffer = np.empty((_buffer_capacity(self._memory_size), keys.shape[1]), dtype=np.float32)
        self._memory_keys_cos_buffer = np.empty((_buffer_capacity(self._memory_size), keys_cos.shape[1]), dtype=np.float32)
        self._memory_score_buffer = np.empty((_buffer_capacity(self._memory_size),), dtype=np.float32)
        self._memory_keys_buffer[: self._memory_size] = keys
        self._memory_keys_cos_buffer[: self._memory_size] = keys_cos
        self._memory_score_buffer[: self._memory_size] = scores
        self._refresh_public_memory_views()
        if self._use_faiss:
            self._faiss_index = self._new_faiss_index(keys.shape[1])
            indexed_vectors = self._memory_keys_cos if self._similarity_metric == "cosine" else self._memory_keys
            self._faiss_index.add(np.ascontiguousarray(indexed_vectors.astype(np.float32)))
        else:
            self._faiss_index = None

    def _ensure_search_backend(self):
        if self._uses_faiss_id_map():
            memory_size = 0 if self._memory_ids is None else len(self._memory_ids)
            if memory_size == 0:
                if self._faiss_index is not None:
                    self._reset_memory_store()
                return
            if self._faiss_index is None or self._faiss_index.ntotal != memory_size:
                self._rebuild_search_backend_from_public_memory()
            return
        if self._memory_score is None or np.asarray(self._memory_score).size == 0:
            if self._memory_size != 0:
                self._reset_memory_store()
            return
        if not self._backend_matches_public_memory():
            self._rebuild_search_backend_from_public_memory()

    def _selected_memory_scores(self, selection_idx: np.ndarray) -> np.ndarray:
        selection_idx = np.asarray(selection_idx, dtype=np.int64).reshape(-1)
        if selection_idx.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._uses_faiss_id_map():
            return np.asarray(
                [self._memory_score_by_id[int(memory_id)] for memory_id in selection_idx],
                dtype=np.float32,
            )
        self._ensure_public_memory_views()
        return self._memory_score[selection_idx].astype(np.float32)

    def _trim_oldest_memory(self):
        if self._uses_faiss_id_map():
            if self._memory_ids is None or len(self._memory_ids) == 0:
                return
            oldest_id = int(self._memory_ids.popleft())
            self._memory_key_queue.popleft()
            self._memory_key_cos_queue.popleft()
            self._memory_score_queue.popleft()
            self._memory_score_by_id.pop(oldest_id, None)
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
        if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            removed = self._faiss_index.remove_ids(faiss.IDSelectorRange(0, 1))
            if removed != 1:
                self._faiss_index = None
        if self._memory_size > 1:
            self._memory_keys_buffer[: self._memory_size - 1] = self._memory_keys_buffer[1 : self._memory_size]
            self._memory_keys_cos_buffer[: self._memory_size - 1] = self._memory_keys_cos_buffer[1 : self._memory_size]
            self._memory_score_buffer[: self._memory_size - 1] = self._memory_score_buffer[1 : self._memory_size]
        self._memory_size -= 1
        self._refresh_public_memory_views()

    def _search_memory(self, query_key: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_search_backend()
        if self._uses_faiss_id_map():
            memory_size = 0 if self._memory_ids is None else len(self._memory_ids)
        else:
            self._ensure_public_memory_views()
            if self._memory_score is None:
                return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
            memory_size = int(self._memory_score.shape[0])
        if memory_size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        top_k = memory_size if int(k) <= 0 else min(int(k), memory_size)
        if self._use_faiss and self._faiss_index is not None:
            query = np.asarray(query_key, dtype=np.float32).reshape(1, -1)
            if self._similarity_metric == "cosine":
                query = _l2_normalize_rows(query)
            distances, indices = self._faiss_index.search(np.ascontiguousarray(query), top_k)
            top_idx = indices[0].astype(np.int64)
            valid = top_idx >= 0
            top_idx = top_idx[valid]
            scores = distances[0].astype(np.float32)[valid]
            if self._similarity_metric == "l2":
                scores = (-np.sqrt(np.maximum(scores, 0.0))).astype(np.float32)
            return scores, top_idx
        sims = self._compute_similarity_scores(query_key)
        if top_k < memory_size:
            top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        else:
            top_idx = np.arange(memory_size)
        return sims[top_idx].astype(np.float32), top_idx.astype(np.int64)

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

    def _append_memory(self, key: np.ndarray, score: float):
        self._ensure_search_backend()
        key = np.asarray(key, dtype=np.float32).reshape(1, -1)
        score = float(score)
        key_cos = _l2_normalize_rows(key)
        if self._uses_faiss_id_map():
            if self._memory_ids is None:
                self._memory_ids = deque()
                self._memory_key_queue = deque()
                self._memory_key_cos_queue = deque()
                self._memory_score_queue = deque()
                self._memory_score_by_id = {}
            if self._memory_ids and self._memory_key_queue and self._memory_key_queue[0].shape[0] != key.shape[1]:
                raise ValueError(f"Memory key dimension changed from {self._memory_key_queue[0].shape[0]} to {key.shape[1]}")
            if self._faiss_index is None:
                self._faiss_index = self._new_faiss_index(key.shape[1])
            if self._max_memory is not None and len(self._memory_ids) >= self._max_memory:
                self._trim_oldest_memory()
                if self._faiss_index is None:
                    self._rebuild_search_backend_from_public_memory()
            memory_id = int(self._next_memory_id)
            self._next_memory_id += 1
            self._memory_ids.append(memory_id)
            self._memory_key_queue.append(key[0].copy())
            self._memory_key_cos_queue.append(key_cos[0].copy())
            self._memory_score_queue.append(np.float32(score))
            self._memory_score_by_id[memory_id] = np.float32(score)
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
            self._trim_oldest_memory()
            if self._use_faiss and self._memory_size > 0 and self._faiss_index is None:
                self._rebuild_search_backend_from_public_memory()
        self._ensure_memory_capacity(dim=key.shape[1], target_size=self._memory_size + 1)
        insert_idx = self._memory_size
        self._memory_keys_buffer[insert_idx] = key[0]
        self._memory_keys_cos_buffer[insert_idx] = key_cos[0]
        self._memory_score_buffer[insert_idx] = np.float32(score)
        self._memory_size += 1
        self._refresh_public_memory_views()
        if self._use_faiss:
            if self._faiss_index is None:
                self._faiss_index = self._new_faiss_index(key.shape[1])
            indexed_key = key_cos if self._similarity_metric == "cosine" else key
            self._faiss_index.add(np.ascontiguousarray(indexed_key))

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        return None

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del alpha, calib_artifact, mix_calib_data, mix_calib_artifact

        self._reset_memory_store()
        self._ewma_scale = None
        self._last_query_key = None
        self._last_scale_before_update = None
        self._last_base_key_source = None

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
        base_keys = self._extract_base_key_batch(fc_result.state, calib_data.X_calib)
        if base_keys.shape[0] != abs_resid_cal.shape[0]:
            raise ValueError(f"Mismatch between key rows ({base_keys.shape[0]}) and residuals ({abs_resid_cal.shape[0]})")

        y_pre = _safe_float_array(_as_numpy(calib_data.Y_pre_calib).reshape(-1))
        calib_size = abs_resid_cal.shape[0]
        situation_keys = []
        for idx in range(calib_size):
            y_history = np.concatenate([y_pre, y_cal[:idx]]).astype(np.float32)
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

        keep_frac = float(np.clip(self._memory_use_calib_frac, 0.0, 1.0))
        if calib_size == 0 or keep_frac == 0.0:
            raise ValueError("Calibration set for RACP memory is empty after filtering")
        keep_count = max(1, int(round(calib_size * keep_frac)))
        memory_start_idx = calib_size - keep_count if self._use_recent_calib else 0
        memory_end_idx = calib_size if self._use_recent_calib else keep_count

        ewma_scale = self._init_scale(abs_resid_cal, memory_start_idx)
        for idx in range(memory_start_idx, memory_end_idx):
            b_t = max(float(ewma_scale), self._ewma_min_scale)
            score = float(abs_resid_cal[idx] / (b_t + self._eps)) if self._normalize_score else float(abs_resid_cal[idx])
            self._append_memory(situation_keys[idx], score)
            ewma_scale = self._ewma_update(ewma_scale, float(abs_resid_cal[idx]))
        self._ewma_scale = float(ewma_scale)

        artifact = PICalibArtifacts(fc_Y_hat=y_hat_cal, eps=abs_resid_cal.reshape(-1, 1))
        self._ensure_public_memory_views()
        artifact.add_info = {
            "memory_start_idx": int(memory_start_idx),
            "memory_end_idx": int(memory_end_idx),
            "base_key_source_used": self._last_base_key_source,
            "similarity_metric": self._similarity_metric,
            "use_residual_context": self._use_residual_context,
            "memory_size": int(self._memory_score.shape[0]) if self._memory_score is not None else 0,
            "retrieval_backend": "faiss" if self._use_faiss else "numpy",
        }
        return artifact

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if not self.model_ready():
            raise ValueError("RACP model is not calibrated")

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
        y_history = _safe_float_array(_as_numpy(pred_data.Y_past).reshape(-1))
        eps_history = _safe_float_array(_as_numpy(pred_data.eps_past).reshape(-1)) if (self._use_residual_context and pred_data.eps_past is not None) else None
        query_key = self._build_situation_key(
            base_key=base_key,
            y_hat=y_hat_scalar,
            y_history=y_history,
            eps_history=eps_history,
        )

        b_t = max(float(self._ewma_scale), self._ewma_min_scale)
        top_scores, top_idx = self._search_memory(query_key, self._k)
        weights = _softmax(top_scores, temperature=self._temperature)
        if self._use_finite_sample_correction:
            if self._local_sample_size_mode == "effective":
                local_n = _effective_sample_size(weights)
            else:
                local_n = len(top_idx)
            quantile_level = _conformal_quantile_level(pred_data.alpha, local_n)
        else:
            quantile_level = 1.0 - float(pred_data.alpha)
        selected_scores = self._selected_memory_scores(top_idx)
        score_q = _weighted_quantile(selected_scores, q=quantile_level, weights=weights)
        radius = float(b_t * score_q) if self._normalize_score else float(score_q)
        self._ensure_public_memory_views()
        if self._use_global_score_floor and self._memory_score is not None:
            global_weights = np.ones_like(self._memory_score, dtype=np.float32)
            if self._use_finite_sample_correction:
                global_q = _conformal_quantile_level(pred_data.alpha, len(self._memory_score))
            else:
                global_q = 1.0 - float(pred_data.alpha)
            global_score_q = _weighted_quantile(self._memory_score, q=global_q, weights=global_weights)
            global_radius = float(b_t * global_score_q) if self._normalize_score else float(global_score_q)
            radius = max(radius, global_radius)
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
        b_t = self._last_scale_before_update if self._last_scale_before_update is not None else max(float(self._ewma_scale), self._ewma_min_scale)
        if self._online_memory and self._last_query_key is not None:
            score_new = float(abs_resid / (b_t + self._eps)) if self._normalize_score else float(abs_resid)
            self._append_memory(self._last_query_key, score_new)
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
            and self._memory_score is not None
            and self._memory_score.shape[0] > 0
            and self._ewma_scale is not None
        )
