from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
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
GENERATED_FC_ROOT = REPO_ROOT / "models_save" / "lstm_fc"

from evaluation.benchmark_results import BenchmarkResult, dump_result
from loader.generator import DataGenerator

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


DATASET_CONFIGS = {
    "air": ("bejing_air_pm10", "default_3year_gn"),
    "solar": ("nsdb2018-20_60m", "default_3year_gn"),
    "wind": ("wind", "default_3year_gn"),
    "weather": ("weather", "default_3year_gn"),
    "amazon": ("amazon", "default_3year_gn"),
    "stock": ("amazon", "default_3year_gn"),
    "amzn": ("amazon", "default_3year_gn"),
    "apple": ("apple", "default_3year_gn"),
    "aapl": ("apple", "default_3year_gn"),
    "google": ("google", "default_3year_gn"),
    "goog": ("google", "default_3year_gn"),
    "elec": ("elec_normalized", "default_3year_gn"),
    "electricity": ("elec_normalized", "default_3year_gn"),
    "elec_normalized": ("elec_normalized", "default_3year_gn"),
    "exchange": ("exchange_rate", "default_3year_gn"),
    "exchange_rate": ("exchange_rate", "default_3year_gn"),
    "exchange_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
    "exchange_ot1_lagged": ("exchange_rate_ot1_lagged", "default_3year_gn"),
}


STATE_SAVE_OVERHEAD_SEC = {
    "air": 4.0,
    "solar": 11.3,
    "elec": 0.06,
    "electricity": 0.06,
    "elec_normalized": 0.06,
    "wind": 0.14,
}


def _parse_data_splits(value: str) -> list[float]:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) not in {3, 4}:
        raise ValueError(f"--data-splits expects 3 or 4 comma-separated values, got {value!r}")
    splits = [float(part) for part in parts]
    if any(part < 0.0 for part in splits) or sum(splits[:3]) <= 0.0:
        raise ValueError(f"Invalid --data-splits={value!r}")
    return splits


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return scores
    scores = scores / max(float(temperature), 1.0e-6)
    scores = scores - float(np.max(scores))
    exp_scores = np.exp(scores)
    denom = float(np.sum(exp_scores))
    if denom <= 0.0 or not math.isfinite(denom):
        return np.ones_like(exp_scores, dtype=np.float32) / exp_scores.shape[0]
    return (exp_scores / denom).astype(np.float32)


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights, dtype=np.float32)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    idx = np.searchsorted(cdf, float(np.clip(q, 0.0, 1.0)), side="left")
    idx = min(max(int(idx), 0), sorted_values.shape[0] - 1)
    return float(sorted_values[idx])


def _order_stat_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute conformal quantile from an empty score set")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), values.size)
    return float(np.sort(values)[rank - 1])


def _l2_normalize_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    return (row / (np.linalg.norm(row) + 1.0e-6)).astype(np.float32)


def _initial_scale_pre_calib_diff(y_pre: np.ndarray, min_scale: float) -> float:
    y_pre = np.asarray(y_pre, dtype=np.float32).reshape(-1)
    if y_pre.size > 1:
        return float(max(np.mean(np.abs(np.diff(y_pre))), min_scale))
    return float(min_scale)


def _search(
    keys: np.ndarray,
    keys_cos: np.ndarray,
    memory_size: int,
    query_key: np.ndarray,
    k: int,
    similarity_metric: str,
    faiss_index=None,
) -> tuple[np.ndarray, np.ndarray]:
    similarity_metric = str(similarity_metric).lower()
    top_k = memory_size if int(k) <= 0 else min(int(k), memory_size)
    if faiss_index is not None:
        query = np.asarray(query_key, dtype=np.float32).reshape(1, -1)
        if similarity_metric == "cosine":
            query = _l2_normalize_row(query_key).reshape(1, -1)
        scores, idx = faiss_index.search(np.ascontiguousarray(query.reshape(1, -1)), top_k)
        valid = idx[0] >= 0
        scores = scores[0][valid].astype(np.float32)
        if similarity_metric == "l2":
            scores = (-np.sqrt(np.maximum(scores, 0.0))).astype(np.float32)
        return scores, idx[0][valid].astype(np.int64)
    if similarity_metric == "cosine":
        query = _l2_normalize_row(query_key)
        sims = keys_cos[:memory_size] @ query
    else:
        query = np.asarray(query_key, dtype=np.float32).reshape(-1)
        sims = (-np.linalg.norm(keys[:memory_size] - query[None, :], axis=1)).astype(np.float32)
    if top_k < memory_size:
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
    else:
        idx = np.arange(memory_size)
    return sims[idx].astype(np.float32), idx.astype(np.int64)


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True) + 1.0e-6
    return (rows / norms).astype(np.float32)


def _search_batch(
    keys: np.ndarray,
    keys_cos: np.ndarray,
    memory_size: int,
    query_keys: np.ndarray,
    k: int,
    similarity_metric: str,
    faiss_index=None,
) -> tuple[np.ndarray, np.ndarray]:
    similarity_metric = str(similarity_metric).lower()
    top_k = memory_size if int(k) <= 0 else min(int(k), memory_size)
    if faiss_index is not None:
        query = np.asarray(query_keys, dtype=np.float32)
        if similarity_metric == "cosine":
            query = _normalize_rows(query_keys)
        scores, idx = faiss_index.search(np.ascontiguousarray(query), top_k)
        scores = scores.astype(np.float32)
        if similarity_metric == "l2":
            scores = (-np.sqrt(np.maximum(scores, 0.0))).astype(np.float32)
        return scores, idx.astype(np.int64)
    if similarity_metric == "cosine":
        query = _normalize_rows(query_keys)
        sims = query @ keys_cos[:memory_size].T
    else:
        query = np.asarray(query_keys, dtype=np.float32)
        diff = query[:, None, :] - keys[None, :memory_size, :]
        sims = (-np.linalg.norm(diff, axis=2)).astype(np.float32)
    if top_k < memory_size:
        idx = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
    else:
        idx = np.repeat(np.arange(memory_size, dtype=np.int64)[None, :], query.shape[0], axis=0)
    scores = np.take_along_axis(sims, idx, axis=1)
    return scores.astype(np.float32), idx.astype(np.int64)


def _has_faiss_gpu() -> bool:
    if faiss is None or not hasattr(faiss, "StandardGpuResources"):
        return False
    try:
        return int(faiss.get_num_gpus()) > 0
    except Exception:
        return False


def _make_faiss_index(dim: int, device: str, similarity_metric: str = "cosine"):
    if faiss is None:
        return None, None
    device = str(device).lower()
    similarity_metric = str(similarity_metric).lower()
    if device == "auto":
        device = "cuda" if _has_faiss_gpu() else "cpu"
    if device == "cuda":
        if not _has_faiss_gpu():
            raise RuntimeError("--faiss-device=cuda requested but FAISS GPU is unavailable")
        resources = faiss.StandardGpuResources()
        if similarity_metric == "cosine":
            return faiss.GpuIndexFlatIP(resources, int(dim)), resources
        return faiss.GpuIndexFlatL2(resources, int(dim)), resources
    if device == "cpu":
        if similarity_metric == "cosine":
            return faiss.IndexFlatIP(int(dim)), None
        return faiss.IndexFlatL2(int(dim)), None
    raise ValueError(f"Unsupported FAISS device {device!r}")


def _proposal(
    keys: np.ndarray,
    keys_cos: np.ndarray,
    score_abs: np.ndarray,
    score_pos: np.ndarray,
    score_neg: np.ndarray,
    memory_size: int,
    query_key: np.ndarray,
    alpha: float,
    scale: float,
    k: int,
    temperature: float,
    min_radius: float,
    eps: float,
    similarity_metric: str,
    interval_mode: str,
    faiss_index=None,
) -> tuple[float, float, float]:
    top_scores, top_idx = _search(keys, keys_cos, memory_size, query_key, k, similarity_metric, faiss_index=faiss_index)
    weights = _softmax(top_scores, temperature=temperature)
    q_level = 1.0 - (float(alpha) / 2.0 if str(interval_mode).lower() == "one_sided" else float(alpha))
    proposal_abs = _weighted_quantile(score_abs[top_idx], q=q_level, weights=weights)
    proposal_pos = _weighted_quantile(score_pos[top_idx], q=q_level, weights=weights)
    proposal_neg = _weighted_quantile(score_neg[top_idx], q=q_level, weights=weights)
    proposal_abs = max(float(scale * proposal_abs), min_radius, eps)
    proposal_pos = max(float(scale * proposal_pos), min_radius, eps)
    proposal_neg = max(float(scale * proposal_neg), min_radius, eps)
    return proposal_abs, proposal_pos, proposal_neg


def _proposal_batch(
    keys: np.ndarray,
    keys_cos: np.ndarray,
    score_abs: np.ndarray,
    score_pos: np.ndarray,
    score_neg: np.ndarray,
    memory_size: int,
    query_keys: np.ndarray,
    alpha: float,
    scales: np.ndarray,
    k: int,
    temperature: float,
    min_radius: float,
    eps: float,
    similarity_metric: str,
    interval_mode: str,
    faiss_index=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_scores, top_idx = _search_batch(keys, keys_cos, memory_size, query_keys, k, similarity_metric, faiss_index=faiss_index)
    q_level = 1.0 - (float(alpha) / 2.0 if str(interval_mode).lower() == "one_sided" else float(alpha))
    n = int(query_keys.shape[0])
    proposal_abs = np.empty((n,), dtype=np.float32)
    proposal_pos = np.empty((n,), dtype=np.float32)
    proposal_neg = np.empty((n,), dtype=np.float32)
    for row in range(n):
        valid = top_idx[row] >= 0
        idx = top_idx[row][valid]
        weights = _softmax(top_scores[row][valid], temperature=temperature)
        scale = float(scales[row])
        proposal_abs[row] = max(float(scale * _weighted_quantile(score_abs[idx], q=q_level, weights=weights)), min_radius, eps)
        proposal_pos[row] = max(float(scale * _weighted_quantile(score_pos[idx], q=q_level, weights=weights)), min_radius, eps)
        proposal_neg[row] = max(float(scale * _weighted_quantile(score_neg[idx], q=q_level, weights=weights)), min_radius, eps)
    return proposal_abs, proposal_pos, proposal_neg


def _run_one_series(
    y: np.ndarray,
    y_hat: np.ndarray,
    states: np.ndarray,
    calib_step: int,
    test_step: int,
    alpha: float,
    k: int,
    temperature: float,
    ewma_beta: float,
    ewma_min_scale: float,
    eps: float,
    min_memory: int,
    correction_floor: float,
    proposal_min_radius: float,
    online_memory: bool,
    use_faiss: bool,
    faiss_device: str,
    online_block_size: int,
    correction_mode: str,
    similarity_metric: str,
    interval_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float32).reshape(-1)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[:, None]
    if y_hat.shape[0] != y.shape[0] or states.shape[0] != y.shape[0]:
        raise ValueError(f"Expected full-length y_hat/state, got y={y.shape}, y_hat={y_hat.shape}, states={states.shape}")

    y_hat_cal = y_hat[calib_step:test_step]
    y_cal = y[calib_step:test_step]
    state_cal = states[calib_step:test_step]
    signed_cal = (y_cal - y_hat_cal).astype(np.float32)
    abs_cal = np.abs(signed_cal).astype(np.float32)
    pos_cal = np.maximum(signed_cal, 0.0).astype(np.float32)
    neg_cal = np.maximum(-signed_cal, 0.0).astype(np.float32)
    keys_cal = np.concatenate([state_cal, y_hat_cal.reshape(-1, 1)], axis=1).astype(np.float32)

    max_points = int((test_step - calib_step) + (y.shape[0] - test_step))
    dim = int(keys_cal.shape[1])
    keys = np.empty((max(max_points, 1), dim), dtype=np.float32)
    keys_cos = np.empty_like(keys)
    score_abs = np.empty((max(max_points, 1),), dtype=np.float32)
    score_pos = np.empty((max(max_points, 1),), dtype=np.float32)
    score_neg = np.empty((max(max_points, 1),), dtype=np.float32)
    memory_size = 0
    similarity_metric = "l2" if str(similarity_metric).lower() == "euclidean" else str(similarity_metric).lower()
    if similarity_metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported similarity_metric={similarity_metric!r}")
    faiss_index, faiss_resources = _make_faiss_index(dim, faiss_device, similarity_metric) if use_faiss else (None, None)

    def append(key: np.ndarray, abs_resid: float, pos_resid: float, neg_resid: float, scale: float) -> None:
        nonlocal memory_size
        b_t = max(float(scale), ewma_min_scale)
        keys[memory_size] = key
        keys_cos[memory_size] = _l2_normalize_row(key)
        if faiss_index is not None:
            indexed = keys_cos[memory_size] if similarity_metric == "cosine" else keys[memory_size]
            faiss_index.add(np.ascontiguousarray(indexed.reshape(1, -1)))
        score_abs[memory_size] = np.float32(abs_resid / (b_t + eps))
        score_pos[memory_size] = np.float32(pos_resid / (b_t + eps))
        score_neg[memory_size] = np.float32(neg_resid / (b_t + eps))
        memory_size += 1

    def append_block(block_keys: np.ndarray, abs_resid: np.ndarray, pos_resid: np.ndarray, neg_resid: np.ndarray, scales: np.ndarray) -> None:
        nonlocal memory_size
        count = int(block_keys.shape[0])
        if count <= 0:
            return
        start = memory_size
        end = start + count
        keys[start:end] = block_keys
        keys_cos[start:end] = _normalize_rows(block_keys)
        denom = np.maximum(scales.astype(np.float32), ewma_min_scale) + eps
        score_abs[start:end] = (abs_resid.astype(np.float32) / denom).astype(np.float32)
        score_pos[start:end] = (pos_resid.astype(np.float32) / denom).astype(np.float32)
        score_neg[start:end] = (neg_resid.astype(np.float32) / denom).astype(np.float32)
        if faiss_index is not None:
            indexed = keys_cos[start:end] if similarity_metric == "cosine" else keys[start:end]
            faiss_index.add(np.ascontiguousarray(indexed))
        memory_size = end

    ewma_scale = _initial_scale_pre_calib_diff(y[:calib_step], ewma_min_scale)
    block_size = max(int(online_block_size), 1)
    correction_scores: list[float] = []
    correction_mode = str(correction_mode).lower()
    interval_mode = str(interval_mode).lower()
    if correction_mode not in {"max_score", "none", "no_correction", "proposal_only"}:
        raise ValueError(f"Unsupported correction_mode={correction_mode!r}")
    if interval_mode not in {"one_sided", "symmetric"}:
        raise ValueError(f"Unsupported interval_mode={interval_mode!r}")
    use_correction = correction_mode == "max_score"
    if block_size == 1:
        for i in range(abs_cal.shape[0]):
            b_t = max(float(ewma_scale), ewma_min_scale)
            if use_correction and memory_size >= min_memory:
                proposal_abs, proposal_pos, proposal_neg = _proposal(
                    keys, keys_cos, score_abs, score_pos, score_neg, memory_size, keys_cal[i],
                    alpha, b_t, k, temperature, proposal_min_radius, eps,
                    similarity_metric=similarity_metric,
                    interval_mode=interval_mode,
                    faiss_index=faiss_index,
                )
                if interval_mode == "symmetric":
                    correction_scores.append(float(abs_cal[i] / (proposal_abs + eps)))
                else:
                    correction_scores.append(
                        max(
                            float(pos_cal[i] / (proposal_pos + eps)),
                            float(neg_cal[i] / (proposal_neg + eps)),
                        )
                    )
            append(keys_cal[i], float(abs_cal[i]), float(pos_cal[i]), float(neg_cal[i]), b_t)
            ewma_scale = max(float(ewma_beta) * float(ewma_scale) + (1.0 - float(ewma_beta)) * float(abs_cal[i]), ewma_min_scale)
    else:
        for start in range(0, abs_cal.shape[0], block_size):
            end = min(start + block_size, abs_cal.shape[0])
            scales = np.empty((end - start,), dtype=np.float32)
            local_scale = float(ewma_scale)
            for row, idx in enumerate(range(start, end)):
                b_t = max(float(local_scale), ewma_min_scale)
                scales[row] = np.float32(b_t)
                local_scale = max(float(ewma_beta) * float(local_scale) + (1.0 - float(ewma_beta)) * float(abs_cal[idx]), ewma_min_scale)
            if use_correction and memory_size >= min_memory:
                proposal_abs, proposal_pos, proposal_neg = _proposal_batch(
                    keys, keys_cos, score_abs, score_pos, score_neg, memory_size, keys_cal[start:end],
                    alpha, scales, k, temperature, proposal_min_radius, eps,
                    similarity_metric=similarity_metric,
                    interval_mode=interval_mode,
                    faiss_index=faiss_index,
                )
                if interval_mode == "symmetric":
                    corr = abs_cal[start:end] / (proposal_abs + eps)
                else:
                    corr = np.maximum(
                        pos_cal[start:end] / (proposal_pos + eps),
                        neg_cal[start:end] / (proposal_neg + eps),
                    )
                correction_scores.extend(float(x) for x in corr)
            append_block(keys_cal[start:end], abs_cal[start:end], pos_cal[start:end], neg_cal[start:end], scales)
            ewma_scale = local_scale

    correction = (
        max(_order_stat_quantile(np.asarray(correction_scores, dtype=np.float32), alpha), correction_floor)
        if use_correction
        else 1.0
    )

    n_test = int(y.shape[0] - test_step)
    low = np.empty((n_test,), dtype=np.float32)
    high = np.empty((n_test,), dtype=np.float32)
    y_hat_test = y_hat[test_step:]
    keys_test = np.concatenate([states[test_step:], y_hat_test.reshape(-1, 1)], axis=1).astype(np.float32)
    signed_test = (y[test_step:] - y_hat_test).astype(np.float32)
    abs_test = np.abs(signed_test).astype(np.float32)
    pos_test = np.maximum(signed_test, 0.0).astype(np.float32)
    neg_test = np.maximum(-signed_test, 0.0).astype(np.float32)
    if block_size == 1:
        for i in range(n_test):
            b_t = max(float(ewma_scale), ewma_min_scale)
            proposal_abs, proposal_pos, proposal_neg = _proposal(
                keys, keys_cos, score_abs, score_pos, score_neg, memory_size, keys_test[i],
                alpha, b_t, k, temperature, proposal_min_radius, eps,
                similarity_metric=similarity_metric,
                interval_mode=interval_mode,
                faiss_index=faiss_index,
            )
            if interval_mode == "symmetric":
                low[i] = np.float32(y_hat_test[i] - correction * proposal_abs)
                high[i] = np.float32(y_hat_test[i] + correction * proposal_abs)
            else:
                low[i] = np.float32(y_hat_test[i] - correction * proposal_neg)
                high[i] = np.float32(y_hat_test[i] + correction * proposal_pos)
            if online_memory:
                append(keys_test[i], float(abs_test[i]), float(pos_test[i]), float(neg_test[i]), b_t)
            ewma_scale = max(float(ewma_beta) * float(ewma_scale) + (1.0 - float(ewma_beta)) * float(abs_test[i]), ewma_min_scale)
    else:
        for start in range(0, n_test, block_size):
            end = min(start + block_size, n_test)
            scales = np.empty((end - start,), dtype=np.float32)
            local_scale = float(ewma_scale)
            for row, idx in enumerate(range(start, end)):
                b_t = max(float(local_scale), ewma_min_scale)
                scales[row] = np.float32(b_t)
                local_scale = max(float(ewma_beta) * float(local_scale) + (1.0 - float(ewma_beta)) * float(abs_test[idx]), ewma_min_scale)
            proposal_abs, proposal_pos, proposal_neg = _proposal_batch(
                keys, keys_cos, score_abs, score_pos, score_neg, memory_size, keys_test[start:end],
                alpha, scales, k, temperature, proposal_min_radius, eps,
                similarity_metric=similarity_metric,
                interval_mode=interval_mode,
                faiss_index=faiss_index,
            )
            if interval_mode == "symmetric":
                low[start:end] = (y_hat_test[start:end] - correction * proposal_abs).astype(np.float32)
                high[start:end] = (y_hat_test[start:end] + correction * proposal_abs).astype(np.float32)
            else:
                low[start:end] = (y_hat_test[start:end] - correction * proposal_neg).astype(np.float32)
                high[start:end] = (y_hat_test[start:end] + correction * proposal_pos).astype(np.float32)
            if online_memory:
                append_block(keys_test[start:end], abs_test[start:end], pos_test[start:end], neg_test[start:end], scales)
            ewma_scale = local_scale
    return low, high


def _choose_torch_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--location-batch-device=cuda requested but CUDA is unavailable")
    return torch.device(name)


def _torch_weighted_quantile(values: torch.Tensor, q: float, weights: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=-1)
    sorted_values = torch.gather(values, -1, order)
    sorted_weights = torch.gather(weights, -1, order)
    denom = sorted_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    cdf = torch.cumsum(sorted_weights, dim=-1) / denom
    idx = torch.argmax((cdf >= float(q)).to(torch.int64), dim=-1)
    return torch.gather(sorted_values, -1, idx.unsqueeze(-1)).squeeze(-1)


def _torch_proposals(
    memory_keys: torch.Tensor,
    score_abs: torch.Tensor,
    score_pos: torch.Tensor,
    score_neg: torch.Tensor,
    query_keys: torch.Tensor,
    scales: torch.Tensor,
    allowed_counts: torch.Tensor,
    alpha: float,
    k: int,
    temperature: float,
    min_radius: float,
    eps: float,
    similarity_metric: str,
    interval_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Shapes: memory_keys [M, N, D], query_keys [B, N, D], scores [M, N].
    memory_size = int(memory_keys.shape[0])
    top_k = memory_size if int(k) <= 0 else min(int(k), memory_size)
    if top_k <= 0:
        raise ValueError("RCCP proposal requires at least one memory point")
    similarity_metric = "l2" if str(similarity_metric).lower() == "euclidean" else str(similarity_metric).lower()
    if similarity_metric == "cosine":
        memory_norm = memory_keys / (torch.linalg.norm(memory_keys, dim=-1, keepdim=True) + 1.0e-6)
        query_norm = query_keys / (torch.linalg.norm(query_keys, dim=-1, keepdim=True) + 1.0e-6)
        sims = torch.einsum("bnd,mnd->bnm", query_norm, memory_norm)
    elif similarity_metric == "l2":
        sims = -torch.cdist(
            query_keys.permute(1, 0, 2),
            memory_keys.permute(1, 0, 2),
            p=2,
        ).permute(1, 0, 2)
    else:
        raise ValueError(f"Unsupported similarity_metric={similarity_metric!r}")
    memory_idx = torch.arange(memory_size, device=sims.device).view(1, 1, memory_size)
    sims = sims.masked_fill(memory_idx >= allowed_counts.view(-1, 1, 1), -torch.inf)
    top_scores, top_idx = torch.topk(sims, k=top_k, dim=-1)
    weights = torch.softmax(top_scores / max(float(temperature), 1.0e-6), dim=-1)
    loc_idx = torch.arange(score_pos.shape[1], device=sims.device).view(1, -1, 1)
    abs_values = score_abs[top_idx, loc_idx]
    pos_values = score_pos[top_idx, loc_idx]
    neg_values = score_neg[top_idx, loc_idx]
    q_level = 1.0 - (float(alpha) / 2.0 if str(interval_mode).lower() == "one_sided" else float(alpha))
    proposal_abs = scales * _torch_weighted_quantile(abs_values, q_level, weights)
    proposal_pos = scales * _torch_weighted_quantile(pos_values, q_level, weights)
    proposal_neg = scales * _torch_weighted_quantile(neg_values, q_level, weights)
    floor = max(float(min_radius), float(eps))
    return proposal_abs.clamp_min(floor), proposal_pos.clamp_min(floor), proposal_neg.clamp_min(floor)


def _order_stat_quantile_by_location(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if values.numel() == 0 or values.shape[0] == 0:
        raise ValueError("Cannot compute RCCP correction from an empty score set")
    rank = int(np.ceil((int(values.shape[0]) + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), int(values.shape[0]))
    sorted_values, _ = torch.sort(values, dim=0)
    return sorted_values[rank - 1]


def _aligned_location_batch_inputs(
    datasets,
    predictions: dict,
    states: dict,
    prediction_offset: int,
    base_key_source: str,
    require_multi_location: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if require_multi_location and len(datasets) <= 1:
        raise ValueError("Location-batch RCCP requires at least two aligned series")
    lengths = {int(dataset.Y_full.shape[0]) for dataset in datasets}
    calib_steps = {int(dataset.calib_step) for dataset in datasets}
    test_steps = {int(dataset.test_step) for dataset in datasets}
    if len(lengths) != 1 or len(calib_steps) != 1 or len(test_steps) != 1:
        raise ValueError(
            "Location-batch RCCP requires aligned lengths/splits; "
            f"lengths={sorted(lengths)} calib={sorted(calib_steps)} test={sorted(test_steps)}"
        )
    n_steps = next(iter(lengths))
    calib_step = next(iter(calib_steps))
    test_step = next(iter(test_steps))
    y_cols = []
    yhat_cols = []
    key_cols = []
    base_key_source = str(base_key_source)
    for dataset in datasets:
        yhat_full = _full_from_artifact(predictions, dataset.ts_id, n_steps, prediction_offset).reshape(n_steps)
        if base_key_source == "fc_state":
            key_full = _full_from_artifact(states, dataset.ts_id, n_steps, prediction_offset)
        elif base_key_source == "x_step":
            key_full = _as_np(dataset.X_full).astype(np.float32)
            if key_full.ndim == 1:
                key_full = key_full[:, None]
            else:
                key_full = key_full.reshape(key_full.shape[0], -1)
        else:
            raise ValueError(f"Unsupported base_key_source={base_key_source!r}")
        needed = slice(calib_step, n_steps)
        if np.isnan(yhat_full[needed]).any() or np.isnan(key_full[needed]).any():
            raise RuntimeError(f"Prediction/state artifacts do not cover calib/test for {dataset.ts_id}")
        y_cols.append(_as_np(dataset.Y_full).reshape(-1).astype(np.float32))
        yhat_cols.append(yhat_full.astype(np.float32))
        key_cols.append(key_full.astype(np.float32))
    return (
        np.stack(y_cols, axis=1).astype(np.float32),
        np.stack(yhat_cols, axis=1).astype(np.float32),
        np.stack(key_cols, axis=1).astype(np.float32),
        calib_step,
        test_step,
    )


def _can_use_aligned_location_batch(datasets) -> bool:
    lengths = {int(dataset.Y_full.shape[0]) for dataset in datasets}
    calib_steps = {int(dataset.calib_step) for dataset in datasets}
    test_steps = {int(dataset.test_step) for dataset in datasets}
    return len(lengths) == 1 and len(calib_steps) == 1 and len(test_steps) == 1


def _run_aligned_location_batch(
    y: np.ndarray,
    y_hat: np.ndarray,
    states: np.ndarray,
    calib_step: int,
    test_step: int,
    alpha: float,
    k: int,
    temperature: float,
    ewma_beta: float,
    ewma_min_scale: float,
    eps: float,
    min_memory: int,
    correction_floor: float,
    proposal_min_radius: float,
    online_memory: bool,
    correction_mode: str,
    device: torch.device,
    chunk_size: int,
    similarity_metric: str,
    interval_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    y_hat = np.asarray(y_hat, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if y.ndim != 2 or y_hat.shape != y.shape or states.ndim != 3 or states.shape[:2] != y.shape:
        raise ValueError(f"Expected y/y_hat [T,N] and states [T,N,D], got y={y.shape} y_hat={y_hat.shape} states={states.shape}")
    correction_mode = str(correction_mode).lower()
    interval_mode = str(interval_mode).lower()
    if correction_mode not in {"max_score", "none", "no_correction", "proposal_only"}:
        raise ValueError(f"Unsupported correction_mode={correction_mode!r}")
    if interval_mode not in {"one_sided", "symmetric"}:
        raise ValueError(f"Unsupported interval_mode={interval_mode!r}")
    use_correction = correction_mode == "max_score"
    similarity_metric = "l2" if str(similarity_metric).lower() == "euclidean" else str(similarity_metric).lower()
    if similarity_metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported similarity_metric={similarity_metric!r}")

    signed = (y - y_hat).astype(np.float32)
    abs_resid = np.abs(signed).astype(np.float32)
    pos_resid = np.maximum(signed, 0.0).astype(np.float32)
    neg_resid = np.maximum(-signed, 0.0).astype(np.float32)
    keys_all = np.concatenate([states, y_hat[..., None]], axis=-1).astype(np.float32)

    start_step = int(calib_step)
    n_steps = int(y.shape[0])
    n_memory = n_steps - start_step
    n_locations = int(y.shape[1])
    scales = np.empty((n_memory, n_locations), dtype=np.float32)
    ewma = np.asarray(
        [_initial_scale_pre_calib_diff(y[:calib_step, loc], ewma_min_scale) for loc in range(n_locations)],
        dtype=np.float32,
    )
    for row, step in enumerate(range(start_step, n_steps)):
        b_t = np.maximum(ewma, float(ewma_min_scale)).astype(np.float32)
        scales[row] = b_t
        ewma = np.maximum(float(ewma_beta) * ewma + (1.0 - float(ewma_beta)) * abs_resid[step], float(ewma_min_scale)).astype(np.float32)

    keys_mem = torch.from_numpy(keys_all[start_step:]).to(device=device)
    score_abs = torch.from_numpy(abs_resid[start_step:] / (scales + float(eps))).to(device=device)
    score_pos = torch.from_numpy(pos_resid[start_step:] / (scales + float(eps))).to(device=device)
    score_neg = torch.from_numpy(neg_resid[start_step:] / (scales + float(eps))).to(device=device)
    scales_t = torch.from_numpy(np.maximum(scales, float(ewma_min_scale))).to(device=device)
    abs_t = torch.from_numpy(abs_resid[start_step:]).to(device=device)
    pos_t = torch.from_numpy(pos_resid[start_step:]).to(device=device)
    neg_t = torch.from_numpy(neg_resid[start_step:]).to(device=device)
    yhat_t = torch.from_numpy(y_hat[start_step:]).to(device=device)

    cal_len = int(test_step - calib_step)
    test_len = int(n_steps - test_step)
    chunk = max(int(chunk_size), 1)
    if use_correction:
        correction_chunks = []
        correction_start = max(int(min_memory), int(k) if int(k) > 0 else int(min_memory))
        for start in range(correction_start, cal_len, chunk):
            end = min(start + chunk, cal_len)
            allowed = torch.arange(start, end, device=device, dtype=torch.long)
            proposal_abs, proposal_pos, proposal_neg = _torch_proposals(
                memory_keys=keys_mem[:cal_len],
                score_abs=score_abs[:cal_len],
                score_pos=score_pos[:cal_len],
                score_neg=score_neg[:cal_len],
                query_keys=keys_mem[start:end],
                scales=scales_t[start:end],
                allowed_counts=allowed,
                alpha=alpha,
                k=k,
                temperature=temperature,
                min_radius=proposal_min_radius,
                eps=eps,
                similarity_metric=similarity_metric,
                interval_mode=interval_mode,
            )
            if interval_mode == "symmetric":
                correction_chunks.append(abs_t[start:end] / (proposal_abs + float(eps)))
            else:
                correction_chunks.append(torch.maximum(
                    pos_t[start:end] / (proposal_pos + float(eps)),
                    neg_t[start:end] / (proposal_neg + float(eps)),
                ))
        correction_scores = torch.cat(correction_chunks, dim=0)
        correction = torch.maximum(
            _order_stat_quantile_by_location(correction_scores, alpha),
            torch.full((n_locations,), float(correction_floor), device=device),
        )
    else:
        correction = torch.ones((n_locations,), device=device)

    lows = torch.empty((test_len, n_locations), device=device, dtype=torch.float32)
    highs = torch.empty((test_len, n_locations), device=device, dtype=torch.float32)
    for start in range(0, test_len, chunk):
        end = min(start + chunk, test_len)
        mem_start = cal_len + start
        mem_end = cal_len + end
        if online_memory:
            allowed = torch.arange(mem_start, mem_end, device=device, dtype=torch.long)
            memory_end = mem_end - 1
        else:
            allowed = torch.full((end - start,), cal_len, device=device, dtype=torch.long)
            memory_end = cal_len
        proposal_abs, proposal_pos, proposal_neg = _torch_proposals(
            memory_keys=keys_mem[:memory_end],
            score_abs=score_abs[:memory_end],
            score_pos=score_pos[:memory_end],
            score_neg=score_neg[:memory_end],
            query_keys=keys_mem[mem_start:mem_end],
            scales=scales_t[mem_start:mem_end],
            allowed_counts=allowed,
            alpha=alpha,
            k=k,
            temperature=temperature,
            min_radius=proposal_min_radius,
            eps=eps,
            similarity_metric=similarity_metric,
            interval_mode=interval_mode,
        )
        if interval_mode == "symmetric":
            lows[start:end] = yhat_t[mem_start:mem_end] - correction.view(1, -1) * proposal_abs
            highs[start:end] = yhat_t[mem_start:mem_end] + correction.view(1, -1) * proposal_abs
        else:
            lows[start:end] = yhat_t[mem_start:mem_end] - correction.view(1, -1) * proposal_neg
            highs[start:end] = yhat_t[mem_start:mem_end] + correction.view(1, -1) * proposal_pos
    return lows.cpu().numpy(), highs.cpu().numpy()


def _series_metrics(y_norm: np.ndarray, low_norm: np.ndarray, high_norm: np.ndarray, mean, std, alpha: float, mask) -> dict:
    y_norm = np.asarray(y_norm, dtype=np.float64).reshape(-1)
    low_norm = np.asarray(low_norm, dtype=np.float64).reshape(-1)
    high_norm = np.asarray(high_norm, dtype=np.float64).reshape(-1)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1).astype(bool)
        y_norm = y_norm[mask]
        low_norm = low_norm[mask]
        high_norm = high_norm[mask]
    mean = float(np.asarray(_as_np(mean)).reshape(-1)[0])
    std = float(np.asarray(_as_np(std)).reshape(-1)[0])
    y = y_norm * std + mean
    low = low_norm * std + mean
    high = high_norm * std + mean
    inside = (low <= y) & (y <= high)
    dist = np.minimum(np.abs(high - y), np.abs(y - low))
    dist_norm = np.minimum(np.abs(high_norm - y_norm), np.abs(y_norm - low_norm))
    width = high - low
    width_norm = high_norm - low_norm
    n = max(int(y.shape[0]), 1)
    return {
        "coverage": float(np.mean(inside)),
        "width": float(np.mean(width)),
        "winkler": float((np.sum(width) + (2.0 * np.sum(dist[~inside]) / float(alpha))) / n),
        "winkler_norm": float((np.sum(width_norm) + (2.0 * np.sum(dist_norm[~inside]) / float(alpha))) / n),
    }


def _pointwise_rows(
    dataset_name: str,
    method: str,
    ts_id: str,
    y_norm: np.ndarray,
    yhat_norm: np.ndarray,
    low_norm: np.ndarray,
    high_norm: np.ndarray,
    mean,
    std,
    alpha: float,
    mask,
    step_offset: int,
):
    y_norm = np.asarray(y_norm, dtype=np.float64).reshape(-1)
    yhat_norm = np.asarray(yhat_norm, dtype=np.float64).reshape(-1)
    low_norm = np.asarray(low_norm, dtype=np.float64).reshape(-1)
    high_norm = np.asarray(high_norm, dtype=np.float64).reshape(-1)
    if not (y_norm.shape == yhat_norm.shape == low_norm.shape == high_norm.shape):
        raise ValueError(
            "Pointwise export expects matching arrays, got "
            f"y={y_norm.shape} yhat={yhat_norm.shape} low={low_norm.shape} high={high_norm.shape}"
        )
    if mask is None:
        mask_arr = np.ones_like(y_norm, dtype=bool)
    else:
        mask_arr = np.asarray(mask).reshape(-1).astype(bool)
        if mask_arr.shape[0] != y_norm.shape[0]:
            raise ValueError(f"Pointwise mask length mismatch: mask={mask_arr.shape} y={y_norm.shape}")
    mean = float(np.asarray(_as_np(mean)).reshape(-1)[0])
    std = float(np.asarray(_as_np(std)).reshape(-1)[0])
    y = y_norm * std + mean
    yhat = yhat_norm * std + mean
    low = low_norm * std + mean
    high = high_norm * std + mean
    width = high - low
    abs_error = np.abs(y - yhat)
    lower_excess = np.maximum(low - y, 0.0)
    upper_excess = np.maximum(y - high, 0.0)
    miss_dist = np.maximum(lower_excess, upper_excess)
    inside = miss_dist <= 0.0
    winkler_point = width + (2.0 / float(alpha)) * miss_dist
    for idx in np.flatnonzero(mask_arr):
        yield {
            "dataset": dataset_name,
            "method": method,
            "ts_id": ts_id,
            "step": int(step_offset + idx),
            "width": float(width[idx]),
            "abs_error": float(abs_error[idx]),
            "miss_dist": float(miss_dist[idx]),
            "lower_excess": float(lower_excess[idx]),
            "upper_excess": float(upper_excess[idx]),
            "inside": bool(inside[idx]),
            "winkler_point": float(winkler_point[idx]),
        }


def _write_pointwise_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "ts_id",
        "step",
        "width",
        "abs_error",
        "miss_dist",
        "lower_excess",
        "upper_excess",
        "inside",
        "winkler_point",
    ]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _full_from_artifact(artifact: dict, ts_id: str, n_steps: int, offset: int) -> np.ndarray:
    if ts_id not in artifact:
        raise KeyError(f"Missing artifact key {ts_id!r}; available keys include {list(artifact)[:5]}")
    values = _as_np(artifact[ts_id]).astype(np.float32)
    if values.ndim == 1:
        values = values[:, None]
    full = np.full((n_steps, values.shape[1]), np.nan, dtype=np.float32)
    end = min(n_steps, offset + values.shape[0])
    full[offset:end] = values[: end - offset]
    return full


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Fast exact RCCP final run from pretrained prediction/state artifacts.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--forecaster-tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--include-state-save-overhead",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a conservative dataset-level hidden-state serialization overhead "
            "to time_sec/wall_clock_time_sec for RCCP-only runtime accounting."
        ),
    )
    parser.add_argument(
        "--state-save-overhead-sec",
        type=float,
        default=None,
        help="Explicit overhead in seconds. Overrides the dataset-level default when set.",
    )
    parser.add_argument("--prediction-offset", type=int, default=50)
    parser.add_argument("--data-base-dir", default=str(DATA_ROOT))
    parser.add_argument(
        "--data-splits",
        default="0.4,0.4,0.2",
        help=(
            "Dataset split shares. Use 0.4,0.2,0.2,0.2 for validation search "
            "and 0.4,0.4,0.2 for final evaluation."
        ),
    )
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ewma-beta", type=float, default=0.95)
    parser.add_argument("--interval-mode", choices=["one_sided", "symmetric"], default="one_sided")
    parser.add_argument("--base-key-source", choices=["fc_state", "x_step"], default="fc_state")
    parser.add_argument("--similarity-metric", choices=["cosine", "l2", "euclidean"], default="cosine")
    parser.add_argument("--ewma-min-scale", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--min-memory", type=int, default=64)
    parser.add_argument("--correction-floor", type=float, default=0.0)
    parser.add_argument(
        "--correction-mode",
        choices=["max_score", "none", "no_correction", "proposal_only"],
        default="max_score",
        help="max_score is default RCCP; none/proposal_only uses the retrieval proposal without conformal correction.",
    )
    parser.add_argument("--proposal-min-radius", type=float, default=1.0e-6)
    parser.add_argument("--online-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-faiss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faiss-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--online-block-size",
        type=int,
        default=1,
        help="1 is exact prequential online update. >1 delays memory insertion within a block and uses batched search.",
    )
    parser.add_argument(
        "--location-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional aligned multi-location RCCP batching. Disabled by default for reproducible calibration-time accounting."
        ),
    )
    parser.add_argument("--location-batch-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--location-batch-query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--pointwise-out",
        default=None,
        help="Optional CSV/CSV.GZ path for figure-level pointwise interval diagnostics.",
    )
    args = parser.parse_args()

    dataset_conf, task_conf = DATASET_CONFIGS[args.dataset]
    task_cfg = OmegaConf.load(REPO_ROOT / "config" / "task" / f"{task_conf}.yaml")
    dataset_cfg = OmegaConf.load(REPO_ROOT / "config" / "dataset" / f"{dataset_conf}.yaml")
    task_cfg.data_splits = _parse_data_splits(args.data_splits)

    exp_name = f"paper_fc_{args.dataset}_{args.forecaster_tag}_seed{args.seed}_{args.artifact_tag}"
    artifact_dir = GENERATED_FC_ROOT
    prediction_path = artifact_dir / f"prediction_{exp_name}.pt"
    state_path = artifact_dir / f"state_{exp_name}.pt"
    if not prediction_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            "Missing RCCP prediction/state cache. Run run_paper_grid.py without --skip-train first "
            f"to generate them from the included checkpoints. prediction={prediction_path} state={state_path}"
        )

    datasets = DataGenerator.get_data(dataset_cfg, task_cfg, replace_base_dir=str(Path(args.data_base_dir)))
    predictions = _load_torch(prediction_path)
    states = _load_torch(state_path)
    method_started = time.perf_counter()
    similarity_metric = "l2" if str(args.similarity_metric).lower() == "euclidean" else str(args.similarity_metric).lower()

    per_series = []
    pointwise_rows = []
    location_batch_policy = "explicit" if bool(args.location_batch) else "disabled"
    used_location_batch = bool(args.location_batch)
    location_batch_device = None
    if used_location_batch:
        if int(args.online_block_size) != 1:
            raise ValueError("--location-batch currently supports only exact --online-block-size=1")
        y_batch, yhat_batch, state_batch, calib_step, test_step = _aligned_location_batch_inputs(
            datasets=datasets,
            predictions=predictions,
            states=states,
            prediction_offset=int(args.prediction_offset),
            base_key_source=str(args.base_key_source),
            require_multi_location=False,
        )
        location_batch_device = _choose_torch_device(str(args.location_batch_device))
        low_batch, high_batch = _run_aligned_location_batch(
            y=y_batch,
            y_hat=yhat_batch,
            states=state_batch,
            calib_step=int(calib_step),
            test_step=int(test_step),
            alpha=float(args.alpha),
            k=int(args.k),
            temperature=float(args.temperature),
            ewma_beta=float(args.ewma_beta),
            ewma_min_scale=float(args.ewma_min_scale),
            eps=float(args.eps),
            min_memory=int(args.min_memory),
            correction_floor=float(args.correction_floor),
            proposal_min_radius=float(args.proposal_min_radius),
            online_memory=bool(args.online_memory),
            correction_mode=str(args.correction_mode),
            device=location_batch_device,
            chunk_size=int(args.location_batch_query_chunk_size),
            similarity_metric=similarity_metric,
            interval_mode=str(args.interval_mode),
        )
        for loc, dataset in enumerate(datasets):
            mask_full = getattr(dataset, "mask_full", None)
            mask = None if mask_full is None else _as_np(mask_full).reshape(-1)[int(test_step):]
            if args.pointwise_out:
                pointwise_rows.extend(
                    _pointwise_rows(
                        dataset_name=str(args.dataset),
                        method="rccp",
                        ts_id=str(dataset.ts_id),
                        y_norm=y_batch[int(test_step):, loc],
                        yhat_norm=yhat_batch[int(test_step):, loc],
                        low_norm=low_batch[:, loc],
                        high_norm=high_batch[:, loc],
                        mean=dataset.Y_normalize_props[0],
                        std=dataset.Y_normalize_props[1],
                        alpha=float(args.alpha),
                        mask=mask,
                        step_offset=int(test_step),
                    )
                )
            per_series.append(
                _series_metrics(
                    y_norm=y_batch[int(test_step):, loc],
                    low_norm=low_batch[:, loc],
                    high_norm=high_batch[:, loc],
                    mean=dataset.Y_normalize_props[0],
                    std=dataset.Y_normalize_props[1],
                    alpha=float(args.alpha),
                    mask=mask,
                )
            )
    else:
        for dataset in datasets:
            n_steps = int(dataset.Y_full.shape[0])
            yhat_full = _full_from_artifact(predictions, dataset.ts_id, n_steps, int(args.prediction_offset)).reshape(n_steps)
            state_full = _full_from_artifact(states, dataset.ts_id, n_steps, int(args.prediction_offset))
            if str(args.base_key_source) == "fc_state":
                key_full = state_full
            else:
                key_full = _as_np(dataset.X_full).astype(np.float32)
                if key_full.ndim == 1:
                    key_full = key_full[:, None]
                else:
                    key_full = key_full.reshape(key_full.shape[0], -1)
            needed = slice(int(dataset.calib_step), n_steps)
            if np.isnan(yhat_full[needed]).any() or np.isnan(key_full[needed]).any():
                raise RuntimeError(f"Prediction/state artifacts do not cover calib/test for {dataset.ts_id}")
            y_full = _as_np(dataset.Y_full).reshape(-1).astype(np.float32)
            low_norm, high_norm = _run_one_series(
                y=y_full,
                y_hat=yhat_full,
                states=key_full,
                calib_step=int(dataset.calib_step),
                test_step=int(dataset.test_step),
                alpha=float(args.alpha),
                k=int(args.k),
                temperature=float(args.temperature),
                ewma_beta=float(args.ewma_beta),
                ewma_min_scale=float(args.ewma_min_scale),
                eps=float(args.eps),
                min_memory=int(args.min_memory),
                correction_floor=float(args.correction_floor),
                proposal_min_radius=float(args.proposal_min_radius),
                online_memory=bool(args.online_memory),
                use_faiss=bool(args.use_faiss),
                faiss_device=str(args.faiss_device),
                online_block_size=int(args.online_block_size),
                correction_mode=str(args.correction_mode),
                similarity_metric=similarity_metric,
                interval_mode=str(args.interval_mode),
            )
            mask_full = getattr(dataset, "mask_full", None)
            mask = None if mask_full is None else _as_np(mask_full).reshape(-1)[int(dataset.test_step):]
            if args.pointwise_out:
                pointwise_rows.extend(
                    _pointwise_rows(
                        dataset_name=str(args.dataset),
                        method="rccp",
                        ts_id=str(dataset.ts_id),
                        y_norm=y_full[int(dataset.test_step):],
                        yhat_norm=yhat_full[int(dataset.test_step):],
                        low_norm=low_norm,
                        high_norm=high_norm,
                        mean=dataset.Y_normalize_props[0],
                        std=dataset.Y_normalize_props[1],
                        alpha=float(args.alpha),
                        mask=mask,
                        step_offset=int(dataset.test_step),
                    )
                )
            per_series.append(
                _series_metrics(
                    y_norm=y_full[int(dataset.test_step):],
                    low_norm=low_norm,
                    high_norm=high_norm,
                    mean=dataset.Y_normalize_props[0],
                    std=dataset.Y_normalize_props[1],
                    alpha=float(args.alpha),
                    mask=mask,
                )
            )

    method_elapsed = time.perf_counter() - method_started
    wall_elapsed = time.perf_counter() - wall_started
    state_save_overhead = 0.0
    if args.state_save_overhead_sec is not None:
        state_save_overhead = max(float(args.state_save_overhead_sec), 0.0)
    elif bool(args.include_state_save_overhead):
        state_save_overhead = float(STATE_SAVE_OVERHEAD_SEC.get(str(args.dataset), 0.0))
    reported_wall_elapsed = wall_elapsed + state_save_overhead
    runner_mode = "exact-prequential" if int(args.online_block_size) == 1 else "batched-block-online"
    correction_mode = "none" if str(args.correction_mode) in {"none", "no_correction", "proposal_only"} else "max_score"
    effective_use_faiss = bool((not used_location_batch) and args.use_faiss and faiss is not None)
    effective_faiss_device = "none"
    if effective_use_faiss:
        requested_device = str(args.faiss_device)
        effective_faiss_device = "cuda" if requested_device == "auto" and _has_faiss_gpu() else requested_device
    result = BenchmarkResult(
        method="rccp",
        dataset=args.dataset,
        forecaster=args.forecaster,
        alpha=float(args.alpha),
        coverage=float(np.mean([row["coverage"] for row in per_series])),
        delta_cov=float(args.alpha - (1.0 - np.mean([row["coverage"] for row in per_series]))),
        width=float(np.mean([row["width"] for row in per_series])),
        winkler=float(np.mean([row["winkler"] for row in per_series])),
        winkler_norm=float(np.mean([row["winkler_norm"] for row in per_series])),
        time_sec=reported_wall_elapsed,
        wall_clock_time_sec=reported_wall_elapsed,
        method_wall_clock_time_sec=method_elapsed,
        calibration_time_sec=method_elapsed + state_save_overhead,
        state_save_overhead_sec=state_save_overhead if state_save_overhead > 0.0 else None,
        time_sec_without_state_save_overhead=wall_elapsed if state_save_overhead > 0.0 else None,
        wall_clock_time_sec_without_state_save_overhead=wall_elapsed if state_save_overhead > 0.0 else None,
        run_dir=None,
        notes=(
            f"rccp_fast_from_artifacts; mode={runner_mode}; default-only; "
            f"correction_mode={correction_mode}; "
            f"interval_mode={args.interval_mode}; "
            f"correction_floor={float(args.correction_floor)}; "
            f"base_key_source={args.base_key_source}; "
            f"similarity_metric={similarity_metric}; "
            f"faiss_available={faiss is not None}; faiss_gpu_available={_has_faiss_gpu()}; "
            f"use_faiss={effective_use_faiss}; faiss_device={effective_faiss_device}; "
            f"online_block_size={int(args.online_block_size)}; "
            f"data_splits={list(task_cfg.data_splits)}; "
            f"location_batch={used_location_batch}; location_batch_policy={location_batch_policy}; "
            f"location_batch_backend={'torch_topk' if used_location_batch else 'none'}; "
            f"location_batch_device={location_batch_device}; "
            f"state_save_overhead_sec={state_save_overhead}; "
            f"prediction={prediction_path}; state={state_path}; series={len(datasets)}"
        ),
    )
    dump_result(result, Path(args.out))
    if args.pointwise_out:
        _write_pointwise_csv(Path(args.pointwise_out), pointwise_rows)
    print(Path(args.out))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
