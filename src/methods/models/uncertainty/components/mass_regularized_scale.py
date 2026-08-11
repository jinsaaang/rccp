from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if values.size != weights.size:
        raise ValueError("weights must match values length")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        weights = np.ones_like(weights, dtype=np.float32)
        weight_sum = float(weights.size)
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / weight_sum
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(max(int(idx), 0), sorted_values.shape[0] - 1)
    return float(sorted_values[idx])


def scores_to_distances(scores: np.ndarray, metric: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    metric = str(metric).lower()
    if metric == "cosine":
        scores = np.clip(scores, -1.0, 1.0)
        return np.sqrt(np.maximum(2.0 - 2.0 * scores, 0.0)).astype(np.float32)
    if metric == "l2":
        return np.maximum(-scores, 0.0).astype(np.float32)
    raise ValueError(f"Unsupported metric: {metric}")


def pairwise_distances(keys: np.ndarray, metric: str) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.float32)
    metric = str(metric).lower()
    if metric == "cosine":
        norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
        keys_norm = keys / norms
        sims = keys_norm @ keys_norm.T
        return scores_to_distances(sims, metric="cosine")
    if metric == "l2":
        diff = keys[:, None, :] - keys[None, :, :]
        return np.linalg.norm(diff, axis=2).astype(np.float32)
    raise ValueError(f"Unsupported metric: {metric}")


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    bandwidth = max(float(bandwidth), 1e-6)
    scaled = np.asarray(distances, dtype=np.float32) / bandwidth
    return np.exp(-0.5 * scaled ** 2).astype(np.float32)


def estimate_bandwidth(
    keys: np.ndarray,
    metric: str,
    quantile: float = 0.10,
    max_points: int = 2048,
) -> float:
    keys = np.asarray(keys, dtype=np.float32)
    if keys.shape[0] <= 1:
        return 1.0
    if keys.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keys = keys[rng.choice(keys.shape[0], size=max_points, replace=False)]
    distances = pairwise_distances(keys, metric=metric)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    non_self = distances[mask]
    positive = non_self[non_self > 1e-8]
    if positive.size == 0:
        return 1.0
    return float(max(np.quantile(positive, float(quantile)), 1e-6))


def estimate_lambda_from_loo_mass(
    keys: np.ndarray,
    metric: str,
    bandwidth: float,
    kernel: str = "gaussian",
    quantile: float = 0.50,
    max_points: int = 2048,
) -> float:
    if kernel != "gaussian":
        raise ValueError(f"Unsupported kernel: {kernel}")
    keys = np.asarray(keys, dtype=np.float32)
    if keys.shape[0] <= 1:
        return 1.0
    if keys.shape[0] > max_points:
        rng = np.random.default_rng(1)
        keys = keys[rng.choice(keys.shape[0], size=max_points, replace=False)]
    distances = pairwise_distances(keys, metric=metric)
    weights = gaussian_kernel(distances, bandwidth=bandwidth)
    np.fill_diagonal(weights, 0.0)
    masses = np.sum(weights, axis=1)
    positive = masses[masses > 1e-8]
    if positive.size == 0:
        return 1.0
    return float(max(np.quantile(positive, float(quantile)), 1e-6))


@dataclass
class ScaleResult:
    scale: np.ndarray
    mass: float
    rho: float
    base_scale: np.ndarray


class MassRegularizedScaleEstimator:
    def __init__(
        self,
        residuals: np.ndarray,
        bandwidth: float,
        lambda_value: float,
        metric: str = "cosine",
        kernel: str = "gaussian",
        scale_quantile: float = 0.5,
        shrinkage_mode: str = "distribution",
        ignore_zero_values: bool = False,
    ):
        residuals = np.asarray(residuals, dtype=np.float32)
        if residuals.ndim == 1:
            residuals = residuals[:, None]
        if residuals.shape[0] == 0:
            raise ValueError("residual memory must be non-empty")
        self.residuals = residuals
        self.bandwidth = max(float(bandwidth), 1e-6)
        self.lambda_value = max(float(lambda_value), 0.0)
        self.metric = str(metric).lower()
        self.kernel = str(kernel).lower()
        self.scale_quantile = float(scale_quantile)
        self.shrinkage_mode = str(shrinkage_mode).lower()
        self.ignore_zero_values = bool(ignore_zero_values)
        if self.kernel != "gaussian":
            raise ValueError(f"Unsupported kernel: {kernel}")
        if self.shrinkage_mode not in {"distribution", "convex_scale"}:
            raise ValueError(f"Unsupported shrinkage_mode: {shrinkage_mode}")
        self.base_scale = self._compute_base_scale()

    def _valid_values_weights(self, values: np.ndarray, weights: np.ndarray):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if self.ignore_zero_values:
            mask = values > 1e-12
            values = values[mask]
            weights = weights[mask]
        return values, weights

    def _compute_base_scale(self) -> np.ndarray:
        scales = []
        for h in range(self.residuals.shape[1]):
            values, weights = self._valid_values_weights(
                self.residuals[:, h],
                np.ones(self.residuals.shape[0], dtype=np.float32),
            )
            if values.size == 0:
                values = self.residuals[:, h]
                weights = np.ones(self.residuals.shape[0], dtype=np.float32)
            scales.append(weighted_quantile(values, self.scale_quantile, weights))
        return np.asarray(scales, dtype=np.float32)

    def compute(self, candidate_indices: np.ndarray, candidate_scores: np.ndarray) -> ScaleResult:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
        if candidate_indices.size == 0:
            return ScaleResult(scale=self.base_scale.copy(), mass=0.0, rho=0.0, base_scale=self.base_scale.copy())
        distances = scores_to_distances(candidate_scores, metric=self.metric)
        local_weights_raw = gaussian_kernel(distances, bandwidth=self.bandwidth)
        mass = float(np.sum(local_weights_raw))
        rho = float(mass / (mass + self.lambda_value)) if (mass + self.lambda_value) > 0 else 0.0
        rho = float(np.clip(rho, 0.0, 1.0))

        if mass <= 1e-12 or rho <= 0.0:
            return ScaleResult(scale=self.base_scale.copy(), mass=mass, rho=0.0, base_scale=self.base_scale.copy())

        local_weights = rho * (local_weights_raw / mass)
        if self.shrinkage_mode == "convex_scale":
            scales = []
            local_weights_norm = local_weights_raw / mass
            for horizon_idx in range(self.residuals.shape[1]):
                local_values, local_weights = self._valid_values_weights(
                    self.residuals[candidate_indices, horizon_idx],
                    local_weights_norm,
                )
                if local_values.size == 0:
                    local_scale = float(self.base_scale[horizon_idx])
                else:
                    local_scale = weighted_quantile(
                        local_values,
                        self.scale_quantile,
                        local_weights,
                    )
                scales.append(rho * local_scale + (1.0 - rho) * float(self.base_scale[horizon_idx]))
            return ScaleResult(scale=np.asarray(scales, dtype=np.float32), mass=mass, rho=rho, base_scale=self.base_scale.copy())

        base_weights = np.full((self.residuals.shape[0],), (1.0 - rho) / self.residuals.shape[0], dtype=np.float32)
        scales = []
        for horizon_idx in range(self.residuals.shape[1]):
            values = np.concatenate([self.residuals[candidate_indices, horizon_idx], self.residuals[:, horizon_idx]])
            weights = np.concatenate([local_weights, base_weights])
            values, weights = self._valid_values_weights(values, weights)
            if values.size == 0:
                scales.append(float(self.base_scale[horizon_idx]))
                continue
            scales.append(weighted_quantile(values, self.scale_quantile, weights))
        return ScaleResult(scale=np.asarray(scales, dtype=np.float32), mass=mass, rho=rho, base_scale=self.base_scale.copy())
