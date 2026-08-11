from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def safe_1d_float(values) -> np.ndarray:
    values = as_numpy(values).astype(np.float32).reshape(-1)
    return values


def fixed_tail(values: np.ndarray, length: int) -> np.ndarray:
    values = safe_1d_float(values)
    length = max(int(length), 1)
    if values.size == 0:
        return np.zeros((length,), dtype=np.float32)
    tail = values[-length:]
    if tail.size < length:
        pad_value = float(tail[0]) if tail.size > 0 else 0.0
        tail = np.pad(tail, (length - tail.size, 0), mode="constant", constant_values=pad_value)
    return tail.astype(np.float32)


def normalize_with_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = safe_1d_float(values)
    reference = safe_1d_float(reference)
    if reference.size == 0:
        return values.astype(np.float32)
    center = float(np.mean(reference))
    scale = float(np.std(reference) + 1e-6)
    return ((values - center) / scale).astype(np.float32)


def build_raw_window_forecast_key(
    y_history,
    y_hat,
    context_length: int,
    mode: str = "raw_window_plus_forecast",
    fc_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    mode = str(mode).lower()
    y_tail = fixed_tail(y_history, context_length)
    y_hat_arr = safe_1d_float(y_hat)
    norm_tail = normalize_with_reference(y_tail, y_tail)
    norm_forecast = normalize_with_reference(y_hat_arr, y_tail)

    if mode == "raw_window":
        return norm_tail.astype(np.float32)
    if mode == "forecast":
        return norm_forecast.astype(np.float32)
    if mode == "raw_window_plus_forecast":
        return np.concatenate([norm_tail, norm_forecast]).astype(np.float32)
    if mode == "fc_state":
        if fc_state is None:
            raise ValueError("key mode 'fc_state' requires a forecast state")
        return safe_1d_float(fc_state).astype(np.float32)
    if mode == "fc_state_plus_forecast":
        if fc_state is None:
            raise ValueError("key mode 'fc_state_plus_forecast' requires a forecast state")
        return np.concatenate([safe_1d_float(fc_state), norm_forecast]).astype(np.float32)
    raise ValueError(f"Unsupported retrieval key mode: {mode}")
