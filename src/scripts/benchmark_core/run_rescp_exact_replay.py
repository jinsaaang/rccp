from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "source"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=["node", "channel"])
    return df


def _load_beijing_target(data_dir: Path) -> pd.DataFrame:
    station_frames = {}
    for file in sorted(data_dir.glob("PRSA*.csv")):
        station = file.stem.replace("PRSA_Data_", "").split("_")[0]
        frame = pd.read_csv(file)
        frame["datetime"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
        frame = frame.set_index("datetime")
        station_frames[station] = frame[["PM10"]].ffill().bfill()
    common_index = None
    for frame in station_frames.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    values = {}
    for station, frame in station_frames.items():
        values[(station, "PM10_0")] = frame.reindex(common_index)["PM10"]
    out = pd.DataFrame(values, index=common_index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["node", "channel"])
    return out


class _CopiedReservoir:
    def __init__(
        self,
        n_internal_units=512,
        spectral_radius=1.3,
        leak=0.95,
        connectivity=0.2,
        input_scaling=0.25,
        noise_level=0.0,
        seed=801363602,
    ):
        self._n_internal_units = int(n_internal_units)
        self._input_scaling = float(input_scaling)
        self._noise_level = float(noise_level)
        self._leak = leak
        self._seed = int(seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        self._input_weights = None
        internal = sparse.rand(
            self._n_internal_units,
            self._n_internal_units,
            density=float(connectivity),
        ).todense()
        internal[np.where(internal > 0)] -= 0.5
        eigvals, _ = np.linalg.eig(internal)
        internal /= np.abs(eigvals).max() / float(spectral_radius)
        self._internal_weights = torch.tensor(internal, dtype=torch.float32)

    def to(self, device: torch.device):
        self._internal_weights = self._internal_weights.to(device)
        if self._input_weights is not None:
            self._input_weights = self._input_weights.to(device)
        return self

    def get_states(self, x: torch.Tensor) -> torch.Tensor:
        n_nodes, n_steps, n_features = x.shape
        if self._input_weights is None:
            self._input_weights = (
                2.0 * np.random.binomial(1, 0.5, [self._n_internal_units, n_features]) - 1.0
            ) * self._input_scaling
            self._input_weights = torch.tensor(self._input_weights, device=x.device, dtype=torch.float32)
        state = torch.zeros(n_nodes, self._n_internal_units, device=x.device, dtype=torch.float32)
        states = torch.empty(n_nodes, n_steps, self._n_internal_units, device=x.device, dtype=torch.float32)
        for step in range(n_steps):
            current = x[:, step, :]
            pre = self._internal_weights @ state.T + self._input_weights @ current.T
            if self._noise_level:
                pre += torch.rand(self._n_internal_units, n_nodes, device=x.device) * self._noise_level
            state = (1.0 - float(self._leak)) * state + torch.tanh(pre).T
            states[:, step, :] = state
        return states


def _sample_parallel(cal_residuals: torch.Tensor, similarity: torch.Tensor, sample_size: int) -> torch.Tensor:
    idxs = torch.multinomial(similarity.t(), num_samples=sample_size, replacement=True)
    col_idx = torch.arange(cal_residuals.shape[1], dtype=torch.long, device=cal_residuals.device).unsqueeze(1)
    return cal_residuals[idxs, col_idx].t()


def _run_sampler(cal_states, cal_residuals, test_states, y_hat, y_true, alpha=0.1):
    temperature = 0.1
    n_quantiles = 100
    past_residuals_window = 3200
    eta = 0.0
    target_coverage = 1 - alpha
    running_alpha = alpha
    cal_residuals = cal_residuals.clone()
    cal_states = cal_states.clone()
    if past_residuals_window < cal_residuals.shape[0]:
        cal_residuals = cal_residuals[-past_residuals_window:]
        cal_states = cal_states[-past_residuals_window:]
    sample_size = past_residuals_window
    running_sample_size = sample_size
    lower_quantiles = torch.zeros((test_states.shape[0], test_states.shape[1], n_quantiles), device=test_states.device)
    upper_quantiles = torch.zeros_like(lower_quantiles)
    coverages = torch.zeros(test_states.shape[0], test_states.shape[1], device=test_states.device)
    rolling_coverage = torch.zeros(test_states.shape[0], device=test_states.device)
    for i in tqdm(range(len(test_states)), desc="rescp-exact", leave=False):
        test_state = test_states[i].reshape(1, test_states.shape[1], test_states.shape[2])
        scores = torch.einsum("cnh,tnh->cn", cal_states, test_state)
        weights = torch.arange(cal_states.shape[0], device=cal_states.device).unsqueeze(1)
        weights = weights / weights.sum()
        similarity = F.softmax(scores / temperature, dim=0)
        similarity = similarity * weights
        similarity = similarity / similarity.sum(dim=0, keepdim=True)
        sampled = _sample_parallel(cal_residuals.squeeze(-1), similarity, running_sample_size)
        lower_betas = torch.linspace(1.0e-3, running_alpha - 1.0e-3, n_quantiles, device=test_states.device)
        upper_betas = 1 - running_alpha + lower_betas
        lower_quantiles[i] = torch.quantile(sampled, lower_betas, dim=0).t()
        upper_quantiles[i] = torch.quantile(sampled, upper_betas, dim=0).t()
        _, idx_min = torch.min(upper_quantiles[i] - lower_quantiles[i], dim=1)
        node_idx = torch.arange(lower_quantiles.shape[1], device=test_states.device)
        lo = y_hat[i].squeeze() + lower_quantiles[i][node_idx, idx_min]
        hi = y_hat[i].squeeze() + upper_quantiles[i][node_idx, idx_min]
        coverages[i] = (lo <= y_true[i].squeeze()) & (y_true[i].squeeze() <= hi)
        rolling_coverage[i] = coverages[: i + 1].mean()
        running_alpha = torch.clip(
            torch.tensor(running_alpha, device=test_states.device) + eta * (rolling_coverage[i] - target_coverage),
            1.0e-3,
            0.3,
        )
        old_len = cal_states.shape[0]
        cal_states = torch.roll(cal_states, -1, dims=0)
        cal_states[-1] = test_state
        assert old_len == cal_states.shape[0]
        cal_residuals = torch.roll(cal_residuals, -1, dims=0)
        cal_residuals[-1] = y_true[i] - y_hat[i]
    quantiles = torch.cat([lower_quantiles, upper_quantiles], dim=2)
    lower_q = quantiles[:, :, :n_quantiles]
    upper_q = quantiles[:, :, n_quantiles:]
    idx = torch.argmin(upper_q - lower_q, dim=2)
    lower_best = lower_q.gather(dim=2, index=idx.unsqueeze(2)).squeeze(2)
    upper_best = upper_q.gather(dim=2, index=idx.unsqueeze(2)).squeeze(2)
    return y_hat.squeeze(-1) + lower_best, y_hat.squeeze(-1) + upper_best


def _metrics(lower, upper, y, mask, alpha):
    valid = mask.bool()
    covered = ((lower <= y) & (y <= upper))[valid].float().mean()
    width = (upper - lower)[valid].float().mean()
    penalty = (torch.clamp(lower - y, min=0) + torch.clamp(y - upper, min=0))[valid]
    winkler = ((upper - lower)[valid] + (2 / alpha) * penalty).float().mean()
    return {
        "coverage": float(covered.cpu()),
        "delta_cov": float((covered - (1 - alpha)).cpu()),
        "width": float(width.cpu()),
        "winkler": float(winkler.cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "methods/upstream/rescp/reservoir_conformal_prediction/logs/base/beijing/rnn/2026-05-16/00-07-55"))
    parser.add_argument("--data-dir", default=str(DATA_ROOT / "bejing_air_quality"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results/raw/rescp_exact_replay_air_rnn.json"))
    parser.add_argument("--calib-start", type=int, default=16844)
    parser.add_argument("--calib-end", type=int, default=26891)
    parser.add_argument("--test-start", type=int, default=28056)
    parser.add_argument("--test-end", type=int, default=35040)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    residuals_input = _normalize_columns(pd.read_hdf(artifact_dir / "residuals.h5", key="input"))
    residuals_target = _normalize_columns(pd.read_hdf(artifact_dir / "residuals.h5", key="target"))
    target_mask = _normalize_columns(pd.read_hdf(artifact_dir / "residuals.h5", key="target_mask"))
    lower_clip = residuals_input.quantile(0.005, axis=0)
    upper_clip = residuals_input.quantile(0.995, axis=0)
    residuals_input = residuals_input.clip(lower_clip, upper_clip, axis=1)
    residuals_target = residuals_target.clip(lower_clip, upper_clip, axis=1)
    target = _load_beijing_target(Path(args.data_dir))
    stations = list(residuals_input.columns.get_level_values(0))
    target = target.loc[:, [(station, "PM10_0") for station in stations]]

    indices = np.load(artifact_dir / "indices.npz")
    valid_input_idx = indices["valid_input_indices"].astype(int)
    valid_target_idx = indices["valid_target_indices"].astype(int)
    start_at = int(valid_input_idx[0])
    end_at = int(valid_input_idx[-1])
    seq_index = np.arange(start_at, end_at + 1)

    n_nodes = len(stations)
    input_timeline = np.full((len(seq_index), n_nodes, 1), np.nan, dtype=np.float32)
    input_timeline[valid_input_idx - start_at, :, 0] = residuals_input.to_numpy(dtype=np.float32)

    calib_indices = np.arange(args.calib_start, args.calib_end)
    test_indices = np.arange(args.test_start, args.test_end)
    cal_residuals_raw = torch.from_numpy(input_timeline[calib_indices - start_at]).float()

    scaler_mean = np.nanmean(input_timeline[calib_indices - start_at])
    scaler_std = np.nanstd(input_timeline[calib_indices - start_at])
    scaled = (input_timeline - scaler_mean) / scaler_std
    scaled = np.nan_to_num(scaled, nan=0.0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = _CopiedReservoir().to(device)
    states_after = encoder.get_states(torch.from_numpy(scaled).float().permute(1, 0, 2).to(device))
    states_after = states_after.permute(1, 0, 2)
    zero = torch.zeros(1, n_nodes, states_after.shape[-1], device=device)
    states_before = torch.cat([zero, states_after[:-1]], dim=0)
    norm = torch.linalg.norm(states_before, dim=-1, keepdim=True)
    states_before = torch.where(norm > 0, states_before / norm, states_before)

    cal_states = states_before[calib_indices - start_at]
    test_states = states_before[test_indices - start_at]

    residual_target_timeline = np.full((len(target), n_nodes), np.nan, dtype=np.float32)
    residual_target_timeline[valid_target_idx, :] = residuals_target.to_numpy(dtype=np.float32)
    mask_timeline = np.zeros((len(target), n_nodes), dtype=bool)
    mask_timeline[valid_target_idx, :] = target_mask.to_numpy(dtype=bool)
    y_np = target.to_numpy(dtype=np.float32)
    y_test = torch.from_numpy(y_np[test_indices]).float().to(device)
    test_residuals = torch.from_numpy(residual_target_timeline[test_indices]).float().to(device)
    y_hat = (y_test - test_residuals).unsqueeze(-1)
    y_true = y_test.unsqueeze(-1)
    mask = torch.from_numpy(mask_timeline[test_indices]).to(device)

    lower, upper = _run_sampler(
        cal_states=cal_states,
        cal_residuals=cal_residuals_raw.to(device),
        test_states=test_states,
        y_hat=y_hat,
        y_true=y_true,
        alpha=0.1,
    )
    result = _metrics(lower, upper, y_test, mask, 0.1)
    result["method"] = "rescp_exact_replay"
    result["dataset"] = "air"
    result["forecaster"] = "upstream_rnn32_artifact"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
