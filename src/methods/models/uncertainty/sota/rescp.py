"""ResCP (Reservoir Conformal Prediction) ported as a PIModel.

Adapted from Neglia et al., ICLR 2026 (``methods/upstream/rescp``). This port
keeps the algorithmic core:
- A randomly initialised ESN encodes the residual sequence into reservoir states.
- Test/calibration states are L2-normalised so similarity reduces to a dot
  product.
- Conformity weights = softmax(similarity / temperature) elementwise multiplied
  by a linear-decay schedule, then re-normalised.
- ``n_quantiles`` candidate (lower, upper) pairs are drawn and the narrowest is
  used (Eq. 12 of the paper).

Differences from upstream:
- The default path is the upstream-style node-parallel residual sampler. Each
  location is a separate node/column, so calibration residuals are never pooled
  across locations.
"""
from __future__ import annotations

from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
)
from utils.calc_np import calc_residuals
from utils.utils import get_device


class RescpModel(PIModel):
    """ResCP as a central PIModel implementation.

    Args (passed via kwargs):
        hidden_size: Reservoir dimension D_h.
        spectral_radius: ``ρ(W_h)``. Paper Table 4 best per (dataset, backbone).
        leaking_rate: ``l``.
        input_scaling: ``W_x`` scale.
        density: Reservoir connectivity. Defaults to ``0.2`` (paper).
        activation: ``tanh`` (paper) or ``relu``.
        temperature: Softmax τ for similarity reweighting. ``0.1`` (paper).
        decay: ``linear`` (paper), ``exponential``, or ``none``.
        decay_rate: Used by exponential decay only.
        similarity: ``cosine`` (paper), ``euclidean``, or ``angular``.
        sliding_window: ``past_residuals_window``. If ``None``, uses all calib.
        eta: ACI step (``0`` disables online α update).
        n_quantiles: Number of β candidates for Eq. 12. ``1`` = symmetric.
        seed: Reservoir RNG seed.
    """

    def __init__(self, **kwargs):
        super().__init__(
            use_dedicated_calibration=True,
            fc_prediction_out_modes=(PredictionOutputType.POINT,),
        )
        self._hidden_size = int(kwargs.get("hidden_size", 512))
        self._spectral_radius = float(kwargs.get("spectral_radius", 0.9))
        self._leaking_rate = float(kwargs.get("leaking_rate", 0.75))
        self._input_scaling = float(kwargs.get("input_scaling", 0.7))
        self._density = float(kwargs.get("density", 0.2))
        self._activation = str(kwargs.get("activation", "tanh"))
        self._temperature = float(kwargs.get("temperature", 0.1))
        self._decay = str(kwargs.get("decay", "linear"))
        self._decay_rate = float(kwargs.get("decay_rate", 0.99))
        self._similarity = str(kwargs.get("similarity", "cosine"))
        self._sliding_window = kwargs.get("sliding_window", None)
        if isinstance(self._sliding_window, str) and self._sliding_window.lower() == "all":
            self._sliding_window = None
        if self._sliding_window is not None:
            self._sliding_window = int(self._sliding_window)
        self._eta = float(kwargs.get("eta", 0.0))
        self._n_quantiles = int(kwargs.get("n_quantiles", 1))
        self._topk_residuals = kwargs.get("topk_residuals", None)
        if self._topk_residuals is not None and str(self._topk_residuals).lower() != "none":
            self._topk_residuals = int(self._topk_residuals)
        else:
            self._topk_residuals = None
        self._proposal_correction = bool(kwargs.get("proposal_correction", False))
        self._proposal_correction_style = str(kwargs.get("proposal_correction_style", "multiplicative_radius")).lower()
        if self._proposal_correction_style in {"multiplicative", "radius", "ratio"}:
            self._proposal_correction_style = "multiplicative_radius"
        if self._proposal_correction_style in {"additive", "cqr", "cqr_slack"}:
            self._proposal_correction_style = "additive_slack"
        if self._proposal_correction_style not in {"multiplicative_radius", "additive_slack"}:
            raise ValueError(f"Unsupported proposal_correction_style={self._proposal_correction_style!r}")
        self._proposal_correction_floor = float(kwargs.get("proposal_correction_floor", 1.0))
        self._proposal_correction_max_points = kwargs.get("proposal_correction_max_points", None)
        if (
            self._proposal_correction_max_points is not None
            and str(self._proposal_correction_max_points).lower() != "none"
        ):
            self._proposal_correction_max_points = int(self._proposal_correction_max_points)
        else:
            self._proposal_correction_max_points = None
        self._proposal_correction_ewma_beta = kwargs.get("proposal_correction_ewma_beta", None)
        if (
            self._proposal_correction_ewma_beta is not None
            and str(self._proposal_correction_ewma_beta).lower() != "none"
        ):
            self._proposal_correction_ewma_beta = float(self._proposal_correction_ewma_beta)
        else:
            self._proposal_correction_ewma_beta = None
        self._proposal_correction_ewma_min_scale = float(kwargs.get("proposal_correction_ewma_min_scale", 1e-6))
        self._seed = int(kwargs.get("seed", 0))
        self._eps_floor = float(kwargs.get("eps_floor", 1e-6))
        self._scale_reservoir_input = bool(kwargs.get("scale_reservoir_input", True))
        self._reservoir_scale_axis = str(kwargs.get("reservoir_scale_axis", "global"))
        self._location_batch = bool(kwargs.get("location_batch", False))
        self._sampler_cal_size = kwargs.get("sampler_cal_size", None)
        if self._sampler_cal_size is not None and str(self._sampler_cal_size).lower() != "all":
            self._sampler_cal_size = int(self._sampler_cal_size)
        self._sampler_val_fraction = float(kwargs.get("sampler_val_fraction", 0.0))
        self._samples_offset = int(kwargs.get("samples_offset", 0))
        self._sampler_start_offset = int(kwargs.get("sampler_start_offset", 0))
        self._test_warmup_skip = int(kwargs.get("test_warmup_skip", 0))
        self._zero_test_warmup_residuals = bool(kwargs.get("zero_test_warmup_residuals", False))

        self._running_alpha: dict[str, float] = {}
        self._parallel_ts_ids: list[str] = []
        self._parallel_artifacts: dict[str, PICalibArtifacts] = {}
        self._parallel_cal_residuals: Optional[torch.Tensor] = None
        self._parallel_cal_residuals_full: Optional[torch.Tensor] = None
        self._parallel_predictions: dict[float, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._device = torch.device("cpu")
        self._ready = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _calibrate(self, calib_data: List[PICalibData], alphas, **kwargs) -> List[PICalibArtifacts]:
        self._device = self._resolve_device(kwargs)
        artifacts = self._calibrate_parallel(calib_data)
        self._ready = True
        return artifacts

    def _calibrate_parallel(self, calib_data: List[PICalibData]) -> List[PICalibArtifacts]:
        self._parallel_ts_ids = [cd.ts_id for cd in calib_data]
        residuals = []
        artifacts = []
        for cd in calib_data:
            Y_hat = self._forcast_service.predict(
                FCPredictionData(
                    ts_id=cd.ts_id,
                    X_past=cd.X_pre_calib,
                    Y_past=cd.Y_pre_calib,
                    X_step=cd.X_calib,
                    step_offset=cd.step_offset,
                ),
                retrieve_tensor=False,
            ).point
            eps_calib_np = calc_residuals(y_hat=Y_hat, y=_to_numpy(cd.Y_calib))
            eps_calib = torch.from_numpy(np.asarray(eps_calib_np).reshape(-1, 1)).float()
            residuals.append(eps_calib)
            artifact = PICalibArtifacts()
            artifact.fc_Y_hat = Y_hat
            artifact.eps = eps_calib_np
            artifacts.append(artifact)
            self._parallel_artifacts[cd.ts_id] = artifact
        lengths = {r.shape[0] for r in residuals}
        if len(lengths) != 1:
            raise ValueError(f"Parallel ResCP requires aligned calibration lengths, got {sorted(lengths)}")
        cal_full = torch.stack(residuals, dim=1)  # [n_cal, n_nodes, 1]
        self._parallel_cal_residuals_full = cal_full
        self._parallel_cal_residuals = self._select_parallel_calibration(cal_full)
        return artifacts

    def _resolve_device(self, kwargs: dict) -> torch.device:
        exp = kwargs.get("experiment_config")
        gpu_id = getattr(exp, "gpu_id", None) if exp is not None else None
        if gpu_id is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return get_device(gpu_id)

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        del calib_artifact, mix_calib_data, mix_calib_artifact
        self._running_alpha[calib_data.ts_id] = float(alpha)
        return self._parallel_artifacts[calib_data.ts_id]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        alpha = kwargs.get("alpha")
        if alpha is not None:
            for ts_id in self._running_alpha:
                self._running_alpha[ts_id] = float(alpha)
            if float(alpha) not in self._parallel_predictions:
                dataset = kwargs.get("dataset")
                other_datasets = kwargs.get("other_datasets")
                if dataset is not None and other_datasets is not None:
                    self._precompute_parallel_predictions(float(alpha), [dataset, *other_datasets])

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        ts_id = pred_data.ts_id
        alpha = float(pred_data.alpha)
        if alpha not in self._parallel_predictions:
            raise RuntimeError("ResCP predictions were not precomputed before evaluation.")
        lo, hi, y_hat = self._parallel_predictions[alpha][ts_id]
        idx = int(pred_data.step_offset_prediction)
        return PIModelPrediction(
            pred_interval=(lo[idx].reshape(-1), hi[idx].reshape(-1)),
            fc_Y_hat=y_hat[idx].reshape(-1),
        )

    # ------------------------------------------------------------------
    # PIModel boilerplate
    # ------------------------------------------------------------------
    def model_ready(self):
        return self._ready

    def required_past_len(self) -> Tuple[int, int]:
        # ResCP keeps its own residual/state memory. The evaluator only needs
        # to pass a minimal history for the point forecaster call.
        return 0, 1

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.ts_id in self._parallel_ts_ids

    @property
    def can_handle_different_alpha(self):
        return True

    def _select_parallel_calibration(self, cal_full: torch.Tensor) -> torch.Tensor:
        start = self._sampler_start_offset
        if self._sampler_cal_size is not None and str(self._sampler_cal_size).lower() != "all":
            return cal_full[start: start + int(self._sampler_cal_size)]
        n_cal = cal_full.shape[0] - start
        if self._sampler_val_fraction > 0:
            val_len = int(self._sampler_val_fraction * n_cal)
            end = start + max(1, n_cal - val_len - self._samples_offset)
            return cal_full[start:end]
        if self._samples_offset > 0 and n_cal > self._samples_offset:
            return cal_full[start:-self._samples_offset]
        return cal_full[start:]

    def _precompute_parallel_predictions(self, alpha: float, datasets):
        by_id = {d.ts_id: d for d in datasets}
        missing = [ts_id for ts_id in self._parallel_ts_ids if ts_id not in by_id]
        if missing:
            raise ValueError(f"Parallel ResCP missing datasets for ts_ids: {missing}")
        ordered = [by_id[ts_id] for ts_id in self._parallel_ts_ids]
        test_lengths = {int(d.no_test_steps) for d in ordered}
        if len(test_lengths) != 1:
            raise ValueError(f"Parallel ResCP requires aligned test lengths, got {sorted(test_lengths)}")
        n_test = next(iter(test_lengths))

        y_hats = []
        y_true = []
        for d in ordered:
            y_hat = self._forcast_service.predict(
                FCPredictionData(
                    ts_id=d.ts_id,
                    X_past=d.X_full[: d.test_step],
                    Y_past=d.Y_full[: d.test_step],
                    X_step=d.X_test,
                    step_offset=d.test_step,
                ),
                retrieve_tensor=True,
            ).point.detach().cpu().reshape(n_test, 1)
            y_hats.append(y_hat)
            y_true.append(d.Y_test.detach().cpu().reshape(n_test, 1).float())
        y_hat_test = torch.stack(y_hats, dim=1).to(self._device)  # [n_test, n_nodes, 1]
        y_test = torch.stack(y_true, dim=1).to(self._device)
        test_residuals = y_test - y_hat_test
        state_test_residuals = test_residuals.clone()
        if self._zero_test_warmup_residuals and self._test_warmup_skip > 0:
            state_test_residuals[: self._test_warmup_skip] = 0.0

        cal_full = self._parallel_cal_residuals_full.to(self._device)
        cal_residuals = self._parallel_cal_residuals.to(self._device)
        if self._location_batch:
            all_residuals = torch.cat([cal_full, state_test_residuals], dim=0)
            state_input = self._scale_parallel_reservoir_input(all_residuals, cal_full)
            states_before = self._encode_parallel_residual_states(state_input)
            cal_start = self._sampler_start_offset
            cal_states = states_before[cal_start: cal_start + cal_residuals.shape[0]]
            test_states = states_before[cal_full.shape[0] : cal_full.shape[0] + n_test]

            lower, upper = self._run_parallel_sampler(
                cal_states=cal_states,
                cal_residuals=cal_residuals,
                test_states=test_states,
                y_hat_test=y_hat_test,
                y_test=y_test,
                alpha=alpha,
            )
        else:
            lower, upper = self._run_sequential_location_sampler(
                cal_full=cal_full,
                cal_residuals=cal_residuals,
                state_test_residuals=state_test_residuals,
                y_hat_test=y_hat_test,
                y_test=y_test,
                alpha=alpha,
            )
        self._parallel_predictions[alpha] = {}
        for node, ts_id in enumerate(self._parallel_ts_ids):
            self._parallel_predictions[alpha][ts_id] = (
                lower[:, node, :].detach().cpu(),
                upper[:, node, :].detach().cpu(),
                y_hat_test[:, node, :].detach().cpu(),
            )

    def _encode_parallel_residual_states(self, residuals: torch.Tensor) -> torch.Tensor:
        encoder = _UpstreamSamplingReservoir(
            n_internal_units=self._hidden_size,
            spectral_radius=self._spectral_radius,
            leak=self._leaking_rate,
            connectivity=self._density,
            input_scaling=self._input_scaling,
            seed=self._seed,
        )
        encoder.to(self._device)
        # Upstream encoder consumes [n_nodes, n_steps, n_features].
        states_after = encoder.get_states(residuals.permute(1, 0, 2).to(self._device), bidir=False)
        states_after = states_after.permute(1, 0, 2)  # [n_steps, n_nodes, hidden]
        zero = states_after.new_zeros(1, states_after.shape[1], states_after.shape[2])
        states_before = torch.cat([zero, states_after[:-1]], dim=0)
        norm = torch.linalg.norm(states_before, dim=-1, keepdim=True)
        return torch.where(norm > 0, states_before / norm.clamp_min(self._eps_floor), states_before)

    def _run_sequential_location_sampler(
        self,
        cal_full: torch.Tensor,
        cal_residuals: torch.Tensor,
        state_test_residuals: torch.Tensor,
        y_hat_test: torch.Tensor,
        y_test: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = int(cal_full.shape[1])
        lower = torch.empty_like(y_hat_test)
        upper = torch.empty_like(y_hat_test)
        global_scale_stats = self._reservoir_input_scale_stats(cal_full)
        for node in range(n_nodes):
            cal_full_node = cal_full[:, node : node + 1, :]
            cal_residuals_node = cal_residuals[:, node : node + 1, :]
            test_residuals_node = state_test_residuals[:, node : node + 1, :]
            all_residuals = torch.cat([cal_full_node, test_residuals_node], dim=0)
            if self._reservoir_scale_axis == "global":
                state_input = self._apply_reservoir_input_scale(all_residuals, global_scale_stats)
            else:
                state_input = self._scale_parallel_reservoir_input(all_residuals, cal_full_node)
            states_before = self._encode_parallel_residual_states(state_input)
            cal_start = self._sampler_start_offset
            cal_states = states_before[cal_start : cal_start + cal_residuals_node.shape[0]]
            test_states = states_before[cal_full_node.shape[0] : cal_full_node.shape[0] + y_test.shape[0]]
            low_node, high_node = self._run_parallel_sampler(
                cal_states=cal_states,
                cal_residuals=cal_residuals_node,
                test_states=test_states,
                y_hat_test=y_hat_test[:, node : node + 1, :],
                y_test=y_test[:, node : node + 1, :],
                alpha=alpha,
            )
            lower[:, node : node + 1, :] = low_node
            upper[:, node : node + 1, :] = high_node
        return lower, upper

    def _reservoir_input_scale_stats(self, cal_residuals: torch.Tensor):
        if not self._scale_reservoir_input:
            return None
        if self._reservoir_scale_axis == "node":
            dims = (0,)
        elif self._reservoir_scale_axis == "global":
            dims = (0, 1)
        else:
            raise ValueError(f"Unknown reservoir_scale_axis={self._reservoir_scale_axis!r}")
        mean = cal_residuals.mean(dim=dims, keepdim=True)
        std = cal_residuals.std(dim=dims, keepdim=True, unbiased=False).clamp_min(self._eps_floor)
        return mean, std

    def _apply_reservoir_input_scale(self, residuals: torch.Tensor, stats) -> torch.Tensor:
        if stats is None:
            return residuals
        mean, std = stats
        return (residuals - mean) / std

    def _scale_parallel_reservoir_input(self, residuals: torch.Tensor, cal_residuals: torch.Tensor) -> torch.Tensor:
        if not self._scale_reservoir_input:
            return residuals
        return self._apply_reservoir_input_scale(residuals, self._reservoir_input_scale_stats(cal_residuals))

    def _run_parallel_sampler(
        self,
        cal_states: torch.Tensor,
        cal_residuals: torch.Tensor,
        test_states: torch.Tensor,
        y_hat_test: torch.Tensor,
        y_test: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_fixed_window = self._sliding_window is not None
        if use_fixed_window and self._sliding_window < cal_residuals.shape[0]:
            cal_states = cal_states[-self._sliding_window :]
            cal_residuals = cal_residuals[-self._sliding_window :]
        sample_size = int(self._sliding_window or cal_residuals.shape[0])
        running_sample_size = sample_size
        proposal_correction = None
        proposal_correction_scale = None
        if self._proposal_correction:
            proposal_correction, proposal_correction_scale = self._estimate_parallel_proposal_correction(
                cal_states=cal_states,
                cal_residuals=cal_residuals,
                alpha=alpha,
                sample_size=sample_size,
            )
        running_alpha = float(alpha)
        target_coverage = 1.0 - float(alpha)
        coverages = torch.zeros(test_states.shape[0], test_states.shape[1], device=self._device)
        lowers = torch.zeros_like(y_hat_test)
        uppers = torch.zeros_like(y_hat_test)

        for i in range(test_states.shape[0]):
            if i < self._test_warmup_skip:
                lowers[i] = y_hat_test[i]
                uppers[i] = y_hat_test[i]
                continue
            test_state = test_states[i : i + 1]
            similarity = self._compute_parallel_weights(cal_states, test_state)
            sampled = self._sample_parallel_residuals(cal_residuals.squeeze(-1), similarity, running_sample_size)
            if self._n_quantiles > 1:
                lower_betas = torch.linspace(1e-3, running_alpha - 1e-3, self._n_quantiles, device=self._device)
                upper_betas = 1 - running_alpha + lower_betas
            else:
                lower_betas = torch.tensor([running_alpha / 2.0], device=self._device)
                upper_betas = torch.tensor([1 - running_alpha / 2.0], device=self._device)
            lower_q = torch.quantile(sampled, lower_betas, dim=0).t()
            upper_q = torch.quantile(sampled, upper_betas, dim=0).t()
            best = torch.argmin(upper_q - lower_q, dim=1)
            node_idx = torch.arange(lower_q.shape[0], device=self._device)
            best_low = lower_q[node_idx, best].reshape(-1, 1)
            best_high = upper_q[node_idx, best].reshape(-1, 1)
            if proposal_correction is None:
                lowers[i] = y_hat_test[i] + best_low
                uppers[i] = y_hat_test[i] + best_high
            elif self._proposal_correction_style == "additive_slack":
                slack_scale = self._proposal_scale_for_step(proposal_correction_scale, best_low)
                slack = proposal_correction * slack_scale
                lowers[i] = y_hat_test[i] + best_low - slack
                uppers[i] = y_hat_test[i] + best_high + slack
            else:
                lower_radius = (-best_low).clamp_min(self._eps_floor)
                upper_radius = best_high.clamp_min(self._eps_floor)
                lowers[i] = y_hat_test[i] - proposal_correction * lower_radius
                uppers[i] = y_hat_test[i] + proposal_correction * upper_radius

            coverages[i] = ((lowers[i] <= y_test[i]) & (y_test[i] <= uppers[i])).squeeze(-1)
            if self._eta != 0.0:
                rolling_coverage = coverages[: i + 1].mean()
                running_alpha = float(torch.clip(
                    torch.tensor(running_alpha + self._eta * (rolling_coverage - target_coverage), device=self._device),
                    1.0e-3,
                    0.3,
                ).item())

            new_residual = (y_test[i] - y_hat_test[i]).unsqueeze(0)
            proposal_correction_scale = self._update_proposal_correction_scale(
                proposal_correction_scale,
                new_residual.squeeze(0).squeeze(-1).abs(),
            )
            if use_fixed_window:
                cal_states = torch.roll(cal_states, -1, dims=0)
                cal_residuals = torch.roll(cal_residuals, -1, dims=0)
                cal_states[-1] = test_state
                cal_residuals[-1] = new_residual
            else:
                cal_states = torch.cat([cal_states, test_state], dim=0)
                cal_residuals = torch.cat([cal_residuals, new_residual], dim=0)
                running_sample_size += 1
        return lowers, uppers

    def _compute_parallel_weights(
        self,
        cal_states: torch.Tensor,
        test_state: torch.Tensor,
        exclude_index: Optional[int] = None,
    ) -> torch.Tensor:
        if self._similarity != "cosine":
            raise NotImplementedError("Parallel ResCP currently mirrors upstream cosine similarity only.")
        scores = torch.einsum("cnh,tnh->cn", cal_states, test_state)
        if exclude_index is not None:
            scores[int(exclude_index)] = torch.finfo(scores.dtype).min
        similarity = F.softmax(scores / self._temperature, dim=0)
        if self._decay == "linear":
            weights = torch.arange(cal_states.shape[0], device=cal_states.device, dtype=similarity.dtype).unsqueeze(1)
            weights = weights / weights.sum().clamp_min(1.0)
            similarity = similarity * weights
            similarity = similarity / similarity.sum(dim=0, keepdim=True).clamp_min(self._eps_floor)
        elif self._decay == "none":
            pass
        else:
            raise NotImplementedError("Parallel ResCP currently mirrors upstream linear/no decay only.")
        if self._topk_residuals is not None and self._topk_residuals > 0 and self._topk_residuals < similarity.shape[0]:
            idx = torch.topk(similarity, k=int(self._topk_residuals), dim=0).indices
            truncated = torch.zeros_like(similarity)
            truncated.scatter_(0, idx, similarity.gather(0, idx))
            similarity = truncated / truncated.sum(dim=0, keepdim=True).clamp_min(self._eps_floor)
        return similarity

    def _estimate_parallel_proposal_correction(
        self,
        cal_states: torch.Tensor,
        cal_residuals: torch.Tensor,
        alpha: float,
        sample_size: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            n_cal = int(cal_states.shape[0])
            if n_cal <= 1:
                return torch.tensor(self._proposal_correction_floor, device=self._device), None
            if self._n_quantiles > 1:
                lower_betas = torch.linspace(1e-3, float(alpha) - 1e-3, self._n_quantiles, device=self._device)
                upper_betas = 1 - float(alpha) + lower_betas
            else:
                lower_betas = torch.tensor([float(alpha) / 2.0], device=self._device)
                upper_betas = torch.tensor([1 - float(alpha) / 2.0], device=self._device)
            scores = []
            residual_matrix = cal_residuals.squeeze(-1)
            min_memory = int(self._topk_residuals or 64)
            min_memory = max(1, min(min_memory, n_cal - 1))
            correction_scale = self._init_proposal_correction_scale(residual_matrix[:min_memory])
            if self._proposal_correction_max_points is not None and self._proposal_correction_max_points < n_cal - min_memory:
                eval_idx = torch.linspace(
                    min_memory,
                    n_cal - 1,
                    int(self._proposal_correction_max_points),
                    device=self._device,
                ).round().long().unique()
            else:
                eval_idx = torch.arange(min_memory, n_cal, device=self._device)
            for idx in eval_idx.tolist():
                memory_states = cal_states[:idx]
                memory_residuals = residual_matrix[:idx]
                test_state = cal_states[idx : idx + 1]
                similarity = self._compute_parallel_weights(memory_states, test_state)
                sampled = self._sample_parallel_residuals(
                    memory_residuals,
                    similarity,
                    min(int(sample_size), int(memory_residuals.shape[0])),
                )
                lower_q = torch.quantile(sampled, lower_betas, dim=0).t()
                upper_q = torch.quantile(sampled, upper_betas, dim=0).t()
                best = torch.argmin(upper_q - lower_q, dim=1)
                node_idx = torch.arange(lower_q.shape[0], device=self._device)
                best_low = lower_q[node_idx, best]
                best_high = upper_q[node_idx, best]
                residual = residual_matrix[int(idx)]
                if self._proposal_correction_style == "additive_slack":
                    raw_score = torch.maximum(best_low - residual, residual - best_high).clamp_min(0.0)
                    scale = self._proposal_scale_for_step(correction_scale, raw_score)
                    score = raw_score / scale
                else:
                    lower_radius = (-best_low).clamp_min(self._eps_floor)
                    upper_radius = best_high.clamp_min(self._eps_floor)
                    score = torch.where(residual >= 0, residual / upper_radius, -residual / lower_radius)
                scores.append(score)
                correction_scale = self._update_proposal_correction_scale(correction_scale, residual.abs())
            if not scores:
                return torch.tensor(self._proposal_correction_floor, device=self._device), correction_scale
            all_scores = torch.cat(scores).nan_to_num(posinf=1.0e6, neginf=0.0)
            sorted_scores = torch.sort(all_scores).values
            rank = int(np.ceil((int(sorted_scores.numel()) + 1) * (1.0 - float(alpha)))) - 1
            rank = min(max(rank, 0), int(sorted_scores.numel()) - 1)
            correction = sorted_scores[rank]
            correction = correction.clamp_min(self._proposal_correction_floor)
            return correction.reshape(1, 1), correction_scale
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def _init_proposal_correction_scale(self, residuals: torch.Tensor) -> Optional[torch.Tensor]:
        if self._proposal_correction_ewma_beta is None:
            return None
        if residuals.numel() == 0:
            return None
        scale = residuals.abs().mean(dim=0)
        return scale.clamp_min(self._proposal_correction_ewma_min_scale)

    def _proposal_scale_for_step(
        self,
        scale: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        if scale is None:
            return torch.ones_like(like).clamp_min(self._proposal_correction_ewma_min_scale)
        return scale.to(device=like.device, dtype=like.dtype).reshape_as(like).clamp_min(
            self._proposal_correction_ewma_min_scale
        )

    def _update_proposal_correction_scale(
        self,
        scale: Optional[torch.Tensor],
        abs_residual: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if scale is None or self._proposal_correction_ewma_beta is None:
            return scale
        beta = float(self._proposal_correction_ewma_beta)
        return (beta * scale + (1.0 - beta) * abs_residual.to(scale.device, scale.dtype)).clamp_min(
            self._proposal_correction_ewma_min_scale
        )

    @staticmethod
    def _sample_parallel_residuals(cal_residuals: torch.Tensor, similarity: torch.Tensor, sample_size: int) -> torch.Tensor:
        idxs = torch.multinomial(similarity.t(), num_samples=sample_size, replacement=True)
        col_idx = torch.arange(cal_residuals.shape[1], dtype=torch.long, device=cal_residuals.device).unsqueeze(1)
        return cal_residuals[idxs, col_idx].transpose(0, 1)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _UpstreamSamplingReservoir:
    """Small central copy of ResCP's ``torch_reservoir_computing.Reservoir``.

    The sampling baseline uses this reservoir for residual-state retrieval, not
    the TSL reservoir layer.
    """

    def __init__(
        self,
        n_internal_units: int,
        spectral_radius: float,
        leak: Optional[float],
        connectivity: float,
        input_scaling: float,
        seed: int,
    ):
        self._n_internal_units = int(n_internal_units)
        self._input_scaling = float(input_scaling)
        self._leak = leak
        self._seed = int(seed)
        self._input_weights = None
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
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

    def get_states(self, x: torch.Tensor, bidir: bool = False, initial_state=None) -> torch.Tensor:
        if bidir:
            raise NotImplementedError("ResCP sampling uses unidirectional reservoir states.")
        n_nodes, _, n_features = x.shape
        if self._input_weights is None:
            self._input_weights = (
                2.0 * np.random.binomial(1, 0.5, [self._n_internal_units, n_features]) - 1.0
            ) * self._input_scaling
            self._input_weights = torch.tensor(self._input_weights, device=x.device, dtype=torch.float32)
        if initial_state is None:
            state = torch.zeros(n_nodes, self._n_internal_units, device=x.device, dtype=torch.float32)
        else:
            state = initial_state.to(device=x.device, dtype=torch.float32)
        states = torch.empty(n_nodes, x.shape[1], self._n_internal_units, device=x.device, dtype=torch.float32)
        for t in range(x.shape[1]):
            current = x[:, t, :]
            pre = self._internal_weights @ state.T + self._input_weights @ current.T
            if self._leak is None:
                state = torch.tanh(pre).T
            else:
                state = (1.0 - float(self._leak)) * state + torch.tanh(pre).T
            states[:, t, :] = state
        return states
