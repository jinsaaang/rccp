from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.pi_base import PICalibArtifacts, PICalibData, PIModel, PIModelPrediction, PIPredictionStepData
from utils.calc_np import calc_residuals


_GAMMA_GRIDS = {
    "paper": [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128],
    "default": [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128],
    "agaci_figure_fast": [0.032, 0.064, 0.128, 0.256],
    "agaci_figure_extended": [0.008, 0.016, 0.032, 0.064, 0.128, 0.256],
    "wide": [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256],
}


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_gammas(gammas: Optional[Iterable[float]], gamma_grid_name: str) -> np.ndarray:
    if gammas is None:
        try:
            gammas = _GAMMA_GRIDS[str(gamma_grid_name)]
        except KeyError as exc:
            raise ValueError(f"Unknown gamma_grid_name={gamma_grid_name!r}") from exc
    arr = np.asarray(list(gammas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("gammas must be a non-empty 1D list")
    if np.any(arr <= 0):
        raise ValueError("all gamma values must be positive")
    return arr


def _pinball(u: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * u - np.minimum(u, 0.0)


def _clip_alpha(value, min_alpha: float, max_alpha: float):
    return np.clip(value, min_alpha, max_alpha)


def _conformal_width(scores: np.ndarray, alpha: float, finite_sample: bool) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("cannot compute a conformal width from an empty score memory")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if finite_sample:
        rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
        rank = min(max(rank, 1), scores.size)
        return float(np.sort(scores)[rank - 1])
    return float(np.quantile(scores, 1.0 - alpha))


def _tail_beta(scores: np.ndarray, score: float) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= float(score)))


class _AgACIController:
    def __init__(
        self,
        alpha: float,
        gammas: np.ndarray,
        alpha_init: Optional[float],
        eps: float,
        min_alpha: float,
        max_alpha: float,
    ):
        self.alpha = float(alpha)
        self.gammas = gammas.astype(np.float64)
        self.eps = float(eps)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        init = self.alpha if alpha_init is None else float(alpha_init)
        self.expert_alphas = np.full(self.gammas.shape[0], init, dtype=np.float64)
        self.expert_probs = np.full(self.gammas.shape[0], 1.0 / self.gammas.shape[0], dtype=np.float64)
        self.expert_sq_losses = np.zeros(self.gammas.shape[0], dtype=np.float64)
        self.expert_etas = np.zeros(self.gammas.shape[0], dtype=np.float64)
        self.expert_l_values = np.zeros(self.gammas.shape[0], dtype=np.float64)
        self.expert_max_losses = np.zeros(self.gammas.shape[0], dtype=np.float64)

    def current_alpha(self) -> float:
        return float(_clip_alpha(np.sum(self.expert_probs * self.expert_alphas), self.min_alpha, self.max_alpha))

    def update(self, beta: float) -> None:
        alpha_seq = self.current_alpha()
        err = float(alpha_seq > beta)
        expert_losses = (err - self.alpha) * (self.expert_alphas - alpha_seq)
        self.expert_sq_losses += expert_losses**2
        self.expert_max_losses = np.maximum(self.expert_max_losses, np.abs(expert_losses))
        expert_e_vals = 2 ** (np.ceil(np.log2(np.abs(self.expert_max_losses) + self.eps)) + 1)
        self.expert_l_values += 0.5 * (
            expert_losses * (1.0 + self.expert_etas * expert_losses)
            + expert_e_vals * (self.expert_etas * expert_losses > 0.5)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_terms = np.sqrt(np.log(len(self.gammas)) / self.expert_sq_losses)
        self.expert_etas = np.minimum(1.0 / expert_e_vals, sqrt_terms)
        self.expert_etas = np.nan_to_num(self.expert_etas, nan=1.0 / expert_e_vals, posinf=1.0 / expert_e_vals)
        self.expert_alphas = _clip_alpha(
            self.expert_alphas + self.gammas * (self.alpha - (self.expert_alphas > beta).astype(np.float64)),
            self.min_alpha,
            self.max_alpha,
        )
        logits = -self.expert_etas * self.expert_l_values
        logits = logits - np.max(logits)
        weights = self.expert_etas * np.exp(logits)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            self.expert_probs = np.full(len(self.gammas), 1.0 / len(self.gammas), dtype=np.float64)
        else:
            self.expert_probs = weights / total


class _DtACIController:
    def __init__(
        self,
        alpha: float,
        gammas: np.ndarray,
        alpha_init: Optional[float],
        sigma: float,
        eta: float,
        eta_adapt: bool,
        eta_lookback: int,
        selection_mode: str,
        min_alpha: float,
        max_alpha: float,
        random_state: int,
    ):
        self.alpha = float(alpha)
        self.gammas = gammas.astype(np.float64)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.eta_adapt = bool(eta_adapt)
        self.eta_lookback = int(eta_lookback)
        self.selection_mode = str(selection_mode)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        init = self.alpha if alpha_init is None else float(alpha_init)
        self.expert_alphas = np.full(self.gammas.shape[0], init, dtype=np.float64)
        self.expert_probs = np.full(self.gammas.shape[0], 1.0 / self.gammas.shape[0], dtype=np.float64)
        self.loss_seq: List[float] = []
        self.rng = np.random.default_rng(int(random_state))
        if self.selection_mode not in {"mean", "sample"}:
            raise ValueError("selection_mode must be either 'mean' or 'sample'")

    def current_alpha(self) -> float:
        if self.selection_mode == "sample":
            idx = int(self.rng.choice(len(self.gammas), p=self.expert_probs))
            value = self.expert_alphas[idx]
        else:
            value = float(np.sum(self.expert_probs * self.expert_alphas))
        return float(_clip_alpha(value, self.min_alpha, self.max_alpha))

    def update(self, beta: float) -> None:
        eta = self.eta
        if self.eta_adapt and len(self.loss_seq) >= self.eta_lookback > 0:
            window = np.asarray(self.loss_seq[-self.eta_lookback :], dtype=np.float64)
            denom = float(np.sum(window**2))
            if denom > 0:
                eta = float(np.sqrt((np.log(2 * len(self.gammas) * self.eta_lookback) + 1.0) / denom))
        expert_losses = _pinball(float(beta) - self.expert_alphas, self.alpha)
        self.loss_seq.append(float(np.sum(expert_losses * self.expert_probs)))
        self.expert_alphas = _clip_alpha(
            self.expert_alphas + self.gammas * (self.alpha - (self.expert_alphas > beta).astype(np.float64)),
            self.min_alpha,
            self.max_alpha,
        )
        if np.isfinite(eta):
            logits = -float(eta) * expert_losses
            logits = logits - np.max(logits)
            bar_weights = self.expert_probs * np.exp(logits)
            total = float(np.sum(bar_weights))
            if not np.isfinite(total) or total <= 0:
                self.expert_probs = np.full(len(self.gammas), 1.0 / len(self.gammas), dtype=np.float64)
            else:
                next_probs = (1.0 - self.sigma) * bar_weights / total + self.sigma / len(self.gammas)
                self.expert_probs = next_probs / np.sum(next_probs)
        else:
            best = int(np.argmin(expert_losses))
            self.expert_probs = np.zeros(len(self.gammas), dtype=np.float64)
            self.expert_probs[best] = 1.0


class _AdaptiveQuantileCP(PIModel):
    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._gamma_grid_name = kwargs.get("gamma_grid_name", "paper")
        self._gammas = _resolve_gammas(kwargs.get("gammas"), self._gamma_grid_name)
        self._alpha_init = kwargs.get("alpha_init", None)
        self._finite_sample = bool(kwargs.get("finite_sample", True))
        self._online_memory = bool(kwargs.get("online_memory", True))
        self._max_memory = kwargs.get("max_memory", None)
        self._min_alpha = float(kwargs.get("min_alpha", 1.0e-4))
        self._max_alpha = float(kwargs.get("max_alpha", 0.999))
        self._scores: Optional[List[float]] = None
        self._controller = None
        self._last_scores_for_beta: Optional[np.ndarray] = None

    def _new_controller(self, alpha: float):
        raise NotImplementedError

    def _calibrate(self, calib_data: List[PICalibData], alphas, **kwargs):
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
        y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=calib_data.ts_id,
                X_past=calib_data.X_pre_calib,
                Y_past=calib_data.Y_pre_calib,
                X_step=calib_data.X_calib,
                step_offset=calib_data.step_offset,
            ),
            retrieve_tensor=False,
        ).point
        scores = np.abs(calc_residuals(y_hat=y_hat, y=calib_data.Y_calib.numpy())).reshape(-1)
        self._scores = [float(v) for v in scores]
        self._controller = self._new_controller(float(alpha))
        return PICalibArtifacts(fc_Y_hat=y_hat, eps=scores.reshape(-1, 1))

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._controller = self._new_controller(float(kwargs["alpha"]))
        self._last_scores_for_beta = None

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if self._scores is None or self._controller is None:
            raise ValueError(f"{self.__class__.__name__} is not calibrated")
        y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=pred_data.X_past,
                Y_past=pred_data.Y_past,
                X_step=pred_data.X_step,
                step_offset=pred_data.step_offset_overall,
            )
        ).point
        scores = np.asarray(self._scores, dtype=np.float64)
        alpha_t = self._controller.current_alpha()
        width = _conformal_width(scores, alpha_t, finite_sample=self._finite_sample)
        self._last_scores_for_beta = scores
        return PIModelPrediction(pred_interval=(y_hat - width, y_hat + width), fc_Y_hat=y_hat)

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        del pred_data, kwargs
        if self._scores is None or self._controller is None:
            return
        y = _as_numpy(Y_step).reshape(-1)
        y_hat = _as_numpy(pred_result.fc_Y_hat).reshape(-1)
        if y.size == 0 or y_hat.size == 0:
            return
        score = float(abs(y[0] - y_hat[0]))
        beta = _tail_beta(
            self._last_scores_for_beta if self._last_scores_for_beta is not None else np.asarray(self._scores),
            score,
        )
        self._controller.update(beta)
        if self._online_memory:
            self._scores.append(score)
            if self._max_memory is not None:
                max_memory = int(self._max_memory)
                if max_memory > 0 and len(self._scores) > max_memory:
                    self._scores = self._scores[-max_memory:]
        self._last_scores_for_beta = None

    def model_ready(self):
        return self._scores is not None and len(self._scores) > 0

    @property
    def can_handle_different_alpha(self):
        return True


class AgACIModel(_AdaptiveQuantileCP):
    """Aggregated Adaptive Conformal Inference with a point-forecast quantile constructor."""

    def __init__(self, **kwargs):
        self._eps = float(kwargs.get("eps", 0.001))
        super().__init__(**kwargs)

    def _new_controller(self, alpha: float):
        return _AgACIController(
            alpha=alpha,
            gammas=self._gammas,
            alpha_init=self._alpha_init,
            eps=self._eps,
            min_alpha=self._min_alpha,
            max_alpha=self._max_alpha,
        )


class DtACIModel(_AdaptiveQuantileCP):
    """Dynamically tuned Adaptive Conformal Inference with a point-forecast quantile constructor."""

    def __init__(self, **kwargs):
        self._sigma = float(kwargs.get("sigma", 1.0 / 1000.0))
        self._eta = float(kwargs.get("eta", 2.72))
        self._eta_adapt = bool(kwargs.get("eta_adapt", False))
        self._eta_lookback = int(kwargs.get("eta_lookback", 500))
        self._selection_mode = str(kwargs.get("selection_mode", "mean"))
        self._random_state = int(kwargs.get("random_state", 0))
        super().__init__(**kwargs)

    def _new_controller(self, alpha: float):
        return _DtACIController(
            alpha=alpha,
            gammas=self._gammas,
            alpha_init=self._alpha_init,
            sigma=self._sigma,
            eta=self._eta,
            eta_adapt=self._eta_adapt,
            eta_lookback=self._eta_lookback,
            selection_mode=self._selection_mode,
            min_alpha=self._min_alpha,
            max_alpha=self._max_alpha,
            random_state=self._random_state,
        )
