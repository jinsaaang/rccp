from __future__ import annotations

import copy
import json
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.pi_base import (
    PICalibArtifacts,
    PICalibData,
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
)

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except ImportError:
    BoundedModule = None
    BoundedTensor = None
    PerturbationLpNorm = None


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _safe_1d(values) -> np.ndarray:
    return _as_numpy(values).astype(np.float32).reshape(-1)


def _upstream_feature_radius(values: np.ndarray, alpha: float) -> float:
    """Feature-space radius from upstream FeatErrorErrFunc.apply_inverse."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("CT-SSF calibration scores must be non-empty")
    sorted_desc = np.sort(values)[::-1]
    border = int(np.floor(float(alpha) * (sorted_desc.size + 1))) - 1
    border = min(max(border, 0), sorted_desc.size - 1)
    return float(sorted_desc[border])


def _feature_norm(diff: torch.Tensor, norm_name: str) -> torch.Tensor:
    if norm_name in {"inf", "infinity", "linf", "l_inf"}:
        return diff.abs().amax(dim=1)
    if norm_name in {"2", "l2"}:
        return torch.linalg.vector_norm(diff, ord=2, dim=1)
    raise ValueError(f"Unsupported CT-SSF feature_norm: {norm_name}")


class _CTSSFHead(nn.Module):
    """Upstream-style CT-SSF g head over a precomputed semantic feature z."""

    def __init__(self, in_shape: int, out_shape: int = 1, hidden_size: int = 64):
        super().__init__()
        self.out_shape = int(out_shape)
        self.g = nn.Sequential(
            nn.Linear(int(in_shape), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.out_shape),
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.g(z)


class _GlobalStateCTSSFCore:
    """Upstream CT-SSF g/inv_g/LiRPA logic over precomputed global states."""

    def __init__(
        self,
        *,
        feature_norm: str,
        inv_lr: float,
        inv_step: int,
        inv_optimizer: str,
        inverse_batch_size: int,
        certification_method: str,
        eps: float,
        head_hidden_size,
        head_epochs: int,
        head_batch_size: int,
        head_lr: float,
        head_weight_decay: float,
        head_val_ratio: float,
        head_cv_random_state: int,
        head_seed: Optional[int],
    ):
        self.feature_norm = str(feature_norm).lower()
        self.inv_lr = float(inv_lr)
        self.inv_step = int(inv_step)
        self.inv_optimizer = str(inv_optimizer).lower()
        self.inverse_batch_size = int(inverse_batch_size)
        self.certification_method = str(certification_method).lower()
        self.eps = float(eps)
        self.head_hidden_size = head_hidden_size
        self.head_epochs = int(head_epochs)
        self.head_batch_size = int(head_batch_size)
        self.head_lr = float(head_lr)
        self.head_weight_decay = float(head_weight_decay)
        self.head_val_ratio = float(head_val_ratio)
        self.head_cv_random_state = int(head_cv_random_state)
        self.head_seed = None if head_seed is None else int(head_seed)
        self.head: Optional[_CTSSFHead] = None
        self.device = torch.device("cpu")

    def fit_g(self, train_states: np.ndarray, train_y: np.ndarray, device: torch.device) -> dict:
        train_states = np.asarray(train_states, dtype=np.float32).reshape(train_states.shape[0], -1)
        train_y = np.asarray(train_y, dtype=np.float32).reshape(-1, 1)
        if train_states.shape[0] != train_y.shape[0]:
            raise ValueError("CT-SSF head training length mismatch between states and labels")
        if not np.isfinite(train_states).all() or not np.isfinite(train_y).all():
            raise ValueError("CT-SSF head training data contains non-finite values")
        if train_states.shape[0] < 2:
            raise ValueError("CT-SSF head training requires at least two samples")

        state_dim = int(train_states.shape[1])
        hidden_size = state_dim if self.head_hidden_size is None else int(self.head_hidden_size)
        if self.head_seed is None:
            head = _CTSSFHead(in_shape=state_dim, out_shape=1, hidden_size=hidden_size).to(device)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.head_seed)
                head = _CTSSFHead(in_shape=state_dim, out_shape=1, hidden_size=hidden_size).to(device)

        x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
            train_states,
            train_y,
            test_size=self.head_val_ratio,
            random_state=self.head_cv_random_state,
        )
        x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
        x_val = torch.as_tensor(x_val_np, dtype=torch.float32, device=device)
        y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=device)
        x_all = torch.as_tensor(train_states, dtype=torch.float32, device=device)
        y_all = torch.as_tensor(train_y, dtype=torch.float32, device=device)

        candidate = copy.deepcopy(head).to(device)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=self.head_lr, weight_decay=self.head_weight_decay)
        best_epoch = self.head_epochs
        best_cnt = int(1e10)
        best_val = float("inf")
        cnt = 0
        for epoch in range(self.head_epochs):
            _, cnt = self._train_epoch(candidate, x_train, y_train, optimizer, cnt=cnt)
            candidate.eval()
            with torch.no_grad():
                val_loss = float(F.mse_loss(candidate(x_val).reshape_as(y_val), y_val).detach().cpu())
            if val_loss <= best_val:
                best_val = val_loss
                best_epoch = epoch
                best_cnt = cnt

        optimizer = torch.optim.Adam(head.parameters(), lr=self.head_lr, weight_decay=self.head_weight_decay)
        cnt = 0
        for _ in range(best_epoch + 1):
            if cnt > best_cnt:
                break
            _, cnt = self._train_epoch(head, x_all, y_all, optimizer, cnt=cnt, best_cnt=best_cnt)

        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self.head = head
        self.device = device
        with torch.no_grad():
            train_pred = head(x_all).reshape_as(y_all)
            train_mse = float(F.mse_loss(train_pred, y_all).detach().cpu())
        return {
            "head_source": "global_state_upstream_mlp_g",
            "head_hidden_size": int(hidden_size),
            "head_epochs": int(self.head_epochs),
            "head_batch_size": int(self.head_batch_size),
            "head_lr": float(self.head_lr),
            "head_weight_decay": float(self.head_weight_decay),
            "head_val_ratio": float(self.head_val_ratio),
            "head_cv_random_state": int(self.head_cv_random_state),
            "head_seed": None if self.head_seed is None else int(self.head_seed),
            "head_best_epoch": int(best_epoch),
            "head_best_batches": int(best_cnt),
            "head_best_val_mse": float(best_val),
            "head_train_mse": float(train_mse),
        }

    def _train_epoch(
        self,
        model: nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer,
        *,
        cnt: int = 0,
        best_cnt: int = int(1e10),
    ) -> tuple[float, int]:
        model.train()
        shuffle_idx = np.arange(x_train.shape[0])
        np.random.shuffle(shuffle_idx)
        order = torch.as_tensor(shuffle_idx, dtype=torch.long, device=x_train.device)
        x_train = x_train[order]
        y_train = y_train[order]
        losses = []
        for idx in range(0, x_train.shape[0], self.head_batch_size):
            cnt += 1
            optimizer.zero_grad()
            batch_x = x_train[idx : min(idx + self.head_batch_size, x_train.shape[0])]
            batch_y = y_train[idx : min(idx + self.head_batch_size, y_train.shape[0])]
            pred = model(batch_x)
            loss = F.mse_loss(pred.reshape_as(batch_y), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if cnt >= best_cnt:
                break
        return float(np.mean(losses)) if losses else float("nan"), cnt

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise ValueError("CT-SSF head g is not initialized")
        out = self.head(z)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        elif out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        return out[:, :1]

    def inverse_features(self, z_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_true = z_pred.detach().clone().requires_grad_(True)
        if self.inv_optimizer == "adam":
            optimizer = torch.optim.Adam([z_true], lr=self.inv_lr)
        elif self.inv_optimizer == "sgd":
            optimizer = torch.optim.SGD([z_true], lr=self.inv_lr)
        else:
            raise ValueError(f"Unsupported CT-SSF inv_optimizer: {self.inv_optimizer}")
        for _ in range(self.inv_step):
            pred = self.decoder_forward(z_true)
            loss = F.mse_loss(pred.reshape_as(y), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return z_true.detach()

    def score_features(self, states: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.head is None:
            raise ValueError("CT-SSF head g is not initialized")
        z_all = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        y_all = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
        scores = []
        for start in range(0, z_all.shape[0], self.inverse_batch_size):
            end = min(start + self.inverse_batch_size, z_all.shape[0])
            z_pred = z_all[start:end]
            z_true = self.inverse_features(z_pred, y_all[start:end])
            scores.append(_feature_norm(z_pred - z_true, self.feature_norm).detach().cpu())
        return torch.cat(scores, dim=0).numpy().astype(np.float32)

    def bound_feature_ball(self, z: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
        if BoundedModule is None or BoundedTensor is None or PerturbationLpNorm is None:
            raise ImportError("auto_LiRPA is required for CT-SSF certification")
        if self.head is None:
            raise ValueError("CT-SSF head g is not initialized")
        method_map = {
            "0": "IBP",
            "ibp": "IBP",
            "1": "IBP+backward",
            "ibp+backward": "IBP+backward",
            "2": "backward",
            "backward": "backward",
            "3": "CROWN-Optimized",
            "crown-optimized": "CROWN-Optimized",
        }
        method = method_map.get(self.certification_method, self.certification_method)
        lirpa_model = BoundedModule(self.head.g, torch.empty_like(z), device=z.device)
        norm = np.inf if self.feature_norm in {"inf", "infinity", "linf", "l_inf"} else 2
        bounded_input = BoundedTensor(z, PerturbationLpNorm(norm=norm, eps=float(radius)))
        if "Optimized" in method:
            lirpa_model.set_bound_opts(
                {"optimize_bound_args": {"ob_iteration": 20, "ob_lr": 0.1, "ob_verbose": 0}}
            )
        lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
        return lb.reshape(z.shape[0], -1)[:, :1], ub.reshape(z.shape[0], -1)[:, :1]


class CTSSFHiddenStateModel(PIModel):
    """
    CT-SSF with the shared central forecaster as encoder.

    This ports the core CT-SSF procedure into the central benchmark protocol:
    the forecaster hidden state is the semantic feature z, a separate
    upstream-style MLP is trained as g(z), calibration scores are
    ||z_pred - inv_g(y)||, and prediction intervals are auto-LiRPA certified
    output bounds for a feature-space ball around z.
    """

    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._feature_norm = str(kwargs.get("feature_norm", "inf")).lower()
        self._eps = float(kwargs.get("eps", 1e-6))
        self._inv_lr = float(kwargs.get("inv_lr", kwargs.get("feat_lr", 1e-3)))
        self._inv_step = int(kwargs.get("inv_step", kwargs.get("feat_step", 80)))
        self._inv_optimizer = str(kwargs.get("inv_optimizer", "sgd")).lower()
        self._inverse_batch_size = int(kwargs.get("inverse_batch_size", 512))
        self._certification_method = str(kwargs.get("certification_method", "ibp")).lower()
        self._output_slack = bool(kwargs.get("output_slack", True))
        self._head_hidden_size = kwargs.get("head_hidden_size", 64)
        self._head_epochs = int(kwargs.get("head_epochs", kwargs.get("epochs", 10)))
        self._head_batch_size = int(kwargs.get("head_batch_size", kwargs.get("batch_size", 64)))
        self._head_lr = float(kwargs.get("head_lr", kwargs.get("lr", 5e-4)))
        self._head_weight_decay = float(kwargs.get("head_weight_decay", kwargs.get("wd", 1e-6)))
        self._head_val_ratio = float(kwargs.get("head_val_ratio", 0.05))
        self._head_cv_random_state = int(kwargs.get("head_cv_random_state", 1))
        head_seed = kwargs.get("head_seed", None)
        self._head_seed = None if head_seed is None else int(head_seed)
        self._debug = bool(kwargs.get("debug", False))
        self._debug_pred_printed = False
        self._active_alpha: Optional[float] = None
        self._active_q: Optional[float] = None
        self._active_output_slack: float = 0.0
        self._head: Optional[nn.Module] = None
        self._calib_scores: Optional[np.ndarray] = None
        self._device = torch.device("cpu")
        self._head_fit_time_sec_total: float = 0.0
        self._core = _GlobalStateCTSSFCore(
            feature_norm=self._feature_norm,
            inv_lr=self._inv_lr,
            inv_step=self._inv_step,
            inv_optimizer=self._inv_optimizer,
            inverse_batch_size=self._inverse_batch_size,
            certification_method=self._certification_method,
            eps=self._eps,
            head_hidden_size=self._head_hidden_size,
            head_epochs=self._head_epochs,
            head_batch_size=self._head_batch_size,
            head_lr=self._head_lr,
            head_weight_decay=self._head_weight_decay,
            head_val_ratio=self._head_val_ratio,
            head_cv_random_state=self._head_cv_random_state,
            head_seed=self._head_seed,
        )

    def _debug_record(self, payload: dict) -> None:
        if not self._debug:
            return
        with open("ct_ssf_debug.jsonl", "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, sort_keys=True) + "\n")

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        return None

    def _get_shared_forecaster(self):
        fc_models = getattr(self._forcast_service, "_fc_models", None)
        if not fc_models:
            return None
        return fc_models[0]

    def _get_prediction_offset(self) -> int:
        fc_model = self._get_shared_forecaster()
        offset = getattr(fc_model, "_prediction_offset", None)
        if offset is None:
            offset = getattr(fc_model, "_seq_len", 0)
        return int(offset or 0)

    def _forecast_states(self, ts_id: str, x: torch.Tensor, y: torch.Tensor, start_offset: int) -> torch.Tensor:
        if x is None or y is None:
            raise ValueError("CT-SSF head training requires pre-calibration X/Y data")
        if x.shape[0] <= start_offset:
            raise ValueError(
                f"CT-SSF needs more pre-calibration points than the forecast offset: "
                f"len={x.shape[0]}, offset={start_offset}"
            )
        result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=ts_id,
                X_past=x[:start_offset],
                Y_past=y[:start_offset],
                X_step=x[start_offset:],
                step_offset=start_offset,
            )
        )
        if result.state is None:
            raise ValueError("CT-SSF requires global forecaster hidden states")
        return torch.as_tensor(result.state, dtype=torch.float32)

    def _prepare_head(self, train_states: np.ndarray, train_y: np.ndarray, device: torch.device) -> dict:
        started = time.perf_counter()
        head_info = self._core.fit_g(train_states, train_y, device=device)
        head_fit_time_sec = time.perf_counter() - started
        self._head_fit_time_sec_total += head_fit_time_sec
        self._head = self._core.head
        self._device = self._core.device
        head_info = {
            **head_info,
            "head_fit_time_sec": float(head_fit_time_sec),
            "head_fit_time_sec_total": float(self._head_fit_time_sec_total),
        }
        return head_info

    def _decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._core.decoder_forward(z)

    def _inverse_features(self, z_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._core.inverse_features(z_pred, y)

    def _score_features(self, states: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._core.score_features(states, y)

    def _bound_with_lirpa(self, z: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self._core.bound_feature_ball(z, radius)

    def _bound_feature_ball(self, z: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bound_with_lirpa(z, radius)

    def _bound_feature_ball_batched(self, states: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32).reshape(states.shape[0], -1)
        lowers = []
        uppers = []
        for start in range(0, states.shape[0], self._inverse_batch_size):
            end = min(start + self._inverse_batch_size, states.shape[0])
            z = torch.as_tensor(states[start:end], dtype=torch.float32, device=self._device)
            lower, upper = self._bound_feature_ball(z, radius)
            lowers.append(lower.detach().cpu().numpy().reshape(-1))
            uppers.append(upper.detach().cpu().numpy().reshape(-1))
        return np.concatenate(lowers, axis=0), np.concatenate(uppers, axis=0)

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data,
        mix_calib_artifact,
    ) -> PICalibArtifacts:
        del calib_artifact, mix_calib_data, mix_calib_artifact

        fc_result = self._forcast_service.predict(
            FCPredictionData(
                ts_id=calib_data.ts_id,
                X_past=calib_data.X_pre_calib,
                Y_past=calib_data.Y_pre_calib,
                X_step=calib_data.X_calib,
                step_offset=calib_data.step_offset,
            )
        )
        if fc_result.state is None:
            raise ValueError("CT-SSF hidden-state adapter requires a forecaster state")

        y_cal = _safe_1d(calib_data.Y_calib)
        states = _as_numpy(fc_result.state).astype(np.float32)
        states = states.reshape(states.shape[0], -1)
        if y_cal.shape[0] != states.shape[0]:
            raise ValueError("CT-SSF calibration length mismatch among y and hidden states")

        device = fc_result.point.device if isinstance(fc_result.point, torch.Tensor) else torch.device("cpu")
        offset = self._get_prediction_offset()
        train_states = self._forecast_states(calib_data.ts_id, calib_data.X_pre_calib, calib_data.Y_pre_calib, offset)
        train_y = _safe_1d(calib_data.Y_pre_calib[offset:])
        head_info = self._prepare_head(_as_numpy(train_states), train_y, device=device)
        self._calib_scores = self._score_features(states, y_cal)
        self.pre_predict(alpha=alpha)

        with torch.no_grad():
            z_cal = torch.as_tensor(states, dtype=torch.float32, device=self._device)
            y_hat_head = self._decoder_forward(z_cal).to(device=fc_result.point.device)
        y_hat = _safe_1d(y_hat_head)
        y_hat_shared = _safe_1d(fc_result.point)
        abs_resid = np.abs(y_cal - y_hat).astype(np.float32)
        if not np.isfinite(self._calib_scores).all():
            raise ValueError("CT-SSF calibration feature scores contain non-finite values")
        slack_scores = np.zeros_like(abs_resid, dtype=np.float32)
        if self._output_slack:
            cal_lower, cal_upper = self._bound_feature_ball_batched(states, float(self._active_q) + self._eps)
            slack_scores = np.maximum.reduce([
                cal_lower - y_cal,
                y_cal - cal_upper,
                np.zeros_like(y_cal, dtype=np.float32),
            ]).astype(np.float32)
            self._active_output_slack = _upstream_feature_radius(slack_scores, float(alpha))
        if self._debug:
            shared_mse = float(np.mean((y_cal - y_hat_shared) ** 2))
            head_mse = float(np.mean((y_cal - y_hat) ** 2))
            self._debug_record({
                "stage": "calibrate",
                "ts_id": str(calib_data.ts_id),
                "alpha": float(alpha),
                "inv_lr": float(self._inv_lr),
                "inv_step": int(self._inv_step),
                "head_train_mse": float(head_info.get("head_train_mse")),
                "head_cal_mse": head_mse,
                "shared_cal_mse": shared_mse,
                "q": float(self._active_q),
                "score_min": float(np.min(self._calib_scores)),
                "score_mean": float(np.mean(self._calib_scores)),
                "score_max": float(np.max(self._calib_scores)),
                "output_slack": float(self._active_output_slack),
                "slack_score_mean": float(np.mean(slack_scores)),
                "slack_score_max": float(np.max(slack_scores)),
                "head_yhat_mean": float(np.mean(y_hat)),
                "shared_yhat_mean": float(np.mean(y_hat_shared)),
                "y_mean": float(np.mean(y_cal)),
            })
        artifact = PICalibArtifacts(fc_Y_hat=y_hat_head, eps=abs_resid.reshape(-1, 1), fc_state_step=fc_result.state)
        artifact.add_info = {
            "method": "ct_ssf_global_encoder_upstream_g",
            "feature_norm": self._feature_norm,
            "state_dim": int(states.shape[1]),
            "inv_lr": float(self._inv_lr),
            "inv_step": int(self._inv_step),
            "certification_method": self._certification_method,
            "active_alpha": float(alpha),
            "active_q_feature_radius": float(self._active_q),
            "active_output_slack": float(self._active_output_slack),
            "center_forecaster": "shared",
            "calib_score_min": float(np.min(self._calib_scores)),
            "calib_score_max": float(np.max(self._calib_scores)),
            "calib_head_mse": float(np.mean((y_cal - y_hat) ** 2)),
            **head_info,
        }
        return artifact

    def pre_predict(self, **kwargs):
        alpha = kwargs.get("alpha", None)
        if alpha is None or self._calib_scores is None:
            return
        self._active_alpha = float(alpha)
        self._active_q = _upstream_feature_radius(self._calib_scores, self._active_alpha)

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        del kwargs
        if self._active_q is None:
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
        if fc_result.state is None:
            raise ValueError("CT-SSF requires forecaster states at prediction time")
        states = _as_numpy(fc_result.state).astype(np.float32).reshape(fc_result.state.shape[0], -1)
        z = torch.as_tensor(states, dtype=torch.float32, device=self._device)
        lower, upper = self._bound_feature_ball(z, float(self._active_q) + self._eps)
        lower = lower.to(device=fc_result.point.device).reshape_as(fc_result.point)
        upper = upper.to(device=fc_result.point.device).reshape_as(fc_result.point)
        if self._output_slack:
            lower = lower - float(self._active_output_slack)
            upper = upper + float(self._active_output_slack)
        with torch.no_grad():
            y_hat_head = self._decoder_forward(z).to(device=fc_result.point.device).reshape_as(fc_result.point)
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not torch.isfinite(y_hat_head).all():
            raise ValueError("CT-SSF produced non-finite prediction values")
        if self._debug and not self._debug_pred_printed:
            width = _safe_1d(upper - lower)
            y_hat = _safe_1d(y_hat_head)
            shared_yhat = _safe_1d(fc_result.point)
            self._debug_record({
                "stage": "predict",
                "ts_id": str(pred_data.ts_id),
                "alpha": float(pred_data.alpha),
                "q": float(self._active_q),
                "output_slack": float(self._active_output_slack),
                "width_mean": float(np.mean(width)),
                "width_min": float(np.min(width)),
                "width_max": float(np.max(width)),
                "head_yhat_mean": float(np.mean(y_hat)),
                "shared_yhat_mean": float(np.mean(shared_yhat)),
            })
            self._debug_pred_printed = True
        return PIModelPrediction(pred_interval=(lower, upper), fc_Y_hat=y_hat_head)

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.X_step is not None

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return self._calib_scores is not None and self._head is not None

    def method_runtime_info(self) -> dict:
        return {
            "ct_ssf_g_fit_time_sec": float(self._head_fit_time_sec_total),
        }
