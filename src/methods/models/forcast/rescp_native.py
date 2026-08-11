"""ResCP-style node-batched forecasters for central experiments.

These forecasters keep the central PIModel/ForcastService interface, but train
one model over a stacked ``[time, nodes, features]`` view of all locations in a
dataset. This mirrors the ResCP upstream backbone setting more closely than the
HopCPT-derived global many-to-one forecaster, which treats each location as an
independent sample stream.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tsl.nn import maybe_cat_exog
from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import RNN

from loader.dataset import ChronoSplittedTsDataset
from models.forcast.forcast_base import (
    FCModelPrediction,
    FCPredictionData,
    FcSingleModelPrediction,
    ForcastMode,
    ForcastModel,
    PredictionOutputType,
)
from utils.utils import get_device

LOGGER = logging.getLogger(__name__)


class _NodeWindowDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        target_indices: torch.Tensor,
        window: int,
        delay: int,
        input_mode: str,
    ):
        self._features = features
        self._target = target
        self._mask = mask
        self._target_indices = target_indices.long()
        self._window = int(window)
        self._delay = int(delay)
        self._input_mode = str(input_mode)

    def __len__(self):
        return int(self._target_indices.numel())

    def __getitem__(self, index):
        target_idx = int(self._target_indices[index])
        input_end = target_idx - self._delay
        input_start = input_end - self._window
        target_history = self._target[input_start:input_end]
        exog_history = self._features[input_start:input_end]
        if self._input_mode == "upstream_x_u":
            item = {"x": target_history.float(), "u": exog_history.float()}
        else:
            item = {"x": torch.cat([target_history, exog_history], dim=-1).float()}
        item["y"] = self._target[target_idx].float()
        if self._mask is not None:
            item["mask"] = self._mask[target_idx].float()
        return {
            **item,
        }


class _UpstreamBackboneAdapter(nn.Module):
    """Central copy/wrapper of the exact backbone classes used by ResCP."""

    def __init__(self, model_name: str, input_size: int, exog_size: int, model_kwargs: dict):
        super().__init__()
        self.hidden_size = int(model_kwargs.get("hidden_size", 32))
        model_name = str(model_name)
        if model_name == "rnn":
            self.model = _CopiedResCPRNNModel(
                input_size=input_size,
                output_size=1,
                horizon=1,
                exog_size=exog_size,
                hidden_size=self.hidden_size,
                n_layers=int(model_kwargs.get("n_layers", model_kwargs.get("num_layers", 1))),
                dropout=float(model_kwargs.get("dropout", 0.0)),
                cell_type=str(model_kwargs.get("cell_type", "gru")),
            )
        elif model_name == "transformer":
            from tsl.nn.models import TransformerModel

            kwargs = dict(
                input_size=input_size,
                exog_size=exog_size,
                output_size=1,
                horizon=1,
                hidden_size=self.hidden_size,
                ff_size=int(model_kwargs.get("ff_size", self.hidden_size * 2)),
                n_heads=int(model_kwargs.get("n_heads", 2)),
                n_layers=int(model_kwargs.get("n_layers", 3)),
                dropout=float(model_kwargs.get("dropout", 0.1)),
            )
            if hasattr(TransformerModel, "filter_model_args_"):
                TransformerModel.filter_model_args_(kwargs)
            self.model = TransformerModel(**kwargs)
        else:
            raise ValueError(f"Unsupported upstream ResCP backbone: {model_name}")

    def forward(self, x, u=None):
        if isinstance(self.model, _CopiedResCPRNNModel):
            out, state = self.model.forward_with_state(x, u)
        else:
            out = self.model(x, u)
            state = None
        if out.ndim == 4:
            y_hat = out[:, 0]
        elif out.ndim == 3:
            y_hat = out
        elif out.ndim == 2:
            y_hat = out.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected upstream backbone output shape: {tuple(out.shape)}")
        return y_hat, state


class _CopiedResCPRNNModel(nn.Module):
    """Copied from ResCP upstream ``src/lib/nn/base/rnn_model.py``.

    This keeps the central code self-contained while preserving the upstream
    architecture: Linear input encoder -> TSL RNN/GRU -> LinearReadout.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon: int,
        exog_size: int = 0,
        hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = "gru",
    ):
        super().__init__()
        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)
        self.rnn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            return_only_last_state=True,
            dropout=dropout,
            cell=cell_type,
        )
        self.readout = LinearReadout(input_size=hidden_size, output_size=output_size, horizon=horizon)

    def forward(self, x, u=None):
        state = self.encode(x, u)
        return self.readout(state)

    def encode(self, x, u=None):
        x = maybe_cat_exog(x, u)
        x = self.input_encoder(x)
        return self.rnn(x)

    def forward_with_state(self, x, u=None):
        state = self.encode(x, u)
        return self.readout(state), state


class _ResCPNativeRNNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        exog_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        cell_type: str = "gru",
    ):
        super().__init__()
        self._adapter = _UpstreamBackboneAdapter(
            "rnn",
            input_size=input_size,
            exog_size=exog_size,
            model_kwargs=dict(
                hidden_size=hidden_size,
                n_layers=n_layers,
                dropout=dropout,
                cell_type=cell_type,
            )
        )
        self.hidden_size = self._adapter.hidden_size

    def forward(self, x, u=None):
        return self._adapter(x, u)


class _ResCPNativeTransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        exog_size: int,
        hidden_size: int,
        ff_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self._adapter = _UpstreamBackboneAdapter(
            "transformer",
            input_size=input_size,
            exog_size=exog_size,
            model_kwargs=dict(
                hidden_size=hidden_size,
                ff_size=ff_size,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                max_seq_len=max_seq_len,
            ),
        )
        self.hidden_size = self._adapter.hidden_size

    def forward(self, x, u=None):
        return self._adapter(x, u)


def _sinusoidal_encoding(max_seq_len: int, hidden_size: int) -> torch.Tensor:
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    encoding = torch.zeros(1, max_seq_len, hidden_size)
    encoding[0, :, 0::2] = torch.sin(position * div_term)
    encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[-1]])
    return encoding


@dataclass
class _StackedData:
    ts_ids: list[str]
    features: torch.Tensor
    target: torch.Tensor
    mask: Optional[torch.Tensor]
    calib_step: int
    test_step: int


class _ResCPNativeBase(ForcastModel):
    def __init__(self, no_x_features, model_params, **kwargs):
        super().__init__(
            forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
            supported_outputs=(PredictionOutputType.POINT,),
        )
        self._window = int(model_params.get("window", model_params.get("seq_len", 24)))
        self._delay = int(model_params.get("delay", 0))
        self._batch_size = int(model_params.get("batch_size", 32))
        self._input_mode = str(model_params.get("input_mode", "upstream_x_u"))
        self._append_y_history = bool(model_params.get("append_y_history", True))
        self._loss_name = str(model_params.get("loss", "mae")).lower()
        self._val_source = str(model_params.get("val_source", "calib_front"))
        self._val_calib_fraction = float(model_params.get("val_calib_fraction", 0.25))
        self._limit_train_batches = model_params.get("limit_train_batches", 100)
        if self._limit_train_batches is not None:
            self._limit_train_batches = int(self._limit_train_batches)
        self._n_epochs = int(model_params.get("n_epochs", 200))
        self._val_every = int(model_params.get("val_every", 1))
        self._patience = int(model_params.get("patience", 50))
        self._lr = float(model_params.get("lr", 0.003))
        self._weight_decay = float(model_params.get("weight_decay", 0.0))
        self._grad_clip_val = model_params.get("grad_clip_val", 5.0)
        self._scheduler_factor = float(model_params.get("scheduler_factor", 0.5))
        self._scheduler_patience = int(model_params.get("scheduler_patience", 5))
        self._save_prediction = kwargs.get("save_predictions", False)
        self._pre_trained_predictions_paths = model_params.get("pre_trained_predictions_paths", None)
        self._pre_trained_states_path = model_params.get("pre_trained_states_path", None)
        self._pre_trained_model_path = model_params.get("pre_trained_model_path", None)
        self._predictions: dict[str, torch.Tensor] = {}
        self._states: dict[str, torch.Tensor] = {}
        self._prediction_offset = self._window + self._delay
        if self._input_mode == "upstream_x_u":
            self._input_size = 1
            self._exog_size = int(no_x_features)
        else:
            self._input_size = int(no_x_features) + (1 if self._append_y_history else 0)
            self._exog_size = 0
        self._model = self._build_model(self._input_size, self._exog_size, model_params)

    def _build_model(self, input_size: int, exog_size: int, model_params) -> nn.Module:
        raise NotImplementedError

    @property
    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return False

    @property
    def fc_state_dim(self):
        return self._model.hidden_size

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[tuple[FCModelPrediction, Optional[int]]]:
        raise NotImplementedError("ResCP-native forecasters are trained through train_global().")

    def train_global(self, datasets: Iterable[ChronoSplittedTsDataset], alphas, trainer_config, experiment_config):
        datasets = list(datasets)
        stacked = self._stack_datasets(datasets)
        device = get_device(getattr(experiment_config, "gpu_id", -1))
        self._model.to(device)

        if self._pre_trained_model_path:
            state = torch.load(str(self._pre_trained_model_path), map_location=device)
            self._model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)

        if self._pre_trained_predictions_paths is None:
            self._fit(stacked, trainer_config, device)
            self._predictions, self._states = self._predict_all(stacked, device)
            if self._save_prediction:
                torch.save(self._predictions, f"{experiment_config.experiment_dir}/predictions.pt")
                torch.save(self._states, f"{experiment_config.experiment_dir}/states.pt")
                torch.save({"state_dict": self._model.state_dict()}, f"{experiment_config.experiment_dir}/model.pt")
        else:
            self._predictions = torch.load(self._pre_trained_predictions_paths, map_location="cpu")
            if self._pre_trained_states_path is not None:
                self._states = torch.load(self._pre_trained_states_path, map_location="cpu")
            else:
                self._states = {}

    def _stack_datasets(self, datasets: list[ChronoSplittedTsDataset]) -> _StackedData:
        if not datasets:
            raise ValueError("ResCP-native forecaster received no datasets.")
        lengths = {int(d.X_full.shape[0]) for d in datasets}
        x_dims = {int(d.X_full.shape[1]) for d in datasets}
        calib_steps = {int(d.calib_step) for d in datasets}
        test_steps = {int(d.test_step) for d in datasets}
        if len(lengths) != 1 or len(x_dims) != 1 or len(calib_steps) != 1 or len(test_steps) != 1:
            raise ValueError("ResCP-native forecaster requires aligned series with identical length, feature dimension, and split points.")
        x = torch.stack([d.X_full.float() for d in datasets], dim=1)
        y = torch.stack([d.Y_full.float().reshape(-1, 1) for d in datasets], dim=1)
        masks = [getattr(d, "mask_full", None) for d in datasets]
        mask = None
        if all(m is not None for m in masks):
            mask = torch.stack([m.float().reshape(-1, 1) for m in masks], dim=1)
        features = x if self._input_mode == "upstream_x_u" else (torch.cat([y, x], dim=-1) if self._append_y_history else x)
        return _StackedData(
            ts_ids=[d.ts_id for d in datasets],
            features=features,
            target=y,
            mask=mask,
            calib_step=next(iter(calib_steps)),
            test_step=next(iter(test_steps)),
        )

    def _target_indices(self, start: int, end: int) -> torch.Tensor:
        start = max(int(start), self._prediction_offset)
        end = int(end)
        if end <= start:
            return torch.empty(0, dtype=torch.long)
        return torch.arange(start, end, dtype=torch.long)

    def _build_loader(self, stacked: _StackedData, target_indices: torch.Tensor, shuffle: bool) -> DataLoader:
        dataset = _NodeWindowDataset(
            features=stacked.features,
            target=stacked.target,
            mask=stacked.mask,
            target_indices=target_indices,
            window=self._window,
            delay=self._delay,
            input_mode=self._input_mode,
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=False)

    def _train_val_indices(self, stacked: _StackedData) -> tuple[torch.Tensor, torch.Tensor]:
        train_idx = self._target_indices(self._prediction_offset, stacked.calib_step)
        if self._val_source == "calib_front":
            val_end = stacked.calib_step + max(1, int((stacked.test_step - stacked.calib_step) * self._val_calib_fraction))
            val_idx = self._target_indices(stacked.calib_step, val_end)
        elif self._val_source == "calib_all":
            val_idx = self._target_indices(stacked.calib_step, stacked.test_step)
        elif self._val_source == "train_tail":
            split = int(train_idx.numel() * 0.75)
            val_idx = train_idx[split:]
            train_idx = train_idx[:split]
        else:
            raise ValueError(f"Unsupported ResCP-native validation source: {self._val_source}")
        return train_idx, val_idx

    def _fit(self, stacked: _StackedData, trainer_config, device: torch.device):
        train_idx, val_idx = self._train_val_indices(stacked)
        if train_idx.numel() == 0 or val_idx.numel() == 0:
            raise ValueError(f"Invalid train/val indices: train={train_idx.numel()}, val={val_idx.numel()}")
        train_loader = self._build_loader(stacked, train_idx, shuffle=True)
        val_loader = self._build_loader(stacked, val_idx, shuffle=False)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self._scheduler_factor,
            patience=self._scheduler_patience,
            threshold=1e-4,
            threshold_mode="abs",
            min_lr=1e-6,
        )
        n_epochs = self._n_epochs
        val_every = self._val_every
        patience = self._patience
        grad_clip = None if self._grad_clip_val is None else {"max_norm": float(self._grad_clip_val)}

        best_epoch = 0
        best_score = float("inf")
        best_state = copy.deepcopy(self._model.state_dict())
        for epoch in range(1, n_epochs + 1):
            self._model.train()
            for batch_no, batch in enumerate(train_loader):
                if self._limit_train_batches is not None and batch_no >= self._limit_train_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                u = batch.get("u")
                y_hat, _ = self._model(batch["x"].to(device), None if u is None else u.to(device))
                loss = self._loss(y_hat, batch["y"].to(device), batch.get("mask"), device)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), **grad_clip)
                optimizer.step()

            if epoch % val_every != 0:
                continue
            val_score = self._evaluate_mae(val_loader, device)
            if scheduler is not None:
                try:
                    scheduler.step(val_score)
                except TypeError:
                    scheduler.step(epoch=epoch)
            if val_score < best_score:
                best_epoch = epoch
                best_score = val_score
                best_state = copy.deepcopy(self._model.state_dict())
            elif patience is not None and epoch > best_epoch + patience:
                LOGGER.info("ResCP-native forecaster early stop at epoch %s; best epoch %s val_mae %.6f", epoch, best_epoch, best_score)
                break
        self._model.load_state_dict(best_state)
        LOGGER.info("ResCP-native forecaster selected epoch %s with val_mae %.6f", best_epoch, best_score)

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        err = torch.abs(y_hat - y) if self._loss_name in {"mae", "l1"} else (y_hat - y).square()
        if mask is None:
            return err.mean()
        mask = mask.to(device)
        return (err * mask).sum() / mask.sum().clamp_min(1.0)

    def _evaluate_mae(self, loader: DataLoader, device: torch.device) -> float:
        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                y = batch["y"].to(device)
                u = batch.get("u")
                y_hat, _ = self._model(batch["x"].to(device), None if u is None else u.to(device))
                mask = batch.get("mask")
                err = torch.abs(y_hat - y)
                if mask is None:
                    total += err.sum().item()
                    count += y.numel()
                else:
                    mask = mask.to(device)
                    total += (err * mask).sum().item()
                    count += int(mask.sum().item())
        return total / max(count, 1)

    def _predict_all(self, stacked: _StackedData, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        idx = self._target_indices(self._prediction_offset, stacked.target.shape[0])
        loader = self._build_loader(stacked, idx, shuffle=False)
        pred_chunks = []
        state_chunks = []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                u = batch.get("u")
                y_hat, state = self._model(batch["x"].to(device), None if u is None else u.to(device))
                pred_chunks.append(y_hat.cpu())
                if state is not None:
                    state_chunks.append(state.cpu())
        preds = torch.cat(pred_chunks, dim=0)  # [time, nodes, 1]
        states = torch.cat(state_chunks, dim=0) if state_chunks else None  # [time, nodes, hidden]
        pred_by_ts = {}
        state_by_ts = {}
        for node, ts_id in enumerate(stacked.ts_ids):
            pred_by_ts[ts_id] = preds[:, node, :]
            if states is not None:
                state_by_ts[ts_id] = states[:, node, :]
        return pred_by_ts, state_by_ts

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        start = int(pred_data.step_offset) - self._prediction_offset
        end = start + pred_data.no_fc_steps
        prediction = self._predictions[pred_data.ts_id][start:end].to(device=pred_data.X_step.device)
        state = self._states.get(pred_data.ts_id)
        if state is not None:
            state = state[start:end].to(device=pred_data.X_step.device)
        return FcSingleModelPrediction(point=prediction, state=state)


class ResCPNativeRNN(_ResCPNativeBase):
    def _build_model(self, input_size: int, exog_size: int, model_params) -> nn.Module:
        conf = model_params.get("rnn_conf", {})
        return _ResCPNativeRNNModel(
            input_size=input_size,
            exog_size=exog_size,
            hidden_size=int(conf.get("hidden_size", 32)),
            n_layers=int(conf.get("n_layers", conf.get("num_layers", 1))),
            dropout=float(conf.get("dropout", model_params.get("dropout", 0.0))),
            cell_type=str(conf.get("cell_type", "gru")),
        )


class ResCPNativeTransformer(_ResCPNativeBase):
    def _build_model(self, input_size: int, exog_size: int, model_params) -> nn.Module:
        conf = model_params.get("transformer_conf", {})
        hidden_size = int(conf.get("hidden_size", 32))
        return _ResCPNativeTransformerModel(
            input_size=input_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            ff_size=int(conf.get("ff_size", hidden_size * 2)),
            n_heads=int(conf.get("n_heads", 2)),
            n_layers=int(conf.get("n_layers", 3)),
            dropout=float(conf.get("dropout", model_params.get("dropout", 0.1))),
            max_seq_len=int(model_params.get("max_seq_len", 512)),
        )
