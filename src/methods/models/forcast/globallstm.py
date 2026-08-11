import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Iterable

import hydra
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from loader.dataset import ChronoSplittedTsDataset
from models.base_model import BaseModel
from models.forcast.forcast_base import ForcastModel, FCPredictionData, FCModelPrediction, ForcastMode, \
    PredictionOutputType, FcSingleModelPrediction
from models.forcast.mdn_lstm import MDNLSTMManyToOne
from models.uncertainty.components.eps_ctx_encode import FcModel
from models.uncertainty.components.mdn import MNDCoef
from trainer.evaluator import Evaluator
from trainer.prep_iterator import ManyToOneIterator, CompleteDataset
from trainer.utils import map_merge_dicts, merge_dicts, batch_to_device_all, map_identity

LOGGER = logging.getLogger(__name__)

class SimpleLSTM(BaseModel):
    def __init__(self, input_size, lstm_conf, dropout, use_mc_dropout, **kwargs):
        super().__init__()
        self._loss_name = str(kwargs.pop("loss", "mse")).lower()
        self._mc_samples = 500 if use_mc_dropout else None
        self._constructor_args = dict(
            input_size=input_size,
            lstm_conf=lstm_conf,
            dropout=dropout,
            use_mc_dropout=use_mc_dropout,
            loss=self._loss_name,
            **kwargs,
        )
        self._lstm = nn.LSTM(input_size=input_size, batch_first=True, **lstm_conf)
        self._dropout_rate = dropout
        if self._dropout_rate > 0:
            self._dropout = nn.Dropout(p=self._dropout_rate)
        self._fc = FcModel(input_dim=lstm_conf['hidden_size'], out_dim=1, hidden=())

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    def forward(self, x, h=None, **kwargs):
        lstm_out, (h, c) = self._lstm(x, h)
        mc_y_hat = None
        features = lstm_out[:, -1, :]
        if self._dropout_rate > 0:
            y_hat = self._fc(self._dropout(features)).unsqueeze(1)
            if not self.training and self._mc_samples is not None:
                # In Evalution we need the samples
                self._dropout.train(True)
                tmp = []
                for i in range(self._mc_samples):
                    tmp.append(self._fc(self._dropout(features)).unsqueeze(1))
                mc_y_hat = torch.cat(tmp, dim=2)
                del tmp
                self._dropout.train(False)  # TODO
        else:
            assert self._mc_samples is None
            y_hat = self._fc(features).unsqueeze(1)
        if mc_y_hat is not None:
            return dict(y_hat=y_hat, mc_y_hat=mc_y_hat, state=features)
        else:
            return dict(y_hat=y_hat, state=features)

    def get_loss_func(self, **kwargs):
        loss_func = L1Loss() if self._loss_name in {"mae", "l1"} else MSELoss()
        loss_key = "mae" if self._loss_name in {"mae", "l1"} else "mse"
        def loss(y_hat, y, **kwargs):
            loss_val = loss_func(input=y_hat, target=y)
            return loss_val, {loss_key: loss_val}

        return loss

    def _get_constructor_parameters(self) -> dict:
        return self._constructor_args

    def get_train_fingerprint(self) -> dict:
        raise NotImplemented("asdf")


class SimpleGRU(BaseModel):
    def __init__(self, input_size, rnn_conf, dropout, **kwargs):
        super().__init__()
        self._loss_name = str(kwargs.pop("loss", "mse")).lower()
        hidden_size = int(rnn_conf["hidden_size"])
        n_layers = int(rnn_conf.get("num_layers", rnn_conf.get("n_layers", 1)))
        rnn_dropout = float(rnn_conf.get("dropout", 0.0)) if n_layers > 1 else 0.0
        self._constructor_args = dict(
            input_size=input_size,
            rnn_conf=rnn_conf,
            dropout=dropout,
            loss=self._loss_name,
            **kwargs,
        )
        self._input_encoder = nn.Linear(input_size, hidden_size)
        self._rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self._dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self._fc = FcModel(input_dim=hidden_size, out_dim=1, hidden=())

    @property
    def hidden_size(self):
        return self._rnn.hidden_size

    def forward(self, x, h=None, **kwargs):
        z = self._input_encoder(x)
        rnn_out, h = self._rnn(z, h)
        features = rnn_out[:, -1, :]
        y_hat = self._fc(self._dropout(features)).unsqueeze(1)
        return dict(y_hat=y_hat, state=features)

    def get_loss_func(self, **kwargs):
        loss_func = L1Loss() if self._loss_name in {"mae", "l1"} else MSELoss()
        loss_key = "mae" if self._loss_name in {"mae", "l1"} else "mse"

        def loss(y_hat, y, **kwargs):
            loss_val = loss_func(input=y_hat, target=y)
            return loss_val, {loss_key: loss_val}

        return loss

    def _get_constructor_parameters(self) -> dict:
        return self._constructor_args

    def get_train_fingerprint(self) -> dict:
        raise NotImplemented("asdf")


class SimpleTransformerDecoder(BaseModel):
    def __init__(self, input_size, transformer_conf, dropout, max_seq_len=512, **kwargs):
        super().__init__()
        self._loss_name = str(kwargs.pop("loss", "mse")).lower()
        self._hidden_size = int(transformer_conf['hidden_size'])
        n_heads = int(transformer_conf.get('n_heads', 4))
        if self._hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size={self._hidden_size} must be divisible by n_heads={n_heads}")
        ff_size = int(transformer_conf.get('ff_size', self._hidden_size * 2))
        n_layers = int(transformer_conf.get('n_layers', 3))
        transformer_dropout = float(transformer_conf.get('dropout', dropout))
        self._constructor_args = dict(
            input_size=input_size,
            transformer_conf=transformer_conf,
            dropout=dropout,
            max_seq_len=max_seq_len,
            loss=self._loss_name,
            **kwargs,
        )
        self._input_proj = nn.Linear(input_size, self._hidden_size)
        self.register_buffer(
            '_pos_embedding',
            self._build_sinusoidal_positional_encoding(int(max_seq_len), self._hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_size,
            dropout=transformer_dropout,
            batch_first=True,
            activation=F.elu,
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self._readout = nn.Sequential(
            nn.Linear(self._hidden_size, ff_size),
            nn.ELU(),
            nn.Dropout(p=transformer_dropout) if transformer_dropout > 0 else nn.Identity(),
            nn.Linear(ff_size, 1),
        )

    @staticmethod
    def _build_sinusoidal_positional_encoding(max_seq_len, hidden_size):
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(1, max_seq_len, hidden_size)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term[:encoding[0, :, 1::2].shape[-1]])
        return encoding

    @property
    def hidden_size(self):
        return self._hidden_size

    def forward(self, x, **kwargs):
        seq_len = x.shape[1]
        if seq_len > self._pos_embedding.shape[1]:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self._pos_embedding.shape[1]}")
        z = self._input_proj(x) + self._pos_embedding[:, :seq_len, :]
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1,
        )
        encoded = self._transformer(z, mask=mask)
        features = encoded[:, -1, :]
        y_hat = self._readout(features).unsqueeze(1)
        return dict(y_hat=y_hat, state=features)

    def get_loss_func(self, **kwargs):
        loss_func = L1Loss() if self._loss_name in {"mae", "l1"} else MSELoss()
        loss_key = "mae" if self._loss_name in {"mae", "l1"} else "mse"

        def loss(y_hat, y, **kwargs):
            loss_val = loss_func(input=y_hat, target=y)
            return loss_val, {loss_key: loss_val}

        return loss

    def _get_constructor_parameters(self) -> dict:
        return self._constructor_args

    def get_train_fingerprint(self) -> dict:
        raise NotImplemented("asdf")


class GlobalLSTM(ForcastModel):

    def __init__(self, no_x_features, model_params, **kwargs) -> None:
        self._use_mc_dropout = model_params['use_mc_dropout']
        self._use_with_mdn = model_params.get('use_with_mdn', False)
        self._append_y_history = model_params.get('append_y_history', True)
        self._loss_name = str(model_params.get('loss', 'mse')).lower()
        assert not (self._use_mc_dropout and self._use_with_mdn)
        super().__init__(forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
                         supported_outputs=(PredictionOutputType.QUANTILE,) if self._use_mc_dropout or self._use_with_mdn else (PredictionOutputType.POINT, ))
        input_size = no_x_features + (1 if self._append_y_history else 0)
        if not self._use_with_mdn:
            self._lstm_model = SimpleLSTM(input_size=input_size, lstm_conf=model_params['lstm_conf'],
                                          dropout=model_params['dropout'], use_mc_dropout=self._use_mc_dropout,
                                          loss=self._loss_name)
        else:
            self._lstm_model = MDNLSTMManyToOne(input_size=input_size, lstm_conf=model_params['lstm_conf'],
                                                dropout=model_params['dropout'], mdn_conf=model_params['mdn_conf'])
        self._seq_len = model_params['seq_len']
        self._batch_size = model_params['batch_size']
        self._train_split = float(model_params.get('train_split', 0.75))
        self._val_source = str(model_params.get('val_source', 'train_tail'))
        self._val_calib_fraction = float(model_params.get('val_calib_fraction', 0.25))
        self._train_with_calib = model_params['train_with_calib']
        self._train_only = kwargs.get('train_only', False)
        self._save_prediction = kwargs.get('save_predictions', False)
        self._pre_trained_model_path = model_params.get('pre_trained_model_path', None)
        self._plot_eval_after_train = model_params['plot_eval_after_train']
        self._pre_trained_predictions_paths = model_params.get('pre_trained_predictions_paths', None)
        self._pre_trained_states_path = model_params.get('pre_trained_states_path', None)
        self._predictions = dict()
        self._states = dict()
        self._prediction_offset = None


    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        raise NotImplemented("Asdf")

    def train_global(self, datasets, alphas, trainer_config, experiment_config):
        # 1) Check if model available -> Load model / Otherwise train
        if self._pre_trained_model_path is None:
            trainer = hydra.utils.instantiate(
                trainer_config,
                experiment_config=experiment_config,
                model=self._lstm_model,
                get_data_loader=lambda num_worker=0: (
                    self._get_dataloader(datasets, is_val=False, num_worker=num_worker),
                    self._get_dataloader(datasets, is_val=True, num_worker=num_worker),
                ),
                move_batch_to_device=batch_to_device_all(),
                map_to_model_in=map_identity(),
                loss_func=self._lstm_model.get_loss_func(),
                map_to_loss_in=merge_dicts(),
                val_metrics=self._point_metric_collection(),
                train_metrics=self._point_metric_collection(),
                map_to_metrics_in=map_merge_dicts({"y_hat": "preds", "y": "target"}),
            )
            trainer.train()
            if self._plot_eval_after_train:
                # Log Evalution Prediciton after Training
                predicitons, _, metrics = self._create_predictions_point(datasets, trainer, evaluate_mode='train')
                for dataset in datasets:
                    if self._train_with_calib:
                        split_point = int((dataset.no_train_steps + dataset.no_calib_steps) * self._train_split)
                    else:
                        split_point = int(dataset.no_train_steps * self._train_split)
                    y = dataset.Y_train[split_point:].cpu()
                    y_hat = predicitons[dataset.ts_id][split_point - self._seq_len:].cpu()
                    data = pd.DataFrame(torch.cat((y, y_hat), dim=1).numpy(), columns=["Y", "Y_hat"])
                    fig, ax = plt.subplots()
                    sns.lineplot(data=data, ax=ax)
                    LOGGER.debug(f"Created validation plot for {dataset.ts_id}: {fig}")
        else:
            self._lstm_model.load_state(self._pre_trained_model_path)

        if self._pre_trained_predictions_paths is None and not self._train_only:
            evaluator = Evaluator(
                gpu_id=experiment_config.gpu_id,
                move_batch_to_device=batch_to_device_all(),
                map_to_model_in=map_identity(),
                val_metrics=self._point_metric_collection(),
                map_to_metrics_in=map_merge_dicts({"y_hat": "preds", "y": "target"}),
            )
            # 2) Create all predictions for train, calib and test
            self._predictions, self._states, _ = self._create_predictions_point(datasets, evaluator, evaluate_mode='all')
            if self._save_prediction:
                torch.save(self._predictions, f"{experiment_config.experiment_dir}/predictions.pt")
                torch.save(self._states, f"{experiment_config.experiment_dir}/states.pt")
        elif not self._train_only:
            self._predictions = torch.load(self._pre_trained_predictions_paths, map_location='cpu')
            if self._pre_trained_states_path is not None and Path(self._pre_trained_states_path).exists():
                self._states = torch.load(self._pre_trained_states_path, map_location='cpu')
            else:
                evaluator = Evaluator(
                    gpu_id=experiment_config.gpu_id,
                    move_batch_to_device=batch_to_device_all(),
                    map_to_model_in=map_identity(),
                    val_metrics=self._point_metric_collection(),
                    map_to_metrics_in=map_merge_dicts({"y_hat": "preds", "y": "target"}),
                )
                _, self._states, _ = self._create_predictions_point(datasets, evaluator, evaluate_mode='all')
                if self._pre_trained_states_path is not None:
                    state_dir = Path(self._pre_trained_states_path).parent
                    state_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self._states, self._pre_trained_states_path)

        self._prediction_offset = self._seq_len

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        start = pred_data.step_offset - self._prediction_offset
        end = start + pred_data.no_fc_steps
        prediction = self._predictions[pred_data.ts_id][start:end].to(device=pred_data.X_step.device)
        state = None
        if pred_data.ts_id in self._states:
            state = self._states[pred_data.ts_id][start:end].to(device=pred_data.X_step.device)
        if self._use_mc_dropout or self._use_with_mdn:
            # ToDo Hack only use 500 mc samples
            assert prediction.shape[1] > 100
            use_mc_samples = 500
            prediction = prediction[:, :use_mc_samples]
            lower = torch.quantile(prediction, pred_data.alpha / 2, dim=1, keepdim=True)
            upper = torch.quantile(prediction, (1 - pred_data.alpha / 2), dim=1, keepdim=True)
            return FcSingleModelPrediction(quantile=(lower, upper), state=state)
        else:
            assert prediction.shape[1] == 1
            return FcSingleModelPrediction(point=prediction, state=state)

    def _create_predictions_point(self, datasets, evaluator, evaluate_mode='val'):
        predictions = {}
        states = {}
        metrics = {}
        def get_pred_split(dataset: ChronoSplittedTsDataset):
            if evaluate_mode == 'val':
                return torch.cat((dataset.X_train[-self._seq_len:], dataset.X_calib), dim=0),\
                    torch.cat((dataset.Y_train[-self._seq_len:], dataset.Y_calib), dim=0),
            elif evaluate_mode == 'train':
                return dataset.X_train, dataset.Y_train
            elif evaluate_mode == 'test':
                return torch.cat((dataset.X_calib[-self._seq_len:], dataset.X_test), dim=0), \
                    torch.cat((dataset.Y_calib[-self._seq_len:], dataset.Y_calib), dim=0),
            elif evaluate_mode == 'all':
                return dataset.X_full, dataset.Y_full
            else:
                raise ValueError("Invalid mode!")
        for dataset in datasets:
            iterator = ManyToOneIterator(
                [dataset],
                seq_len=self._seq_len,
                split_func=get_pred_split,
                include_target_history=self._append_y_history,
            )
            dataloader = DataLoader(CompleteDataset(iterator), batch_size=self._batch_size, shuffle=False, drop_last=False)
            model_out, metric_vals = evaluator.eval(dataloader, self._lstm_model, eval_text=f"mode {evaluate_mode} - TS {dataset.ts_id}")
            if self._use_mc_dropout:
                predictions[dataset.ts_id] = torch.concat([o['mc_y_hat'] for o in model_out], dim=0).squeeze(1).to(device='cpu')
            elif self._use_with_mdn:
                pi = torch.concat([o['mdn_coef'].pi for o in model_out], dim=0)
                mu = torch.concat([o['mdn_coef'].mu for o in model_out], dim=0)
                sigma = torch.concat([o['mdn_coef'].sigma for o in model_out], dim=0)
                samples = self._lstm_model.sample_from_mdn_coef(MNDCoef(pi=pi, mu=mu, sigma=sigma))
                predictions[dataset.ts_id] = samples.squeeze(2).to(device='cpu')
            else:
                predictions[dataset.ts_id] = torch.concat([o['y_hat'] for o in model_out], dim=0).squeeze(1)
            if all('state' in o and o['state'] is not None for o in model_out):
                states[dataset.ts_id] = torch.concat([o['state'] for o in model_out], dim=0).to(device='cpu')
            metrics[dataset.ts_id] = metric_vals
        return predictions, states, metrics

    def _get_dataloader(self, datasets, is_val=False, num_worker=0):
        def get_split(dataset: ChronoSplittedTsDataset):
            if self._val_source == 'calib_front':
                if not dataset.has_calib_set:
                    raise ValueError("calib_front validation requires a calibration split")
                val_end = dataset.calib_step + max(1, int(dataset.no_calib_steps * self._val_calib_fraction))
                if is_val:
                    return dataset.X_full[dataset.calib_step:val_end], dataset.Y_full[dataset.calib_step:val_end]
                else:
                    return dataset.X_train, dataset.Y_train
            if self._val_source != 'train_tail':
                raise ValueError(f"Unsupported forecaster validation source: {self._val_source}")
            if self._train_with_calib:
                split_point = int((dataset.no_train_steps + dataset.no_calib_steps) * self._train_split)
                if is_val:
                    return dataset.X_full[split_point:dataset.test_step], dataset.Y_full[split_point:dataset.test_step]
                else:
                    return dataset.X_full[:split_point], dataset.Y_full[:split_point]
            else:
                split_point = int(dataset.no_train_steps * self._train_split)
                if is_val:
                    return dataset.X_train[split_point:], dataset.Y_train[split_point:]
                else:
                    return dataset.X_train[:split_point], dataset.Y_train[:split_point]
        iterator = ManyToOneIterator(
            datasets,
            seq_len=self._seq_len,
            split_func=get_split,
            include_target_history=self._append_y_history,
        )
        return DataLoader(CompleteDataset(iterator), batch_size=self._batch_size, shuffle=True, num_workers=num_worker)

    @property
    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return False

    @property
    def fc_state_dim(self):
        return self._lstm_model.hidden_size

    def _point_metric_collection(self):
        metric = MeanAbsoluteError() if self._loss_name in {"mae", "l1"} else MeanSquaredError()
        return MetricCollection(metric)


class GlobalRNN(GlobalLSTM):
    def __init__(self, no_x_features, model_params, **kwargs) -> None:
        self._use_mc_dropout = False
        self._use_with_mdn = False
        self._append_y_history = model_params.get('append_y_history', True)
        self._loss_name = str(model_params.get('loss', 'mse')).lower()
        ForcastModel.__init__(self, forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
                              supported_outputs=(PredictionOutputType.POINT,))
        input_size = no_x_features + (1 if self._append_y_history else 0)
        self._lstm_model = SimpleGRU(
            input_size=input_size,
            rnn_conf=model_params['rnn_conf'],
            dropout=model_params['dropout'],
            loss=self._loss_name,
        )
        self._seq_len = model_params['seq_len']
        self._batch_size = model_params['batch_size']
        self._train_split = float(model_params.get('train_split', 0.75))
        self._val_source = str(model_params.get('val_source', 'train_tail'))
        self._val_calib_fraction = float(model_params.get('val_calib_fraction', 0.25))
        self._train_with_calib = model_params['train_with_calib']
        self._train_only = kwargs.get('train_only', False)
        self._save_prediction = kwargs.get('save_predictions', False)
        self._pre_trained_model_path = model_params.get('pre_trained_model_path', None)
        self._plot_eval_after_train = model_params['plot_eval_after_train']
        self._pre_trained_predictions_paths = model_params.get('pre_trained_predictions_paths', None)
        self._pre_trained_states_path = model_params.get('pre_trained_states_path', None)
        self._predictions = dict()
        self._states = dict()
        self._prediction_offset = None


class GlobalTransformerDecoder(GlobalLSTM):
    def __init__(self, no_x_features, model_params, **kwargs) -> None:
        self._use_mc_dropout = False
        self._use_with_mdn = False
        self._append_y_history = model_params.get('append_y_history', True)
        self._loss_name = str(model_params.get('loss', 'mse')).lower()
        ForcastModel.__init__(self, forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
                              supported_outputs=(PredictionOutputType.POINT,))
        input_size = no_x_features + (1 if self._append_y_history else 0)
        self._lstm_model = SimpleTransformerDecoder(
            input_size=input_size,
            transformer_conf=model_params['transformer_conf'],
            dropout=model_params['dropout'],
            max_seq_len=model_params.get('max_seq_len', 512),
            loss=self._loss_name,
        )
        self._seq_len = model_params['seq_len']
        self._batch_size = model_params['batch_size']
        self._train_split = float(model_params.get('train_split', 0.75))
        self._val_source = str(model_params.get('val_source', 'train_tail'))
        self._val_calib_fraction = float(model_params.get('val_calib_fraction', 0.25))
        self._train_with_calib = model_params['train_with_calib']
        self._train_only = kwargs.get('train_only', False)
        self._save_prediction = kwargs.get('save_predictions', False)
        self._pre_trained_model_path = model_params.get('pre_trained_model_path', None)
        self._plot_eval_after_train = model_params['plot_eval_after_train']
        self._pre_trained_predictions_paths = model_params.get('pre_trained_predictions_paths', None)
        self._pre_trained_states_path = model_params.get('pre_trained_states_path', None)
        self._predictions = dict()
        self._states = dict()
        self._prediction_offset = None

    @property
    def fc_state_dim(self):
        return self._lstm_model.hidden_size
