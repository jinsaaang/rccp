import random
import collections
import datetime

import matplotlib.pyplot as plt

from reservoir_computing.modules import RC_forecaster

from reservoir_conformal_prediction.src.quantile_regressor import (
    QuantileRegressorLightning,
)
from reservoir_conformal_prediction.src.reservoir_conformal_residual_sampler import (
    ConformalResidualSamplerLightning,
    ConformalResidualSamplerNumpy,
)
from reservoir_conformal_prediction.src.torch_reservoir_computing.modules import (
    RC_forecaster_torch,
)
from reservoir_conformal_prediction.src.utils.utils import (
    minimize_intervals,
)
import numpy as np

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb


class ReservoirDataset(torch.utils.data.Dataset):
    """Dataset for the reservoir states and the residuals of the time series.
    This class is used for the training and validation of the quantile regression model.

    Parameters
    ----------
    X_states : torch.Tensor
        Reservoir states of the time series. Shape (n_samples, n_internal_units)
    y_residuals : torch.Tensor
        Residuals of the time series. Shape (n_samples,)
    y_true : torch.Tensor
        True values of the time series. Shape (n_samples,)
    y_pred : torch.Tensor
        Point predictions of the time series. Shape (n_samples,)
    """

    def __init__(self, X_states, y_residuals, y_true, y_pred):
        self.X_states = X_states
        self.y_residuals = y_residuals
        self.y_true = y_true
        self.y_pred = y_pred

    def n_features(self):
        """Returns the number of features in the dataset.

        Returns
        -------
        n_features : int
            Number of features in the dataset.
        """

        return self.X_states.shape[1]

    def __getitem__(self, index):
        return (
            self.X_states[index],
            self.y_residuals[index],
            self.y_true[index],
            self.y_pred[index],
        )

    def __len__(self):
        return len(self.X_states)


class BucketBatchSampler(torch.utils.data.Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.shape[0]))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        # shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = collections.OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [
                indices[i : (i + self.batch_size)]
                for i in range(0, len(indices), self.batch_size)
            ]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        random.shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


class ReservoirConformalPrediction:
    """Reservoir conformal prediction for time series forecasting.

    Parameters
    ----------
    reservoir_params : dict
        Dictionary containing the parameters of the reservoir.
    backend : str
        Backend to use for the reservoir. Can be 'numpy' or 'torch'.
    offline_logging : bool
        Whether to log the results offline or online. Default is True.
    project : str
        Name of the project for wandb logging. Default is 'ReservoirConformalPrediction'.
    """

    def __init__(
        self,
        reservoir_params: dict,
        backend: str = "numpy",
        wandb_logger: WandbLogger = None,
    ):
        """

        Parameters
        ----------
        reservoir_params : dict
            Dictionary containing the parameters of the reservoir.
        backend : str
            Backend to use for the reservoir. Can be 'numpy' or 'torch'.
        offline_logging : bool
            Whether to log the results offline or online. Default is True.
        project : str
            Name of the project for wandb logging. Default is 'ReservoirConformalPrediction'.
        """
        self.backend = backend
        self.reservoir_params = reservoir_params
        if backend == "numpy":
            self.default_instance = np.ndarray
            if reservoir_params is not None:
                self.reservoir = RC_forecaster(**reservoir_params)
            else:
                self.reservoir = None
        elif backend == "torch":
            self.default_instance = torch.Tensor
            self.wandb_logger = wandb_logger

            if reservoir_params is not None:
                self.reservoir = RC_forecaster_torch(**reservoir_params)
                reservoir_params.pop("seed", None)
                self.wandb_logger.experiment.config.update(reservoir_params)
            else:
                self.reservoir = None

    def to(self, device):
        """Move the reservoir to the specified device.

        Parameters
        ----------
        device : str
            Device to move the reservoir to. Can be 'cpu' or 'cuda'.

        Returns
        -------
        self : ReservoirConformalPrediction
            The reservoir conformal prediction object on the specified device.
        """
        if self.backend == "torch":
            if self.reservoir is not None:
                self.reservoir.to(device)
        else:
            raise NotImplementedError(
                f"Moving to {device} is not supported for {self.backend} backend."
            )
        return self

    def get_states(self, X, previous_state=None):
        """Computes the reservoir states for the input time series.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
             Input time series. Shape (n_samples, n_features)
        previous_state : np.ndarray or torch.Tensor
            Previous state of the reservoir. Shape (1, n_internal_units).
            - If None, the reservoir is initialized to zero.
            - If not None, the reservoir is initialized to the given state.

        Returns
        -------
        states : np.ndarray
            Reservoir states of the input time series. Shape (n_samples, n_internal_units)
        """

        assert isinstance(
            X, self.default_instance
        ), f"Expected {self.default_instance}, got {type(X)}"

        states = self.reservoir._reservoir.get_states(
            X[None, :, :], bidir=False, initial_state=previous_state
        )
        return states[0]

    def init_data(self, X_cal, y_cal, y_hat_cal, X_val, y_val, y_hat_val):
        """Initializes the data for the quantile regression model.

        Parameters
        ----------
        X_cal : np.ndarray or torch.Tensor
            Training set for the quantile regression model. These are the endogenous + eventually exogenous variables in the calibration set. Shape (n_cal, n_features)
        y_cal : np.ndarray or torch.Tensor
            Training samples of the time series. These are the ground truth samples from the calibration set. Shape (n_cal,)
        y_hat_cal : np.ndarray or torch.Tensor
            Point predictions for the calibration set. Shape (n_cal,)
        X_val : np.ndarray or torch.Tensor
            Validation set. Reservoir states of the validation set. Shape (n_val, n_features)
        y_val : np.ndarray or torch.Tensor
            Validation labels. Residuals of the validation set. Shape (n_val,)
        y_hat_val : np.ndarray or torch.Tensor
            Point predictions for the validation set. Shape (n_val,)
        """
        if X_cal.ndim == 2:
            cal_residuals = y_cal - y_hat_cal
            # shift the residuals by one time step
            shifted_cal_residuals = torch.vstack(
                [
                    torch.zeros((1, 1), device=X_cal.device),
                    cal_residuals[:-1].reshape(-1, 1),
                ]
            )
            val_residuals = y_val - y_hat_val
            shifted_val_residuals = torch.vstack(
                [
                    cal_residuals[-1].reshape(1, 1),
                    val_residuals[:-1].reshape(-1, 1),
                ]
            )
            # -----------------------------------------------------------------------------------------------------
            # if X_cal shape is (n_cal, n_features) and has exogenous variables, then compute the
            # reservoir states for the time series (first column) and concatenate them with the exogenous variables
            # -----------------------------------------------------------------------------------------------------
            if X_cal.shape[1] > 1:
                # calibration set time series and residuals states
                ts_states = self.get_states(X_cal[:, 0].reshape(-1, 1))
                res_states = self.get_states(shifted_cal_residuals)
                X_cal = torch.hstack([ts_states, res_states, X_cal[:, 1:]])
                # X_cal shape: (n_cal, 2 * n_internal_units + n_exogenous_features)

                # histogram plot of the reservoir states
                plt.hist(ts_states.flatten().cpu().numpy(), bins=100)
                plt.title("Reservoir states histogram")
                plt.xlabel("Reservoir states")
                plt.ylabel("Frequency")
                self.wandb_logger.experiment.log(
                    {"Reservoir states histogram": wandb.Image(plt)}
                )

                # validation set time series and residuals states
                # use the last state of the calibration set as the previous state for the validation set
                ts_prev = ts_states[-1].reshape(1, -1)
                res_prev = res_states[-1].reshape(1, -1)
                self.last_cal_state = ts_prev
                self.last_cal_res_state = res_prev
                ts_states = self.get_states(
                    X_val[:, 0].reshape(-1, 1), previous_state=ts_prev
                )
                res_states = self.get_states(
                    shifted_val_residuals, previous_state=res_prev
                )
                X_val = torch.hstack([ts_states, res_states, X_val[:, 1:]])
                # X_val shape: (n_val, 2 * n_internal_units + n_exogenous_features)
            # -----------------------------------------------------------------------------------------------------
            # if X_cal shape is (n_cal, 1) and has no exogenous variables, then just compute the reservoir states
            # for the time series itself
            # -----------------------------------------------------------------------------------------------------
            elif X_cal.shape[1] == 1:
                # calibration set time series and residuals states
                ts_states = self.get_states(X_cal)
                res_states = self.get_states(shifted_cal_residuals)
                X_cal = torch.hstack([ts_states, res_states])
                # X_cal shape: (n_cal, 2 * n_internal_units)

                # histogram plot of the reservoir states
                plt.hist(ts_states.flatten().cpu().numpy(), bins=100)
                plt.title("Reservoir states histogram")
                plt.xlabel("Reservoir states")
                plt.ylabel("Frequency")
                self.wandb_logger.experiment.log(
                    {"Reservoir states histogram": wandb.Image(plt)}
                )

                # validation set time series and residuals states
                # use the last state of the calibration set as the previous state for the validation set
                ts_prev = ts_states[-1].reshape(1, -1)
                res_prev = res_states[-1].reshape(1, -1)
                self.last_cal_state = ts_prev
                self.last_cal_res_state = res_prev
                ts_states = self.get_states(X_val, previous_state=ts_prev)
                res_states = self.get_states(
                    shifted_val_residuals, previous_state=res_prev
                )
                X_val = torch.hstack([ts_states, res_states])
                # X_val shape: (n_val, 2 * n_internal_units)
        elif X_cal.ndim == 3:
            # if we have multiple time series in the calibration set, we need to compute the calibration residuals
            # for each time series, shift them and finally concatenate them
            cal_residuals_list = []
            shifted_cal_residuals = []
            for i in range(y_cal.shape[0]):
                i_cal_residuals = y_cal[i] - y_hat_cal[i]
                cal_residuals_list.append(i_cal_residuals)
                shifted_cal_residuals.append(
                    torch.vstack(
                        [
                            torch.zeros((1, 1), device=X_cal.device),
                            i_cal_residuals[:-1].reshape(-1, 1),
                        ]
                    )
                )

            shifted_cal_residuals = torch.stack(shifted_cal_residuals, dim=0)

            y_cal = torch.cat([y_cal[i] for i in range(y_cal.shape[0])], dim=0)
            y_hat_cal = torch.cat(
                [y_hat_cal[i] for i in range(y_hat_cal.shape[0])], dim=0
            )
            cal_residuals = y_cal - y_hat_cal
            val_residuals = y_val - y_hat_val

            # get the last calibration residual for the target time series to shift the validation residuals
            shifted_val_residuals = torch.vstack(
                [
                    cal_residuals_list[0][-1].reshape(1, 1),
                    val_residuals[:-1].reshape(-1, 1),
                ]
            )

            if X_cal.shape[2] > 1:
                # -----------------------------------------------------------------------------------------------------
                # if X_cal shape is (n_timeseries, n_cal, n_features), do the same as above but for each time series
                # and then concatenate the results over the first dimension
                # -----------------------------------------------------------------------------------------------------
                # get the reservoir states for both the time series and the residuals for each time series
                # and concatenate them with the exogenous variables
                ts_states_list = []
                res_states_list = []
                X_cal_exog_list = []
                for i in range(X_cal.shape[0]):
                    ts_states_list.append(
                        self.get_states(X_cal[i, :, 0].reshape(-1, 1))
                    )
                    res_states_list.append(
                        self.get_states(shifted_cal_residuals[i].reshape(-1, 1))
                    )
                    X_cal_exog_list.append(X_cal[i, :, 1:])
                ts_states = torch.vstack(ts_states_list)
                res_states = torch.vstack(res_states_list)
                X_cal_exog = torch.vstack(X_cal_exog_list)
                X_cal = torch.hstack(
                    [
                        ts_states,
                        res_states,
                        X_cal_exog,
                    ]
                )
                # X_cal shape: (sum(n_cal), 2 * n_internal_units + n_exogenous_features)
                # where sum(n_cal) = sum of the number of calibration samples for each time series

                # histogram plot of the reservoir states
                plt.hist(ts_states.flatten().cpu().numpy(), bins=100)
                plt.title("Reservoir states histogram")
                plt.xlabel("Reservoir states")
                plt.ylabel("Frequency")
                self.wandb_logger.experiment.log(
                    {"Reservoir states histogram": wandb.Image(plt)}
                )

                # validation set time series and residuals states
                ts_prev = ts_states_list[0][-1].reshape(1, -1)
                res_prev = res_states_list[0][-1].reshape(1, -1)
                self.last_cal_state = ts_prev
                self.last_cal_res_state = res_prev
                ts_states = self.get_states(
                    X_val[:, 0].reshape(-1, 1), previous_state=ts_prev
                )
                res_states = self.get_states(
                    shifted_val_residuals, previous_state=res_prev
                )
                X_val = torch.hstack([ts_states, res_states, X_val[:, 1:]])
                # X_val shape: (n_val, 2 * n_internal_units + n_exogenous_features)
            elif X_cal.shape[2] == 1:
                # -----------------------------------------------------------------------------------------------------
                # if X_cal shape is (n_timeseries, n_cal, 1) and has no exogenous variables, then just compute the reservoir states
                # for the time series itself for each time series, and then concatenate the results over the first dimensions
                # -----------------------------------------------------------------------------------------------------
                # get the reservoir states for both the time series and the residuals for each time series
                ts_states_list = []
                res_states_list = []
                for i in range(X_cal.shape[0]):
                    ts_states_list.append(
                        self.get_states(X_cal[i, :, 0].reshape(-1, 1))
                    )
                    res_states_list.append(
                        self.get_states(shifted_cal_residuals[i].reshape(-1, 1))
                    )
                ts_states = torch.vstack(ts_states_list)
                res_states = torch.vstack(res_states_list)
                X_cal = torch.hstack(
                    [
                        ts_states,
                        res_states,
                    ]
                )
                # X_cal shape: (sum(n_cal), 2 * n_internal_units)
                # where sum(n_cal) = sum of the number of calibration samples for each time series

                # histogram plot of the reservoir states
                plt.hist(ts_states.flatten().cpu().numpy(), bins=100)
                plt.title("Reservoir states histogram")
                plt.xlabel("Reservoir states")
                plt.ylabel("Frequency")
                self.wandb_logger.experiment.log(
                    {"Reservoir states histogram": wandb.Image(plt)}
                )

                ts_prev = ts_states_list[0][-1].reshape(1, -1)
                res_prev = res_states_list[0][-1].reshape(1, -1)
                self.last_cal_state = ts_prev
                self.last_cal_res_state = res_prev
                ts_states = self.get_states(
                    X_val[:, 0].reshape(-1, 1), previous_state=ts_prev
                )
                res_states = self.get_states(
                    shifted_val_residuals, previous_state=res_prev
                )
                X_val = torch.hstack([ts_states, res_states])
                # X_val shape: (n_val, 2 * n_internal_units)

            plt.close()

            # create the datasets for the training and validation sets
            self.train_dataset = ReservoirDataset(
                X_cal, cal_residuals, y_cal, y_hat_cal
            )
            self.val_dataset = ReservoirDataset(X_val, val_residuals, y_val, y_hat_val)

    def init_model(
        self,
        in_size,
        n_hidden,
        hidden_dims,
        alpha,
        n_quantiles,
        n_epochs,
        lr=1e-3,
        weight_decay=0.01,
        width_penalty=0.0,
    ):
        """Initializes the quantile regression model and the PyTorch Lightning trainer.
        This method is called after the data has been initialized.

        Parameters
        ----------
        in_size : int
            Number of input features for the quantile regression model.
        n_hidden : int
            Number of hidden layers for the quantile regression model.
        hidden_dims : list
            List of the number of neurons in each hidden layer for the quantile regression model.
        alpha : float
            Significance level for the quantile regression model.
        n_quantiles : int
            Number of quantiles for the quantile regression model.
        n_epochs : int
            Number of epochs for the quantile regression model.
        lr : float
            Learning rate for the quantile regression model.
        """
        self.model = QuantileRegressorLightning(
            in_size=in_size,
            n_hidden=n_hidden,
            hidden_dims=hidden_dims,
            alpha=alpha,
            n_quantiles=n_quantiles,
            lr=lr,
            weight_decay=weight_decay,
            width_penalty=width_penalty,
        )

        self.trainer = L.Trainer(
            max_epochs=n_epochs,
            devices=1,
            accelerator="auto",
            logger=self.wandb_logger,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10),
                LearningRateMonitor(logging_interval="step"),
            ],
        )

    def fit(self, batch_size=512):
        """Fits the quantile regression model to the training set.

        Parameters
        ----------
        batch_size : int
            Batch size for the training set.

        Returns
        -------
        model : QuantileRegressorLightning
            Fitted quantile regression model.
        """

        self.wandb_logger.experiment.config.update({"batch_size": batch_size})

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.trainer.fit(self.model, train_loader, val_loader)

        return self.model

    def compute_intervals_numpy(
        self,
        X_cal,
        cal_residuals,
        X_val,
        y_val,
        y_hat_val,
        alpha=0.1,
        T=1,
        past_residuals_window=1000,
        eta=0.1,
        n_quantiles=100,
        use_softmax=True,
    ):
        self.model = ConformalResidualSamplerNumpy(
            cal_residuals,
            self.reservoir,
            alpha,
            T,
            past_residuals_window,
            eta,
            n_quantiles,
            use_softmax=use_softmax,
        )

        cal_states = self.model.compute_calibration_states(X_cal)
        prev = cal_states[-1].reshape(1, -1)
        cal_states = cal_states / np.linalg.norm(cal_states, axis=1, keepdims=True)

        coverage, width = self.model.compute_intervals(
            X_val,
            y_hat_val,
            y_val,
            previous_ts_state=prev,
        )
        self.lower_bounds = self.model.lower_bounds
        self.upper_bounds = self.model.upper_bounds
        self.coverages = self.model.coverages
        self.widths = self.model.widths
        return coverage, width

    def compute_intervals_univariate(
        self,
        X_cal,
        cal_residuals,
        X_val,
        y_val,
        y_hat_val,
        alpha=0.1,
        T=1,
        past_residuals_window=1000,
        eta=0.1,
        n_quantiles=100,
        similarity="cosine",
        decay="linear",
        decay_rate=0.99,
        in_sweep=False,
    ):
        assert isinstance(
            X_cal, self.default_instance
        ), f"Expected {self.default_instance}, got {type(X_cal)}"
        assert isinstance(
            cal_residuals, self.default_instance
        ), f"Expected {self.default_instance}, got {type(cal_residuals)}"
        assert isinstance(
            X_val, self.default_instance
        ), f"Expected {self.default_instance}, got {type(X_val)}"
        assert isinstance(
            y_val, self.default_instance
        ), f"Expected {self.default_instance}, got {type(y_val)}"
        assert isinstance(
            y_hat_val, self.default_instance
        ), f"Expected {self.default_instance}, got {type(y_hat_val)}"

        if self.backend == "numpy":
            return self.compute_intervals_numpy()
        else:
            self.model = ConformalResidualSamplerLightning(
                self.reservoir,
                cal_residuals,
                alpha,
                T,
                past_residuals_window,
                eta,
                n_quantiles,
                similarity=similarity,
                decay=decay,
                decay_rate=decay_rate,
            )
            cal_states = self.model.model.compute_calibration_states(X_cal)
            prev = cal_states[-1].reshape(1, -1)

            # normalize the calibration states
            self.model.model.cal_states = cal_states / torch.linalg.norm(
                cal_states, dim=1, keepdim=True
            )
            self.cal_states = self.model.model.cal_states

            assert torch.isclose(
                torch.linalg.norm(self.cal_states, dim=1), torch.tensor(1.0)
            ).all(), f"Calibration states are not normalized. Got norms {torch.linalg.norm(self.cal_states, dim=1)} instead."

            coverage, width = self.model.compute_intervals(
                X_val,
                y_hat_val,
                y_val,
                previous_ts_state=prev,
                in_sweep=in_sweep,
            )
            self.lower_bounds = self.model.lower_bounds
            self.upper_bounds = self.model.upper_bounds
            self.coverages = self.model.coverages
            self.widths = self.model.widths
            return coverage, width

    def compute_intervals_univariate_test(
        self,
        X_test,
        y_hat_test,
        y_test,
    ):
        coverage, width = self.model.compute_intervals(
            X_test,
            y_hat_test,
            y_test,
            previous_ts_state=self.model.last_state,
        )
        self.test_lower_bounds = self.model.lower_bounds
        self.test_upper_bounds = self.model.upper_bounds
        self.test_coverages = self.model.coverages
        self.test_widths = self.model.widths
        return coverage, width
