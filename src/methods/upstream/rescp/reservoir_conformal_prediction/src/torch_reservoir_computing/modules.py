import torch
import numpy as np
import time

from reservoir_conformal_prediction.src.torch_reservoir_computing.reservoir import (
    Reservoir,
)


class RC_forecaster_torch(object):
    r"""Class to perform time series forecasting with RC.

    The training and test data are multidimensional arrays of shape ``[T,V]``, with

    - ``T`` = number of time steps in each sample,
    - ``V`` = number of variables in each sample.

    Given a time series ``X``, the training data are supposed to be as follows:

        ``Xtr, Ytr = X[0:-forecast_horizon,:], X[forecast_horizon:,:]``

    Once trained, the model can be used to compute prediction ``forecast_horizon`` steps ahead:

            ``Yhat[t,:] = Xte[t+forecast_horizon,:]``

    **Reservoir parameters:**

    :param reservoir: object of class ``Reservoir`` (default ``None``)
        Precomputed reservoir. If ``None``, the following structural hyperparameters must be specified.
    :param n_internal_units: int (default ``100``)
        Processing units in the reservoir.
    :param spectral_radius: float (default ``0.99``)
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    :param leak: float (default ``None``)
        Amount of leakage in the reservoir state update.
    :param connectivity: float (default ``0.3``)
        Percentage of nonzero connection weights.
    :param input_scaling: float (default ``0.2``)
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    :param noise_level: float (default ``0.0``)
        Standard deviation of the Gaussian noise injected in the state update.
    :param n_drop: int (default ``0``)
        Number of transient states to drop.
    :param circle: bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection
        has the same weight.

    **Dimensionality reduction parameters:**

    :param dimred_method: str (default ``None``)
        Procedure for reducing the number of features in the sequence of reservoir states.
        Possible options are: ``None`` (no dimensionality reduction), ``'pca'`` (standard PCA),
        or ``'tenpca'`` (tensorial PCA for multivariate time series data).
    :param n_dim: int (default ``None``)
        Number of resulting dimensions after the dimensionality reduction procedure.

    **Readout parameters:**

    :param w_ridge: float (default ``1.0``)
        Regularization parameter of the ridge regression readout.
    """

    def __init__(
        self,
        # reservoir
        reservoir=None,
        n_internal_units=100,
        spectral_radius=0.99,
        leak=None,
        connectivity=0.3,
        input_scaling=0.2,
        noise_level=0.0,
        n_drop=0,
        circle=False,
        # dim red
        dimred_method=None,
        n_dim=None,
        # readout
        w_ridge=1.0,
        seed=0,
    ):
        self.n_drop = n_drop
        self.dimred_method = dimred_method

        # Initialize reservoir
        if reservoir is None:
            self._reservoir = Reservoir(
                n_internal_units=n_internal_units,
                spectral_radius=spectral_radius,
                leak=leak,
                connectivity=connectivity,
                input_scaling=input_scaling,
                noise_level=noise_level,
                circle=circle,
                seed=seed,
            )
        else:
            self._reservoir = reservoir

        # Initialize dimensionality reduction method
        if dimred_method is not None:
            if dimred_method.lower() == "pca":
                # self._dim_red = PCA(n_components=n_dim)
                pass
            else:
                raise RuntimeError("Invalid dimred method ID")

        # Initialize readout
        self.w_ridge = w_ridge
        self.readout = None

    def fit(self, X, Y, verbose=True):
        """Train the RC model for forecasting.

        Parameters:
        -----------
        X : np.ndarray
            Array of shape ``[T, V]`` representing the training data.

        Y : np.ndarray
            Array of shape ``[T, V]`` representing the target values.

        verbose : bool
            If ``True``, print the training time.

        Returns:
        --------
        red_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the time steps used for training.
        """

        time_start = time.time()

        # ============ Compute reservoir states ============
        res_states = self._reservoir.get_states(
            X[None, :, :], n_drop=self.n_drop, bidir=False
        )

        # ============ Dimensionality reduction of the reservoir states ============
        if self.dimred_method is not None:
            if self.dimred_method.lower() == "pca":
                # fit
                self.U, self.s, self.V = torch.pca_lowrank(res_states[0], center=True)
                # transform
                red_states = torch.mm(res_states[0], self.V[:, : self.n_dim])
        else:  # Skip dimensionality reduction
            red_states = res_states[0]

        self._fitted_states = red_states
        print(red_states.shape)

        # ============ Train readout ============
        self.readout = (
            (
                red_states.T @ red_states
                + self.w_ridge
                * torch.eye(red_states.shape[1], device=red_states.device)
            ).inverse()
            @ red_states.T
            @ Y[self.n_drop :, :]
        )
        print(self.readout.shape)

        if verbose:
            tot_time = (time.time() - time_start) / 60
            print(f"Training completed in {tot_time:.2f} min")

        return red_states

    def predict(self, Xte, return_states=False):
        r"""Computes predictions for out-of-sample (test) data.

        Parameters:
        -----------
        Xte : np.ndarray
            Array of shape ``[T, V]`` representing the test data.

        return_states : bool
            If ``True``, return the predicted states.

        Returns:
        --------
        Yhat : np.ndarray
            Array of shape ``[T, V]`` representing the predicted values.

        red_states_te : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the new time steps.
        """

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(
            Xte[None, :, :], n_drop=self.n_drop, bidir=False
        )

        # ============ Dimensionality reduction of the reservoir states ============
        if self.dimred_method is not None:
            if self.dimred_method.lower() == "pca":
                red_states_te = torch.mm(res_states_te[0], self.V[:, : self.n_dim])
        else:  # Skip dimensionality reduction
            red_states_te = res_states_te[0]

        self._predicted_states = red_states_te

        # ============ Apply readout ============
        Yhat = red_states_te @ self.readout

        if return_states:
            return Yhat, red_states_te
        return Yhat

    def get_fitted_states(self):
        r"""Return the fitted reservoir states.

        Returns:
        --------
        fitted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the fitted reservoir states.
        """
        return self._fitted_states

    def get_predicted_states(self):
        r"""Return the predicted reservoir states.

        Returns:
        --------
        predicted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the predicted reservoir states.
        """
        return self._predicted_states

    def to(self, device):
        r"""Move the model to the specified device.

        Parameters:
        -----------
        device : str or torch.device
            The device to move the model to (e.g., "cpu", "cuda").
        """
        self._reservoir.to(device)
        if self.dimred_method is not None:
            if self.dimred_method.lower() == "pca":
                self.U = self.U.to(device)
                self.s = self.s.to(device)
                self.V = self.V.to(device)

        return self
