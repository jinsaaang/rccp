from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from torch import Tensor
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from tsl.nn.models.base_model import BaseModel


class ARIMAModel(BaseModel):
    r"""ARIMA model for time series forecasting using statsmodels.

    Args:
        order (tuple): The (p, d, q) order of the ARIMA model where:
            - p is the number of autoregressive terms
            - d is the number of differences needed for stationarity
            - q is the number of moving average terms
        seasonal_order (tuple, optional): The (P, D, Q, S) seasonal order of the model.
            Set to (0, 0, 0, 0) for non-seasonal ARIMA.
        enforce_stationarity (bool, optional): Whether to enforce stationarity of the
            autoregressive parameters. (default: True)
        enforce_invertibility (bool, optional): Whether to enforce invertibility of the
            moving average parameters. (default: True)
        trend (str, optional): Parameter controlling the deterministic trend.
            Can be 'n', 'c', 't', 'ct' for no trend, constant, linear trend, or
            constant+linear trend. (default: 'n')
    """

    return_type = Tensor

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ):
        super(ARIMAModel, self).__init__()

        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.delay = None
        self.fitted_models = []  # Store fitted models for each series

    def _fit_series(self, series: np.ndarray, exog: Optional[np.ndarray]) -> ARIMA:
        """Fit ARIMA model to a single time series."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    series,
                    exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )
                fitted_model = model.fit()
                return fitted_model
        except Exception as e:
            # Fallback to simpler model if fitting fails
            print(f"ARIMA fitting failed for series: {e}")
            print("Falling back to AR(1) model with no trend")
            try:
                # Try with no trend first
                model = ARIMA(series, exog=exog, order=(1, 0, 0), trend="n")
                return model.fit()
            except Exception as e2:
                # If still fails, try without exogenous variables
                print(f"AR(1) with exog failed: {e2}")
                print("Falling back to AR(1) without exogenous variables")
                model = ARIMA(series, order=(1, 0, 0), trend="c")
                return model.fit()

    def fit(self, y: Tensor, u: Optional[Tensor] = None) -> "ARIMAModel":
        """Fit ARIMA models to the input time series.

        Note: Only fits models for the target variable (first feature).
        Additional features are used as exogenous variables.

        Args:
            y: Target tensor of shape [batches, time_steps, nodes, 1]
            u: Optional exogenous variables of shape [batches, time_steps, nodes, exog_features]
               or [batches, time_steps, exog_features] for global exogenous variables

        Returns:
            self: The fitted model
        """
        # Convert to numpy for statsmodels
        y_np = y.detach().cpu().numpy()
        n_samples, n_nodes, _ = y_np.shape

        # Process exogenous variables if provided
        u_np = None
        if u is not None:
            u_np = u.detach().cpu().numpy()
            # Local or Global/Local exogenous variables [n_samples, n_nodes, exog_features]
            if u_np.ndim == 3:
                pass  # Already in correct shape
            elif u_np.ndim == 2:
                u_np = np.expand_dims(u_np, axis=1)  # [batches, 1, exog_features]
                u_np = np.repeat(
                    u_np, n_nodes, axis=1
                )  # [batches, nodes, exog_features]

        # Fit ARIMA model for each node
        # Note: We typically only fit models for the target variable (first feature)
        # Other features can be used as exogenous variables
        for node in tqdm(range(n_nodes)):
            # Only fit for the first feature (target variable)
            series = y_np[:, node, 0]  # [n_samples,]

            # Extract exogenous variables for this series
            exog = None
            if u_np is not None:
                exog = u_np[:, node, :]  # [n_samples, exog_features]

            # Skip if series has no variation
            if np.var(series) > 1e-8:
                fitted_model = self._fit_series(
                    series,
                    exog,
                )
                self.fitted_models.append(fitted_model)
            else:
                # For constant series, store the mean value
                self.fitted_models.append(np.mean(series))

        self.start_prediction = n_samples
        return self

    def forward(
        self,
        y: Tensor,
        u: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate forecasts using fitted ARIMA models.

        Note: This method only predicts the target variable (first feature).
        Additional features in y are treated as exogenous variables during fitting.

        Args:
            y: Target tensor of shape [batches, time_steps, nodes, features]
            u: Optional exogenous variables for the input period of shape
               [batches, time_steps, nodes, exog_features] or [batches, time_steps, exog_features]

        Returns:
            Tensor: Forecasted values of shape [batches, horizon, nodes, 1]
                   Only the target variable (first feature) is predicted
        """
        # y: [batches, time_steps, nodes, features]
        # u: [batches, time_steps, (nodes), exog_features]
        y_np = y.detach().cpu().numpy().squeeze(1)  # [n_samples, n_nodes, n_features]
        n_samples, n_nodes, n_features = y_np.shape

        # Process exogenous variables if provided
        u_np = None
        if u is not None:
            u_np = (
                u.detach().cpu().numpy().squeeze(1)
            )  # [n_samples, n_nodes, exog_features]
            # Local or Global/Local exogenous variables [n_samples, n_nodes, exog_features]
            if u_np.ndim == 3:
                pass  # Already in correct shape
            elif u_np.ndim == 2:
                u_np = np.expand_dims(u_np, axis=1)  # [batches, 1, exog_features]
                u_np = np.repeat(
                    u_np, n_nodes, axis=1
                )  # [batches, nodes, exog_features]

        print(y_np.shape, u_np.shape if u_np is not None else None)
        forecasts = np.zeros((y_np.shape[0], y_np.shape[1], 1))

        for node in range(n_nodes):
            fitted_model = self.fitted_models[node]

            if isinstance(fitted_model, (int, float)):
                # Constant series case
                forecasts[:, node, 0] = fitted_model
            else:
                y_np_node = y_np[:, node, 0]  # Target variable for this node
                if u_np is not None:
                    # Use exogenous variables for this node
                    u_np_node = u_np[:, node, :]
                else:
                    u_np_node = None
                try:
                    # Generate forecast
                    fitted_model = fitted_model.append(
                        y_np_node,
                        exog=u_np_node,
                        refit=False,
                    )
                    self.fitted_models[node] = fitted_model  # Update with new model
                    forecast = fitted_model.predict(
                        start=self.start_prediction - self.delay,
                        end=self.start_prediction + n_samples - 1,
                        exog=u_np_node,
                    ).reshape(n_samples + self.delay, 1)[:n_samples]
                    forecasts[:, node, :] = forecast
                except Exception as e:
                    print(f"Forecast failed: {e}")
                    # Fallback to last observed value
                    last_value = y_np[-1, node, 0]
                    forecasts[:, node, 0] = last_value

        self.start_prediction += n_samples

        forecasts[: self.delay + 1, :, :] = y_np[self.delay + 1, :, :]
        # Convert back to tensor
        return torch.tensor(forecasts, dtype=y.dtype, device=y.device).unsqueeze(1)

    def predict(
        self,
        x: Tensor,
        horizon: int = 1,
        u: Optional[Tensor] = None,
        u_future: Optional[Tensor] = None,
    ) -> Tensor:
        """Alias for forward method to maintain compatibility."""
        return self.forward(x, horizon, u, u_future)


class MultiVariateARIMAModel(ARIMAModel):
    r"""ARIMA model that handles multivariate time series by fitting separate
    ARIMA models for each variable.

    This is essentially the same as ARIMAModel but with clearer naming for
    multivariate scenarios.

    Args:
        order (tuple): The (p, d, q) order of the ARIMA model.
        seasonal_order (tuple, optional): The (P, D, Q, S) seasonal order.
        enforce_stationarity (bool, optional): Whether to enforce stationarity.
        enforce_invertibility (bool, optional): Whether to enforce invertibility.
        trend (str, optional): Parameter controlling the deterministic trend.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        trend: str = "n",
    ):
        super(MultiVariateARIMAModel, self).__init__(
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            trend=trend,
        )
