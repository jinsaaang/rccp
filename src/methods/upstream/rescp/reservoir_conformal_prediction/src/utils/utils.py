import torch
import numpy as np


def minimize_intervals(pred_quantiles, y_hat, y, return_quantiles=False):
    if pred_quantiles.ndim == 2:
        lower_quantiles = pred_quantiles[:, : (pred_quantiles.shape[1] // 2)]
        upper_quantiles = pred_quantiles[:, (pred_quantiles.shape[1] // 2) :]

        # Find the quantiles that minimize the interval size
        idxs_min = torch.argmin(upper_quantiles - lower_quantiles, dim=1)

        # select the best quantiles
        best_lower_quantiles = lower_quantiles[torch.arange(len(y_hat)), idxs_min]
        best_upper_quantiles = upper_quantiles[torch.arange(len(y_hat)), idxs_min]

        # reshape to match the shape of y_hat
        best_lower_quantiles = best_lower_quantiles.reshape(y.shape)
        best_upper_quantiles = best_upper_quantiles.reshape(y.shape)

        # compute the lower and upper bounds
        lower_bounds = y_hat + best_lower_quantiles
        upper_bounds = y_hat + best_upper_quantiles
        # lower_bounds = lower_bounds[torch.arange(len(y_hat)), idxs_min].reshape(y.shape)
        # upper_bounds = upper_bounds[torch.arange(len(y_hat)), idxs_min].reshape(y.shape)
    else:
        lower_quantiles = pred_quantiles[:, :, : (pred_quantiles.shape[2] // 2)]
        upper_quantiles = pred_quantiles[:, :, (pred_quantiles.shape[2] // 2) :]
        # n_samples, n_nodes, n_quantiles/2

        # Find the quantiles that minimize the interval size
        idxs_min = torch.argmin(upper_quantiles - lower_quantiles, dim=2)
        # n_samples, n_nodes

        # select the best quantiles
        best_lower_quantiles = lower_quantiles.gather(
            dim=2, index=idxs_min.unsqueeze(2)
        ).squeeze(
            2
        )  # n_samples, n_nodes
        best_upper_quantiles = upper_quantiles.gather(
            dim=2, index=idxs_min.unsqueeze(2)
        ).squeeze(
            2
        )  # n_samples, n_nodes

        # reshape to match the shape of y_hat
        best_lower_quantiles = best_lower_quantiles.reshape(
            y.shape
        )  # n_samples, n_nodes
        best_upper_quantiles = best_upper_quantiles.reshape(
            y.shape
        )  # n_samples, n_nodes

        # compute the lower and upper bounds
        lower_bounds = y_hat + best_lower_quantiles  # n_samples, n_nodes
        upper_bounds = y_hat + best_upper_quantiles  # n_samples, n_nodes
        # lower_bounds = lower_bounds[torch.arange(len(y_hat)), idxs_min].reshape(y.shape)
        # upper_bounds = upper_bounds[torch.arange(len(y_hat)), idxs_min].reshape(y.shape)

    if return_quantiles:
        return lower_bounds, upper_bounds, best_lower_quantiles, best_upper_quantiles
    else:
        return lower_bounds, upper_bounds


def minimize_intervals_numpy(pred_quantiles, y_hat, y):
    # Find the quantiles that minimize the interval size
    lower_quantiles = pred_quantiles[:, : (pred_quantiles.shape[1] // 2)]
    upper_quantiles = pred_quantiles[:, (pred_quantiles.shape[1] // 2) :]
    lower_bounds = y_hat + lower_quantiles
    upper_bounds = y_hat + upper_quantiles
    idxs_min = np.argmin(upper_quantiles - lower_quantiles, axis=1)
    lower_bounds = lower_bounds[np.arange(len(y_hat)), idxs_min].reshape(y.shape)
    upper_bounds = upper_bounds[np.arange(len(y_hat)), idxs_min].reshape(y.shape)
    return lower_bounds, upper_bounds


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def weighted_quantile_torch(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    # values = torch.tensor(values)
    # quantiles = torch.tensor(quantiles)
    if sample_weight is None:
        sample_weight = torch.ones(len(values), device=values.device)
    # sample_weight = torch.tensor(sample_weight)
    assert torch.all(quantiles >= 0) and torch.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = torch.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = torch.cumsum(sample_weight, dim=0) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= torch.sum(sample_weight)
    return torch_interp(quantiles, weighted_quantiles, values)


def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


def winkler_score_torch(lower_bounds, upper_bounds, y, alpha):
    """Compute the Winkler score of the prediction intervals.
    Parameters
    ----------
    lower_bound : np.ndarray
        Lower bounds of the prediction intervals. Shape (n_test,)
    upper_bound : np.ndarray
        Upper bounds of the prediction intervals. Shape (n_test,)
    y : np.ndarray
        True values of the test set. Shape (n_test,)

    Returns
    -------
    winkler_score : float
        Winkler score of the prediction intervals.
    """

    upper_bounds = upper_bounds.squeeze()
    lower_bounds = lower_bounds.squeeze()
    y = y.squeeze()

    widths = upper_bounds - lower_bounds

    above_interval = y > upper_bounds
    below_interval = y < lower_bounds

    penalty = torch.zeros_like(widths, device=upper_bounds.device)
    penalty[above_interval] = (
        2 * (y[above_interval] - upper_bounds[above_interval]) / alpha
    )
    penalty[below_interval] = (
        2 * (lower_bounds[below_interval] - y[below_interval]) / alpha
    )
    winkler_score = widths + penalty
    return torch.mean(winkler_score)


def winkler_score(lower_bounds, upper_bounds, y, alpha):
    """Compute the Winkler score of the prediction intervals.
    Parameters
    ----------
    lower_bound : np.ndarray
        Lower bounds of the prediction intervals. Shape (n_test,)
    upper_bound : np.ndarray
        Upper bounds of the prediction intervals. Shape (n_test,)
    y : np.ndarray
        True values of the test set. Shape (n_test,)

    Returns
    -------
    winkler_score : float
        Winkler score of the prediction intervals.
    """

    upper_bounds = upper_bounds.squeeze()
    lower_bounds = lower_bounds.squeeze()
    y = y.squeeze()

    widths = upper_bounds - lower_bounds

    above_interval = y > upper_bounds
    below_interval = y < lower_bounds

    penalty = np.zeros_like(widths)
    penalty[above_interval] = (
        2 * (y[above_interval] - upper_bounds[above_interval]) / alpha
    )
    penalty[below_interval] = (
        2 * (lower_bounds[below_interval] - y[below_interval]) / alpha
    )
    winkler_score = widths + penalty
    return np.mean(winkler_score)
