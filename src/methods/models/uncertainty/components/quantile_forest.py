from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from quantile_forest import RandomForestQuantileRegressor


@dataclass
class QuantileForestPrediction:
    point: Optional[np.ndarray]
    interval: np.ndarray


class QuantileRegressionForestCompat:
    """
    Small compatibility wrapper for the old `doubt.QuantileRegressionForest` API.

    SPIC only needs:
    - `fit(X, y)`
    - `predict(X, alpha)` returning `(point, [[low, high], ...])`

    We keep the return signature stable so the rest of the legacy SPIC code does
    not need to know which backend implementation is used.
    """

    def __init__(self, **kwargs):
        self._model = RandomForestQuantileRegressor(**kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self._model.fit(X, y)
        return self

    def predict(self, X, alpha):
        X = np.asarray(X)
        alpha = float(alpha)
        alpha = min(max(alpha, 1e-6), 1 - 1e-6)
        q_low = alpha / 2.0
        q_high = 1.0 - (alpha / 2.0)
        low = np.asarray(self._model.predict(X, quantiles=q_low))
        high = np.asarray(self._model.predict(X, quantiles=q_high))
        point = np.asarray(self._model.predict(X, quantiles=0.5))
        interval = np.stack((low, high), axis=1)
        return point, interval
