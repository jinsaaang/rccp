import os
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import Literal

from tsl.datasets.prototypes import DatetimeDataset


class ElectricityRome(DatetimeDataset):
    """
    Electricity consumption coming from a backbone of the energy supply network in the city of Rome.
    The time series is sampled every 10 minutes.
    The dataset has been originally presented in [Bianchi, Filippo Maria, et al. "Short-term electric load forecasting using echo state networks and PCA decomposition." Ieee Access 3 (2015)](https://doi.org/10.1109/ACCESS.2015.2485943).
    """

    similarity_options = [None]

    def __init__(self, root: str = "../data/elec_rome", freq: str = None, **kwargs):
        self.root = root
        self.temporal_aggregation = "nearest"

        df, mask = self.load()
        super().__init__(
            target=df,
            mask=mask,
            freq=freq,
            temporal_aggregation=self.temporal_aggregation,
            name="ElectricityRome",
        )

    def load(self):
        df = self.load_raw()
        mask = df["consumption"].notna().to_numpy().astype("uint8")
        df["datetime"] = pd.date_range(
            start="2020-01-01", periods=len(df), freq="10min"
        )
        df = df.set_index("datetime")
        return df, mask

    def load_raw(self, *args, **kwargs):
        self.maybe_build()
        path = os.path.join(self.root, "Elec_Rome.npz")
        data = np.load(path)["X"]
        df = pd.DataFrame(data=data, columns=["consumption"])
        return df

    def resample(
        self,
        freq=None,
        aggr: str = None,
        keep: Literal["first", "last", False] = "first",
        mask_tolerance: float = 0.0,
    ) -> "DatetimeDataset":
        """"""
        self_copy = deepcopy(self)
        self_copy.resample_(freq, aggr, keep, mask_tolerance)
        return self_copy
