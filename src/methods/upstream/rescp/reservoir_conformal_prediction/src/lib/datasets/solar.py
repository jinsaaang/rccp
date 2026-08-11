import os
import glob
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import Literal, Union, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from tsl.datasets.prototypes import DatetimeDataset

from src.lib.utils.data_utils import create_multiindex_for_multiple_stations


class Solar(DatetimeDataset):
    """
    Solar dataset.
    Contains half-hourly measurements of solar radiation and meteorological data.
    """

    similarity_options = [
        "correlation",
        "cosine",
        "mutual_information",
        "dtw",
        "combined_meteorological",
    ]

    def __init__(
        self,
        root: str = "../data/solar",
        freq=None,
        include_covariates=True,
    ):
        self.root = root
        self.target_channel = "dhi"
        self.include_covariates = include_covariates
        self.temporal_aggregation = "nearest"

        if include_covariates:
            print("Loading dataset with covariates...")
            df, mask, self.__covariates = self.load(
                target=self.target_channel, freq=freq, return_covariates=True
            )
            print(f"Covariates loaded: {list(self.__covariates.keys())}\n")
        else:
            print("Loading dataset without covariates...")
            df, mask = self.load(
                target=self.target_channel, freq=freq, return_covariates=False
            )
            self.__covariates = None
            print("No covariates loaded.\n")

        super().__init__(
            target=df,
            mask=mask,
            freq=freq,
            similarity_score="correlation",  # Default similarity method
            temporal_aggregation=self.temporal_aggregation,
            name="Solar",
        )

    # Process all solar station files
    def process_all_solar_stations(
        self,
        data_dir="../data/solar/",
    ):
        """
        Process all solar station files and create MultiIndex structure.

        Args:
            data_dir: Directory containing CSV files
            air10: If True, keep PM10 and drop PM2.5; if False, keep PM2.5 and drop PM10
            wd_encode: How to handle wind direction ('encode', 'one-hot', 'drop')

        Returns:
            DataFrame with MultiIndex columns (node=station, channel=measurement)
        """

        files = [file for file in glob.glob(os.path.join(data_dir, "*.csv"))]
        print(f"Found {len(files)} files in {data_dir}")

        station_dict = {}
        for file in files:
            # Extract station name from filename
            station_name = file.split("/")[-1].split(".")[0]

            # Load data
            data = pd.read_csv(file)

            # Create datetime index
            datetime = pd.to_datetime(data[["year", "month", "day", "hour", "minute"]])
            data["datetime"] = datetime
            data = data.drop(
                columns=["Unnamed: 0", "year", "month", "day", "hour", "minute"],
                axis=1,
            )
            data = data.set_index("datetime")

            station_dict[station_name] = data

        # Create MultiIndex structure
        df = create_multiindex_for_multiple_stations(station_dict)
        return df

    def load_raw(self):
        """
        Load the dataset, returning the target DataFrame and mask.
        """
        self.maybe_build()
        df = self.process_all_solar_stations(
            data_dir=self.root,
        )

        return df

    def load(self, target="dhi", freq=None, return_covariates=True):
        """
        Load the dataset, returning the target DataFrame, mask, and optionally covariates.

        Args:
            target: Primary target variable (e.g., "PM10")
            wd_encode: Wind direction encoding method
            return_covariates: If True, return covariates separately

        Returns:
            If return_covariates=False: (target_df, mask)
            If return_covariates=True: (target_df, mask, covariates_dict)
        """
        df = self.load_raw()
        if freq is not None:
            df = df.resample(freq).apply(self.temporal_aggregation)

        # Separate target from covariates
        target_df, covariates_dict = self._separate_target_and_covariates(df, target)

        # Create mask for target variable
        masks = []
        for station in target_df.columns.get_level_values(0).unique():
            station_data = target_df[station]
            masks.append(station_data[target].notna().to_numpy().astype("uint8"))

        mask = np.stack(masks, axis=1)

        if return_covariates:
            return target_df, mask, covariates_dict
        else:
            return target_df, mask

    def _separate_target_and_covariates(self, df, target_channel):
        """
        Separate the primary target channel from other channels (covariates).

        Args:
            df: MultiIndex DataFrame with (station, channel) structure
            target_channel: Name of the primary target channel

        Returns:
            target_df: DataFrame with only target channel
            covariates_dict: Dictionary of covariate DataFrames
        """
        stations = df.columns.get_level_values(0).unique()
        channels = df.columns.get_level_values(1).unique()

        # Create target DataFrame (only primary channel)
        target_data = {}
        for station in stations:
            if target_channel in df[station].columns:
                target_data[(station, target_channel)] = df[station][target_channel]

        target_df = pd.DataFrame(target_data, index=df.index)
        target_df.columns = pd.MultiIndex.from_tuples(
            target_df.columns, names=["node", "channel"]
        )

        # Create covariates dictionary
        covariates_dict = {}
        covariate_channels = [ch for ch in channels if ch != target_channel]

        for cov_channel in covariate_channels:
            cov_data = {}
            for station in stations:
                if cov_channel in df[station].columns:
                    cov_data[(station, cov_channel)] = df[station][cov_channel]

            if cov_data:  # Only add if there's data
                cov_df = pd.DataFrame(cov_data, index=df.index)
                cov_df.columns = pd.MultiIndex.from_tuples(
                    cov_df.columns, names=["node", "channel"]
                )
                covariates_dict[cov_channel] = cov_df

        # Fill missing values in all DataFrames
        target_df = target_df.ffill().bfill()
        for key in covariates_dict:
            covariates_dict[key] = covariates_dict[key].ffill().bfill()

        return target_df, covariates_dict

    def get_covariates_for_tsl(self):
        """
        Get covariates in a format suitable for TSL SpatioTemporalDataset.

        Returns:
            dict: Dictionary where keys are covariate names and values are
                  (DataFrame, pattern) tuples for TSL
        """
        if self.__covariates is None:
            return {}

        tsl_covariates = {}
        for cov_name, cov_df in self.__covariates.items():
            # For TSL, we need to specify the pattern
            # 't n f' means time x nodes x features
            tsl_covariates[cov_name] = (cov_df, "t n f")

        return tsl_covariates

    def get_target_only(self):
        """
        Get only the target DataFrame (primary channel).

        Returns:
            DataFrame: Target data with MultiIndex (node, channel)
        """
        return self.dataframe()

    def get_covariate_channels(self):
        """
        Get list of available covariate channel names.

        Returns:
            list: Names of covariate channels
        """
        if self.__covariates is None:
            return []
        return list(self.__covariates.keys())

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

    def compute_similarity(self, method: str = "correlation", **kwargs) -> np.ndarray:
        """
        Compute similarity matrix between stations for adjacency matrix construction.

        Args:
            method: Similarity method to use. Options:
                   - "correlation": Pearson correlation of target time series
                   - "cosine": Cosine similarity of target time series
                   - "mutual_information": Mutual information between stations
                   - "dtw": Dynamic Time Warping similarity (requires dtaidistance)
                   - "combined_meteorological": Weighted combination of meteorological features
            **kwargs: Additional parameters for specific methods

        Returns:
            np.ndarray: Similarity matrix (n_stations x n_stations)
        """
        if method not in self.similarity_options:
            raise ValueError(
                f"Method {method} not supported. Choose from {self.similarity_options}"
            )

        target_df = self.get_target_only()
        stations = target_df.columns.get_level_values(0).unique()
        n_stations = len(stations)

        if method == "correlation":
            return self._compute_correlation_similarity(target_df, stations)
        elif method == "cosine":
            return self._compute_cosine_similarity(target_df, stations)
        elif method == "mutual_information":
            return self._compute_mutual_information_similarity(
                target_df, stations, **kwargs
            )
        elif method == "dtw":
            return self._compute_dtw_similarity(target_df, stations, **kwargs)
        elif method == "combined_meteorological":
            return self._compute_combined_meteorological_similarity(stations, **kwargs)
        else:
            raise NotImplementedError(f"Method {method} not implemented yet")

    def _compute_correlation_similarity(
        self, target_df: pd.DataFrame, stations: list, **kwargs: Any
    ) -> np.ndarray:
        """
        Compute Pearson correlation similarity matrix.

        Args:
            target_df: Target DataFrame with MultiIndex columns
            stations: List of station names

        Returns:
            np.ndarray: Correlation similarity matrix
        """
        n_stations = len(stations)
        similarity_matrix = np.ones((n_stations, n_stations))

        for i, station_i in enumerate(stations):
            for j, station_j in enumerate(stations):
                if i != j:
                    series_i = target_df[station_i, self.target_channel].dropna()
                    series_j = target_df[station_j, self.target_channel].dropna()

                    # Find common time indices
                    common_idx = series_i.index.intersection(series_j.index)
                    if len(common_idx) > 10:  # Minimum threshold for correlation
                        corr, _ = pearsonr(
                            series_i.loc[common_idx], series_j.loc[common_idx]
                        )
                        similarity_matrix[i, j] = (
                            abs(corr) if not np.isnan(corr) else 0.0
                        )
                    else:
                        similarity_matrix[i, j] = 0.0

        return similarity_matrix

    def _compute_cosine_similarity(
        self, target_df: pd.DataFrame, stations: list, **kwargs: Any
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix.

        Args:
            target_df: Target DataFrame with MultiIndex columns
            stations: List of station names

        Returns:
            np.ndarray: Cosine similarity matrix
        """
        # Prepare data matrix (stations x time)
        data_matrix = []
        for station in stations:
            series = target_df[station, self.target_channel].fillna(
                0
            )  # Fill NaN with 0
            data_matrix.append(series.values)

        data_matrix = np.array(data_matrix)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(data_matrix)

        return similarity_matrix

    def _compute_mutual_information_similarity(
        self, target_df: pd.DataFrame, stations: list, bins: int = 50, **kwargs: Any
    ) -> np.ndarray:
        """
        Compute mutual information similarity matrix.

        Args:
            target_df: Target DataFrame with MultiIndex columns
            stations: List of station names
            bins: Number of bins for discretization

        Returns:
            np.ndarray: Mutual information similarity matrix
        """
        from sklearn.feature_selection import mutual_info_regression

        n_stations = len(stations)
        similarity_matrix = np.ones((n_stations, n_stations))

        for i, station_i in enumerate(stations):
            for j, station_j in enumerate(stations):
                if i != j:
                    series_i = target_df[station_i, self.target_channel].dropna()
                    series_j = target_df[station_j, self.target_channel].dropna()

                    # Find common time indices
                    common_idx = series_i.index.intersection(series_j.index)
                    if len(common_idx) > 50:  # Minimum threshold for MI
                        x = series_i.loc[common_idx].values.reshape(-1, 1)
                        y = series_j.loc[common_idx].values

                        mi = mutual_info_regression(x, y, discrete_features=False)[0]
                        similarity_matrix[i, j] = mi
                    else:
                        similarity_matrix[i, j] = 0.0

        # Normalize to [0,1]
        max_mi = np.max(similarity_matrix)
        if max_mi > 0:
            similarity_matrix = similarity_matrix / max_mi

        return similarity_matrix

    def _compute_dtw_similarity(
        self,
        target_df: pd.DataFrame,
        stations: list,
        window_size: int = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute Dynamic Time Warping similarity matrix.

        Args:
            target_df: Target DataFrame with MultiIndex columns
            stations: List of station names
            window_size: DTW window constraint

        Returns:
            np.ndarray: DTW similarity matrix
        """
        try:
            from dtaidistance import dtw
        except ImportError:
            raise ImportError(
                "dtaidistance package required for DTW similarity. Install with: pip install dtaidistance"
            )

        n_stations = len(stations)
        similarity_matrix = np.ones((n_stations, n_stations))

        # Compute DTW distances
        for i, station_i in enumerate(stations):
            for j, station_j in enumerate(stations):
                if i != j:
                    series_i = (
                        target_df[station_i, self.target_channel]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                    series_j = (
                        target_df[station_j, self.target_channel]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )

                    # Use a subset for DTW computation (it's expensive)
                    subset_size = min(1000, len(series_i))
                    s1 = series_i.iloc[:subset_size].values
                    s2 = series_j.iloc[:subset_size].values

                    dtw_distance = dtw.distance(s1, s2, window=window_size)
                    # Convert distance to similarity (inverse relationship)
                    similarity_matrix[i, j] = 1.0 / (1.0 + dtw_distance)

        return similarity_matrix

    def _compute_combined_meteorological_similarity(
        self, stations: list, weights: Dict[str, float] = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Compute similarity based on combined meteorological features.

        Args:
            stations: List of station names
            weights: Weights for different meteorological variables

        Returns:
            np.ndarray: Combined meteorological similarity matrix
        """
        if weights is None:
            weights = {
                "dhi": 0.3,
                "dni": 0.2,
                "air_temperature": 0.2,
                "wind_speed": 0.1,
                "relative_humidity": 0.1,
                "solar_zenith_angle": 0.1,
            }

        # Get the full dataset with all variables
        full_df = self.load_raw()

        n_stations = len(stations)
        similarity_matrix = np.ones((n_stations, n_stations))

        # Available meteorological variables
        available_vars = set()
        for station in stations:
            if station in full_df.columns.get_level_values(0):
                available_vars.update(full_df[station].columns)

        # Filter weights to only include available variables
        filtered_weights = {k: v for k, v in weights.items() if k in available_vars}

        if not filtered_weights:
            print(
                "Warning: No meteorological variables found. Falling back to correlation similarity."
            )
            return self._compute_correlation_similarity(
                self.get_target_only(), stations
            )

        # Normalize weights
        total_weight = sum(filtered_weights.values())
        filtered_weights = {k: v / total_weight for k, v in filtered_weights.items()}

        # Compute weighted similarity
        for var_name, weight in filtered_weights.items():
            var_similarity = np.ones((n_stations, n_stations))

            for i, station_i in enumerate(stations):
                for j, station_j in enumerate(stations):
                    if (
                        i != j
                        and station_i in full_df.columns.get_level_values(0)
                        and station_j in full_df.columns.get_level_values(0)
                    ):
                        if (
                            var_name in full_df[station_i].columns
                            and var_name in full_df[station_j].columns
                        ):
                            series_i = full_df[station_i, var_name].dropna()
                            series_j = full_df[station_j, var_name].dropna()

                            # Find common time indices
                            common_idx = series_i.index.intersection(series_j.index)
                            if len(common_idx) > 10:
                                corr, _ = pearsonr(
                                    series_i.loc[common_idx], series_j.loc[common_idx]
                                )
                                var_similarity[i, j] = (
                                    abs(corr) if not np.isnan(corr) else 0.0
                                )
                            else:
                                var_similarity[i, j] = 0.0

            # Add weighted contribution
            if weight > 0:
                similarity_matrix = (
                    similarity_matrix * (1 - weight) + var_similarity * weight
                )

        return similarity_matrix

    def get_adjacency_matrix(
        self,
        method: str = "correlation",
        threshold: float = 0.5,
        k_neighbors: int = None,
        binary: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute adjacency matrix from similarity matrix.

        Args:
            method: Similarity method to use
            threshold: Similarity threshold for binary adjacency (if binary=True)
            k_neighbors: If specified, keep only top-k neighbors for each node
            binary: Whether to return binary adjacency matrix
            **kwargs: Additional parameters for similarity computation

        Returns:
            np.ndarray: Adjacency matrix
        """
        similarity_matrix = self.compute_similarity_matrix(method=method, **kwargs)

        # Remove self-loops (set diagonal to 0)
        np.fill_diagonal(similarity_matrix, 0)

        if k_neighbors is not None:
            # Keep only top-k neighbors for each node
            adjacency_matrix = np.zeros_like(similarity_matrix)
            for i in range(similarity_matrix.shape[0]):
                # Get indices of top-k neighbors
                top_k_indices = np.argsort(similarity_matrix[i])[-k_neighbors:]
                adjacency_matrix[i, top_k_indices] = similarity_matrix[i, top_k_indices]

            # Make symmetric (if i is neighbor of j, then j is neighbor of i)
            adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        else:
            adjacency_matrix = similarity_matrix.copy()

        if binary:
            # Apply threshold to create binary adjacency matrix
            adjacency_matrix = (adjacency_matrix >= threshold).astype(float)

        return adjacency_matrix
