"""Central copies of ResCP-style dataset loaders.

These loaders preserve the original ResCP data layout without importing the
upstream repository at runtime: target is kept as ``x`` and covariates/time
features are kept as ``u`` for node-batched forecasters.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


RESCP_REPRO_DATASETS = {"rescp-air-10", "rescp-beijing-air-pm10"}


def load_rescp_repro_data(dataset_type: str, data_path: str | Path, **kwargs):
    if dataset_type in RESCP_REPRO_DATASETS:
        return load_rescp_beijing_air_data(data_path, target="PM10", **kwargs)
    raise ValueError(f"Dataset {dataset_type} not supported by ResCP reproduction loader.")


def load_rescp_beijing_air_data(
    data_path: str | Path,
    target: str = "PM10",
    wd_encode: str = "encode",
    include_temporal: bool = True,
):
    df = _process_all_air_quality_stations(
        data_path,
        air10=(target == "PM10"),
        wd_encode=wd_encode,
    )
    target_df, covariates = _separate_target_and_covariates(df, target)
    mask = _target_mask(target_df, target)
    features = _rescp_covariate_tensor(target_df.index, covariates, include_temporal=include_temporal)
    target_values = target_df.to_numpy(dtype=np.float32).reshape(target_df.shape[0], -1, 1)
    stations = list(target_df.columns.get_level_values(0).unique())

    table_combos = []
    for node_idx, station in enumerate(stations):
        table_combos.append(
            (
                torch.from_numpy(features[:, node_idx, :]).float(),
                torch.from_numpy(target_values[:, node_idx, 0]).float(),
                f"rescp-air-10-{station}",
                torch.from_numpy(mask[:, node_idx]).float(),
            )
        )
    return table_combos


def _process_all_air_quality_stations(
    data_dir: str | Path,
    air10: bool = True,
    wd_encode: str = "encode",
) -> pd.DataFrame:
    files = [
        file
        for file in glob.glob(os.path.join(str(data_dir), "*.csv"))
        if "PRSA" in file
    ]

    station_dict = {}
    for file in files:
        station_name = (
            os.path.basename(file)
            .split(".")[0]
            .replace("PRSA_Data_", "")
            .split("_")[0]
        )
        data = pd.read_csv(file)
        data["datetime"] = pd.to_datetime(data[["year", "month", "day", "hour"]])
        data = data.drop(columns=["year", "month", "day", "hour", "No", "station"], axis=1)
        data = data.set_index("datetime")

        if air10:
            data = data.drop(columns=["PM2.5"], axis=1)
        else:
            data = data.drop(columns=["PM10"], axis=1)

        if wd_encode == "drop":
            data.drop(columns=["wd"], inplace=True, axis=1)
        elif wd_encode == "one-hot":
            data = pd.get_dummies(data, columns=["wd"])
        elif wd_encode == "encode":
            data["wd"] = data["wd"].fillna(value="Unknown")
            data["wd_h"] = data["wd"].apply(lambda x: _encode_direction(x, True))
            data["wd_v"] = data["wd"].apply(lambda x: _encode_direction(x, False))
            data.drop(columns=["wd"], inplace=True, axis=1)
        else:
            raise ValueError(f"Unsupported wd_encode: {wd_encode}")

        station_dict[station_name] = data
    return _create_multiindex_for_multiple_stations(station_dict)


def _create_multiindex_for_multiple_stations(station_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_columns = set()
    for df in station_data_dict.values():
        all_columns.update(df.columns)
    common_columns = list(all_columns)

    common_index = None
    for df in station_data_dict.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    result_data = {}
    for station_name, df in station_data_dict.items():
        df_aligned = df.reindex(common_index)
        for channel in common_columns:
            if channel in df_aligned.columns:
                result_data[(station_name, channel)] = df_aligned[channel]
            else:
                result_data[(station_name, channel)] = np.nan

    result_df = pd.DataFrame(result_data, index=common_index)
    result_df.columns = pd.MultiIndex.from_tuples(result_df.columns, names=["node", "channel"])
    return result_df


def _separate_target_and_covariates(df: pd.DataFrame, target_channel: str):
    stations = df.columns.get_level_values(0).unique()
    channels = df.columns.get_level_values(1).unique()

    target_data = {}
    for station in stations:
        if target_channel in df[station].columns:
            target_data[(station, target_channel)] = df[station][target_channel]
    target_df = pd.DataFrame(target_data, index=df.index)
    target_df.columns = pd.MultiIndex.from_tuples(target_df.columns, names=["node", "channel"])

    covariates = {}
    covariate_channels = [ch for ch in channels if ch != target_channel]
    for cov_channel in covariate_channels:
        cov_data = {}
        for station in stations:
            if cov_channel in df[station].columns:
                cov_data[(station, cov_channel)] = df[station][cov_channel]
        if cov_data:
            cov_df = pd.DataFrame(cov_data, index=df.index)
            cov_df.columns = pd.MultiIndex.from_tuples(cov_df.columns, names=["node", "channel"])
            covariates[cov_channel] = cov_df.ffill().bfill()

    target_df = target_df.ffill().bfill()
    return target_df, covariates


def _target_mask(target_df: pd.DataFrame, target: str) -> np.ndarray:
    masks = []
    for station in target_df.columns.get_level_values(0).unique():
        masks.append(target_df[station][target].notna().to_numpy().astype("uint8"))
    return np.stack(masks, axis=1)


def _rescp_covariate_tensor(
    datetime_index: Iterable[pd.Timestamp],
    covariates: dict[str, pd.DataFrame],
    include_temporal: bool,
) -> np.ndarray:
    local_features = []
    n_nodes = None
    for cov_df in covariates.values():
        values = cov_df.to_numpy(dtype=np.float32)
        if n_nodes is None:
            n_nodes = len(cov_df.columns.get_level_values(0).unique())
        local_features.append(values.reshape(values.shape[0], n_nodes, -1))
    if not local_features:
        raise ValueError("ResCP Beijing loader requires local covariates.")
    features = [np.concatenate(local_features, axis=-1)]
    if include_temporal:
        temporal = _datetime_features(datetime_index).astype(np.float32)
        temporal = np.repeat(temporal[:, None, :], n_nodes, axis=1)
        features.append(temporal)
    return np.concatenate(features, axis=-1).astype(np.float32)


def _datetime_features(datetime_index: Iterable[pd.Timestamp]) -> np.ndarray:
    index = pd.DatetimeIndex(datetime_index)
    seconds = (
        index.hour.to_numpy() * 3600
        + index.minute.to_numpy() * 60
        + index.second.to_numpy()
    ).astype(np.float32)
    day_angle = 2 * np.pi * seconds / (24 * 3600)
    day_features = np.stack([np.sin(day_angle), np.cos(day_angle)], axis=1)

    weekday = index.weekday.to_numpy()
    weekday_features = np.zeros((len(index), 7), dtype=np.float32)
    weekday_features[np.arange(len(index)), weekday] = 1.0
    return np.concatenate([day_features, weekday_features], axis=1)


def _encode_direction(direction, horizontal):
    if horizontal:
        if direction in ["N", "S"]:
            return 0
        if direction in ["NNW", "SSW"]:
            return -0.5
        if direction in ["NW", "SW"]:
            return -0.7
        if direction in ["WNW", "WSW"]:
            return -0.86
        if direction == "W":
            return -1
        if direction in ["NNE", "SSE"]:
            return 0.5
        if direction in ["NE", "SE"]:
            return 0.7
        if direction in ["ENE", "ESE"]:
            return 0.86
        if direction == "E":
            return 1
    else:
        if direction in ["W", "E"]:
            return 0
        if direction in ["WSW", "ESE"]:
            return -0.5
        if direction in ["SW", "SE"]:
            return -0.7
        if direction in ["SSW", "SSE"]:
            return -0.86
        if direction == "S":
            return -1
        if direction in ["WNW", "ENE"]:
            return 0.5
        if direction in ["NW", "NE"]:
            return 0.7
        if direction in ["NNW", "NNE"]:
            return 0.86
        if direction == "N":
            return 1
    if direction == "Unknown":
        return 0
    raise ValueError("Invalid Dir")
