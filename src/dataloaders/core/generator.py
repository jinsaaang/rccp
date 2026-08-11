from collections import OrderedDict
from pathlib import Path, PureWindowsPath
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch

from enbPI.wrappers.enbPI_loader import EnbPI_DATASETS, demo_code_data_load
from config import TSDataConfig, TaskConfig
from loader.dataset import ChronoSplittedTsDataset, SimpleTsDataset, HydroDataset
from loader.jena_airpress import get_jena_data, JENA_TYPE_PREFIX
from loader.rescp_repro import RESCP_REPRO_DATASETS, load_rescp_repro_data
from loader.sapflux import SAPFLUX_TYPE_PREFIX, get_sapflux_data
from loader.toy_loader import TOY_DATA_PREFIX, get_toy_data
try:
    from loader.hydrology import HYDRO_TYPE_PREFIX, get_hydro_data
except Exception:
    # Hydrology support is optional for non-hydro experiments.
    HYDRO_TYPE_PREFIX = "hydro"
    get_hydro_data = None

NSDB_DATASET = ['nsdb-60m-2020', 'nsdb-60m-2019', 'nsdb-60m-2018-20', 'nsdb-30m-2020', 'nsdb-30m-2019',
                'nsdb-30m-2018-20', 'nsdb-60m-2019-wCoord', 'nsdb-60m-2018-20-wCoord']
AIR_QUALITY = ['air-25', 'air-10', 'air-25-half', 'air-10-half']
LOCAL_TABULAR_DATASETS = [
    'wind',
    'weather',
    'exchange-rate',
    'exchange-rate-ot1',
    'exchange-rate-ot1-lagged',
    'amazon',
    'stock-amzn',
    'apple',
    'stock-aapl',
    'google',
    'stock-goog',
    'elec-normalized',
    'electricity-normalized',
]
LOCAL_TABULAR_MULTI_DATASETS = ['exchange-rate-multi', 'exchange-rate-rescp']


def get_ts_id(dataset_type, data_path):
    if dataset_type is None:
        return str(dataset_type)
    path_str = str(data_path)
    name = Path(path_str).name
    if "\\" in name or name == path_str:
        name = PureWindowsPath(path_str).name
    stem = name.rsplit(".", 1)[0]
    return str(dataset_type) + stem


class DataGenerator:

    @staticmethod
    def get_data(data_config: TSDataConfig, task_config: TaskConfig, replace_base_dir,
                 X_norm_param = None, Y_norm_param = None, hydro_static_norm_param=None) -> [ChronoSplittedTsDataset]:
        datasets = []
        if data_config.dataset_type.startswith(TOY_DATA_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_toy_data(data_config.dataset_type, p)  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config))
        elif data_config.dataset_type.startswith(JENA_TYPE_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_jena_data(data_config.dataset_type, p)  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config))
        elif data_config.dataset_type.startswith(SAPFLUX_TYPE_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_sapflux_data(data_config.dataset_type, p)  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config))
        elif data_config.dataset_type.startswith(HYDRO_TYPE_PREFIX):
            if get_hydro_data is None:
                raise ModuleNotFoundError(
                    "Hydrology dataset requested but hydrology dependencies are unavailable."
                )
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            # Returns a list of (X, Y, id, number of the FC model's train samples) Tuples and indices of the
            # static attributes.
            table_combos, attribute_indices, attribute_norm_params =\
                get_hydro_data(data_config.dataset_type, p, hydro_static_norm_param=hydro_static_norm_param,
                               **data_config.add_config)
            for X_full, Y_full, ts_id, train_steps, calib_steps in table_combos:
                # Add the number of training steps to the task config so the dataset generator can do its split
                # consistent with the train/test split in the pretrained neuralhydrology model.
                task_config.add_config = (dict(task_config.add_config) if task_config.add_config else {}) \
                    | {'train_steps': int(train_steps)}
                if calib_steps is not None:
                    task_config.add_config = dict(task_config.add_config) | {'calib_steps': int(calib_steps)}
                # Pass the static attribute indices to the hydro dataset so it knows which variables should be ignored
                # during normalization.
                dataset_init = \
                    lambda **kwargs: HydroDataset(**(kwargs | {'static_attribute_indices': attribute_indices,
                                                               'static_attribute_norm_param': attribute_norm_params}))
                datasets.append(DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config, dataset_init))
        elif data_config.dataset_type in RESCP_REPRO_DATASETS:
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            loader_kwargs = dict(data_config.add_config or {})
            loader_kwargs.pop("skip_global_norm", None)
            table_combos = load_rescp_repro_data(
                data_config.dataset_type,
                path,
                **loader_kwargs,
            )
            datasets.extend(DataGenerator._rescp_repro_tables_to_datasets(table_combos, task_config))
        else:
            for path in data_config.paths:
                path = path.replace("/some_base_dir/data", replace_base_dir)
                p = Path(path)
                if data_config.dataset_type in LOCAL_TABULAR_MULTI_DATASETS:
                    table_combos = load_local_tabular_multi_data(data_config.dataset_type, path)
                    for X_full, Y_full, ts_id in table_combos:
                        datasets.append(DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config))
                elif p.is_dir():
                    datasets.extend([DataGenerator._get_data_single(data_config.dataset_type, str(sub_path.resolve()), task_config)
                                     for sub_path in p.glob("*.csv")])
                else:
                    datasets.append(DataGenerator._get_data_single(data_config.dataset_type, path, task_config))
        skip_global_norm = bool((data_config.add_config or {}).get("skip_global_norm", False))
        if hasattr(task_config, "global_norm") and task_config.global_norm and not skip_global_norm:
            DataGenerator._global_normalize(datasets, X_norm_param=X_norm_param, Y_norm_param=Y_norm_param)
        return datasets

    @staticmethod
    def _get_data_single(dataset_type, data_path, task_config: TaskConfig) -> ChronoSplittedTsDataset:
        if dataset_type in EnbPI_DATASETS:
            X_full, Y_full = demo_code_data_load(dataset_type, data_path)
        elif dataset_type in NSDB_DATASET:
            X_full, Y_full = load_nsdb_data(dataset_type, data_path)
        elif dataset_type in AIR_QUALITY:
            X_full, Y_full = load_bejing_air_data(dataset_type, data_path)
        elif dataset_type in LOCAL_TABULAR_DATASETS:
            X_full, Y_full = load_local_tabular_data(dataset_type, data_path)
        else:
            raise ValueError(f"Dataset {dataset_type} not supported!")
        return DataGenerator._table_to_dataset(X_full, Y_full, get_ts_id(dataset_type, data_path), task_config)

    @staticmethod
    def _global_normalize(datasets, X_norm_param = None, Y_norm_param = None):
        if X_norm_param is None:
            X_train_calib_all = torch.concat(([d.X_train for d in datasets] + [d.X_calib for d in datasets]))
            X_mean = torch.mean(X_train_calib_all, dim=0)
            X_std = torch.std(X_train_calib_all, dim=0)
            del X_train_calib_all
        else:
            X_mean, X_std = X_norm_param
        if Y_norm_param is None:
            Y_train_calib_all = torch.concat(([d.Y_train for d in datasets] + [d.Y_calib for d in datasets]))
            Y_mean = torch.mean(Y_train_calib_all, dim=0)
            Y_std = torch.std(Y_train_calib_all, dim=0)
            del Y_train_calib_all
        else:
            Y_mean, Y_std = Y_norm_param
        X_std = torch.where(torch.isfinite(X_std) & (X_std > 0), X_std, torch.ones_like(X_std))
        Y_std = torch.where(torch.isfinite(Y_std) & (Y_std > 0), Y_std, torch.ones_like(Y_std))
        for data in datasets:
            data.global_normalize(X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

    @staticmethod
    def _global_normalize_train_only(datasets):
        X_train_all = torch.concat([d.X_train for d in datasets])
        Y_train_all = torch.concat([d.Y_train for d in datasets])
        X_mean = torch.mean(X_train_all, dim=0)
        Y_mean = torch.mean(Y_train_all, dim=0)
        X_std = torch.std(X_train_all, dim=0)
        Y_std = torch.std(Y_train_all, dim=0)
        X_std = torch.where(X_std == 0, torch.ones_like(X_std), X_std)
        Y_std = torch.where(Y_std == 0, torch.ones_like(Y_std), Y_std)
        for data in datasets:
            data.global_normalize(X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

    @staticmethod
    def _rescp_repro_tables_to_datasets(table_combos, task_config: TaskConfig):
        datasets = []
        for X_full, Y_full, ts_id, mask_full in table_combos:
            if task_config.add_config is not None and task_config.add_config.get('max_steps'):
                max_steps = int(task_config.add_config['max_steps'])
                X_full = X_full[:max_steps]
                Y_full = Y_full[:max_steps]
                mask_full = mask_full[:max_steps]
            if task_config.add_config is not None and task_config.add_config.get('train_steps'):
                train_steps = task_config.add_config['train_steps']
                if task_config.add_config.get("no_calib", False):
                    cal_steps = 0
                elif 'calib_steps' in task_config.add_config:
                    cal_steps = task_config.add_config['calib_steps']
                else:
                    cal_steps = (X_full.shape[0] - train_steps) // 2
                split_points = np.array([train_steps, train_steps + cal_steps])
            else:
                if len(task_config.data_splits) == 4:
                    drop_tail = float(task_config.data_splits[3])
                    if drop_tail < 0 or drop_tail >= 1:
                        raise ValueError(f"drop_tail must be in [0, 1), got {drop_tail}")
                    keep_len = int(round(X_full.shape[0] * (1.0 - drop_tail)))
                    X_full = X_full[:keep_len]
                    Y_full = Y_full[:keep_len]
                    mask_full = mask_full[:keep_len]
                split_def = DataGenerator._get_needed_splits(task_config)
                split_points, _ = DataGenerator._chronological_split(X_full.shape[0], split_def)
            dataset = DataGenerator._create_dataset(
                ts_id,
                X_full,
                Y_full,
                split_points,
                norm=False,
                dataset_init=ChronoSplittedTsDataset,
            )
            mask_tensor = mask_full.reshape(-1, 1).float()
            warmup_skip = int((task_config.add_config or {}).get("mask_test_warmup_skip", 0))
            if warmup_skip > 0:
                test_start = int(split_points[1])
                mask_tensor[test_start: test_start + warmup_skip] = 0.0
            dataset.mask_full = mask_tensor
            datasets.append(dataset)
        DataGenerator._global_normalize_train_only(datasets)
        return datasets

    @staticmethod
    def _table_to_dataset(X_full, Y_full, ts_id, task_config: TaskConfig, dataset_init = ChronoSplittedTsDataset):
        # If the absolute number of train_steps is known, ignore the split config and do a 50/50 split for cal/test.
        if task_config.add_config is not None and task_config.add_config.get('train_steps'):
            train_steps = task_config.add_config['train_steps']
            if task_config.add_config.get("no_calib", False):
                cal_steps = 0
            elif 'calib_steps' in task_config.add_config:
                cal_steps = task_config.add_config['calib_steps']
            else:
                cal_steps = (X_full.shape[0] - train_steps) // 2
            split_points = np.array([train_steps, train_steps + cal_steps])
        else:
            # Optional 4-element split used by HP search:
            # [train, proper_calib, validation_as_test, heldout_real_test].
            # The fourth share is dropped from the tail before creating the
            # temporary validation dataset, so HP selection never sees the
            # real final test block.
            if len(task_config.data_splits) == 4:
                drop_tail = float(task_config.data_splits[3])
                if drop_tail < 0 or drop_tail >= 1:
                    raise ValueError(f"drop_tail must be in [0, 1), got {drop_tail}")
                keep_len = int(round(X_full.shape[0] * (1.0 - drop_tail)))
                X_full = X_full[:keep_len]
                Y_full = Y_full[:keep_len]
            split_def = DataGenerator._get_needed_splits(task_config)
            split_points, _ = DataGenerator._chronological_split(X_full.shape[0], split_def)
        return DataGenerator._create_dataset(ts_id, X_full, Y_full, split_points,
                                             norm=(not hasattr(task_config, "global_norm")) or (not task_config.global_norm),
                                             dataset_init=dataset_init)

    @staticmethod
    def _chronological_split(full_len, split_def):
        split_points = ((np.array(list(split_def.values())) / sum(split_def.values())) * full_len).astype(int)[:-1]
        split_points = np.cumsum(split_points)
        return split_points, split_def

    @staticmethod
    def _create_dataset(ts_id, X_full, Y_full, split_points, norm, dataset_init) -> ChronoSplittedTsDataset:
        if len(split_points) > 1:
            assert len(split_points) == 2
            return dataset_init(ts_id=ts_id, X=X_full, Y=Y_full, test_step=split_points[1].item(),
                                           calib_step=split_points[0].item(), normalize=norm)
        else:
            return dataset_init(ts_id=ts_id, X=X_full, Y=Y_full, test_step=split_points[0].item(),
                                           normalize=norm)

    @staticmethod
    def _get_needed_splits(task_config: TaskConfig) -> Dict[str, float]:
        if len(task_config.data_splits) < 3:
            raise ValueError("3 splits needed!")
        if len(task_config.data_splits) > 4:
            raise ValueError("More than 4 splits!")
        triple = task_config.data_splits[:3]
        _sum = sum(triple)
        splits = OrderedDict()
        splits['train'] = triple[0] / _sum
        splits['calib'] = triple[1] / _sum
        splits['test'] = triple[2] / _sum
        return splits

    #
    # Legacy Methods for original EnbPI Implementation
    #

    @staticmethod
    def get_data_legacy(data_config: TSDataConfig, task_config: TaskConfig) -> List[Tuple[dict, dict]]:
        data = []
        for path in data_config.paths:
            if data_config.dataset_type in EnbPI_DATASETS:
                X_full, Y_full = demo_code_data_load(data_config.dataset_type, path)
                splits = DataGenerator._get_needed_splits(task_config)
                # In Legacy there is no calib
                legacy_splits = OrderedDict()
                legacy_splits['train'] = splits['train'] + splits['calib']
                legacy_splits['test'] = splits['test']
                split_points, split_def = DataGenerator._chronological_split(X_full.shape[0], legacy_splits)
                data.append((DataGenerator._create_dataset_legacy(X_full, Y_full, split_points, split_def),
                            dict(ts_id=get_ts_id(dataset_type=data_config.dataset_type, data_path=path))))
            else:
                raise ValueError(f"Dataset {data_config.dataset_type} not supported!")
        return data

    @staticmethod
    def _create_dataset_legacy(X_full, Y_full, split_points, split_def):
        splits = dict()
        for idx, split_name in enumerate(split_def.keys()):
            start_idx = 0 if idx == 0 else split_points[idx - 1]
            if idx == len(split_points):  # Last Split
                splits[split_name] = SimpleTsDataset(X_full[start_idx:], Y_full[start_idx:])
            else:
                end_idx = split_points[idx]
                splits[split_name] = SimpleTsDataset(X_full[start_idx:end_idx], Y_full[start_idx:end_idx])
        return splits


def load_nsdb_data(dataset_type, data_path):
    data = pd.read_csv(data_path)
    if dataset_type.startswith("nsdb-60m"):
        drop_every_n = 2
    elif dataset_type.startswith("nsdb-30m"):
        drop_every_n = 1
    else:
        raise ValueError("Type not supported")
    append_coords = dataset_type.endswith("-wCoord")
    data = data.iloc[::drop_every_n, :]
    Y_full = data['dhi']
    X_full = data.loc[:, data.columns != 'dhi']
    X_full.drop(columns=X_full.columns[0:6], inplace=True)  # Drop Date Stuff
    if append_coords:
        p = Path(data_path)
        coordinates = pd.read_csv(p.parent.parent / "solar_coordinates.csv")
        lat = coordinates[coordinates["location"] == p.stem]['latitude'].item()
        long = coordinates[coordinates["location"] == p.stem]['longitude'].item()
        X_full["latitude"] = lat
        X_full["longitude"] = long
    return torch.from_numpy(X_full.to_numpy()).float(), torch.from_numpy(Y_full.to_numpy()).float()


def load_bejing_air_data(dataset_type, data_path, wd_encode='encode', imputer=None):
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data:
        data.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
    data.drop(columns=['No', 'station'], inplace=True, axis=1)
    data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True, axis=1)
    data_wd = data['wd'].fillna(value="Unknown")
    if imputer is not None:
        data = pd.DataFrame(imputer.fit_transform(data.loc[:, data.columns != 'wd']))
    else:
        data = data.loc[:, data.columns != 'wd'].fillna(method="ffill")
        data = data.fillna(method="bfill")
    data['wd'] = data_wd
    if dataset_type.startswith("air-25"):
        Y_full = data['PM2.5']
    elif dataset_type.startswith("air-10"):
        Y_full = data['PM10']
    data.drop(columns=['PM2.5', 'PM10'], inplace=True, axis=1)
    if wd_encode == "drop":
        data.drop(columns=["wd"], inplace=True, axis=1)
    elif wd_encode == "one-hot":
        data = pd.get_dummies(data)
    elif wd_encode == 'encode':
        data['wd_h'] = data['wd'].apply(lambda x: _encode_direction(x, True))
        data['wd_v'] = data['wd'].apply(lambda x: _encode_direction(x, False))
        data.drop(columns=["wd"], inplace=True, axis=1)
    X_full = data
    return torch.from_numpy(X_full.to_numpy()).float(), torch.from_numpy(Y_full.to_numpy()).float()


def load_local_tabular_data(dataset_type, data_path):
    data = pd.read_csv(data_path)
    if dataset_type == 'wind':
        return _load_wind_data(data)
    if dataset_type == 'weather':
        return _load_weather_data(data)
    if dataset_type == 'exchange-rate':
        return _load_exchange_rate_data(data)
    if dataset_type == 'exchange-rate-ot1':
        return _load_exchange_rate_one_step_data(data)
    if dataset_type == 'exchange-rate-ot1-lagged':
        return _load_exchange_rate_one_step_lagged_data(data)
    if dataset_type in ['amazon', 'stock-amzn', 'apple', 'stock-aapl', 'google', 'stock-goog']:
        return _load_amazon_stock_data(data)
    if dataset_type in ['elec-normalized', 'electricity-normalized']:
        return _load_electricity_normalized_data(data)
    raise ValueError(f"Dataset {dataset_type} not supported!")


def load_local_tabular_multi_data(dataset_type, data_path):
    data = pd.read_csv(data_path)
    if dataset_type == 'exchange-rate-multi':
        return _load_exchange_rate_multi_data(data)
    if dataset_type == 'exchange-rate-rescp':
        return _load_exchange_rate_rescp_data(data)
    raise ValueError(f"Dataset {dataset_type} not supported!")


def _add_datetime_features(frame, datetime_values, prefix='time'):
    dt = pd.to_datetime(datetime_values)
    frame[f'{prefix}_hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    frame[f'{prefix}_hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    dayofyear = dt.dt.dayofyear.astype(float)
    frame[f'{prefix}_doy_sin'] = np.sin(2 * np.pi * dayofyear / 366.0)
    frame[f'{prefix}_doy_cos'] = np.cos(2 * np.pi * dayofyear / 366.0)
    weekday = dt.dt.weekday.astype(float)
    frame[f'{prefix}_weekday_sin'] = np.sin(2 * np.pi * weekday / 7.0)
    frame[f'{prefix}_weekday_cos'] = np.cos(2 * np.pi * weekday / 7.0)
    return frame


def _finalize_tabular_xy(features, target):
    features = features.apply(pd.to_numeric, errors='coerce')
    target = pd.to_numeric(target, errors='coerce')
    valid = target.notna()
    features = features.loc[valid].replace([np.inf, -np.inf], np.nan)
    target = target.loc[valid]
    features = features.ffill().bfill().fillna(0.0)
    return torch.from_numpy(features.to_numpy(dtype=np.float32)).float(), \
        torch.from_numpy(target.to_numpy(dtype=np.float32)).float()


def _load_wind_data(data):
    data = data.copy()
    data['Hour Ending'] = pd.to_numeric(data['Hour Ending'].astype(str).str.replace(':00', '', regex=False),
                                        errors='coerce')
    data['Date'] = pd.to_numeric(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date', 'Hour Ending', 'MWH']).sort_values(['Date', 'Hour Ending'])
    base_date = pd.to_datetime(data['Date'].astype(int).astype(str), format='%Y%m%d')
    hour = ((data['Hour Ending'].astype(int) // 100) - 1).clip(lower=0, upper=23)
    timestamps = base_date + pd.to_timedelta(hour, unit='h')
    features = pd.DataFrame(index=data.index)
    features = _add_datetime_features(features, timestamps, prefix='wind')
    return _finalize_tabular_xy(features, data['MWH'])


def _load_weather_data(data):
    data = data.copy()
    if 'OT' not in data.columns:
        raise ValueError("weather dataset expects target column 'OT'")
    timestamps = pd.to_datetime(data['date'])
    features = data.drop(columns=['date', 'OT'])
    features = _add_datetime_features(features, timestamps, prefix='weather')
    return _finalize_tabular_xy(features, data['OT'])


def _load_exchange_rate_data(data):
    data = data.copy()
    if not {'date', 'OT'}.issubset(set(data.columns)):
        raise ValueError("exchange-rate dataset expects 'date' and target column 'OT'")
    timestamps = pd.to_datetime(data['date'])
    features = data.drop(columns=['date', 'OT'])
    features = _add_datetime_features(features, timestamps, prefix='exchange')
    constant_columns = features.columns[features.nunique(dropna=False) <= 1]
    if len(constant_columns) > 0:
        features = features.drop(columns=constant_columns)
    return _finalize_tabular_xy(features, data['OT'])


def _load_exchange_rate_one_step_data(data):
    data = data.copy()
    if not {'date', 'OT'}.issubset(set(data.columns)):
        raise ValueError("exchange-rate-ot1 dataset expects 'date' and target column 'OT'")
    timestamps = pd.to_datetime(data['date']).iloc[:-1].reset_index(drop=True)
    features = data.drop(columns=['date', 'OT']).iloc[:-1].reset_index(drop=True)
    features = _add_datetime_features(features, timestamps, prefix='exchange')
    constant_columns = features.columns[features.nunique(dropna=False) <= 1]
    if len(constant_columns) > 0:
        features = features.drop(columns=constant_columns)
    target = data['OT'].iloc[1:].reset_index(drop=True)
    return _finalize_tabular_xy(features, target)


def _load_exchange_rate_one_step_lagged_data(data):
    data = data.copy()
    if not {'date', 'OT'}.issubset(set(data.columns)):
        raise ValueError("exchange-rate-ot1-lagged dataset expects 'date' and target column 'OT'")
    timestamps = pd.to_datetime(data['date']).iloc[:-1].reset_index(drop=True)
    features = data.drop(columns=['date']).iloc[:-1].reset_index(drop=True)
    features = _add_datetime_features(features, timestamps, prefix='exchange')
    constant_columns = features.columns[features.nunique(dropna=False) <= 1]
    if len(constant_columns) > 0:
        features = features.drop(columns=constant_columns)
    target = data['OT'].iloc[1:].reset_index(drop=True)
    return _finalize_tabular_xy(features, target)


def _load_exchange_rate_multi_data(data):
    data = data.copy()
    if 'date' not in data.columns:
        raise ValueError("exchange-rate-multi dataset expects a 'date' column")
    timestamps = pd.to_datetime(data['date'])
    value_cols = [col for col in data.columns if col != 'date']
    values = data[value_cols].apply(pd.to_numeric, errors='coerce')
    lagged_values = values.shift(1).add_prefix('lag1_')
    time_features = _add_datetime_features(pd.DataFrame(index=data.index), timestamps, prefix='exchange')
    base_features = pd.concat([lagged_values, time_features], axis=1)
    constant_columns = base_features.columns[base_features.nunique(dropna=False) <= 1]
    if len(constant_columns) > 0:
        base_features = base_features.drop(columns=constant_columns)
    valid_features = base_features.notna().all(axis=1)
    table_combos = []
    for target_col in value_cols:
        valid = valid_features & values[target_col].notna()
        features = base_features.loc[valid].copy()
        target = values.loc[valid, target_col]
        X_full, Y_full = _finalize_tabular_xy(features, target)
        safe_target = str(target_col).replace('/', '_')
        table_combos.append((X_full, Y_full, f'exchange-rate-multi_{safe_target}'))
    return table_combos


def _load_exchange_rate_rescp_data(data, delay=7, horizon=1):
    data = data.copy()
    if 'date' not in data.columns:
        raise ValueError("exchange-rate-rescp dataset expects a 'date' column")
    target_shift = int(delay) + int(horizon)
    timestamps = pd.to_datetime(data['date'])
    value_cols = [col for col in data.columns if col != 'date']
    values = data[value_cols].apply(pd.to_numeric, errors='coerce')
    time_features = _add_datetime_features(pd.DataFrame(index=data.index), timestamps, prefix='exchange')
    features_full = pd.concat([values.add_prefix('obs_'), time_features], axis=1)
    constant_columns = features_full.columns[features_full.nunique(dropna=False) <= 1]
    if len(constant_columns) > 0:
        features_full = features_full.drop(columns=constant_columns)
    features = features_full.iloc[:-target_shift].reset_index(drop=True)
    target_values = values.iloc[target_shift:].reset_index(drop=True)
    valid_features = features.notna().all(axis=1)
    table_combos = []
    for target_col in value_cols:
        valid = valid_features & target_values[target_col].notna()
        X_full, Y_full = _finalize_tabular_xy(features.loc[valid].copy(), target_values.loc[valid, target_col])
        safe_target = str(target_col).replace('/', '_')
        table_combos.append((X_full, Y_full, f'exchange-rate-rescp_{safe_target}'))
    return table_combos


def _load_amazon_stock_data(data):
    data = data.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    features = ['open', 'high', 'low', 'volume']
    target = 'close'
    missing = [col for col in [*features, target] if col not in data.columns]
    if missing:
        raise ValueError(f"amazon stock dataset missing required columns: {missing}")
    if 'date' in data.columns:
        timestamps = pd.to_datetime(data['date'])
        feature_frame = _add_datetime_features(data[features].copy(), timestamps, prefix='amazon')
    else:
        feature_frame = data[features].copy()
    return _finalize_tabular_xy(feature_frame, data[target])


def _load_electricity_normalized_data(data):
    data = data.copy()
    data = data.iloc[17760:].reset_index(drop=True)
    if not {'period', 'transfer'}.issubset(set(data.columns)):
        raise ValueError("electricity-normalized dataset expects 'period' and 'transfer' columns")
    period = pd.to_numeric(data['period'], errors='coerce')
    keep_rows = (period > period.iloc[17]) & (period < period.iloc[24])
    data = data.loc[keep_rows].reset_index(drop=True)
    covariate_cols = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand']
    features = data[covariate_cols].copy()
    return _finalize_tabular_xy(features, data['transfer'])


def _encode_direction(direction, horizontal):
    if horizontal:
        if direction in ['N', 'S']:
            return 0
        elif direction in ['NNW', 'SSW']:
            return -0.5
        elif direction in ['NW', 'SW']:
            return -0.7
        elif direction in ['WNW', 'WSW']:
            return -0.86
        elif direction == 'W':
            return -1
        elif direction in ['NNE', 'SSE']:
            return 0.5
        elif direction in ['NE', 'SE']:
            return 0.7
        elif direction in ['ENE', 'ESE']:
            return 0.86
        elif direction == 'E':
            return 1
        elif direction == 'Unknown':
            return 0
        else:
            raise ValueError("Invalid Dir")
    else:
        if direction in ['W', 'E']:
            return 0
        elif direction in ['WSW', 'ESE']:
            return -0.5
        elif direction in ['SW', 'SE']:
            return -0.7
        elif direction in ['SSW', 'SSE']:
            return -0.86
        elif direction == 'S':
            return -1
        elif direction in ['WNW', 'ENE']:
            return 0.5
        elif direction in ['NW', 'NE']:
            return 0.7
        elif direction in ['NNW', 'NNE']:
            return 0.86
        elif direction == 'N':
            return 1
        elif direction == 'Unknown':
            return 0
        else:
            raise ValueError("Invalid Dir")
