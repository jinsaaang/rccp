import pandas as pd
import numpy as np

from tsl.data import SpatioTemporalDataset


def create_residuals_frame(
    residuals, index, channels_index=None, horizon=1, idx_type="datetime"
):
    # Flatten index
    if idx_type == "datetime":
        index = pd.DatetimeIndex(index.reshape(-1))
    elif idx_type == "scalar":
        index = pd.Index(index.reshape(-1))
    else:
        raise ValueError('idx_type must be "datetime" or "scalar"')

    # residuals = np.repeat(np.repeat(np.arange(residuals.shape[0]).reshape(-1, 1, 1), 12, 1), 207, -1)
    residuals = residuals.reshape(-1, *residuals.shape[2:])
    df = pd.DataFrame(data=residuals, index=index, columns=channels_index)

    lagged_residuals = {
        k: g.reset_index(drop=True) for k, g in df.groupby(level=0) if len(g) == horizon
    }
    lagged_residuals = pd.concat(lagged_residuals, axis=0).unstack()

    new_cols = pd.MultiIndex.from_tuples(
        [(x[0], f"{x[1]}_{x[2]}") for x in lagged_residuals.columns],
    )

    lagged_residuals.columns = new_cols
    return lagged_residuals


def filter_indices(
    dataset: SpatioTemporalDataset, indices, valid_indices, filter_by="window"
):
    """
    Remove any sample that does not completely overlap with indices.

    :param dataset: Ref dataset
    :param indices: Indices to filter
    :param valid_indices: Valid indices
    :return: Filtered indices
    """

    expanded_indices = dataset.expand_indices(indices)[filter_by].numpy()

    def is_in_idx(sample):
        return np.all(np.in1d(sample, valid_indices))

    mask = np.apply_along_axis(is_in_idx, 1, expanded_indices)
    return indices[mask]


def parse_and_filter_indices(target_dataset, indices):
    calib_indices = indices["calib_indices"]
    test_indices = indices["test_indices"]

    # filter indices incompatible with  new window lenght:
    calib_indices = calib_indices[np.in1d(calib_indices, target_dataset.indices)]
    test_indices = test_indices[np.in1d(test_indices, target_dataset.indices)]

    valid_input_indices = indices["valid_input_indices"]
    valid_target_indices = indices["valid_target_indices"]

    def filter_indices_(indices_):
        indices_ = filter_indices(
            target_dataset, indices_, valid_input_indices, filter_by="window"
        )
        indices_ = filter_indices(
            target_dataset, indices_, valid_target_indices, filter_by="horizon"
        )
        return indices_

    calib_indices = filter_indices_(calib_indices)
    test_indices = filter_indices_(test_indices)
    overlapping_indices, _ = target_dataset.overlapping_indices(
        calib_indices, test_indices, as_mask=True
    )

    return calib_indices[~overlapping_indices], test_indices


def find_close(el, seq):
    # find closest in seq and check is close
    idx = np.argmin(np.abs(np.array(seq) - el))
    assert np.isclose(seq[idx], el)

    return idx


def create_multiindex_for_multiple_stations(station_data_dict):
    """
    Create MultiIndex structure for multiple time series where:
    - First level (node): Station names
    - Second level (channel): All measurement channels across stations

    Args:
        station_data_dict: Dictionary with station_name -> DataFrame

    Returns:
        DataFrame with MultiIndex columns (node, channel)
    """
    # Find common columns across all stations
    all_columns = set()
    for df in station_data_dict.values():
        all_columns.update(df.columns)
    common_columns = list(all_columns)

    # Find common time index
    common_index = None
    for df in station_data_dict.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    # Create MultiIndex structure
    result_data = {}

    for station_name, df in station_data_dict.items():
        # Align to common index
        df_aligned = df.reindex(common_index)

        for channel in common_columns:
            if channel in df_aligned.columns:
                result_data[(station_name, channel)] = df_aligned[channel]
            else:
                print(
                    f"Warning: Channel '{channel}' not found in station '{station_name}'. Filling with NaN."
                )
                result_data[(station_name, channel)] = np.nan

    # Create DataFrame with MultiIndex
    result_df = pd.DataFrame(result_data, index=common_index)
    result_df.columns = pd.MultiIndex.from_tuples(
        result_df.columns, names=["node", "channel"]
    )

    return result_df
