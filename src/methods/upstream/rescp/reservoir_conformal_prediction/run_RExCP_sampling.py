import os

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from omegaconf import DictConfig

from src.reservoir_conformal_residual_sampler import ConformalResidualSamplerLightning
from src.utils.utils import minimize_intervals
from src.lib.nn.encoders.reservoir_encoder import ReservoirEncoder
from src.lib.nn.encoders.identity_encoder import IdentityEncoder

from src.lib.datasets.air_quality_beijing import AirQualityBeijing
from src.lib.datasets.solar import Solar
from src.lib.datasets.elec_rome import ElectricityRome
from src.lib.metrics.torch_metrics.coverage import (
    MaskedCoverage,
    MaskedDeltaCoverage,
    MaskedPIWidth,
)
from src.lib.metrics.torch_metrics.winkler import MaskedWinklerScore
from src.lib.utils.data_utils import parse_and_filter_indices
from src.lib.nn.utils import encode_dataset
from src.lib.metrics.torch_metrics.wrappers import MaskedMetricWrapper
from tsl import logger

from tsl.data import (
    SpatioTemporalDataset,
    SpatioTemporalDataModule,
    BatchMap,
    BatchMapItem,
)
from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import ExchangeBenchmark
from tsl.experiment import Experiment


def _compute_interval_metrics(lower_bounds, upper_bounds, y, mask, alpha):
    if mask is None:
        valid = torch.ones_like(y, dtype=torch.bool)
    else:
        valid = mask.bool()
    covered = torch.logical_and(y >= lower_bounds, y <= upper_bounds)
    valid_count = valid.sum().clamp_min(1)
    coverage = covered[valid].float().sum() / valid_count
    delta_coverage = coverage - (1 - alpha)
    widths = upper_bounds - lower_bounds
    pi_width = widths[valid].float().mean()
    penalties = torch.clamp(lower_bounds - y, min=0) + torch.clamp(y - upper_bounds, min=0)
    winkler = (widths + (2 / alpha) * penalties)[valid].float().mean()
    return coverage, delta_coverage, pi_width, winkler


def get_encoder_class(encoder_str):
    # Basic models  #####################################################
    if encoder_str == "reservoir":
        encoder = IdentityEncoder  # or LinearEncoder
    else:
        raise NotImplementedError(f'Model "{encoder_str}" not available.')
    return encoder


def get_dataset_encoder_class(encoder_str):
    # Basic models  #####################################################
    if encoder_str == "reservoir":
        encoder = ReservoirEncoder
    else:
        raise NotImplementedError(f'Model "{encoder_str}" not available.')
    return encoder


def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    if name == "beijing":
        dataset = AirQualityBeijing(include_covariates=False)
    elif name == "solar":
        dataset = Solar(freq="60min", include_covariates=False)
    elif name == "exchange":
        dataset = ExchangeBenchmark()
    elif name == "elec":
        dataset = ElectricityRome()
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def _normalize_residual_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        if list(df.columns.names) != ["node", "channel"]:
            df = df.copy()
            df.columns = pd.MultiIndex.from_tuples(
                df.columns.tolist(), names=["node", "channel"]
            )
    return df


def _residual_frame_to_tnf(df, dataset_name: str):
    df = _normalize_residual_frame_columns(df)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        return df
    n_nodes = len(df.columns.get_level_values(0).unique())
    n_features = len(df.columns.get_level_values(1).unique())
    values = df.to_numpy()
    return values.reshape(df.shape[0], n_nodes, n_features)


def run_experiment(cfg: DictConfig):

    def run_validation():
        ##########################################
        # Run method.                            #
        ##########################################

        ########################################
        # Create model                         #
        ########################################

        model = ConformalResidualSamplerLightning(
            cal_residuals=cal_residuals_full.to(device),
            reservoir=None,
            alpha=cfg.alpha,
            T=cfg["T"],
            past_residuals_window=cfg["sliding_window"],
            eta=cfg["aci_gamma"],
            n_quantiles=cfg["n_quantiles"],
            similarity=cfg["similarity"],
            decay=cfg["decay"],
            decay_rate=cfg["decay_rate"],
        )

        model = model.to(device=device)
        model.model.cal_states = X_cal_reservoir_full.to(device)

        with torch.no_grad():
            quantiles = model.model(
                X_val_reservoir_full.to(device),
                y_hat_val_full.to(device),
                y_val_full.to(device),
            ).cpu()  # (n_samples, n_nodes, 2*n_quantiles)

        if quantiles.shape[1] > 2:
            lower_bounds, upper_bounds, lower_quantiles, upper_quantiles = (
                minimize_intervals(
                    quantiles, y_hat_val_full, y_val_full, return_quantiles=True
                )
            )
        else:
            lower_quantiles = quantiles[:, 0].reshape(y_hat_val_full.shape)
            upper_quantiles = quantiles[:, 1].reshape(y_hat_val_full.shape)
            lower_bounds = y_hat_val_full + lower_quantiles
            upper_bounds = y_hat_val_full + upper_quantiles

        assert (lower_bounds.shape == y_val_full.shape) and (
            upper_bounds.shape == y_val_full.shape
        ), f"Expected {y_val_full.shape}, got {lower_bounds.shape} and {upper_bounds.shape}"

        coverage, delta_coverage, pi_width, winkler = _compute_interval_metrics(
            lower_bounds, upper_bounds, y_val_full, val_mask_full, cfg.alpha
        )

        logger.info("\n\nValidation results:")
        logger.info(f"Average validation coverage: {coverage}")
        logger.info(f"Average validation delta coverage: {delta_coverage}")
        logger.info(f"Average validation pi width: {pi_width}")
        logger.info(f"Average validation winkler: {winkler}")

    def run_test():
        ##########################################
        # Run method.                            #
        ##########################################

        ########################################
        # Create model                         #
        ########################################

        model = ConformalResidualSamplerLightning(
            cal_residuals=cal_residuals_full.to(device),
            reservoir=None,
            alpha=cfg.alpha,
            T=cfg["T"],
            past_residuals_window=cfg["sliding_window"],
            eta=cfg["aci_gamma"],
            n_quantiles=cfg["n_quantiles"],
            similarity=cfg["similarity"],
            decay=cfg["decay"],
            decay_rate=cfg["decay_rate"],
        )

        model = model.to(device=device)
        model.model.cal_states = X_cal_reservoir_full.to(device)

        with torch.no_grad():
            quantiles = model.model(
                X_test_reservoir_full.to(device),
                y_hat_test_full.to(device),
                y_test_full.to(device),
            ).cpu()

        if quantiles.shape[1] > 2:
            lower_bounds, upper_bounds, lower_quantiles, upper_quantiles = (
                minimize_intervals(
                    quantiles, y_hat_test_full, y_test_full, return_quantiles=True
                )
            )
        else:
            lower_quantiles = quantiles[:, :, 0].reshape(y_hat_test_full.shape)
            upper_quantiles = quantiles[:, :, 1].reshape(y_hat_test_full.shape)
            lower_bounds = y_hat_test_full + lower_quantiles
            upper_bounds = y_hat_test_full + upper_quantiles

        assert (lower_bounds.shape == y_test_full.shape) and (
            upper_bounds.shape == y_test_full.shape
        ), f"Expected {y_test_full.shape}, got {lower_bounds.shape} and {upper_bounds.shape}"

        coverage, delta_coverage, pi_width, winkler = _compute_interval_metrics(
            lower_bounds, upper_bounds, y_test_full, test_mask_full, cfg.alpha
        )

        logger.info("\n\nTest results:")
        logger.info(f"Average test coverage: {coverage}")
        logger.info(f"Average test delta coverage: {delta_coverage}")
        logger.info(f"Average test pi width: {pi_width:.4f}")
        logger.info(f"Average test winkler: {winkler:.4f}")

    ########################################
    # data module                          #
    ########################################

    local_dir = cfg.src_dir

    residuals_input: pd.DataFrame = pd.read_hdf(
        os.path.join(local_dir, "residuals.h5"), key="input"
    )
    residuals_target: pd.DataFrame = pd.read_hdf(
        os.path.join(local_dir, "residuals.h5"), key="target"
    )
    residuals_input = _normalize_residual_frame_columns(residuals_input)
    residuals_target = _normalize_residual_frame_columns(residuals_target)
    with open(os.path.join(local_dir, "config.yaml"), "r") as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)

    # clip residuals to avoid extreme values
    if cfg.get("clip_residuals"):
        lower_clip = residuals_input.quantile(0.005, axis=0)
        upper_clip = residuals_input.quantile(0.995, axis=0)
        residuals_input = residuals_input.clip(lower_clip, upper_clip, axis=1)
        residuals_target = residuals_target.clip(lower_clip, upper_clip, axis=1)

    assert cfg.dataset.name == src_config["dataset"]["name"]
    dataset = get_dataset(src_config["dataset"])

    ds_index = dataset.index

    try:
        mask_target = pd.read_hdf(
            os.path.join(local_dir, "residuals.h5"), key="target_mask"
        )
        mask_target = mask_target.reindex(index=dataset.index)
    except KeyError:
        mask_target = None

    indices = np.load(os.path.join(local_dir, "indices.npz"))

    # t -> steps, n -> nodes, f -> features
    covariates = dict(
        residuals_input={
            "value": _residual_frame_to_tnf(residuals_input.reindex(index=ds_index), cfg.dataset.name),
            "pattern": "t n f",
        },
        residuals_target={
            "value": _residual_frame_to_tnf(residuals_target.reindex(index=ds_index), cfg.dataset.name),
            "pattern": "t n f",
        },
    )
    if cfg.get("add_exogenous"):
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded("day").values
        weekdays = dataset.datetime_onehot("weekday").values
        temporal_features = np.concatenate([day_sin_cos, weekdays], axis=-1)
        covariates["global"] = temporal_features

        # Get dataset-specific covariates (these are LOCAL features - node-specific)
        if cfg.dataset.name == "beijing" or cfg.dataset.name == "solar":
            dataset_covariates = dataset.get_covariates_for_tsl()
            if dataset_covariates:
                # Process local (node-specific) covariates
                local_covariate_arrays = []
                for cov_name, (cov_df, pattern) in dataset_covariates.items():
                    # cov_df has MultiIndex columns (node, channel)
                    # We need to reshape to (time, nodes, features) format
                    cov_array = cov_df.values
                    n_nodes = len(cov_df.columns.get_level_values(0).unique())
                    n_features_per_node = len(
                        cov_df.columns.get_level_values(1).unique()
                    )

                    # Reshape from (time, nodes*features) to (time, nodes, features)
                    cov_array = cov_array.reshape(
                        cov_array.shape[0], n_nodes, n_features_per_node
                    )
                    local_covariate_arrays.append(cov_array)

                if local_covariate_arrays:
                    # Concatenate all local covariates along the feature dimension (last axis)
                    combined_local_covariates = np.concatenate(
                        local_covariate_arrays, axis=-1
                    )
                    covariates["local"] = (
                        combined_local_covariates  # shape: (time, nodes, features)
                    )

        # If we have both local and global features, we need to handle them properly
        # TSL expects them to be combined in a specific way for the model
        if "local" in covariates and "global" in covariates:
            # Get the number of nodes from local covariates
            n_nodes = covariates["local"].shape[1]
            # Broadcast global features to all nodes and concatenate with local features
            global_broadcasted = np.broadcast_to(
                temporal_features[:, None, :],
                (temporal_features.shape[0], n_nodes, temporal_features.shape[1]),
            )
            # Concatenate local and broadcasted global features
            covariates["u"] = np.concatenate(
                [covariates["local"], global_broadcasted], axis=-1
            )
            # Remove the separate global entry since it's now combined
            del covariates["global"]  # Remove global to avoid confusion
            del covariates["local"]  # Remove local to avoid confusion
        elif "local" not in covariates:
            # If only global features are present, we can use them directly
            covariates["u"] = temporal_features
            del covariates["global"]

    # use residuals as regression targets
    target_map = BatchMap()
    target_map["y"] = BatchMapItem(
        "residuals_target", synch_mode="horizon", pattern="t n f", preprocess=False
    )

    input_map = BatchMap()

    if mask_target is not None:
        input_map["mask_target"] = BatchMapItem(
            ["mask_target"], synch_mode="horizon", pattern="t n f"
        )
        covariates.update(mask_target=mask_target.astype("bool"))

    if "u" in covariates:
        if covariates["u"].ndim == 2:
            input_map["u"] = BatchMapItem("u", synch_mode="window", pattern="t f")
        elif covariates["u"].ndim == 3:
            input_map["u"] = BatchMapItem("u", synch_mode="window", pattern="t n f")
        else:
            raise ValueError(
                f"Unexpected dimension for covariates['u']: {covariates['u'].ndim}"
            )
    else:
        logger.warning(
            "No exogenous variables found in the dataset. "
            "Make sure to set `add_exogenous` to True if you want to use them."
        )

    inputs_ = ["residuals_input"]
    model_input_size = dataset.n_channels * src_config["horizon"]

    if cfg.get("target_as_input", True):
        inputs_.append("target")
        model_input_size += dataset.n_channels

    input_map["x"] = BatchMapItem(inputs_, synch_mode="window", pattern="t n f")

    torch_dataset = SpatioTemporalDataset(
        index=ds_index,
        target=dataset.dataframe(),  # original target time series
        mask=dataset.mask,
        covariates=covariates,  # residuals and eventually exogenous variables (or learnable embeddings)
        window=1,
        stride=src_config["stride"],
        target_map=target_map,
        input_map=input_map,
        delay=src_config.get("delay", 0),
        horizon=1,
    )

    calib_indices, test_indices = parse_and_filter_indices(torch_dataset, indices)

    val_len = int(cfg.val_len * len(calib_indices))
    calib_indices, val_indices = (
        calib_indices[: -val_len - torch_dataset.samples_offset],
        calib_indices[-val_len:],
    )

    calib_splitter = FixedIndicesSplitter(
        train_idxs=calib_indices,
        val_idxs=val_indices,
        test_idxs=test_indices,
    )

    scale_axis = (0,) if cfg.get("scale_axis") == "node" else (0, 1)
    transform = {
        "target": StandardScaler(axis=scale_axis),
        "residuals_target": StandardScaler(axis=scale_axis),
        "residuals_input": StandardScaler(axis=scale_axis),
    }
    if "u" in torch_dataset:
        transform["u"] = StandardScaler(axis=scale_axis)
    # transform = {}

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=calib_splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()

    ########################################
    # Encode dataset                       #
    ########################################

    reservoir_encoder_input_size = torch_dataset.n_channels
    if cfg.preprocess_exogenous:
        reservoir_encoder_input_size += torch_dataset.input_map.u.shape[-1]
    additional_encoder_hparams = dict(seed=cfg.run.seed)

    reservoir_encoder_cls = get_dataset_encoder_class("reservoir")
    reservoir_encoder_kwargs = dict()
    reservoir_encoder_kwargs.update(cfg.model.encoder.hparams)
    reservoir_encoder_kwargs.update(additional_encoder_hparams)

    if torch.cuda.is_available():
        if type(cfg.devices) is int:
            device = torch.device("cuda", cfg.devices)
        elif type(cfg.devices) is list:
            device = torch.device("cuda", cfg.devices[0])
        else:
            raise ValueError(
                f"Invalid device type: {type(cfg.devices)}. Expected int or list."
            )
    else:
        device = torch.device("cpu")

    torch_dataset = encode_dataset(
        torch_dataset,
        encoder_class=reservoir_encoder_cls,
        encoder_kwargs=reservoir_encoder_kwargs,
        # start encoding after training samples
        start_at=calib_indices[0] if cfg.use_residuals else 0,
        hidden_size=cfg.model.encoder.hparams.n_internal_units,
        encode_exogenous=cfg.preprocess_exogenous,
        encode_both=cfg.use_both,
        encode_residuals=cfg.use_residuals,
        l2_normalize=True,
        device=device,
        keep_raw=cfg.keep_raw,
    )

    ########################################
    # Prepare data                         #
    ########################################

    encoded_x = torch_dataset.get_tensor("encoded_x")[0]
    residuals_input, residual_scaler = torch_dataset.get_tensor(
        "residuals_input", preprocess=False
    )
    y, y_scaler = torch_dataset.get_tensor("target", preprocess=False)
    mask = torch_dataset.get_tensor("mask_target")[0]

    # no need to preprocess reservoir encoded data, since it is already standardized
    X_cal_reservoir_full = encoded_x[calib_indices - cfg.delay - 1]
    # do not preprocess residuals, these are the ones needed for sampling
    cal_residuals_full = residuals_input[calib_indices]
    assert (
        cal_residuals_full.shape[0] == X_cal_reservoir_full.shape[0]
    ), f"cal_residuals {cal_residuals_full.shape} != X_cal_reservoir_full {X_cal_reservoir_full.shape}"

    # same here, no need to preprocess encoded data
    X_val_reservoir_full = encoded_x[val_indices - cfg.delay - 1]
    # do not preprocess residuals, these are the ones needed for sampling
    val_residuals_full = residuals_input[val_indices]
    # do not preprocess target, these are the point forecasts used to construct quantiles around
    y_val_full = y[val_indices]
    val_mask_full = mask[val_indices]
    assert (
        val_residuals_full.shape == y_val_full.shape
    ), f"val_residuals {val_residuals_full.shape} != y_val {y_val_full.shape}"
    y_hat_val_full = y_val_full - val_residuals_full

    # same here, no need to preprocess encoded data
    X_test_reservoir_full = encoded_x[test_indices - cfg.delay - 1]
    # do not preprocess residuals, these are the ones needed for sampling
    test_residuals_full = residuals_input[test_indices]
    # do not preprocess target, these are the point forecasts used to construct quantiles around
    y_test_full = y[test_indices]
    test_mask_full = mask[test_indices]
    assert (
        test_residuals_full.shape == y_test_full.shape
    ), f"test_residuals {test_residuals_full.shape} != y_test {y_test_full.shape}"
    y_hat_test_full = y_test_full - test_residuals_full

    logger.info(f"X_cal_reservoir_full shape: {X_cal_reservoir_full.shape}")
    logger.info(f"cal_residuals_full shape: {cal_residuals_full.shape}")
    logger.info(f"X_val_reservoir_full shape: {X_val_reservoir_full.shape}")
    logger.info(f"val_residuals_full shape: {val_residuals_full.shape}")
    logger.info(f"y_hat_val_full shape: {y_hat_val_full.shape}")
    logger.info(f"y_val_full shape: {y_val_full.shape}")
    logger.info(f"X_test_reservoir_full shape: {X_test_reservoir_full.shape}")
    logger.info(f"test_residuals_full shape: {test_residuals_full.shape}")
    logger.info(f"y_hat_test_full shape: {y_hat_test_full.shape}")
    logger.info(f"y_test_full shape: {y_test_full.shape}")

    ########################################
    # Metrics.                             #
    ########################################
    def get_metric_at_alpha(base_metric):
        return MaskedMetricWrapper(metric=base_metric)

    coverage_metric = get_metric_at_alpha(MaskedCoverage())
    delta_coverage_metric = get_metric_at_alpha(MaskedDeltaCoverage(alpha=cfg.alpha))
    pi_width_metric = get_metric_at_alpha(MaskedPIWidth())
    winkler_metric = get_metric_at_alpha(MaskedWinklerScore(alpha=cfg.alpha))

    ########################################
    # Run experiment.                      #
    ########################################

    if cfg.run_validation:
        run_validation()
    else:
        run_test()

    wandb.finish()


if __name__ == "__main__":
    exp = Experiment(
        run_fn=run_experiment, config_path="./configs", config_name="rexcp_sampling"
    )
    res = exp.run()
    logger.info(res)
