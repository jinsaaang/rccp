import os

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.lib.nn.decoder.multiquantile_readout import MultiQuantileDecoder
from src.lib.nn.encoder_decoder_model import EncoderDecoderModel
from src.lib.nn.encoders.reservoir_encoder import ReservoirEncoder
from src.lib.nn.encoders.identity_encoder import IdentityEncoder

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.lib.datasets.air_quality_beijing import AirQualityBeijing
from src.lib.datasets.solar import Solar
from src.lib.datasets.elec_rome import ElectricityRome
from src.lib.engines.quantile_predictor import QuantilePredictor
from src.lib.metrics.torch_metrics.coverage import (
    MaskedCoverage,
    MaskedDeltaCoverage,
    MaskedPIWidth,
)
from src.lib.metrics.torch_metrics.pinball_loss import MaskedMultiPinballLoss
from src.lib.metrics.torch_metrics.winkler import MaskedWinklerScore
from src.lib.utils.data_utils import parse_and_filter_indices, find_close
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

torch.set_float32_matmul_precision("high")


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


def get_decoder_class(decoder_str):
    # Basic models  #####################################################
    if decoder_str == "multiquantile":
        readout = MultiQuantileDecoder
    else:
        raise NotImplementedError(f'Model "{decoder_str}" not available.')
    return readout


def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    if name == "beijing":
        dataset = AirQualityBeijing(include_covariates=True)
    elif name == "solar":
        dataset = Solar(freq="60min", include_covariates=True)
    elif name == "exchange":
        dataset = ExchangeBenchmark()
    elif name == "elec":
        dataset = ElectricityRome()
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


# TARGET_QUANTILES = np.round(np.arange(0.025, 1, 0.025), 3).tolist()
TARGET_QUANTILES = np.round(np.array([0.05, 0.95]), 3).tolist()


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

    residuals_input_value = _residual_frame_to_tnf(residuals_input.reindex(index=ds_index), cfg.dataset.name)
    residuals_target_value = _residual_frame_to_tnf(residuals_target.reindex(index=ds_index), cfg.dataset.name)
    logger.info(f"Residual tensor dataset={cfg.dataset.name} input_shape={getattr(residuals_input_value, 'shape', None)} target_shape={getattr(residuals_target_value, 'shape', None)}")
    covariates = dict(
        residuals_input={
            "value": residuals_input_value,
            "pattern": "t n f",
        },
        residuals_target={
            "value": residuals_target_value,
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

    input_map["input_target_and_residuals"] = BatchMapItem(
        inputs_, synch_mode="horizon", pattern="t n f", preprocess=False
    )

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
    reservoir_encoder_kwargs.pop("emb_size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_dataset = encode_dataset(
        torch_dataset,
        encoder_class=reservoir_encoder_cls,
        encoder_kwargs=reservoir_encoder_kwargs,
        # start encoding after training samples if residuals are used
        # (to avoid encoding the residuals of the training set, which are not available)
        start_at=calib_indices[0] if cfg.use_residuals else 0,
        hidden_size=cfg.model.encoder.hparams.n_internal_units,
        encode_exogenous=cfg.preprocess_exogenous,
        append_exogenous=cfg.get("append_exogenous_after_encoding", False),
        encode_both=cfg.use_both,
        encode_residuals=cfg.use_residuals,
        l2_normalize=False,
        device=device,
        keep_raw=cfg.keep_raw,
    )

    ########################################
    # Create model                         #
    ########################################

    alphas = sorted(cfg.alphas)
    assert alphas[-1] < 0.5

    target_qs = TARGET_QUANTILES

    # Set exogenous size to 0 for the model - exogenous variables are only used during encoding
    d_exog = 0

    model_class = EncoderDecoderModel

    ## Encoder init
    encoder_cls = get_encoder_class(cfg.model.encoder.name)
    encoder_kwargs = dict(
        emb_size=cfg.model.encoder.hparams.emb_size,
        n_instances=torch_dataset.n_nodes,
    )
    encoder_cls.filter_init_args_(encoder_kwargs)
    encoder_kwargs.update(cfg.model.encoder.hparams)

    encoder_kwargs["hidden_size"] = (
        cfg.model.encoder.hparams.n_internal_units * 1
        + (
            d_exog if cfg.get("append_exogenous_after_encoding", False) else 0
        )  # Changed: use separate config
        + cfg.model.encoder.hparams.emb_size
    )

    ## Decoder init
    decoder_cls = get_decoder_class(cfg.model.decoder.name)

    decoder_kwargs = dict(quantiles=target_qs)
    decoder_cls.filter_init_args_(decoder_kwargs)
    decoder_kwargs.update(cfg.model.decoder.hparams)

    decoder_kwargs["hidden_size"] = (
        cfg.model.encoder.hparams.n_internal_units * 1
        + (
            d_exog if cfg.get("append_exogenous_after_encoding", False) else 0
        )  # Changed: use separate config
        + cfg.model.encoder.hparams.emb_size
    )

    model_kwargs = dict(
        encoder_class=encoder_cls,
        encoder_kwargs=encoder_kwargs,
        decoder_class=decoder_cls,
        decoder_kwargs=decoder_kwargs,
        input_size=model_input_size,
        exog_size=d_exog,
        output_size=torch_dataset.n_channels * src_config["horizon"],
        horizon=torch_dataset.horizon,
    )

    model_class.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # Metrics.                             #
    ########################################

    def get_metric_at_alpha(base_metric, target_alpha):
        idx_low = find_close(target_alpha / 2, target_qs)
        idx_high = find_close(1 - target_alpha / 2, target_qs)

        def preprocessing_fn(y_hat):
            return torch.stack((y_hat[idx_low], y_hat[idx_high]))

        return MaskedMetricWrapper(
            metric=base_metric, input_preprocessing=preprocessing_fn
        )

    # Pinball is already used as the training loss. Logging it as a metric sends
    # the stacked quantile tensor directly to tsl's generic metric shape check,
    # which expects y_hat and y to have identical shapes. Keep validation/test
    # metrics focused on interval quantities selected below.
    log_metrics = {}

    for a in alphas:
        log_metrics[f"coverage_at_{int((1 - a) * 100)}"] = get_metric_at_alpha(
            MaskedCoverage(), a
        )
        log_metrics[f"delta_cov_at_{int((1 - a) * 100)}"] = get_metric_at_alpha(
            MaskedDeltaCoverage(alpha=a), a
        )
        log_metrics[f"pi_width_at_{int((1 - a) * 100)}"] = get_metric_at_alpha(
            MaskedPIWidth(), a
        )
        log_metrics[f"winkler_at_{int((1 - a) * 100)}"] = get_metric_at_alpha(
            MaskedWinklerScore(alpha=a), a
        )

    log_metrics = {k.replace(".", "-"): v for k, v in log_metrics.items()}

    if cfg.get("lr_scheduler") is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    predictor = QuantilePredictor(
        model_class=model_class,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        quantiles=target_qs,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    run_args = exp.get_config_dict()
    run_args["model"]["trainable_parameters"] = predictor.trainable_parameters

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor="val_winkler_at_90", patience=cfg.patience, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor="val_winkler_at_90",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.train_batches,
        default_root_dir=cfg.run.dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(
        predictor,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    # run validation one last time to save val error best model
    trainer.validate(predictor, dataloaders=dm.val_dataloader())
    out = trainer.test(predictor, dataloaders=dm.test_dataloader())
    coverage = out[0]["test_coverage_at_90"]
    width = out[0]["test_pi_width_at_90"]
    delta_coverage = out[0]["test_delta_cov_at_90"]
    winkler = out[0]["test_winkler_at_90"]

    logger.info("\n\nTest results:")
    logger.info(f"Average test coverage: {coverage}")
    logger.info(f"Average test delta coverage: {delta_coverage}")
    logger.info(f"Average test pi width: {width}")
    logger.info(f"Average test winkler: {winkler}")

    wandb.finish()

    return "Done"


if __name__ == "__main__":
    exp = Experiment(
        run_fn=run_experiment, config_path="./configs", config_name="rexcp_qr"
    )
    res = exp.run()
    logger.info(res)
