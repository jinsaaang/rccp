import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.lib.engines.base_predictor import BasePredictor
from src.lib.engines.arima_predictor import ARIMAPredictor
from src.lib.utils.data_utils import create_residuals_frame
from tsl import logger
from tsl.data import (
    SpatioTemporalDataset,
    SpatioTemporalDataModule,
    BatchMap,
    BatchMapItem,
)
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, ExchangeBenchmark
from tsl.experiment import Experiment, NeptuneLogger
from src.lib.experiment.wandb_logger import WandbLogger
from tsl.metrics import torch_metrics

from src.lib import config
from tsl.nn.models import TransformerModel
from tsl.utils.casting import torch_to_numpy

from src.lib.nn.base import RNNModel, ARIMAModel  # , MLPModel
from src.lib.datasets.air_quality_beijing import AirQualityBeijing
from src.lib.datasets.solar import Solar
from src.lib.datasets.elec_rome import ElectricityRome

try:
    from src.lib.datasets.gpvar import GPVARDataset
except ModuleNotFoundError:
    GPVARDataset = None

try:
    from src.lib.datasets.air_quality import AirQuality
except ModuleNotFoundError:
    AirQuality = None

try:
    from src.lib.nn.base import STGNNModel
except ModuleNotFoundError:
    STGNNModel = None


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == "rnn":
        model = RNNModel
    elif model_str == "transformer":
        model = TransformerModel
    elif model_str == "stgnn":
        if STGNNModel is None:
            raise ModuleNotFoundError("STGNNModel is unavailable in the current ResCP checkout/environment.")
        model = STGNNModel
    elif model_str == "arima":
        model = ARIMAModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == "la":
        dataset = MetrLA()
    elif name == "air":
        if AirQuality is None:
            raise ModuleNotFoundError("AirQuality dataset is unavailable in the current ResCP checkout.")
        dataset = AirQuality()
    elif name == "beijing":
        dataset = AirQualityBeijing()
    elif name == "solar":
        dataset = Solar(freq="60min")
    elif name == "exchange":
        dataset = ExchangeBenchmark()
    elif name == "elec":
        dataset = ElectricityRome()
    elif name == "gpvar":
        if GPVARDataset is None:
            raise ModuleNotFoundError("GPVARDataset is unavailable in the current ResCP checkout.")
        dataset = GPVARDataset(**dataset_cfg.hparams, p_max=0)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    # Handle covariates properly for TSL
    covariates = {}
    # Get dataset-specific covariates (these are LOCAL features - node-specific)
    if cfg.dataset.name == "beijing" or cfg.dataset.name == "solar":
        target_df = dataset.get_target_only()
        dataset_covariates = dataset.get_covariates_for_tsl()
        if dataset_covariates:
            logger.info(
                f"Available dataset covariates: {list(dataset_covariates.keys())}"
            )
            # Process local (node-specific) covariates
            local_covariate_arrays = []
            for cov_name, (cov_df, pattern) in dataset_covariates.items():
                logger.info(
                    f"Processing local covariate {cov_name} with shape {cov_df.shape}"
                )
                # cov_df has MultiIndex columns (node, channel)
                # We need to reshape to (time, nodes, features) format
                cov_array = cov_df.values
                n_nodes = len(cov_df.columns.get_level_values(0).unique())
                n_features_per_node = len(cov_df.columns.get_level_values(1).unique())

                # Reshape from (time, nodes*features) to (time, nodes, features)
                cov_array = cov_array.reshape(
                    cov_array.shape[0], n_nodes, n_features_per_node
                )
                local_covariate_arrays.append(cov_array)
                logger.info(
                    f"Reshaped local covariate {cov_name} to shape: {cov_array.shape}"
                )

            if local_covariate_arrays:
                # Concatenate all local covariates along the feature dimension (last axis)
                combined_local_covariates = np.concatenate(
                    local_covariate_arrays, axis=-1
                )
                covariates["local"] = (
                    combined_local_covariates  # shape: (time, nodes, features)
                )
                logger.info(
                    f"Combined local covariates into 'local' with shape: {combined_local_covariates.shape}"
                )

    # Add temporal features as GLOBAL features (same for all nodes)
    if cfg.get("add_exogenous"):
        assert cfg.dataset.name not in {"gpvar"}
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded("day").values
        weekdays = dataset.datetime_onehot("weekday").values
        temporal_features = np.concatenate([day_sin_cos, weekdays], axis=-1)
        logger.info(f"Global temporal features shape: {temporal_features.shape}")

        # Add global features separately - they will be broadcast to all nodes by TSL
        covariates["global"] = temporal_features  # shape: (time, features)
        logger.info(
            f"Added global temporal features with shape: {temporal_features.shape}"
        )

        # Log the final covariate structure
        logger.info(f"Final covariates structure:")
        for key, cov in covariates.items():
            logger.info(f"  {key}: {cov.shape}")

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
            logger.info(
                f"Combined local and global features, final 'u' shape: {covariates['u'].shape}"
            )
        elif "local" not in covariates:
            # If only global features are present, we can use them directly
            covariates["u"] = temporal_features
            del covariates["global"]

    if cfg.dataset.name in {"gpvar", "toy", "mso"}:
        ds_index = pd.Index(dataset.index)
        index_type = "scalar"
    else:
        ds_index = dataset.index
        index_type = "datetime"

    torch_dataset = SpatioTemporalDataset(
        index=ds_index,
        target=dataset.dataframe(),
        mask=dataset.mask,
        covariates=covariates,
        horizon=cfg.horizon,
        window=cfg.window if cfg.model.name != "arima" else 1,
        stride=cfg.stride,
        delay=cfg.get("delay", 0),
    )

    if cfg.apply_scaler is False:
        transform = {}
    else:
        scale_axis = (0,) if cfg.get("scale_axis") == "node" else (0, 1)
        transform = {"target": StandardScaler(axis=scale_axis)}
        if "u" in torch_dataset:
            # Apply scaling to exogenous features if they exist
            transform["u"] = StandardScaler(axis=scale_axis)

    # Use continuous splitter to avoid missing residuals
    # from src.lib.data.continuous_splitter import ContinuousSplitter

    # Option 1: Use continuous splitter with no gaps
    # continuous_splitter = ContinuousSplitter(
    #     val_len=cfg.dataset.get("val_len", 0.4),
    #     test_len=cfg.dataset.get("test_len", 0.2),
    # )

    # Option 2: Use default splitter (comment out Option 1 and uncomment this)
    # dataset_splitting = cfg.dataset.get("splitting", {})
    # continuous_splitter = dataset.get_splitter(**dataset_splitting)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.get("splitting", {})),
        # splitter=continuous_splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()

    logger.info(f"Number of training samples: {dm.train_len}")
    logger.info(f"Number of validation samples: {dm.val_len}")
    logger.info(f"Number of test samples: {dm.test_len}")

    ########################################
    # training                             #
    ########################################

    adj = dataset.get_connectivity(
        **cfg.dataset.connectivity, train_slice=dm.train_slice
    )
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if "u" in torch_dataset else 0
    logger.info(
        f"Available covariates in torch_dataset: {list(torch_dataset.covariates.keys()) if torch_dataset.covariates else 'None'}"
    )
    logger.info(
        f"Input map keys: {list(torch_dataset.input_map.keys()) if hasattr(torch_dataset, 'input_map') else 'No input_map'}"
    )
    logger.info(f"Exogenous features size (d_exog): {d_exog}")

    model_kwargs = dict(
        n_nodes=torch_dataset.n_nodes,
        input_size=torch_dataset.n_channels,
        exog_size=d_exog,
        output_size=torch_dataset.n_channels,
        weighted_graph=torch_dataset.edge_weight is not None,
        window=torch_dataset.window,
        horizon=torch_dataset.horizon,
    )

    # logger.info(f"Model kwargs: {model_kwargs}")
    model_cls.filter_model_args_(model_kwargs)
    # logger.info(f"Filtered model kwargs: {model_kwargs}")
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        "mae": torch_metrics.MaskedMAE(),
        "mse": torch_metrics.MaskedMSE(),
        "mre": torch_metrics.MaskedMRE(),
    }

    if cfg.dataset.name in ["la", "bay"]:
        multistep_metrics = {
            "mape": torch_metrics.MaskedMAPE(),
            "mae@15": torch_metrics.MaskedMAE(at=2),
            "mae@30": torch_metrics.MaskedMAE(at=5),
            "mae@60": torch_metrics.MaskedMAE(at=11),
        }
        log_metrics.update(multistep_metrics)

    # setup predictor
    if cfg.model.name == "arima":
        predictor = ARIMAPredictor(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            loss_fn=loss_fn,
            metrics=log_metrics,
            scale_target=cfg.scale_target,
        )
    else:
        if cfg.get("lr_scheduler") is not None:
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
            scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
        else:
            scheduler_class = scheduler_kwargs = None
        predictor = BasePredictor(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=getattr(torch.optim, cfg.optimizer.name),
            optim_kwargs=dict(cfg.optimizer.hparams),
            loss_fn=loss_fn,
            metrics=log_metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scale_target=cfg.scale_target,
        )

    ########################################
    # logging options                      #
    ########################################

    run_args = dict(cfg)  # Use the config directly
    run_args["model_trainable_parameters"] = predictor.trainable_parameters

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor="val_mae", patience=cfg.patience, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor="val_mae",
        mode="min",
    )

    val_batches = 0.25

    trainer = Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.train_batches,
        limit_val_batches=val_batches,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    load_model_path = cfg.get("load_model_path")
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        if cfg.model.name == "arima":
            # ARIMA models are fitted directly on the data
            predictor.fit(dm)
            logger.info(
                f"ARIMA model fitted with {len(dm.trainset)} training samples and {len(dm.valset)} validation samples."
            )
        else:
            # Neural network models use the trainer
            trainer.fit(
                predictor,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
            )

    predictor.freeze()

    ########################################
    # compute residuals                    #
    ########################################

    output = trainer.predict(
        predictor,
        dataloaders=[
            dm.val_dataloader(
                shuffle=False,
                batch_size=dm.val_len if cfg.model.name == "arima" else None,
            ),
            dm.test_dataloader(
                shuffle=False,
                batch_size=dm.test_len if cfg.model.name == "arima" else None,
            ),
        ],
    )  # has size [[len_val], [len_test]]
    output = predictor.collate_prediction_outputs(
        output
    )  # has size [len_val + len_test]
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output["y_hat"], output["y"], output.get("mask", None))

    logger.info(f"y_hat shape: {y_hat.shape}, y_true shape: {y_true.shape}")
    plt.figure(figsize=(18, 6))
    plt.plot(y_true.squeeze()[-1000:], label="True")
    plt.plot(y_hat.squeeze()[-1000:], "--", alpha=0.7, label="Predicted")
    plt.legend()
    plt.savefig("predictions.pdf", bbox_inches="tight")

    residuals = (y_true - y_hat).squeeze(-1)
    calib_indices = dm.valset.indices
    test_indices = dm.testset.indices

    logger.info(
        f"Computed residuals for {len(calib_indices)} calibration samples and {len(test_indices)} test samples. Total: {len(calib_indices) + len(test_indices)} samples."
    )

    # input covariates
    val_index = dm.torch_dataset.data_timestamps(calib_indices)["horizon"]
    test_index = dm.torch_dataset.data_timestamps(test_indices)["horizon"]

    # Remove residuals at the beginning and at the end of the time series that have less than window and horizon time steps respectively
    # The second dimension in the dataframe is nodes x horizon
    # Input: [samples, nodes, horizon]
    # Output: [filtered_samples, nodes x horizon]
    lagged_residuals = create_residuals_frame(
        residuals,
        np.concatenate([val_index, test_index], axis=0),
        channels_index=dataset._columns_multiindex(),
        horizon=cfg.horizon,
        idx_type=index_type,
    )

    # concatenate indices and take the one corresponding to the last time step
    target_index = np.concatenate([val_index, test_index], axis=0)[:, 0]

    # combinations of nodex x horizon
    col_idx = [
        (c[0], f"{c[1]}_{i}")
        for c in dataset._columns_multiindex()
        for i in range(cfg.horizon)
    ]

    # create a dataframe with the residuals arranged in shape [samples, nodes x horizon]
    target_df = pd.DataFrame(
        data=rearrange(residuals, "t h n ... -> t (n ... h)"),
        index=target_index,
        columns=pd.MultiIndex.from_tuples(col_idx),
    )
    if mask is not None:
        mask_df = pd.DataFrame(
            data=rearrange(mask, "t h n ... -> t (n ... h)"),
            index=target_index,
            columns=pd.MultiIndex.from_tuples(col_idx),
        )
    else:
        mask_df = None

    # filter calib and test indices
    valid_input_indices = torch_dataset.index.get_indexer(lagged_residuals.index)
    valid_target_indices = torch_dataset.index.get_indexer(target_index)

    ########################################
    # save residuals for CP                #
    ########################################
    if cfg.save_outputs:
        lagged_residuals.to_hdf(os.path.join(cfg.run.dir, "residuals.h5"), key="input")
        target_df.to_hdf(os.path.join(cfg.run.dir, "residuals.h5"), key="target")
        if mask_df is not None:
            mask_df.to_hdf(os.path.join(cfg.run.dir, "residuals.h5"), key="target_mask")
        np.savez(
            os.path.join(cfg.run.dir, "indices.npz"),
            calib_indices=calib_indices,
            test_indices=test_indices,
            valid_input_indices=valid_input_indices,
            valid_target_indices=valid_target_indices,
        )

    return "done"


if __name__ == "__main__":
    exp = Experiment(
        run_fn=run_experiment, config_path="./configs/training", config_name="default"
    )
    res = exp.run()
    logger.info(res)
