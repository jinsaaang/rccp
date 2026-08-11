import torch
import numpy as np
from typing import Type, Mapping, Optional, Callable
from torchmetrics import Metric

from tsl import logger
from tsl.data import Data
from .base_predictor import BasePredictor


class ARIMAPredictor(BasePredictor):
    """Special predictor for ARIMA models that don't use gradient-based training."""

    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        scale_target: bool = False,
    ):
        # ARIMA doesn't need optimizer, so we pass dummy values
        super(ARIMAPredictor, self).__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.SGD,  # Dummy optimizer
            optim_kwargs={"lr": 0.01},  # Dummy parameters
            loss_fn=loss_fn,
            scale_target=scale_target,
            metrics=metrics,
        )
        self._is_fitted = False

    def fit(self, datamodule):
        """Fit the ARIMA model using training data."""
        # Get training data
        n_samples = datamodule.train_len
        train_dataloader = datamodule.train_dataloader(
            shuffle=False, batch_size=n_samples
        )

        self.delay = datamodule.torch_dataset.delay
        self.model.delay = self.delay

        assert (
            len(train_dataloader) == 1
        ), "ARIMA expects all training data in one batch"

        # Collect all training data
        y_list, x_list, u_list = [], [], []
        for batch in train_dataloader:
            x, y, mask, transform = self._unpack_batch(batch)
            y_list.append(transform["y"].transform(y["y"]))
            x_list.append(transform["x"].transform(x["x"]))
            if "u" in x:
                u_list.append(x["u"])

        # Concatenate all batches
        y_train = torch.cat(y_list, dim=0).squeeze(1)
        x_train = torch.cat(x_list, dim=0).squeeze(1)
        u_train = torch.cat(u_list, dim=0).squeeze(1) if u_list else None

        # Fit the model
        self.model.fit(x_train, u_train)
        self._is_fitted = True

        return self

    def training_step(self, batch, batch_idx):
        """ARIMA doesn't use gradient-based training, so this is a no-op."""
        # Return a dummy loss to satisfy the trainer interface
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        """Validation step for ARIMA."""
        if not self._is_fitted:
            # If model isn't fitted yet, return dummy metrics
            return {"val_mae": torch.tensor(float("inf"))}

        loss = torch.tensor(float("inf"), requires_grad=True)  # Dummy loss

        return loss

    def predict_batch(
        self,
        batch: Data,
        preprocess: bool = False,
        postprocess: bool = True,
        **forward_kwargs,
    ):
        """Override to handle ARIMA prediction with proper horizon."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Unpack batch
        inputs, targets, mask, transform = self._unpack_batch(batch)

        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        # Make prediction with horizon
        y = inputs["x"]
        u = inputs.get("u", None)

        y_hat = self.model.forward(y, u)

        # Note: Don't apply inverse transformation here as it will be done
        # automatically by the TSL framework's predict_batch method
        # This prevents double inverse transformation

        # For metrics computation, we need the original scale targets
        y_original = y
        y_hat_for_metrics = y_hat

        self.test_metrics.update(y_hat_for_metrics.detach(), y_original, mask)
        metrics_dict = self.test_metrics.compute()
        for key, value in metrics_dict.items():
            logger.info(f"{key}: {value:.4f}")

        y_hat = transform["y"].inverse_transform(y_hat)
        return y_hat

    def configure_optimizers(self):
        """ARIMA doesn't need optimizers."""
        # Return a dummy optimizer to satisfy the interface
        return torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=0.01)

    @property
    def trainable_parameters(self) -> int:
        """ARIMA models don't have trainable parameters in the PyTorch sense."""
        return 0
