import os
import numpy as np
import pandas as pd
from typing import List, Mapping, Optional, Union
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger as LightningWandbLogger
from torch import Tensor

from tsl.utils.io import save_figure
from tsl.utils.python_utils import ensure_list


class WandbLogger(LightningWandbLogger):
    """Extensions of PyTorch Lightning
    :class:`~pytorch_lightning.loggers.WandbLogger` with useful logging
    functionalities.

    Args:
        project_name (str): Name of the project in wandb.
        experiment_name (str, optional): Name of the experiment in wandb.
            (default: :obj:`None`)
        tags (list, optional): List of tags to associate with the run.
            (default: :obj:`None`)
        params (Mapping, optional): Mapping of the run's parameters (are logged
            as config in wandb).
            (default: :obj:`None`)
        save_dir (str, optional): Save directory of the experiment, used to
            temporarily log artifacts before upload. If :obj:`None`, then
            defaults to current working directory.
            (default: :obj:`None`)
        debug (bool): If :obj:`True`, then do not log online (i.e., log in
            :obj:`"offline"` mode). Otherwise log online.
            (default: :obj:`False`)
        prefix (str, optional): Root namespace for all metadata logging.
            (default: :obj:`"logs"`)
        upload_stdout (bool): If :obj:`True`, then log also :obj:`stdout` to wandb.
            (default: :obj:`False`)
        **kwargs: Additional parameters for
            :class:`~pytorch_lightning.loggers.WandbLogger`.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        tags: Optional[Union[str, List]] = None,
        params: Optional[Mapping] = None,
        save_dir: Optional[str] = None,
        debug: bool = False,
        prefix: Optional[str] = "logs",
        upload_stdout: bool = False,
        **kwargs,
    ):
        prefix = prefix or ""
        if tags is not None:
            kwargs["tags"] = ensure_list(tags)
        mode = "offline" if debug else "online"

        # Set config from params
        config = params if params is not None else {}

        super(WandbLogger, self).__init__(
            project=project_name,
            name=experiment_name,
            save_dir=save_dir,
            mode=mode,
            prefix=prefix,
            config=config,
            **kwargs,
        )
        self.save_dir = save_dir

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment.

        Returns:
            the root directory where experiment logs get saved
        """
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if value is not None:
            self._save_dir = os.path.abspath(value)
        else:
            self._save_dir = os.getcwd()

    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[Tensor, float, str],
        step: Optional[int] = None,
    ):
        """Log a single metric value.

        Args:
            metric_name (str): Name of the metric
            metric_value: Value to log
            step (int, optional): Step number for the metric
        """
        if hasattr(self.experiment, "log"):
            log_dict = {metric_name: metric_value}
            if step is not None:
                log_dict["step"] = step
            self.experiment.log(log_dict)

    def _artifact_storage_path(self, name, extension: str = None):
        """Create a temporary path for storing artifacts before upload.

        Args:
            name (str): Base name for the artifact
            extension (str, optional): File extension

        Returns:
            Tuple of (file_path, artifact_name)
        """
        # add extension to name
        if extension is not None:
            if not extension.startswith("."):
                extension = "." + extension
            if not name.endswith(extension):
                name += extension
        else:
            _, extension = os.path.splitext(name)

        # save artifact with temporary random id
        from random import choice
        from string import ascii_letters

        rnd = "".join([choice(ascii_letters) for _ in range(16)]) + extension

        # create artifact path
        os.makedirs(self.save_dir, exist_ok=True)
        id_path = os.path.join(self.save_dir, rnd)
        return id_path, name

    def log_artifact(
        self,
        filename: str,
        artifact_name: Optional[str] = None,
        delete_after: bool = False,
    ):
        """Log an artifact file to wandb.

        Args:
            filename (str): Path to the file to upload
            artifact_name (str, optional): Name for the artifact in wandb
            delete_after (bool): Whether to delete the file after upload
        """
        if artifact_name is None:
            # './dir/file.ext' -> 'file.ext'
            artifact_name = os.path.basename(filename)

        if hasattr(self.experiment, "log_artifact"):
            self.experiment.log_artifact(filename, name=artifact_name)
        elif hasattr(self.experiment, "save"):
            # Fallback to save method
            self.experiment.save(filename)

        if delete_after and os.path.exists(filename):
            os.unlink(filename)

    def log_numpy(self, array, name: str = "array"):
        """Log a numpy array object.

        Args:
            array (array_like): The array to be logged.
            name (str): The name of the file. (default: :obj:`'array'`)
        """
        path, name = self._artifact_storage_path(name, extension=".npy")
        np.save(path, array)
        self.log_artifact(path, artifact_name=name, delete_after=True)

    def log_dataframe(self, df: pd.DataFrame, name: str = "dataframe"):
        """Log a dataframe as csv.

        Args:
            df (DataFrame): The dataframe to be logged.
            name (str): The name of the file. (default: :obj:`'dataframe'`)
        """
        path, name = self._artifact_storage_path(name, extension=".csv")
        df.to_csv(path, index=True, index_label="index")
        self.log_artifact(path, artifact_name=name, delete_after=True)

    def log_figure(self, fig, name: str = "figure"):
        """Log a matplotlib figure as html.

        Args:
            fig (Figure): The matplotlib figure to be logged.
            name (str): The name of the file. (default: :obj:`'figure'`)
        """
        path, name = self._artifact_storage_path(name, extension=".html")
        save_figure(fig, path)
        self.log_artifact(path, artifact_name=name, delete_after=True)

    # OLD METHODS (for compatibility)

    def log_pred_df(self, name, idx, y, yhat, label_y="true", label_yhat="pred"):
        """Log a csv containing predictions and true values. Only works for
        univariate timeseries.

        Args:
            name: name of the file
            idx: dataframe idx
            y: true values
            yhat: predictions
            label_y: label for true values
            label_yhat: label for predictions
        """
        y = rearrange(y, "b ... -> b (...)")
        yhat = rearrange(yhat, "b ... -> b (...)")
        if isinstance(label_y, str):
            label_y = [f"{label_y}_{i}" for i in range(y.shape[-1])]
        if isinstance(label_yhat, str):
            label_yhat = [f"{label_yhat}_{i}" for i in range(yhat.shape[-1])]
        df = pd.DataFrame(
            data=np.concatenate([y, yhat], axis=-1),
            columns=label_y + label_yhat,
            index=idx,
        )

        # Create temporary file
        temp_path = os.path.join(self.save_dir, name)
        df.to_csv(temp_path, index=True, index_label="datetime")
        self.log_artifact(temp_path, artifact_name=name, delete_after=True)
