from einops import repeat
from typing import Optional

from einops import rearrange
from torch import Tensor, nn

from src.lib.nn.reservoir import Reservoir

from tsl import logger


class ReservoirEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rec_layers,
        activation="tanh",
        spectral_radius=0.9,
        leaking_rate=0.9,
        input_scaling=1.0,
        density=0.2,
        alpha_decay=False,
        bias=False,
    ):
        super(ReservoirEncoder, self).__init__()
        self.reservoir = Reservoir(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rec_layers,
            leaking_rate=leaking_rate,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            density=density,
            activation=activation,
            alpha_decay=alpha_decay,
            bias=bias,
        )

    def encode(self, x, *args, **kwargs) -> Tensor:
        """"""
        x = self.reservoir(x, *args, **kwargs)

        return x

    def forward(self, x: Tensor, *args, **kwargs):
        # x : [t n f]
        x = rearrange(x, "t n f -> 1 t n f")
        x = self.encode(x, *args, **kwargs)
        x = x[0]
        return x
