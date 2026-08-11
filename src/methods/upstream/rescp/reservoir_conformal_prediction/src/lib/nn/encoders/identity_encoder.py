from typing import Optional

from torch import Tensor, nn

from src.lib.nn.encoders.base_encoder import InputEncoder


class IdentityEncoder(InputEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *args,
        **kwargs,
    ):

        super(IdentityEncoder, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            emb_size=kwargs["emb_size"] if "emb_size" in kwargs else 0,
            n_instances=kwargs["n_instances"] if "n_instances" in kwargs else None,
        )
        self.encoder = nn.Identity()

    def encode(self, x) -> Tensor:
        return self.encoder(x)
