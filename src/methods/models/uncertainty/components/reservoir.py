"""Pure-torch port of the ResCP reservoir (ESN) layer.

Adapted from ``methods/upstream/rescp/reservoir_conformal_prediction/src/lib/nn/reservoir/reservoir.py``
with the following deviations only:
- ``tsl`` imports removed; activation lookup inlined.
- ``self_normalizing_activation`` dropped (unused by the sampling pipeline).
- API kept compatible: ``forward(x, h0, return_last_state)`` and ``get_states``.

Algorithmic behavior (state update rule, weight init, spectral radius scaling,
leak rate, input scaling, sparsity, bias) is unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


_ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": F.relu,
    "identity": lambda x: x,
}


class ReservoirLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spectral_radius: float,
        leaking_rate: float,
        bias: bool = True,
        density: float = 1.0,
        in_scaling: float = 1.0,
        bias_scale: float = 1.0,
        activation: str = "tanh",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = _ACTIVATIONS[activation]
        self.w_ih_scale = in_scaling
        self.b_scale = bias_scale
        self.density = density
        self.hidden_size = hidden_size
        self.alpha = leaking_rate
        self.spectral_radius = spectral_radius

        self.w_ih = nn.Parameter(torch.empty(hidden_size, input_size), requires_grad=False)
        self.w_hh = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=False)
        if bias:
            self.b_ih = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        else:
            self.register_parameter("b_ih", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_ih.data.bernoulli_(p=0.5).mul_(2.0).add_(-1.0)
        self.w_ih.data.mul_(self.w_ih_scale)

        if self.b_ih is not None:
            self.b_ih.data.bernoulli_(p=0.5).mul_(2.0).add_(-1.0)
            self.b_ih.data.mul_(self.b_scale)

        self.w_hh.data.uniform_(-0.5, 0.5)

        if self.density < 1:
            n_units = self.hidden_size * self.hidden_size
            mask = self.w_hh.data.new_ones(n_units)
            drop_idx = torch.randperm(n_units)[: int(n_units * (1 - self.density))]
            mask[drop_idx] = 0.0
            self.w_hh.data.mul_(mask.view(self.hidden_size, self.hidden_size))

        abs_eigs = torch.linalg.eigvals(self.w_hh.data).abs()
        self.w_hh.data.mul_(self.spectral_radius / torch.max(abs_eigs))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_new = self.activation(F.linear(x, self.w_ih, self.b_ih) + F.linear(h, self.w_hh))
        return (1 - self.alpha) * h + self.alpha * h_new


class Reservoir(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        input_scaling: float = 1.0,
        num_layers: int = 1,
        leaking_rate: float = 0.9,
        spectral_radius: float = 0.9,
        density: float = 0.9,
        activation: str = "tanh",
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            layers.append(
                ReservoirLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    in_scaling=input_scaling,
                    density=density,
                    activation=activation,
                    spectral_radius=spectral_radius,
                    leaking_rate=leaking_rate,
                    bias=bias,
                )
            )
        self.reservoir_layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.reservoir_layers:
            layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """``x`` is ``[b, s, n, f]``. By default returns the full state
        sequence ``[b, s, n, L*hidden]``.

        - ``return_last_state``: also return ``out[:, -1]`` and per-layer ``h``.
        - ``return_hidden``: return ``(full_sequence, per_layer_h_final)``.
          Useful for streaming where you want both the trajectory and the
          state needed to resume.
        """
        batch_size, steps, nodes, _ = x.size()
        if h0 is None:
            h0 = x.new_zeros(len(self.reservoir_layers), batch_size * nodes, self.hidden_size)
        x = rearrange(x, "b s n f -> s (b n) f")

        out = []
        h = h0
        for s in range(steps):
            x_s = x[s]
            h_s = []
            for i, layer in enumerate(self.reservoir_layers):
                x_s = layer(x_s, h[i])
                h_s.append(x_s)
            h = torch.stack(h_s)
            out.append(h)

        out = torch.stack(out)  # [s, l, b*n, f]
        out = rearrange(out, "s l (b n) f -> b s n (l f)", b=batch_size, n=nodes)
        if return_last_state:
            return out[:, -1], h
        if return_hidden:
            return out, h
        return out

    def get_states(self, x: torch.Tensor, bidir: bool = False, initial_state=None) -> torch.Tensor:
        """Backwards-compatible helper. ``x`` is ``[b, t, f]``.

        Returns ``[b, t, L*hidden]``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(2)  # b t f -> b t 1 f
        out = self.forward(x, h0=initial_state, return_last_state=False)
        return out.squeeze(2)
