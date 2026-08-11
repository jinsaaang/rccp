from __future__ import annotations

from models.uncertainty.pi_eps_pred_hopfield import EpsPredictionHopfield


class HopCPTModel(EpsPredictionHopfield):
    """Paper-facing HopCPT alias.

    This class intentionally does not override any method logic. It exists so
    benchmark configs can refer to `hopcpt` without exposing the legacy
    `eps_pred_hopfield_*` experiment name.
    """

    pass
