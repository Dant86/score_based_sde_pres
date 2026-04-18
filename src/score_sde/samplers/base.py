from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..sdes.base import SDE, ScoreFn


class Sampler(ABC):
    """Abstract reverse-time sampler.

    Samplers depend only on the SDE and a ScoreFn — they are agnostic to
    how the score is parameterised (ScoreNet, guided, etc.).
    """

    @abstractmethod
    def sample(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        shape: tuple[int, ...],
        n_steps: int,
        device: torch.device,
    ) -> Tensor:
        """Draw samples by running the reverse SDE / ODE from t=T to t=eps.

        Args:
            sde:       forward SDE (provides prior_sampling, sde coefficients)
            score_fn:  s(x,t) ≈ ∇_x log p_t(x)
            shape:     (B, C, H, W)
            n_steps:   number of discretisation steps
            device:    target device

        Returns:
            Tensor of shape `shape`, values clipped to [−1, 1].
        """
