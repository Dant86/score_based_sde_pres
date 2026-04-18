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
            sde (SDE): forward SDE providing prior_sampling and sde coefficients.
            score_fn (ScoreFn): s(x, t) ≈ ∇_x log p_t(x).
            shape (tuple[int, ...]): desired output shape, e.g. (B, C, H, W).
            n_steps (int): number of discretisation steps.
            device (torch.device): target device.

        Returns:
            Tensor: samples of shape ``shape``, values clipped to [−1, 1].
        """
