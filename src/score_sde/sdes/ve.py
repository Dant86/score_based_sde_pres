"""VE-SDE: dx = √(dσ²/dt) dw,  σ(t) = σ_min (σ_max/σ_min)^t  (Song et al. 2021, Eq. 9)."""

import math

import torch
from torch import Tensor

from .base import SDE


class VESDE(SDE):
    """Variance-Exploding SDE with geometric noise schedule σ(t)."""

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        T: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self._log_ratio = math.log(sigma_max / sigma_min)

    def _sigma(self, t: Tensor) -> Tensor:
        """Compute σ(t) = σ_min (σ_max/σ_min)^t.

        Args:
            t (Tensor): continuous time values of shape (B,).

        Returns:
            Tensor: noise standard deviation values of shape (B,).
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        sigma = self._sigma(t)
        drift = torch.zeros_like(x)
        # g(t) = σ(t) √(2 log(σ_max/σ_min))
        diffusion = sigma * math.sqrt(2.0 * self._log_ratio)
        return drift, diffusion

    def marginal_prob(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        std = self._sigma(t)
        mean = x0.clone()
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randn(*shape, device=device) * self.sigma_max

    def prior_logp(self, z: Tensor) -> Tensor:
        n = z[0].numel()
        sigma_max_sq = self.sigma_max**2
        return -0.5 * (
            torch.sum(z**2, dim=(1, 2, 3)) / sigma_max_sq
            + n * math.log(2 * math.pi * sigma_max_sq)
        )
