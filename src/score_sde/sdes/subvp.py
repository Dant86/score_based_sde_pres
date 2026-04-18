"""Sub-VP-SDE: dx = -½ β(t) x dt + √(β(t)(1−e^{-2∫β})) dw  (Song et al. 2021, Eq. 11)."""

import math

import torch
from torch import Tensor

from .base import SDE


class SubVPSDE(SDE):
    """Sub-Variance-Preserving SDE — smaller diffusion coefficient than VP.

    Marginal std is σ̃(t) = 1 − exp(−∫₀ᵗ β(s)ds), compared to VP's
    √(1 − exp(−∫β)).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def _beta(self, t: Tensor) -> Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: Tensor) -> Tensor:
        """Compute log α(t) = −½ ∫₀ᵗ β(s)ds.

        Args:
            t (Tensor): continuous time values of shape (B,).

        Returns:
            Tensor: log of the mean coefficient of shape (B,).
        """
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        beta = self._beta(t)
        log_mean_coeff = self._log_mean_coeff(t)
        # 1 − e^{−2∫β} = 1 − exp(4 · log_mean_coeff)
        discount = 1.0 - torch.exp(4.0 * log_mean_coeff)
        drift = -0.5 * beta[:, None, None, None] * x
        diffusion = torch.sqrt(beta * discount.clamp(min=0.0))
        return drift, diffusion

    def marginal_prob(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        log_mean_coeff = self._log_mean_coeff(t)
        mean = x0 * torch.exp(log_mean_coeff)[:, None, None, None]
        # σ̃(t) = 1 − exp(−∫β) = 1 − exp(2 · log_mean_coeff)
        std = 1.0 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randn(*shape, device=device)

    def prior_logp(self, z: Tensor) -> Tensor:
        n = z[0].numel()
        return -0.5 * (torch.sum(z**2, dim=(1, 2, 3)) + n * math.log(2 * math.pi))
