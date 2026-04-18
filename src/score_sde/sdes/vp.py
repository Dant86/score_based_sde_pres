"""VP-SDE: dx = -½ β(t) x dt + √β(t) dw  (Song et al. 2021, Eq. 11)."""

import math

import torch
from torch import Tensor

from .base import SDE


class VPSDE(SDE):
    """Variance-Preserving SDE with linear noise schedule β(t)."""

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    # ------------------------------------------------------------------ helpers
    def _beta(self, t: Tensor) -> Tensor:
        """β(t) = β_min + t(β_max − β_min).  Shape: (B,)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: Tensor) -> Tensor:
        """log α(t) = −¼ t²(β_max−β_min) − ½ t β_min.  Shape: (B,)."""
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    # ------------------------------------------------------------------ SDE API
    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        beta = self._beta(t)
        drift = -0.5 * beta[:, None, None, None] * x
        diffusion = torch.sqrt(beta)
        return drift, diffusion

    def marginal_prob(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        log_mean_coeff = self._log_mean_coeff(t)
        mean = x0 * torch.exp(log_mean_coeff)[:, None, None, None]
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randn(*shape, device=device)

    def prior_logp(self, z: Tensor) -> Tensor:
        n = z[0].numel()
        return -0.5 * (torch.sum(z**2, dim=(1, 2, 3)) + n * math.log(2 * math.pi))
