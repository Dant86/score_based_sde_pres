from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import Tensor


class ScoreFn(Protocol):
    """Score function s(x, t) ≈ ∇_x log p_t(x)."""

    def __call__(self, x: Tensor, t: Tensor) -> Tensor: ...


class SDE(ABC):
    """Abstract base class for Itô SDEs of the form dx = f(x,t)dt + g(t)dw."""

    T: float = 1.0
    eps: float = 1e-5  # small t floor to avoid singularities at t=0

    @abstractmethod
    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Forward SDE coefficients.

        Returns:
            drift:     f(x, t)  shape (B, C, H, W)
            diffusion: g(t)     shape (B,)
        """

    @abstractmethod
    def marginal_prob(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Parameters of p(x_t | x_0) = N(x_t; mean, std² I).

        Returns:
            mean: shape (B, C, H, W)
            std:  shape (B,)
        """

    @abstractmethod
    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample x_T ~ p_T (the prior)."""

    @abstractmethod
    def prior_logp(self, z: Tensor) -> Tensor:
        """Log-density of the prior p_T. Shape: (B,)."""
