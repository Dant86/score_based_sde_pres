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
        """Return forward SDE coefficients f(x, t) and g(t).

        Args:
            x (Tensor): noisy image tensor of shape (B, C, H, W).
            t (Tensor): continuous time values of shape (B,).

        Returns:
            tuple: drift f(x, t) of shape (B, C, H, W) and diffusion g(t) of shape (B,).
        """

    @abstractmethod
    def marginal_prob(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return parameters of p(x_t | x_0) = N(x_t; mean, std² I).

        Args:
            x0 (Tensor): clean image tensor of shape (B, C, H, W).
            t (Tensor): continuous time values of shape (B,).

        Returns:
            tuple: mean of shape (B, C, H, W) and std of shape (B,).
        """

    @abstractmethod
    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample x_T from the prior p_T.

        Args:
            shape (tuple[int, ...]): desired output shape, e.g. (B, C, H, W).
            device (torch.device): target device for the sample.

        Returns:
            Tensor: sample from the prior distribution.
        """

    @abstractmethod
    def prior_logp(self, z: Tensor) -> Tensor:
        """Compute the log-density of the prior p_T.

        Args:
            z (Tensor): sample tensor of shape (B, C, H, W).

        Returns:
            Tensor: per-sample log-probabilities of shape (B,).
        """
