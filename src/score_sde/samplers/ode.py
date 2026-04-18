"""Probability flow ODE sampler using Heun's method (2nd-order Runge-Kutta)."""

import torch
from torch import Tensor

from ..sdes.base import SDE, ScoreFn
from .base import Sampler


class ProbabilityFlowODE(Sampler):
    """Deterministic sampler via the probability flow ODE (Song et al. 2021, Eq. 13).

    ODE (reverse direction, t: T → 0):
        dx/dt = f(x,t) − ½ g(t)² s(x,t)

    Discretised with Heun's method (trapezoidal correction) for 2nd-order accuracy
    without external ODE solver dependencies.
    """

    def _drift(self, x: Tensor, t: Tensor, sde: SDE, score_fn: ScoreFn) -> Tensor:
        """Compute the probability-flow ODE drift negated for reverse-time integration.

        Args:
            x (Tensor): current sample tensor of shape (B, C, H, W).
            t (Tensor): current time values of shape (B,).
            sde (SDE): forward SDE providing drift and diffusion coefficients.
            score_fn (ScoreFn): s(x, t) ≈ ∇_x log p_t(x).

        Returns:
            Tensor: ODE drift tensor of shape (B, C, H, W).
        """
        f, g = sde.sde(x, t)
        score = score_fn(x, t)
        return f - 0.5 * g[:, None, None, None] ** 2 * score

    def sample(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        shape: tuple[int, ...],
        n_steps: int,
        device: torch.device,
    ) -> Tensor:
        x = sde.prior_sampling(shape, device)
        # n_steps intervals; timesteps[i] > timesteps[i+1] (decreasing)
        timesteps = torch.linspace(sde.T, sde.eps, n_steps + 1, device=device)

        for i in range(n_steps):
            t_curr = timesteps[i].expand(shape[0])
            t_next = timesteps[i + 1].expand(shape[0])
            dt = float(timesteps[i] - timesteps[i + 1])  # positive

            # Heun predictor
            d1 = self._drift(x, t_curr, sde, score_fn)
            x_pred = x - d1 * dt

            # Heun corrector (trapezoidal average)
            d2 = self._drift(x_pred, t_next, sde, score_fn)
            x = x - 0.5 * (d1 + d2) * dt

        return x.clamp(-1.0, 1.0)
