"""Predictor-Corrector sampler: Euler-Maruyama predictor + Langevin corrector."""

import math

import torch
from torch import Tensor

from ..sdes.base import SDE, ScoreFn
from .base import Sampler


class PredictorCorrector(Sampler):
    """PC sampler from Song et al. 2021 (Algorithm 2).

    Predictor: one Euler-Maruyama step.
    Corrector: `n_corrector` steps of annealed Langevin dynamics with SNR-based step size.
    """

    def __init__(self, n_corrector: int = 1, snr: float = 0.16) -> None:
        self.n_corrector = n_corrector
        self.snr = snr

    def _langevin_step(self, x: Tensor, t: Tensor, score_fn: ScoreFn) -> Tensor:
        """Perform one annealed Langevin corrector step.

        Args:
            x (Tensor): current sample tensor of shape (B, C, H, W).
            t (Tensor): current time values of shape (B,).
            score_fn (ScoreFn): s(x, t) ≈ ∇_x log p_t(x).

        Returns:
            Tensor: updated sample tensor of shape (B, C, H, W).
        """
        score = score_fn(x, t)
        noise = torch.randn_like(x)
        # Adaptive step size: α = (snr · ‖z‖ / ‖s‖)²·2
        b = x.shape[0]
        score_norm = score.reshape(b, -1).norm(dim=-1).mean()
        noise_norm = noise.reshape(b, -1).norm(dim=-1).mean()
        step_size = (self.snr * noise_norm / (score_norm + 1e-8)) ** 2 * 2
        return x + step_size * score + math.sqrt(2.0 * float(step_size)) * noise

    def sample(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        shape: tuple[int, ...],
        n_steps: int,
        device: torch.device,
    ) -> Tensor:
        x = sde.prior_sampling(shape, device)
        timesteps = torch.linspace(sde.T, sde.eps, n_steps + 1, device=device)

        for i in range(n_steps):
            t_val = timesteps[i]
            dt = float(timesteps[i] - timesteps[i + 1])
            t = t_val.expand(shape[0])

            # Predictor (Euler-Maruyama)
            drift, diffusion = sde.sde(x, t)
            score = score_fn(x, t)
            g = diffusion[:, None, None, None]
            reverse_drift = drift - g**2 * score
            z = torch.randn_like(x)
            x = x - reverse_drift * dt + g * math.sqrt(dt) * z

            # Corrector (Langevin)
            for _ in range(self.n_corrector):
                x = self._langevin_step(x, t, score_fn)

        return x.clamp(-1.0, 1.0)
