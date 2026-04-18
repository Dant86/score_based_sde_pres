"""Euler-Maruyama discretisation of the reverse-time SDE."""

import math

import torch
from torch import Tensor

from ..sdes.base import SDE, ScoreFn
from .base import Sampler


class EulerMaruyama(Sampler):
    """First-order stochastic sampler.

    Reverse-time update:
        x_{t-dt} = x_t − [f(x_t,t) − g(t)² s(x_t,t)] dt + g(t) √dt z
    """

    def sample(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        shape: tuple[int, ...],
        n_steps: int,
        device: torch.device,
    ) -> Tensor:
        x = sde.prior_sampling(shape, device)
        # n_steps+1 endpoints → n_steps intervals of equal size
        timesteps = torch.linspace(sde.T, sde.eps, n_steps + 1, device=device)

        for i in range(n_steps):
            t_val = timesteps[i]
            dt = float(timesteps[i] - timesteps[i + 1])
            t = t_val.expand(shape[0])

            drift, diffusion = sde.sde(x, t)
            score = score_fn(x, t)
            g = diffusion[:, None, None, None]
            reverse_drift = drift - g**2 * score

            z = torch.randn_like(x)
            x = x - reverse_drift * dt + g * math.sqrt(dt) * z

        return x.clamp(-1.0, 1.0)
