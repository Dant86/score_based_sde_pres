"""Denoising score matching loss (noise-prediction formulation)."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..sdes.base import SDE


def denoising_score_matching_loss(
    model: torch.nn.Module,
    sde: SDE,
    x0: Tensor,
) -> Tensor:
    """Compute E_{t,x_0,ε} [‖ε_θ(x_t, t) − ε‖²].

    Steps:
      1. Sample t ~ Uniform(eps, T)
      2. Sample ε ~ N(0, I)
      3. Compute x_t = mean(x_0, t) + std(t) · ε
      4. Return MSE between predicted and actual noise
    """
    device = x0.device
    batch_size = x0.shape[0]

    t = torch.rand(batch_size, device=device) * (sde.T - sde.eps) + sde.eps
    noise = torch.randn_like(x0)
    mean, std = sde.marginal_prob(x0, t)
    x_t = mean + std[:, None, None, None] * noise

    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)
