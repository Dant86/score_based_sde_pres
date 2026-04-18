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
    """Compute the denoising score matching loss E_{t,x_0,ε}[‖ε_θ(x_t, t) − ε‖²].

    Samples t uniformly from (eps, T), draws ε ~ N(0, I), forms x_t via
    the SDE marginal, and returns the MSE between predicted and actual noise.

    Args:
        model (torch.nn.Module): noise-prediction network ε_θ(x_t, t).
        sde (SDE): forward SDE providing marginal distribution parameters.
        x0 (Tensor): clean image batch of shape (B, C, H, W).

    Returns:
        Tensor: scalar MSE loss averaged over the batch.
    """
    device = x0.device
    batch_size = x0.shape[0]

    t = torch.rand(batch_size, device=device) * (sde.T - sde.eps) + sde.eps
    noise = torch.randn_like(x0)
    mean, std = sde.marginal_prob(x0, t)
    x_t = mean + std[:, None, None, None] * noise

    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)
