"""Batched sample generation using any Sampler."""

import torch
from torch import Tensor

from ..samplers.base import Sampler
from ..sdes.base import SDE, ScoreFn


def generate_samples(
    sde: SDE,
    score_fn: ScoreFn,
    sampler: Sampler,
    n_samples: int,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    image_shape: tuple[int, int, int] = (3, 32, 32),
) -> Tensor:
    """Generate exactly `n_samples` images, accumulating them on CPU.

    Returns:
        Tensor of shape (n_samples, *image_shape) with values in [−1, 1].
    """
    buckets: list[Tensor] = []
    remaining = n_samples

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            samples = sampler.sample(sde, score_fn, (bs, *image_shape), n_steps, device)
            buckets.append(samples.cpu())
            remaining -= bs

    return torch.cat(buckets, dim=0)[:n_samples]
