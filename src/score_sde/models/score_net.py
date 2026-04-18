"""ScoreNet: wraps UNet2DModel and exposes both a noise-prediction forward pass
and a score function s(x,t) = −noise_pred / σ(t)."""

import torch
import torch.nn as nn
from diffusers import UNet2DModel
from torch import Tensor

from ..sdes.base import SDE, ScoreFn


class ScoreNet(nn.Module):
    """Noise-prediction network backed by a diffusers UNet2DModel.

    All attention is disabled by using plain DownBlock2D / UpBlock2D blocks.
    """

    def __init__(
        self,
        sample_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 256),
        layers_per_block: int = 2,
    ) -> None:
        super().__init__()
        n_stages = len(block_out_channels)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            # No attention in any block
            down_block_types=("DownBlock2D",) * n_stages,
            up_block_types=("UpBlock2D",) * n_stages,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict noise ε given noisy image x and continuous time t ∈ [0, 1].

        UNet2DModel expects integer timesteps in [0, 999]; we scale accordingly.
        Returns noise_pred of shape (B, C, H, W).
        """
        t_unet = (t * 999.0).long()
        return self.unet(x, t_unet).sample  # .sample extracts Tensor from UNet2DOutput

    def as_score_fn(self, sde: SDE) -> ScoreFn:
        """Return s(x,t) = −ε_θ(x,t) / σ(t), the approximate score function."""

        def score_fn(x: Tensor, t: Tensor) -> Tensor:
            _, std = sde.marginal_prob(x, t)
            noise_pred = self.forward(x, t)
            return -noise_pred / std[:, None, None, None]

        return score_fn
