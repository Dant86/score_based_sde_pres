"""Noisy classifier for classifier guidance (Song et al. 2021, §4.2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..sdes.base import ScoreFn


class NoisyClassifier(nn.Module):
    """Lightweight classifier trained on noisy images x_t at random t.

    Architecture: three strided conv blocks + adaptive pool + linear head.
    """

    def __init__(self, num_classes: int = 100, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 → 16
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 → 8
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def make_guided_score_fn(
    score_fn: ScoreFn,
    classifier: NoisyClassifier,
    label: int,
    guidance_scale: float,
) -> ScoreFn:
    """Wrap a base score function with classifier guidance.

    Guided score: s̃(x, t) = s(x, t) + γ · ∇_x log p(y | x_t).

    Args:
        score_fn (ScoreFn): base unconditional score function.
        classifier (NoisyClassifier): noisy classifier that must be in eval mode.
        label (int): target class index.
        guidance_scale (float): γ, strength of the classifier guidance.

    Returns:
        ScoreFn: guided score function mapping (x, t) to guided score estimates.
    """

    def guided(x: Tensor, t: Tensor) -> Tensor:
        x_req = x.detach().requires_grad_(True)
        logits = classifier(x_req)
        log_prob = F.log_softmax(logits, dim=-1)
        target = torch.full((x.shape[0],), label, dtype=torch.long, device=x.device)
        loss = F.nll_loss(log_prob, target)
        (grad,) = torch.autograd.grad(loss, x_req)
        base = score_fn(x.detach(), t)
        # Gradient of log p(y|x) points toward higher class probability
        return base - guidance_scale * grad.detach()

    return guided
