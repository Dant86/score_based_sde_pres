"""Tests for ScoreNet: UNet output extraction and score conversion."""

import torch

from score_sde.models.score_net import ScoreNet
from score_sde.sdes.vp import VPSDE
from score_sde.sdes.ve import VESDE

B, C, H, W = 2, 3, 32, 32


def test_forward_shape() -> None:
    model = ScoreNet()
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    noise_pred = model(x, t)
    assert noise_pred.shape == (B, C, H, W), ".sample extraction must preserve spatial dims"


def test_forward_dtype() -> None:
    model = ScoreNet()
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    out = model(x, t)
    assert out.dtype == torch.float32


def test_score_fn_shape_vp() -> None:
    model = ScoreNet()
    sde = VPSDE()
    score_fn = model.as_score_fn(sde)
    x = torch.randn(B, C, H, W)
    t = torch.rand(B) * 0.9 + 0.05  # away from singularities
    score = score_fn(x, t)
    assert score.shape == (B, C, H, W)


def test_score_fn_shape_ve() -> None:
    model = ScoreNet()
    sde = VESDE()
    score_fn = model.as_score_fn(sde)
    x = torch.randn(B, C, H, W)
    t = torch.rand(B) * 0.9 + 0.05
    score = score_fn(x, t)
    assert score.shape == (B, C, H, W)


def test_no_attention_blocks() -> None:
    """Verify the UNet has no attention modules."""
    model = ScoreNet()
    attn_modules = [
        name
        for name, mod in model.named_modules()
        if "attention" in type(mod).__name__.lower()
    ]
    assert len(attn_modules) == 0, f"Found attention modules: {attn_modules}"
