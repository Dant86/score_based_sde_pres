"""Unit tests for all three SDEs."""

import pytest
import torch

from score_sde.sdes.base import SDE
from score_sde.sdes.subvp import SubVPSDE
from score_sde.sdes.ve import VESDE
from score_sde.sdes.vp import VPSDE

DEVICE = torch.device("cpu")
B, C, H, W = 4, 3, 32, 32


@pytest.fixture(params=["vp", "ve", "subvp"])
def sde(request: pytest.FixtureRequest) -> SDE:
    return {"vp": VPSDE(), "ve": VESDE(), "subvp": SubVPSDE()}[request.param]


def test_sde_output_shapes(sde: SDE) -> None:
    x = torch.randn(B, C, H, W)
    t = torch.rand(B) * sde.T
    drift, diffusion = sde.sde(x, t)
    assert drift.shape == (B, C, H, W), "drift must be (B,C,H,W)"
    assert diffusion.shape == (B,), "diffusion must be (B,)"


def test_marginal_prob_shapes(sde: SDE) -> None:
    x0 = torch.randn(B, C, H, W)
    t = torch.rand(B) * sde.T
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, C, H, W)
    assert std.shape == (B,)


def test_marginal_prob_at_t0_approaches_x0(sde: SDE) -> None:
    """At t ≈ 0, mean should be close to x0 for VP and Sub-VP."""
    if isinstance(sde, VESDE):
        pytest.skip("VE-SDE mean is always x0 — no useful test here")
    x0 = torch.randn(B, C, H, W)
    t = torch.full((B,), sde.eps)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.allclose(mean, x0, atol=1e-2), "mean at t≈0 should be close to x0"
    assert (std >= 0.0).all(), "std must be non-negative"


def test_prior_sampling_shape(sde: SDE) -> None:
    shape = (B, C, H, W)
    x = sde.prior_sampling(shape, DEVICE)
    assert x.shape == shape


def test_prior_logp_shape(sde: SDE) -> None:
    x = sde.prior_sampling((B, C, H, W), DEVICE)
    logp = sde.prior_logp(x)
    assert logp.shape == (B,)


def test_ve_marginal_mean_is_x0() -> None:
    sde = VESDE()
    x0 = torch.randn(B, C, H, W)
    t = torch.rand(B)
    mean, _ = sde.marginal_prob(x0, t)
    assert torch.allclose(mean, x0), "VE-SDE marginal mean must equal x0"


def test_subvp_std_less_than_vp_std() -> None:
    """Sub-VP standard deviation should be strictly less than VP at same t."""
    vp = VPSDE()
    subvp = SubVPSDE()
    x0 = torch.randn(B, C, H, W)
    t = torch.linspace(0.1, 0.9, B)
    _, std_vp = vp.marginal_prob(x0, t)
    _, std_subvp = subvp.marginal_prob(x0, t)
    assert (std_subvp < std_vp).all(), "Sub-VP std must be < VP std for all t"
