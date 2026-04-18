"""Smoke tests for all three samplers — shape and range checks only."""

import pytest
import torch
from torch import Tensor

from score_sde.samplers.base import Sampler
from score_sde.samplers.euler_maruyama import EulerMaruyama
from score_sde.samplers.ode import ProbabilityFlowODE
from score_sde.samplers.predictor_corrector import PredictorCorrector
from score_sde.sdes.vp import VPSDE

DEVICE = torch.device("cpu")
SHAPE = (2, 3, 32, 32)
N_STEPS = 5  # tiny for speed


def _zero_score(x: Tensor, t: Tensor) -> Tensor:
    """Dummy score function that returns zeros — enough to test sampler mechanics."""
    return torch.zeros_like(x)


@pytest.fixture(
    params=["euler_maruyama", "predictor_corrector", "ode"],
)
def sampler(request: pytest.FixtureRequest) -> Sampler:
    return {
        "euler_maruyama": EulerMaruyama(),
        "predictor_corrector": PredictorCorrector(n_corrector=1),
        "ode": ProbabilityFlowODE(),
    }[request.param]


def test_sampler_output_shape(sampler: Sampler) -> None:
    sde = VPSDE()
    out = sampler.sample(sde, _zero_score, SHAPE, N_STEPS, DEVICE)
    assert out.shape == SHAPE, f"Expected {SHAPE}, got {out.shape}"


def test_sampler_output_clipped(sampler: Sampler) -> None:
    sde = VPSDE()
    out = sampler.sample(sde, _zero_score, SHAPE, N_STEPS, DEVICE)
    assert out.min() >= -1.0, "Output must be >= -1"
    assert out.max() <= 1.0, "Output must be <= 1"


def test_sampler_deterministic_with_seed(sampler: Sampler) -> None:
    """ODE sampler should be fully deterministic; stochastic ones should differ."""
    sde = VPSDE()
    torch.manual_seed(0)
    out1 = sampler.sample(sde, _zero_score, SHAPE, N_STEPS, DEVICE)
    torch.manual_seed(0)
    out2 = sampler.sample(sde, _zero_score, SHAPE, N_STEPS, DEVICE)
    # With same seed, outputs must match
    assert torch.allclose(out1, out2), "Same seed must give identical output"
