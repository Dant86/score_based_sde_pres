#!/usr/bin/env python
"""Offline evaluation script — loads a frozen checkpoint, computes FID."""

import argparse
import os

import torch

from score_sde.config import EvalConfig, ModelConfig, SDEConfig
from score_sde.data.cifar100 import get_cifar100_loaders
from score_sde.evaluation.fid import compute_fid
from score_sde.evaluation.sampler_runner import generate_samples
from score_sde.guidance.classifier import NoisyClassifier, make_guided_score_fn
from score_sde.models.score_net import ScoreNet
from score_sde.samplers.base import Sampler
from score_sde.samplers.euler_maruyama import EulerMaruyama
from score_sde.samplers.ode import ProbabilityFlowODE
from score_sde.samplers.predictor_corrector import PredictorCorrector
from score_sde.sdes.base import SDE, ScoreFn
from score_sde.sdes.subvp import SubVPSDE
from score_sde.sdes.ve import VESDE
from score_sde.sdes.vp import VPSDE


def build_sde(sde_type: str) -> SDE:
    match sde_type:
        case "vp":
            return VPSDE()
        case "ve":
            return VESDE()
        case "subvp":
            return SubVPSDE()
        case _:
            raise ValueError(f"Unknown SDE: {sde_type!r}")


def build_sampler(sampler_type: str) -> Sampler:
    match sampler_type:
        case "euler_maruyama":
            return EulerMaruyama()
        case "predictor_corrector":
            return PredictorCorrector()
        case "ode":
            return ProbabilityFlowODE()
        case _:
            raise ValueError(f"Unknown sampler: {sampler_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen score-SDE checkpoint.")
    parser.add_argument("--sde", required=True, choices=["vp", "ve", "subvp"])
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--sampler",
        default="euler_maruyama",
        choices=["euler_maruyama", "predictor_corrector", "ode"],
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./eval_output")
    # Classifier guidance
    parser.add_argument("--use-guidance", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--classifier-path", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load score network
    model_cfg = ModelConfig()
    model = ScoreNet(
        sample_size=model_cfg.sample_size,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        block_out_channels=model_cfg.block_out_channels,
        layers_per_block=model_cfg.layers_per_block,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    sde = build_sde(args.sde)
    sampler = build_sampler(args.sampler)

    score_fn: ScoreFn = model.as_score_fn(sde)

    if args.use_guidance and args.classifier_path:
        classifier = NoisyClassifier().to(device)
        cls_ckpt = torch.load(args.classifier_path, map_location=device, weights_only=False)
        classifier.load_state_dict(cls_ckpt)
        classifier.eval()
        score_fn = make_guided_score_fn(score_fn, classifier, args.target_class, args.guidance_scale)
        print(f"Classifier guidance: scale={args.guidance_scale}, class={args.target_class}")

    print(f"Generating {args.n_samples} samples  |  sampler={args.sampler}  |  n_steps={args.n_steps}")
    fake_images = generate_samples(
        sde=sde,
        score_fn=score_fn,
        sampler=sampler,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=device,
    )

    out_path = os.path.join(args.output_dir, f"{args.sde}_{args.sampler}_samples.pt")
    torch.save(fake_images, out_path)
    print(f"Samples saved: {out_path}")

    _, val_loader = get_cifar100_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
    fid_score = compute_fid(val_loader, fake_images, device, n_real=args.n_samples)
    print(f"FID ({args.sde} / {args.sampler}): {fid_score:.2f}")

    cfg = EvalConfig(
        sde=SDEConfig(sde_type=args.sde),
        checkpoint_path=args.ckpt,
        sampler=args.sampler,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
    )
    print(f"Config: {cfg}")


if __name__ == "__main__":
    main()
