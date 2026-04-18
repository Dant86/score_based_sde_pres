#!/usr/bin/env python
"""Local training script — single SDE type per invocation."""

import argparse

import torch

from score_sde.config import ModelConfig, SDEConfig, TrainConfig
from score_sde.data.cifar100 import get_cifar100_loaders
from score_sde.models.score_net import ScoreNet
from score_sde.sdes.base import SDE
from score_sde.sdes.subvp import SubVPSDE
from score_sde.sdes.ve import VESDE
from score_sde.sdes.vp import VPSDE
from score_sde.training.trainer import Trainer


def build_sde(cfg: SDEConfig) -> SDE:
    match cfg.sde_type:
        case "vp":
            return VPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, T=cfg.T)
        case "ve":
            return VESDE(sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max, T=cfg.T)
        case "subvp":
            return SubVPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, T=cfg.T)
        case _:
            raise ValueError(f"Unknown SDE type: {cfg.sde_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a score-based SDE model on CIFAR-100.")
    parser.add_argument("--sde", default="vp", choices=["vp", "ve", "subvp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  SDE: {args.sde}")

    sde_cfg = SDEConfig(sde_type=args.sde)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        sde=sde_cfg,
        model=model_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=f"{args.checkpoint_dir}/{args.sde}",
        resume_from=args.resume_from,
        data_dir=args.data_dir,
    )

    sde = build_sde(sde_cfg)
    model = ScoreNet(
        sample_size=model_cfg.sample_size,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        block_out_channels=model_cfg.block_out_channels,
        layers_per_block=model_cfg.layers_per_block,
    )

    train_loader, _ = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = Trainer(train_cfg, model, sde, train_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()
