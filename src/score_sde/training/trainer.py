"""Training loop with checkpointing and resume support."""

import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig
from ..models.score_net import ScoreNet
from ..sdes.base import SDE
from .losses import denoising_score_matching_loss
from .param_groups import get_param_groups


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: ScoreNet,
        sde: SDE,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.sde = sde
        self.train_loader = train_loader
        self.device = device
        self.start_epoch = 0

        param_groups = get_param_groups(model, config.weight_decay)
        self.optimizer = AdamW(param_groups, lr=config.lr, eps=config.eps)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ---------------------------------------------------------------- checkpoints
    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {path} (epoch {ckpt['epoch']})")

    def _save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        return path

    # ---------------------------------------------------------------- training
    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            for batch in pbar:
                x0: torch.Tensor = batch[0].to(self.device)
                self.optimizer.zero_grad()
                loss = denoising_score_matching_loss(self.model, self.sde, x0)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}: avg_loss={avg:.4f}")

            if (epoch + 1) % self.config.checkpoint_every == 0:
                path = self._save_checkpoint(epoch)
                print(f"  → checkpoint: {path}")

        # Always save final checkpoint
        final = self._save_checkpoint(self.config.epochs - 1)
        print(f"Training complete. Final checkpoint: {final}")
