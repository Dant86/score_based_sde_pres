"""Training loop for the NoisyClassifier used in classifier guidance."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..guidance.classifier import NoisyClassifier
from ..sdes.base import SDE


def train_noisy_classifier(
    sde: SDE,
    train_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/classifier.pt",
) -> NoisyClassifier:
    """Train a NoisyClassifier on CIFAR-100 images corrupted by the forward SDE.

    For each batch the procedure is:

    1. Sample ``t ~ Uniform(eps, T)``.
    2. Corrupt ``x_0`` to ``x_t`` using the SDE marginal distribution.
    3. Compute cross-entropy loss between predicted and true class labels.

    This produces a classifier that can assign class probabilities at any
    noise level, enabling classifier guidance during sampling.

    Args:
        sde (SDE): forward SDE whose marginal is used to corrupt images.
        train_loader (DataLoader): CIFAR-100 training loader yielding
            ``(image, label)`` batches with images in [−1, 1].
        device (torch.device): target device for training.
        n_epochs (int): number of full passes over the training set.
        lr (float): AdamW learning rate.
        checkpoint_path (str): where to save the final classifier weights.

    Returns:
        NoisyClassifier: trained classifier moved to ``device``.
    """
    classifier = NoisyClassifier(num_classes=100).to(device)
    optimizer = AdamW(classifier.parameters(), lr=lr)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    for epoch in range(n_epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Classifier epoch {epoch + 1}/{n_epochs}")

        for x0, labels in pbar:
            x0 = x0.to(device)
            labels = labels.to(device)
            batch_size = x0.shape[0]

            # Sample random noise level and corrupt x0
            t = torch.rand(batch_size, device=device) * (sde.T - sde.eps) + sde.eps
            noise = torch.randn_like(x0)
            mean, std = sde.marginal_prob(x0, t)
            x_t = mean + std[:, None, None, None] * noise

            optimizer.zero_grad()
            logits = classifier(x_t)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += batch_size
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}  acc={acc:.4f}")

    torch.save(classifier.state_dict(), checkpoint_path)
    print(f"Classifier saved: {checkpoint_path}")
    return classifier
