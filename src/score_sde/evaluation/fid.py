"""FID computation via torchmetrics.

Both real and generated images are expected in [−1, 1] (the training normalisation).
FrechetInceptionDistance is initialised with normalize=True, which accepts floats
in [0, 1] — so we rescale once before calling .update().
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(
    real_loader: DataLoader,
    fake_images: Tensor,
    device: torch.device,
    n_real: int = 5000,
) -> float:
    """Compute FID between real CIFAR-100 images and generated images.

    Args:
        real_loader:  DataLoader yielding (image, label) with images in [−1, 1]
        fake_images:  Tensor (N, C, H, W) in [−1, 1], generated images
        device:       compute device
        n_real:       how many real images to use (capped by dataset size)

    Returns:
        FID score (float, lower is better).
    """
    fid: FrechetInceptionDistance = FrechetInceptionDistance(normalize=True).to(device)

    # Feed real images
    seen = 0
    for batch in real_loader:
        imgs: Tensor = batch[0].to(device)
        fid.update((imgs + 1.0) / 2.0, real=True)
        seen += imgs.shape[0]
        if seen >= n_real:
            break

    # Feed fake images
    fake_ds = TensorDataset(fake_images)
    fake_loader = DataLoader(fake_ds, batch_size=64)
    for (batch,) in fake_loader:
        fid.update((batch.to(device) + 1.0) / 2.0, real=False)

    return float(fid.compute())
