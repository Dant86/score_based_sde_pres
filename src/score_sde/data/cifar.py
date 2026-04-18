"""CIFAR-10 / CIFAR-100 DataLoader factory.

Images are normalised to [−1, 1] (mean=0.5, std=0.5 per channel).
"""

from typing import Literal

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_loaders(
    dataset: Literal["cifar10", "cifar100"] = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation DataLoaders for CIFAR-10 or CIFAR-100.

    Training split uses random horizontal flipping; val split does not.

    Args:
        dataset (Literal["cifar10", "cifar100"]): which CIFAR variant to load.
        data_dir (str): directory where data is stored or downloaded.
        batch_size (int): number of samples per batch.
        num_workers (int): number of subprocesses for data loading.
        image_size (int): spatial resolution to resize images to.

    Returns:
        tuple: train DataLoader and validation DataLoader, each yielding
            (image, label) batches with images normalised to [−1, 1].
    """
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]
    )
    val_transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor(), norm]
    )

    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_ds = cls(data_dir, train=True, download=True, transform=train_transform)
    val_ds = cls(data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
